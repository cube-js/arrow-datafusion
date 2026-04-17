// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Stream wrappers for physical operators

use std::cell::Cell;
use std::pin::Pin;
use std::sync::{Arc, Once};
use std::task::{Context, Poll};

use crate::error::DataFusionError;
use crate::execution::context::TaskContext;
use arrow::{
    datatypes::SchemaRef,
    error::{ArrowError, Result as ArrowResult},
    record_batch::RecordBatch,
};
use futures::stream::BoxStream;
use futures::{Future, Stream, StreamExt};
use log::debug;
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::task::{JoinHandle, JoinSet};
use tokio_stream::wrappers::ReceiverStream;

use super::common::AbortOnDropSingle;
use super::displayable;
use super::metrics::BaselineMetrics;
use super::{ExecutionPlan, RecordBatchStream, SendableRecordBatchStream};

thread_local! {
    /// When set, the installed panic hook suppresses the default
    /// stderr output for panics that happen on this thread. The flag
    /// is scoped to the lifetime of a single poll/call of a task
    /// spawned through the builder: panics in that code are already
    /// captured by tokio and re-raised on the consumer thread, so
    /// printing them a second time to stderr is only noise.
    static SUPPRESS_PANIC_OUTPUT: Cell<bool> = const { Cell::new(false) };
}

static PANIC_HOOK_ONCE: Once = Once::new();

/// Install a process-wide panic hook that silences the default stderr
/// output only while a DataFusion-spawned task is running on the
/// current thread. Panics outside those tasks keep the previous hook.
fn install_panic_hook_once() {
    PANIC_HOOK_ONCE.call_once(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if !SUPPRESS_PANIC_OUTPUT.with(|c| c.get()) {
                prev(info);
            }
        }));
    });
}

/// Guard that sets the thread-local flag on creation and clears it on
/// drop, so the flag is restored even when the wrapped code panics.
struct SuppressPanicGuard {
    prev: bool,
}

impl SuppressPanicGuard {
    fn new() -> Self {
        let prev = SUPPRESS_PANIC_OUTPUT.with(|c| c.replace(true));
        Self { prev }
    }
}

impl Drop for SuppressPanicGuard {
    fn drop(&mut self) {
        SUPPRESS_PANIC_OUTPUT.with(|c| c.set(self.prev));
    }
}

pin_project_lite::pin_project! {
    /// Future wrapper that activates the panic-hook suppression flag
    /// around every poll of the inner future. Re-applying on every
    /// poll is required because tokio may migrate the task to a
    /// different worker thread across `.await` points.
    struct SuppressPanicOutput<F> {
        #[pin]
        inner: F,
    }
}

impl<F: Future> Future for SuppressPanicOutput<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<F::Output> {
        let _guard = SuppressPanicGuard::new();
        self.project().inner.poll(cx)
    }
}

/// Builder for [`RecordBatchReceiverStream`] that propagates errors
/// and panics from spawned tasks to the consumer.
///
/// [`RecordBatchReceiverStream`] is used to spawn one or more tasks
/// that produce `RecordBatch`es and send them to a single
/// `Receiver` which can improve parallelism. Previously, panics in
/// those tasks were silently dropped; this builder uses a
/// [`JoinSet`] so that panics are re-raised on the consumer thread
/// and outstanding tasks are aborted when the stream is dropped.
pub struct RecordBatchReceiverStreamBuilder {
    tx: Sender<ArrowResult<RecordBatch>>,
    rx: Receiver<ArrowResult<RecordBatch>>,
    schema: SchemaRef,
    join_set: JoinSet<()>,
}

impl RecordBatchReceiverStreamBuilder {
    /// Create a new builder with an internal buffer of `capacity` batches.
    pub fn new(schema: SchemaRef, capacity: usize) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(capacity);

        Self {
            tx,
            rx,
            schema,
            join_set: JoinSet::new(),
        }
    }

    /// Get a handle for sending [`RecordBatch`]es to the output.
    pub fn tx(&self) -> Sender<ArrowResult<RecordBatch>> {
        self.tx.clone()
    }

    /// Spawn a task that will be aborted if this builder (or the
    /// stream built from it) is dropped.
    ///
    /// Often used to spawn tasks that write to the sender returned
    /// by [`Self::tx`]. Panics inside the task are captured and
    /// re-raised on the consumer thread; they are also silenced
    /// from the default panic hook so the panic is not printed
    /// twice to stderr.
    pub fn spawn<F>(&mut self, task: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        install_panic_hook_once();
        self.join_set.spawn(SuppressPanicOutput { inner: task });
    }

    /// Spawn a blocking task that will be aborted if this builder
    /// (or the stream built from it) is dropped.
    ///
    /// Often used to spawn tasks that write to the sender returned
    /// by [`Self::tx`]. Panics are propagated to the consumer and
    /// silenced from the default panic hook.
    pub fn spawn_blocking<F>(&mut self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        install_panic_hook_once();
        // `JoinSet::spawn_blocking` was added in tokio 1.24; this
        // crate is on 1.22, so we spawn the blocking job via the
        // free function and forward its JoinError into the JoinSet
        // by re-raising any panic inside an adopted async task.
        let handle = tokio::task::spawn_blocking(move || {
            let _guard = SuppressPanicGuard::new();
            f();
        });
        self.join_set.spawn(async move {
            match handle.await {
                Ok(()) => {}
                Err(e) => {
                    if e.is_panic() {
                        std::panic::resume_unwind(e.into_panic());
                    }
                    // Cancellation of the blocking task is only
                    // possible via abort of this adopter task,
                    // which implies the stream is being dropped.
                }
            }
        });
    }

    /// Run a partition of the given `input` [`ExecutionPlan`] on the
    /// tokio threadpool and forward its output batches to this
    /// builder's channel.
    ///
    /// If the input partition produces an error, the error is
    /// forwarded and no further batches are sent from that task.
    pub(crate) fn run_input(
        &mut self,
        input: Arc<dyn ExecutionPlan>,
        partition: usize,
        context: Arc<TaskContext>,
    ) {
        let output = self.tx();

        self.spawn(async move {
            let mut stream = match input.execute(partition, context).await {
                Err(e) => {
                    // If send fails, the plan is being torn down and
                    // there is no place to report the error.
                    let arrow_error = ArrowError::ExternalError(Box::new(e));
                    output.send(Err(arrow_error)).await.ok();
                    debug!(
                        "Stopping execution: error executing input: {}",
                        displayable(input.as_ref()).indent()
                    );
                    return;
                }
                Ok(stream) => stream,
            };

            while let Some(item) = stream.next().await {
                let is_err = item.is_err();

                // If send fails, the consumer is gone; no reason to
                // keep producing.
                if output.send(item).await.is_err() {
                    debug!(
                        "Stopping execution: output is gone, plan cancelling: {}",
                        displayable(input.as_ref()).indent()
                    );
                    return;
                }

                // Stop after the first error so we don't drive every
                // input to completion once one has failed.
                if is_err {
                    debug!(
                        "Stopping execution: plan returned error: {}",
                        displayable(input.as_ref()).indent()
                    );
                    return;
                }
            }
        });
    }

    /// Create a stream of all `RecordBatch`es written to the channel,
    /// propagating any panics from spawned tasks.
    pub fn build(self) -> SendableRecordBatchStream {
        let Self {
            tx,
            rx,
            schema,
            mut join_set,
        } = self;

        // drop our own sender so the receiver closes once all
        // producer tasks have completed
        drop(tx);

        // future that joins every spawned task and re-raises panics
        let check = async move {
            while let Some(result) = join_set.join_next().await {
                match result {
                    Ok(()) => continue,
                    Err(e) => {
                        if e.is_panic() {
                            // resume on the consumer thread. Keep the
                            // suppression flag on for the re-raise
                            // too so the default panic hook stays
                            // silent when the unwind fires here.
                            install_panic_hook_once();
                            let _guard = SuppressPanicGuard::new();
                            std::panic::resume_unwind(e.into_panic());
                        } else {
                            // Only reachable if the task was cancelled,
                            // which only happens when the JoinSet is
                            // dropped, i.e. when this stream has been
                            // dropped, so this code will not run.
                            return Some(Err(ArrowError::ExternalError(Box::new(
                                DataFusionError::Internal(format!(
                                    "Non Panic Task error: {}",
                                    e
                                )),
                            ))));
                        }
                    }
                }
            }
            None
        };

        let check_stream =
            futures::stream::once(check).filter_map(|item| async move { item });

        // Interleave batches from the channel with the join-set
        // check so whichever is ready first produces output.
        let inner =
            futures::stream::select(ReceiverStream::new(rx), check_stream).boxed();

        Box::pin(RecordBatchReceiverStream { schema, inner })
    }
}

/// Adapter for a tokio [`ReceiverStream`] that implements the
/// [`SendableRecordBatchStream`] interface and propagates panics and
/// errors from the tasks writing to the underlying channel. Use
/// [`Self::builder`] to construct one.
pub struct RecordBatchReceiverStream {
    schema: SchemaRef,
    inner: BoxStream<'static, ArrowResult<RecordBatch>>,
}

impl RecordBatchReceiverStream {
    /// Create a builder with an internal buffer of `capacity` batches.
    pub fn builder(
        schema: SchemaRef,
        capacity: usize,
    ) -> RecordBatchReceiverStreamBuilder {
        RecordBatchReceiverStreamBuilder::new(schema, capacity)
    }

    /// Construct a new [`RecordBatchReceiverStream`] which will send
    /// batches of the specified schema from `rx`, while monitoring
    /// `join_handle` for panics.
    ///
    /// The task is aborted if the returned stream is dropped.
    pub fn create(
        schema: &SchemaRef,
        rx: tokio::sync::mpsc::Receiver<ArrowResult<RecordBatch>>,
        join_handle: JoinHandle<()>,
    ) -> SendableRecordBatchStream {
        let schema = schema.clone();

        // Hold the handle in an AbortOnDropSingle so dropping the
        // resulting stream aborts the background task.
        let abort_helper = AbortOnDropSingle::new(join_handle);

        let check = async move {
            match abort_helper.await {
                Ok(()) => None,
                Err(e) => {
                    if e.is_panic() {
                        install_panic_hook_once();
                        let _guard = SuppressPanicGuard::new();
                        std::panic::resume_unwind(e.into_panic());
                    } else {
                        Some(Err(ArrowError::ExternalError(Box::new(
                            DataFusionError::Internal(format!(
                                "Non Panic Task error: {}",
                                e
                            )),
                        ))))
                    }
                }
            }
        };

        let check_stream =
            futures::stream::once(check).filter_map(|item| async move { item });

        let inner =
            futures::stream::select(ReceiverStream::new(rx), check_stream).boxed();

        Box::pin(Self { schema, inner })
    }
}

impl Stream for RecordBatchReceiverStream {
    type Item = ArrowResult<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.inner.poll_next_unpin(cx)
    }
}

impl RecordBatchStream for RecordBatchReceiverStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

/// Combines a [`Stream`] with a [`SchemaRef`] implementing
/// [`SendableRecordBatchStream`] for the combination.
pub struct RecordBatchStreamAdapter<S> {
    schema: SchemaRef,
    stream: S,
}

impl<S> RecordBatchStreamAdapter<S> {
    /// Creates a new [`RecordBatchStreamAdapter`] from the provided schema and stream
    pub fn new(schema: SchemaRef, stream: S) -> Self {
        Self { schema, stream }
    }
}

impl<S> std::fmt::Debug for RecordBatchStreamAdapter<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecordBatchStreamAdapter")
            .field("schema", &self.schema)
            .finish()
    }
}

impl<S> Stream for RecordBatchStreamAdapter<S>
where
    S: Stream<Item = ArrowResult<RecordBatch>> + Unpin,
{
    type Item = ArrowResult<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.stream.poll_next_unpin(cx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.stream.size_hint()
    }
}

impl<S> RecordBatchStream for RecordBatchStreamAdapter<S>
where
    S: Stream<Item = ArrowResult<RecordBatch>> + Unpin,
{
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

/// Stream wrapper that records [`BaselineMetrics`] for a particular
/// [`SendableRecordBatchStream`] (typically a partition).
pub(crate) struct ObservedStream {
    inner: SendableRecordBatchStream,
    baseline_metrics: BaselineMetrics,
}

impl ObservedStream {
    pub fn new(
        inner: SendableRecordBatchStream,
        baseline_metrics: BaselineMetrics,
    ) -> Self {
        Self {
            inner,
            baseline_metrics,
        }
    }
}

impl RecordBatchStream for ObservedStream {
    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }
}

impl Stream for ObservedStream {
    type Item = ArrowResult<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let poll = self.inner.poll_next_unpin(cx);
        self.baseline_metrics.record_poll(poll)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};

    use crate::{
        prelude::SessionContext,
        test::exec::{
            assert_strong_count_converges_to_zero, BlockingExec, MockExec, PanicExec,
        },
    };

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("a", DataType::Float32, true)]))
    }

    #[tokio::test]
    #[should_panic(expected = "PanickingStream did panic")]
    async fn record_batch_receiver_stream_propagates_panics() {
        let schema = schema();

        let num_partitions = 10;
        let input = PanicExec::new(schema.clone(), num_partitions);
        consume(input, 10).await
    }

    #[tokio::test]
    #[should_panic(expected = "PanickingStream did panic: 1")]
    async fn record_batch_receiver_stream_propagates_panics_early_shutdown() {
        let schema = schema();

        // two partitions; the second one panics before the first
        let num_partitions = 2;
        let input = PanicExec::new(schema.clone(), num_partitions)
            .with_partition_panic(0, 10)
            .with_partition_panic(1, 3);

        // The stream should stop after the first panic; since the
        // two partitions interleave (0,1,0,1,0,panic) it should not
        // exceed 5 batches prior to the panic.
        let max_batches = 5;
        consume(input, max_batches).await
    }

    #[tokio::test]
    async fn record_batch_receiver_stream_drop_cancel() {
        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();
        let schema = schema();

        let input = BlockingExec::new(schema.clone(), 1);
        let refs = input.refs();

        let mut builder = RecordBatchReceiverStream::builder(schema, 2);
        builder.run_input(Arc::new(input), 0, task_ctx.clone());
        let stream = builder.build();

        // input should still be present
        assert!(std::sync::Weak::strong_count(&refs) > 0);

        // drop the stream, ensure the refs go to zero
        drop(stream);
        assert_strong_count_converges_to_zero(refs).await;
    }

    /// Ensure that when an error is received from one stream the
    /// [`RecordBatchReceiverStream`] stops early and does not drive
    /// other streams to completion.
    #[tokio::test]
    async fn record_batch_receiver_stream_error_does_not_drive_completion() {
        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();
        let schema = schema();

        let error_stream = MockExec::new(
            vec![
                Err(ArrowError::ComputeError("Test1".to_string())),
                Err(ArrowError::ComputeError("Test2".to_string())),
            ],
            schema.clone(),
        )
        .with_use_task(false);

        let mut builder = RecordBatchReceiverStream::builder(schema, 2);
        builder.run_input(Arc::new(error_stream), 0, task_ctx.clone());
        let mut stream = builder.build();

        // first result should be the first error
        let first_batch = stream.next().await.unwrap();
        let first_err = first_batch.unwrap_err();
        assert_eq!(first_err.to_string(), "Compute error: Test1");

        // no more batches should be produced (second error must not surface)
        assert!(stream.next().await.is_none());
    }

    /// Collect every partition of `input` into a
    /// [`RecordBatchReceiverStream`] and drive it to completion,
    /// panicking if more than `max_batches` are produced.
    async fn consume(input: PanicExec, max_batches: usize) {
        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();

        let input = Arc::new(input);
        let num_partitions = input.output_partitioning().partition_count();

        let mut builder =
            RecordBatchReceiverStream::builder(input.schema(), num_partitions);
        for partition in 0..num_partitions {
            builder.run_input(input.clone(), partition, task_ctx.clone());
        }
        let mut stream = builder.build();

        let mut num_batches = 0;
        while let Some(next) = stream.next().await {
            next.unwrap();
            num_batches += 1;
            assert!(
                num_batches < max_batches,
                "Got the limit of {} batches before seeing panic",
                num_batches
            );
        }
    }
}
