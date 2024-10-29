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

//! Merge Sort implementation

use std::any::Any;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::stream::{Fuse, Stream};
use futures::StreamExt;

use arrow::array::{build_compare, ArrayRef, BooleanArray, DynComparator};
pub use arrow::compute::SortOptions;
use arrow::compute::{
    filter_record_batch, lexsort_to_indices, take, SortColumn, TakeOptions,
};
use arrow::datatypes::SchemaRef;
use arrow::error::Result as ArrowResult;
use arrow::record_batch::RecordBatch;
use itertools::Itertools;

use super::{RecordBatchStream, SendableRecordBatchStream};
use crate::error::{DataFusionError, Result};
use crate::physical_plan::{ExecutionPlan, OptimizerHints, Partitioning};

use crate::cube_ext::util::{cmp_array_row_same_types, lexcmp_array_rows};
use crate::physical_plan::expressions::Column;
use crate::physical_plan::memory::MemoryStream;
use arrow::array::{make_array, MutableArrayData};
use async_trait::async_trait;
use futures::future::join_all;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

/// Sort execution plan
#[derive(Debug)]
pub struct MergeSortExec {
    input: Arc<dyn ExecutionPlan>,
    /// Columns to sort on
    pub columns: Vec<Column>,
}

impl MergeSortExec {
    /// Create a new sort execution plan
    pub fn try_new(input: Arc<dyn ExecutionPlan>, columns: Vec<Column>) -> Result<Self> {
        if columns.is_empty() {
            return Err(DataFusionError::Internal(
                "Empty columns passed for MergeSortExec".to_string(),
            ));
        }
        Ok(Self { input, columns })
    }

    /// Input execution plan
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }
}

#[async_trait]
impl ExecutionPlan for MergeSortExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.input.clone()]
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(1)
    }

    fn with_new_children(
        &self,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(MergeSortExec::try_new(
            children[0].clone(),
            self.columns.clone(),
        )?))
    }

    fn output_hints(&self) -> OptimizerHints {
        // We do want to retain approximate sorting information.  Note that the sorting algorithm's
        // index field in struct Key<'a> makes us see that each input stream's unused sort keys
        // result in sawtoothed runs.

        // For example, if the input streams are sorted by columns A, B, C, D, E, and the sort key
        // is A, B, C, then we want the approximate_sort_order to be [[A, B, C], [D, E]], because
        // for a given value under ABC, the sort order will have multiple increasing (sawtoothing)
        // runs of columns DE the way the input streams get merged (due to the index field usage in
        // struct Key<'a>).

        let mut hints: OptimizerHints = self.input.output_hints();
        let sort_order: Vec<usize> = self.columns.iter().map(|c| c.index()).collect();

        'fallback: {
            if !hints.approximate_sort_order_is_prefix || hints.approximate_sort_order.is_empty() {
                break 'fallback;
            }
            let first_seg: &Vec<usize> = &hints.approximate_sort_order[0];

            let mut sort_order_index: usize = 0;
            let mut approx_index: usize = 0;
            while sort_order_index < sort_order.len() {
                if first_seg[approx_index] == sort_order[sort_order_index] {
                    sort_order_index += 1;
                    approx_index += 1;
                    if approx_index == first_seg.len() {
                        break;
                    }
                } else if hints.single_value_columns.contains(&first_seg[approx_index]) {
                    approx_index += 1;
                    if approx_index == first_seg.len() {
                        break;
                    }
                } else if hints.single_value_columns.contains(&sort_order[sort_order_index]) {
                    sort_order_index += 1;
                } else {
                    // This should not happen.
                    break 'fallback;
                }
            }

            if approx_index > 0 {
                if approx_index != first_seg.len() {
                    let second_seg = first_seg[approx_index..].iter().map(|&x| x).collect_vec();
                    hints.approximate_sort_order.insert(1, second_seg);
                    hints.approximate_sort_order[0].truncate(approx_index);
                } else {
                    // It would be weird if sort_order_index is not equal to sort_order.len() --
                    // another instance of single value columns (we hope).

                    // Nothing to do here.
                }
                hints.approximate_sort_order_is_prefix = true;
            } else {
                // approx_index == 0

                // It's possible we sorted by some single value column, and this means subsequent
                // columns are sawtoothing in separate columns.  Or is it?  Either the input_hints's
                // sort_order is inconsistent with the approximate_sort_order, or we have some
                // particular treatment of single_value_columns in different code deciding whether
                // we can use a MergeExec node, that leads to this case.
                hints.approximate_sort_order_is_prefix = false;
            }

            return hints;

        }

        OptimizerHints::new_sorted(
            Some(sort_order),
            hints.single_value_columns,
        )
    }

    async fn execute(&self, partition: usize) -> Result<SendableRecordBatchStream> {
        if 0 != partition {
            return Err(DataFusionError::Internal(format!(
                "MergeSortExec invalid partition {}",
                partition
            )));
        }

        log::error!("MergeSortExec executing across {} partitions", self.input.output_partitioning().partition_count());

        let inputs = join_all(
            (0..self.input.output_partitioning().partition_count())
                .map(|i| self.input.execute(i))
                .collect::<Vec<_>>(),
        )
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()?;

        if inputs.len() == 1 {
            return Ok(inputs.into_iter().next().unwrap());
        }

        Ok(Box::pin(MergeSortStream::new(
            self.input.schema(),
            inputs,
            self.columns.clone(),
        )))
    }
}

/// Sort execution plan to resort merge join results
#[derive(Debug)]
pub struct MergeReSortExec {
    input: Arc<dyn ExecutionPlan>,
    columns: Vec<Column>,
}

impl MergeReSortExec {
    /// Create a new sort execution plan
    pub fn try_new(input: Arc<dyn ExecutionPlan>, columns: Vec<Column>) -> Result<Self> {
        Ok(Self { input, columns })
    }
}

#[async_trait]
impl ExecutionPlan for MergeReSortExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.input.clone()]
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(1)
    }

    fn with_new_children(
        &self,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(MergeReSortExec::try_new(
            children[0].clone(),
            self.columns.clone(),
        )?))
    }

    async fn execute(&self, partition: usize) -> Result<SendableRecordBatchStream> {
        if 0 != partition {
            return Err(DataFusionError::Internal(format!(
                "MergeReSortExec invalid partition {}",
                partition
            )));
        }

        if 1 != self.input.output_partitioning().partition_count() {
            return Err(DataFusionError::Internal(format!(
                "MergeReSortExec expects only one partition but got {}",
                self.input.output_partitioning().partition_count()
            )));
        }

        let mut stream = self.input.execute(0).await?;
        let mut all_batches = Vec::new();
        while let Some(b) = stream.next().await.transpose()? {
            all_batches.push(b);
        }

        let schema = self.input.schema();
        let sorted_batches = all_batches
            .into_iter()
            .map(|b| -> Result<SendableRecordBatchStream> {
                Ok(Box::pin(MemoryStream::try_new(
                    vec![sort_batch(&self.columns, &schema, b)?],
                    schema.clone(),
                    None,
                )?))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Box::pin(MergeSortStream::new(
            self.input.schema(),
            sorted_batches,
            self.columns.clone(),
        )))
    }
}

fn sort_batch(
    columns: &Vec<Column>,
    schema: &SchemaRef,
    batch: RecordBatch,
) -> ArrowResult<RecordBatch> {
    let columns_to_sort = columns
        .iter()
        .map(|c| -> ArrowResult<SortColumn> {
            Ok(SortColumn {
                values: batch.column(c.index()).clone(),
                options: None,
            })
        })
        .collect::<ArrowResult<Vec<_>>>()?;
    let indices = lexsort_to_indices(columns_to_sort.as_slice(), None)?;

    RecordBatch::try_new(
        schema.clone(),
        batch
            .columns()
            .iter()
            .map(|column| {
                take(
                    column.as_ref(),
                    &indices,
                    // disable bound check overhead since indices are already generated from
                    // the same record batch
                    Some(TakeOptions {
                        check_bounds: false,
                    }),
                )
            })
            .collect::<ArrowResult<Vec<ArrayRef>>>()?,
    )
}

struct MergeSortStream {
    schema: SchemaRef,
    columns: Vec<Column>,
    poll_states: Vec<MergeSortStreamState>,
}

impl MergeSortStream {
    fn new(
        schema: SchemaRef,
        inputs: Vec<SendableRecordBatchStream>,
        columns: Vec<Column>,
    ) -> Self {
        Self {
            schema,
            columns,
            poll_states: inputs
                .into_iter()
                .map(|stream| MergeSortStreamState::new(stream))
                .collect(),
        }
    }
}

struct MergeSortStreamState {
    stream: Fuse<SendableRecordBatchStream>,
    poll_state: Poll<Option<ArrowResult<(usize, RecordBatch)>>>,
}

impl MergeSortStreamState {
    fn new(stream: SendableRecordBatchStream) -> Self {
        Self {
            stream: stream.fuse(),
            poll_state: Poll::Pending,
        }
    }

    pub fn update_state(&mut self, cx: &mut std::task::Context<'_>) {
        if !self.poll_state.is_pending() {
            return;
        }
        let inner = self.stream.poll_next_unpin(cx);
        match inner {
            // skip empty batches and wait for the next poll.
            Poll::Ready(Some(Ok(b))) if b.num_rows() == 0 => {
                cx.waker().wake_by_ref();
                return;
            }
            _ => {}
        }
        self.poll_state = inner.map(|option| match option {
            Some(batch) => Some(batch.map(|b| (0, b))),
            None => None,
        });
    }

    pub fn take_batch(&mut self) -> Option<ArrowResult<(usize, RecordBatch)>> {
        let mut res = Poll::Pending;
        std::mem::swap(&mut res, &mut self.poll_state);
        if let Poll::Ready(option) = &mut res {
            option.take()
        } else {
            panic!(
                "Invalid merge sort state: unexpected empty state: {:?}",
                self.poll_state
            );
        }
    }

    pub fn update_batch(&mut self, new_cursor: usize, batch: RecordBatch) {
        if let Poll::Ready(_) = self.poll_state {
            panic!(
                "Invalid merge sort state: unexpected ready state: {:?}",
                self.poll_state
            );
        } else {
            self.poll_state = if new_cursor == batch.num_rows() {
                Poll::Pending
            } else {
                Poll::Ready(Some(Ok((new_cursor, batch))))
            }
        }
    }
}

impl Stream for MergeSortStream {
    type Item = ArrowResult<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        for state in self.poll_states.iter_mut() {
            state.update_state(cx);
        }

        // TODO: pass the value from ExecutionConfig.
        const MAX_BATCH_ROWS: usize = 4096;
        if self.poll_states.iter().all(|s| s.poll_state.is_ready()) {
            let res = self
                .poll_states
                .iter_mut()
                .map(|s| s.take_batch().transpose())
                .collect::<ArrowResult<Vec<_>>>()
                .and_then(|all_batches| -> ArrowResult<Option<RecordBatch>> {
                    let mut batches = Vec::with_capacity(all_batches.len());
                    let mut batch_indices = Vec::with_capacity(all_batches.len());
                    for (i, b) in all_batches.into_iter().enumerate() {
                        if let Some(b) = b {
                            batch_indices.push(i);
                            batches.push(b);
                        }
                    }
                    if batches.is_empty() {
                        return Ok(None);
                    }
                    let (new_cursors, sorted_batch) = merge_sort(
                        &batches.iter().map(|(c, b)| (*c, b)).collect::<Vec<_>>(),
                        &self.columns,
                        MAX_BATCH_ROWS,
                    )?;

                    assert_eq!(new_cursors.len(), batches.len());
                    for (i, b) in batches.into_iter().enumerate() {
                        self.poll_states[batch_indices[i]]
                            .update_batch(new_cursors[i], b.1);
                    }

                    Ok(Some(sorted_batch))
                });

            Poll::Ready(res.transpose())
        } else {
            Poll::Pending
        }
    }
}

#[tracing::instrument(level = "trace", skip(batches, columns, max_batch_rows))]
fn merge_sort(
    batches: &[(usize, &RecordBatch)],
    columns: &[Column],
    max_batch_rows: usize,
) -> ArrowResult<(Vec<usize>, RecordBatch)> {
    assert!(!columns.is_empty());
    assert!(!batches.is_empty());

    let mut sort_keys = Vec::with_capacity(batches.len());
    let mut pos = Vec::with_capacity(batches.len());
    for (p, b) in batches {
        let mut key_cols = Vec::with_capacity(columns.len());
        for c in columns {
            key_cols.push(b.column(c.index()));
        }

        sort_keys.push(key_cols);
        pos.push(*p);
    }

    struct Key<'a> {
        values: &'a [&'a ArrayRef],
        index: usize,
        row: usize,
    }
    impl PartialEq for Key<'_> {
        fn eq(&self, other: &Self) -> bool {
            self.cmp(other) == Ordering::Equal
        }
    }
    impl Eq for Key<'_> {}
    impl PartialOrd for Key<'_> {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Key<'_> {
        fn cmp(&self, other: &Self) -> Ordering {
            for i in 0..self.values.len() {
                let o = cmp_array_row_same_types(
                    &self.values[i],
                    self.row,
                    &other.values[i],
                    other.row,
                );
                if o != Ordering::Equal {
                    return o;
                }
            }
            self.index.cmp(&other.index) // This comparison makes pop order deterministic.
        }
    }

    let mut candidates = BinaryHeap::with_capacity(sort_keys.len());
    for i in 0..sort_keys.len() {
        if pos[i] == sort_keys[i][0].len() {
            continue;
        }
        let k = Key {
            values: &sort_keys[i],
            index: i,
            row: pos[i],
        };
        candidates.push(Reverse(k));
    }

    let num_cols = batches[0].1.num_columns();
    let mut result_cols = Vec::with_capacity(num_cols);
    let mut num_result_rows = 0;
    for i in 0..num_cols {
        result_cols.push(MutableArrayData::new(
            batches.iter().map(|(_, b)| b.column(i).data()).collect(),
            false,
            max_batch_rows,
        ));
    }
    while let Some(Reverse(c)) = candidates.pop() {
        let mut len = 1;
        if let Some(next) = candidates.peek() {
            loop {
                if num_result_rows + len == max_batch_rows
                    || c.row + len == sort_keys[c.index][0].len()
                {
                    break;
                }
                assert!(
                    lexcmp_array_rows(
                        sort_keys[c.index].iter().map(|a| *a),
                        c.row + len - 1,
                        c.row + len
                    ) <= Ordering::Equal,
                    "unsorted data in merge. row {}. data: {:?}",
                    c.row + len,
                    sort_keys[c.index]
                        .iter()
                        .map(|a| a.slice(pos[c.index] + len - 1, 2))
                        .collect::<Vec<_>>(),
                );
                let k = Key {
                    values: &sort_keys[c.index],
                    index: c.index,
                    row: c.row + len,
                };
                if k.cmp(&next.0) <= Ordering::Equal {
                    len += 1;
                } else {
                    break;
                }
            }
        }
        for i in 0..num_cols {
            result_cols[i].extend(c.index, c.row, c.row + len);
        }
        num_result_rows += len;

        assert_eq!(pos[c.index], c.row);
        pos[c.index] += len;
        if num_result_rows == max_batch_rows
            || pos[c.index] == sort_keys[c.index][0].len()
        {
            break;
        }
        candidates.push(Reverse(Key {
            values: &sort_keys[c.index],
            index: c.index,
            row: pos[c.index],
        }));
    }

    let result_cols: Vec<ArrayRef> = result_cols
        .into_iter()
        .map(|r| make_array(r.freeze()))
        .collect();
    #[cfg(debug_assertions)]
    {
        let key_cols = columns
            .iter()
            .map(|c| &result_cols[c.index()])
            .collect::<Vec<_>>();
        for i in 1..result_cols[0].len() {
            debug_assert!(
                lexcmp_array_rows(key_cols.iter().map(|a| *a), i - 1, i,)
                    <= Ordering::Equal,
                "unsorted data after merge. row {}. data: {:?}",
                i - 1,
                key_cols
                    .iter()
                    .map(|a| a.slice(i - 1, 2))
                    .collect::<Vec<_>>(),
            );
        }
    }
    Ok((
        pos,
        RecordBatch::try_new(batches[0].1.schema(), result_cols)?,
    ))
}

impl RecordBatchStream for MergeSortStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

/// Filter out all but last row by unique key execution plan
#[derive(Debug)]
pub struct LastRowByUniqueKeyExec {
    input: Arc<dyn ExecutionPlan>,
    /// Columns to sort on
    pub unique_key: Vec<Column>,
}

impl LastRowByUniqueKeyExec {
    /// Create a new execution plan
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        unique_key: Vec<Column>,
    ) -> Result<Self> {
        if unique_key.is_empty() {
            return Err(DataFusionError::Internal(
                "Empty unique_key passed for LastRowByUniqueKeyExec".to_string(),
            ));
        }
        Ok(Self { input, unique_key })
    }

    /// Input execution plan
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }
}

#[async_trait]
impl ExecutionPlan for LastRowByUniqueKeyExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.input.clone()]
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(1)
    }

    fn with_new_children(
        &self,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(LastRowByUniqueKeyExec::try_new(
            children[0].clone(),
            self.unique_key.clone(),
        )?))
    }

    fn output_hints(&self) -> OptimizerHints {
        let input_hints = self.input.output_hints();
        OptimizerHints::new_sorted(
            input_hints.sort_order,
            input_hints.single_value_columns,
        )
    }

    async fn execute(&self, partition: usize) -> Result<SendableRecordBatchStream> {
        if 0 != partition {
            return Err(DataFusionError::Internal(format!(
                "LastRowByUniqueKeyExec invalid partition {}",
                partition
            )));
        }

        if self.input.output_partitioning().partition_count() != 1 {
            return Err(DataFusionError::Internal(format!(
                "LastRowByUniqueKeyExec expects only one partition but got {}",
                self.input.output_partitioning().partition_count()
            )));
        }
        let input_stream = self.input.execute(0).await?;

        Ok(Box::pin(LastRowByUniqueKeyExecStream {
            schema: self.input.schema(),
            input: input_stream,
            unique_key: self.unique_key.clone(),
            current_record_batch: None,
        }))
    }
}

/// Filter out all but last row by unique key stream
struct LastRowByUniqueKeyExecStream {
    /// Output schema, which is the same as the input schema for this operator
    schema: SchemaRef,
    /// The input stream to filter.
    input: SendableRecordBatchStream,
    /// Key columns
    unique_key: Vec<Column>,
    /// Current Record Batch
    current_record_batch: Option<RecordBatch>,
}

impl LastRowByUniqueKeyExecStream {
    fn row_equals(comparators: &Vec<DynComparator>, a: usize, b: usize) -> bool {
        for comparator in comparators.iter().rev() {
            if comparator(a, b) != Ordering::Equal {
                return false;
            }
        }
        true
    }

    #[tracing::instrument(level = "trace", skip(self, next_batch))]
    fn keep_only_last_rows_by_key(
        &mut self,
        next_batch: Option<RecordBatch>,
    ) -> ArrowResult<RecordBatch> {
        let batch = self.current_record_batch.take().unwrap();
        let num_rows = batch.num_rows();
        let mut builder = BooleanArray::builder(num_rows);
        let key_columns = self
            .unique_key
            .iter()
            .map(|k| batch.column(k.index()).clone())
            .collect::<Vec<ArrayRef>>();
        let mut requires_filtering = false;
        let self_column_comparators = key_columns
            .iter()
            .map(|c| build_compare(c.as_ref(), c.as_ref()))
            .collect::<ArrowResult<Vec<_>>>()?;
        for i in 0..num_rows {
            let filter_value = if i == num_rows - 1 && next_batch.is_none() {
                true
            } else if i == num_rows - 1 {
                let next_key_columns = self
                    .unique_key
                    .iter()
                    .map(|k| next_batch.as_ref().unwrap().column(k.index()).clone())
                    .collect::<Vec<ArrayRef>>();
                let next_column_comparators = key_columns
                    .iter()
                    .zip(next_key_columns.iter())
                    .map(|(c, n)| build_compare(c.as_ref(), n.as_ref()))
                    .collect::<ArrowResult<Vec<_>>>()?;
                !Self::row_equals(&next_column_comparators, i, 0)
            } else {
                !Self::row_equals(&self_column_comparators, i, i + 1)
            };
            if !filter_value {
                requires_filtering = true;
            }
            builder.append_value(filter_value)?;
        }
        self.current_record_batch = next_batch;
        if requires_filtering {
            let filter_array = builder.finish();
            filter_record_batch(&batch, &filter_array)
        } else {
            Ok(batch)
        }
    }
}

impl Stream for LastRowByUniqueKeyExecStream {
    type Item = ArrowResult<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        self.input.poll_next_unpin(cx).map(|x| {
            match x {
                Some(Ok(batch)) => {
                    if self.current_record_batch.is_none() {
                        let schema = batch.schema();
                        self.current_record_batch = Some(batch);
                        // TODO get rid of empty batch. Returning Poll::Pending here results in stuck stream.
                        Some(Ok(RecordBatch::new_empty(schema)))
                    } else {
                        Some(self.keep_only_last_rows_by_key(Some(batch)))
                    }
                }
                None => {
                    if self.current_record_batch.is_some() {
                        Some(self.keep_only_last_rows_by_key(None))
                    } else {
                        None
                    }
                }
                other => other,
            }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.input.size_hint();
        (lower, upper.map(|u| u + 1))
    }
}

impl RecordBatchStream for LastRowByUniqueKeyExecStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow::compute::kernels::concat::concat;
    use crate::physical_plan::collect;
    use crate::physical_plan::memory::MemoryExec;
    use arrow::array::*;
    use arrow::datatypes::*;
    use itertools::Itertools;

    #[tokio::test]
    async fn two_inputs_three_batches() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::UInt32, true),
            Field::new("b", DataType::UInt64, true),
        ]));

        // define data.
        let batch1_1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![
                    None,
                    None,
                    Some(1),
                    Some(1),
                    Some(3),
                    Some(5),
                    Some(5),
                ])),
                Arc::new(UInt64Array::from(vec![
                    Some(1),
                    Some(2),
                    Some(1),
                    Some(2),
                    Some(2),
                    None,
                    Some(2),
                ])),
            ],
        )?;

        let batch1_2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![
                    Some(7),
                    Some(8),
                    Some(8),
                    Some(8),
                    Some(9),
                ])),
                Arc::new(UInt64Array::from(vec![
                    Some(1),
                    Some(2),
                    Some(2),
                    Some(3),
                    None,
                ])),
            ],
        )?;

        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![Some(3), Some(5), Some(10)])),
                Arc::new(UInt64Array::from(vec![Some(2), Some(2), None])),
            ],
        )?;

        let sort_exec = Arc::new(MergeSortExec::try_new(
            Arc::new(MemoryExec::try_new(
                &vec![vec![batch1_1, batch1_2], vec![batch2]],
                schema.clone(),
                None,
            )?),
            vec![col("a", &schema), col("b", &schema)],
        )?);

        assert_eq!(DataType::UInt32, *sort_exec.schema().field(0).data_type());
        assert_eq!(DataType::UInt64, *sort_exec.schema().field(1).data_type());

        let result: Vec<RecordBatch> = collect(sort_exec).await?;
        assert_eq!(result.len(), 3);

        assert_eq!(
            vec![
                (None, Some("1".to_owned())),
                (None, Some("2".to_owned())),
                (Some("1".to_owned()), Some("1".to_owned())),
                (Some("1".to_owned()), Some("2".to_owned())),
                (Some("3".to_owned()), Some("2".to_owned())),
                (Some("3".to_owned()), Some("2".to_owned())),
                (Some("5".to_owned()), None),
                (Some("5".to_owned()), Some("2".to_owned())),
            ],
            transform_batch_for_assert(&result[0])
        );

        assert_eq!(
            vec![
                (Some("5".to_owned()), Some("2".to_owned())),
                (Some("7".to_owned()), Some("1".to_owned())),
                (Some("8".to_owned()), Some("2".to_owned())),
                (Some("8".to_owned()), Some("2".to_owned())),
                (Some("8".to_owned()), Some("3".to_owned())),
                (Some("9".to_owned()), None),
            ],
            transform_batch_for_assert(&result[1])
        );

        assert_eq!(
            vec![(Some("10".to_owned()), None),],
            transform_batch_for_assert(&result[2])
        );

        Ok(())
    }

    #[tokio::test]
    async fn resort() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::UInt32, true),
            Field::new("b", DataType::UInt64, true),
        ]));

        // define data.
        let batch1_1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![
                    None,
                    None,
                    Some(1),
                    Some(1),
                    Some(3),
                    Some(5),
                    Some(5),
                ])),
                Arc::new(UInt64Array::from(vec![
                    Some(1),
                    Some(2),
                    Some(1),
                    Some(2),
                    Some(2),
                    None,
                    Some(2),
                ])),
            ],
        )?;

        let batch1_2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![
                    Some(7),
                    Some(8),
                    Some(8),
                    Some(8),
                    Some(9),
                ])),
                Arc::new(UInt64Array::from(vec![
                    Some(1),
                    Some(2),
                    Some(2),
                    Some(3),
                    None,
                ])),
            ],
        )?;

        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![Some(3), Some(5), Some(10)])),
                Arc::new(UInt64Array::from(vec![Some(2), Some(2), None])),
            ],
        )?;

        let sort_exec = Arc::new(MergeReSortExec::try_new(
            Arc::new(MemoryExec::try_new(
                &vec![vec![batch1_2, batch1_1, batch2]],
                schema.clone(),
                None,
            )?),
            vec![col("a", &schema), col("b", &schema)],
        )?);

        assert_eq!(DataType::UInt32, *sort_exec.schema().field(0).data_type());
        assert_eq!(DataType::UInt64, *sort_exec.schema().field(1).data_type());

        let result: Vec<RecordBatch> = collect(sort_exec).await?;
        assert_eq!(result.len(), 3);

        assert_eq!(
            vec![
                (None, Some("1".to_owned())),
                (None, Some("2".to_owned())),
                (Some("1".to_owned()), Some("1".to_owned())),
                (Some("1".to_owned()), Some("2".to_owned())),
                (Some("3".to_owned()), Some("2".to_owned())),
                (Some("3".to_owned()), Some("2".to_owned())),
                (Some("5".to_owned()), None),
                (Some("5".to_owned()), Some("2".to_owned())),
            ],
            transform_batch_for_assert(&result[0])
        );

        assert_eq!(
            vec![
                (Some("5".to_owned()), Some("2".to_owned())),
                (Some("7".to_owned()), Some("1".to_owned())),
                (Some("8".to_owned()), Some("2".to_owned())),
                (Some("8".to_owned()), Some("2".to_owned())),
                (Some("8".to_owned()), Some("3".to_owned())),
                (Some("9".to_owned()), None),
            ],
            transform_batch_for_assert(&result[1])
        );

        assert_eq!(
            vec![(Some("10".to_owned()), None),],
            transform_batch_for_assert(&result[2])
        );

        Ok(())
    }

    #[tokio::test]
    async fn empty_batches() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::UInt32, true),
            Field::new("b", DataType::UInt64, true),
        ]));

        // define data.
        let batch1_1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(Vec::<u32>::new())),
                Arc::new(UInt64Array::from(Vec::<u64>::new())),
            ],
        )?;

        let batch1_2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![
                    Some(7),
                    Some(8),
                    Some(8),
                    Some(8),
                    Some(9),
                ])),
                Arc::new(UInt64Array::from(vec![
                    Some(1),
                    Some(2),
                    Some(2),
                    Some(3),
                    None,
                ])),
            ],
        )?;

        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(Vec::<u32>::new())),
                Arc::new(UInt64Array::from(Vec::<u64>::new())),
            ],
        )?;

        let sort_exec = Arc::new(MergeSortExec::try_new(
            Arc::new(MemoryExec::try_new(
                &vec![
                    vec![batch2.clone()],
                    vec![batch1_1.clone(), batch1_1, batch1_2],
                ],
                schema.clone(),
                None,
            )?),
            vec![col("a", &schema), col("b", &schema)],
        )?);

        assert_eq!(DataType::UInt32, *sort_exec.schema().field(0).data_type());
        assert_eq!(DataType::UInt64, *sort_exec.schema().field(1).data_type());

        let result: Vec<RecordBatch> = collect(sort_exec).await?;
        assert_eq!(result.len(), 1);
        assert_eq!(
            transform_batch_for_assert(&result[0]),
            vec![
                (Some("7".to_owned()), Some("1".to_owned())),
                (Some("8".to_owned()), Some("2".to_owned())),
                (Some("8".to_owned()), Some("2".to_owned())),
                (Some("8".to_owned()), Some("3".to_owned())),
                (Some("9".to_owned()), None),
            ]
        );

        Ok(())
    }

    fn ints_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true)]))
    }

    fn ints(d: Vec<i64>) -> RecordBatch {
        RecordBatch::try_new(ints_schema(), vec![Arc::new(Int64Array::from(d))]).unwrap()
    }

    fn to_ints(rs: Vec<RecordBatch>) -> Vec<Vec<i64>> {
        rs.into_iter()
            .map(|r| {
                r.columns()[0]
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect()
    }

    #[tokio::test]
    async fn multiple_inputs_order() {
        let p1 = vec![ints(vec![1, 3])];
        let p2 = vec![ints(vec![2, 4, 6]), ints(vec![8, 9])];
        let p3 = vec![ints(vec![5, 7, 10])];

        let schema = ints_schema();
        let inp = Arc::new(
            MemoryExec::try_new(&vec![p1, p2, p3], schema.clone(), None).unwrap(),
        );
        let r = collect(Arc::new(
            MergeSortExec::try_new(inp, vec![col("a", &schema)]).unwrap(),
        ))
        .await
        .unwrap();
        assert_eq!(
            to_ints(r),
            vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9], vec![10]]
        );
    }

    #[tokio::test]
    async fn empty_batches_2() {
        let p1 = vec![ints(vec![1, 2])];
        let p2 = vec![ints(vec![]), ints(vec![0])];

        let schema = ints_schema();
        let inp =
            Arc::new(MemoryExec::try_new(&vec![p1, p2], schema.clone(), None).unwrap());
        let r = collect(Arc::new(
            MergeSortExec::try_new(inp, vec![col("a", &schema)]).unwrap(),
        ))
        .await
        .unwrap();
        assert_eq!(
            to_ints(r).into_iter().flatten().collect_vec(),
            vec![0, 1, 2],
        );
    }

    fn transform_batch_for_assert(
        batch: &RecordBatch,
    ) -> Vec<(Option<String>, Option<String>)> {
        let columns = batch.columns();

        assert_eq!(DataType::UInt32, *columns[0].data_type());
        assert_eq!(DataType::UInt64, *columns[1].data_type());

        let a = as_primitive_array::<UInt32Type>(&columns[0]);
        let b = as_primitive_array::<UInt64Type>(&columns[1]);

        // convert result to strings to allow comparing to expected result containing NaN
        let result: Vec<(Option<String>, Option<String>)> = (0..batch.num_rows())
            .map(|i| {
                let aval = if a.is_valid(i) {
                    Some(a.value(i).to_string())
                } else {
                    None
                };
                let bval = if b.is_valid(i) {
                    Some(b.value(i).to_string())
                } else {
                    None
                };
                (aval, bval)
            })
            .collect();
        result
    }

    #[test]
    fn test_merge_sort() {
        let array_1: ArrayRef = Arc::new(UInt64Array::from(vec![1, 2, 2, 3, 5, 10, 20]));
        let array_2: ArrayRef = Arc::new(UInt64Array::from(vec![4, 8, 9, 15]));
        let array_3: ArrayRef = Arc::new(UInt64Array::from(vec![4, 7, 9, 15]));
        let arrays = vec![&array_1, &array_2, &array_3];
        let res = test_merge(arrays);

        assert_eq!(
            res.as_any().downcast_ref::<UInt64Array>().unwrap(),
            &UInt64Array::from(vec![1, 2, 2, 3, 4, 4, 5, 7, 8, 9, 9, 10, 15, 15, 20])
        )
    }

    #[test]
    fn merge_sort_with_nulls() {
        let array_1: ArrayRef = Arc::new(UInt64Array::from(vec![
            None,
            None,
            Some(1),
            Some(2),
            Some(2),
            Some(3),
            Some(5),
            Some(10),
            Some(20),
        ]));
        let array_2: ArrayRef = Arc::new(UInt64Array::from(vec![
            None,
            None,
            None,
            None,
            Some(4),
            Some(8),
            Some(9),
            Some(15),
        ]));
        let array_3: ArrayRef = Arc::new(UInt64Array::from(vec![
            None,
            Some(4),
            Some(7),
            Some(9),
            Some(15),
        ]));
        let arrays = vec![&array_1, &array_2, &array_3];
        let res = test_merge(arrays);

        assert_eq!(
            res.as_any().downcast_ref::<UInt64Array>().unwrap(),
            &UInt64Array::from(vec![
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(1),
                Some(2),
                Some(2),
                Some(3),
                Some(4),
                Some(4),
                Some(5),
                Some(7),
                Some(8),
                Some(9),
                Some(9),
                Some(10),
                Some(15),
                Some(15),
                Some(20)
            ])
        )
    }

    #[test]
    fn single_array() {
        let array_1: ArrayRef = Arc::new(UInt64Array::from(vec![1, 2, 2, 3, 5, 10, 20]));
        let arrays = vec![&array_1];
        let res = test_merge(arrays);

        assert_eq!(
            res.as_any().downcast_ref::<UInt64Array>().unwrap(),
            &UInt64Array::from(vec![1, 2, 2, 3, 5, 10, 20])
        )
    }

    #[test]
    fn empty_array() {
        let array_1: ArrayRef = Arc::new(UInt64Array::from(vec![1, 2, 2, 3, 5, 10, 20]));
        let array_2: ArrayRef = Arc::new(UInt64Array::from(Vec::<u64>::new()));
        let arrays = vec![&array_1, &array_2];
        let res = test_merge(arrays);

        assert_eq!(
            res.as_any().downcast_ref::<UInt64Array>().unwrap(),
            &UInt64Array::from(vec![1, 2, 2, 3, 5, 10, 20])
        )
    }

    #[test]
    fn two_empty_arrays() {
        let array_1: ArrayRef = Arc::new(UInt64Array::from(Vec::<u64>::new()));
        let array_2: ArrayRef = Arc::new(UInt64Array::from(Vec::<u64>::new()));
        let arrays = vec![&array_1, &array_2];
        let res = test_merge(arrays);

        assert_eq!(
            res.as_any().downcast_ref::<UInt64Array>().unwrap(),
            &UInt64Array::from(Vec::<u64>::new())
        )
    }

    #[tokio::test]
    async fn last_row_by_unique_key_exec() {
        let p1 = vec![
            ints(vec![1, 1, 2, 3, 4, 5, 5, 6, 7]),
            ints(vec![8, 9, 9, 10]),
            ints(vec![11, 12, 13]),
        ];

        let schema = ints_schema();
        let inp = Arc::new(MemoryExec::try_new(&vec![p1], schema.clone(), None).unwrap());
        let r = collect(Arc::new(
            LastRowByUniqueKeyExec::try_new(inp, vec![col("a", &schema)]).unwrap(),
        ))
        .await
        .unwrap();
        assert_eq!(
            to_ints(r),
            vec![
                vec![],
                vec![1, 2, 3, 4, 5, 6, 7],
                vec![8, 9, 10],
                vec![11, 12, 13]
            ]
        );
    }

    fn test_merge(arrays: Vec<&ArrayRef>) -> ArrayRef {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "a",
            arrays[0].data_type().clone(),
            true,
        )]));

        let mut arrays = arrays.into_iter().map(|a| a.clone()).collect_vec();
        let mut results = Vec::new();
        // Make sure we consume all batches.
        while !arrays.is_empty() {
            let mut batches = Vec::with_capacity(arrays.len());
            for a in &arrays {
                if a.is_empty() {
                    batches.push(RecordBatch::new_empty(schema.clone()));
                } else {
                    batches.push(
                        RecordBatch::try_new(schema.clone(), vec![a.clone()]).unwrap(),
                    );
                };
            }

            let (indices, b) = merge_sort(
                &batches.iter().map(|b| (0, b)).collect_vec(),
                &schema
                    .fields()
                    .iter()
                    .enumerate()
                    .map(|(i, f)| Column::new(f.name(), i))
                    .collect_vec(),
                128, // increase this if you want larger batches in tests.
            )
            .unwrap();
            results.push(b.column(0).clone());
            for i in (0..arrays.len()).rev() {
                // reverse order for remove.
                if arrays[i].len() == indices[i] {
                    arrays.remove(i);
                    continue;
                }
                let a = &mut arrays[i];
                *a = a.slice(indices[i], a.len() - indices[i]);
            }
        }

        concat(&results.iter().map(|a| a.as_ref()).collect_vec()).unwrap()
    }

    fn col(name: &str, schema: &Schema) -> Column {
        Column::new_with_schema(name, schema).unwrap()
    }
}
