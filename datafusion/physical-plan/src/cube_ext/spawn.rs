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

use std::future::Future;
use crate::cube_ext::catch_unwind::{
    async_try_with_catch_unwind, try_with_catch_unwind, PanicError,
};
use datafusion_common_runtime::SpawnedTask;
use futures::sink::SinkExt;
use tokio::task::{AbortHandle, JoinHandle, JoinSet};
use tracing_futures::Instrument;

/// Calls [tokio::spawn] and additionally enables tracing of the spawned task as part of the current
/// computation. This is CubeStore approach to tracing, so all code must use this function instead
/// of replace [tokio::spawn].
pub fn spawn<T>(task: T) -> JoinHandle<T::Output>
where
    T: Future + Send + 'static,
    T::Output: Send + 'static,
{
    if let Some(s) = new_subtask_span() {
        tokio::spawn(async move {
            let _p = s.parent; // ensure parent stays alive.
            task.instrument(s.child).await
        })
    } else {
        tokio::spawn(task)
    }
}

/// Propagates current span to blocking operation. See [spawn] for details.
pub fn spawn_blocking<F, R>(f: F) -> JoinHandle<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    if let Some(s) = new_subtask_span() {
        tokio::task::spawn_blocking(move || {
            let _p = s.parent; // ensure parent stays alive.
            s.child.in_scope(f)
        })
    } else {
        tokio::task::spawn_blocking(f)
    }
}

/// Propagates current span to the spawned task.  See [spawn] for details.
pub fn spawn_on_joinset<F, T>(join_set: &mut JoinSet<T>, task: F) -> AbortHandle
    where
        F: Future<Output = T>,
        F: Send + 'static,
        T: Send + 'static,
{
    if let Some(s) = new_subtask_span() {
        join_set.spawn(async move {
            let _p = s.parent;  // ensure parent stays alive.
            task.instrument(s.child).await
        })
    } else {
        join_set.spawn(task)
    }
}

/// Propagates current span to the blocking operation.  See [spawn] for details.
pub fn spawn_blocking_on_joinset<F, T>(join_set: &mut JoinSet<T>, f: F) -> AbortHandle
    where
        F: FnOnce() -> T,
        F: Send + 'static,
        T: Send + 'static,
{
    if let Some(s) = new_subtask_span() {
        join_set.spawn_blocking(move || {
            let _p = s.parent;  // ensure parent stays alive.
            s.child.in_scope(f)
        })
    } else {
        join_set.spawn_blocking(f)
    }
}

/// Propagates current span to the spawned task.  See [spawn] for details.
pub fn spawn_spawned_task<T>(task: T) -> SpawnedTask<T::Output>
where
    T: Future + Send + 'static,
    T::Output: Send + 'static,
{
    if let Some(s) = new_subtask_span() {
        SpawnedTask::spawn(async move {
            let _p = s.parent; // ensure parent stays alive.
            task.instrument(s.child).await
        })
    } else {
        SpawnedTask::spawn(task)
    }
}

struct SpawnSpans {
    parent: tracing::Span,
    child: tracing::Span,
}

fn new_subtask_span() -> Option<SpawnSpans> {
    let parent = tracing::Span::current();
    if parent.is_disabled() {
        return None;
    }
    // TODO: ensure this is always enabled.
    let child = tracing::info_span!(parent: &parent, "subtask");
    Some(SpawnSpans { parent, child })
}

/// Executes future [f] in a new tokio thread. Catches panics.
pub fn spawn_with_catch_unwind<F, T, E>(f: F) -> JoinHandle<Result<T, E>>
where
    F: Future<Output = Result<T, E>> + Send + 'static,
    T: Send + 'static,
    E: From<PanicError> + Send + 'static,
{
    let task = async move {
        match async_try_with_catch_unwind(f).await {
            Ok(result) => result,
            Err(panic) => Err(E::from(panic)),
        }
    };
    spawn(task)
}

/// Executes future [f] in a new tokio thread. Feeds the result into [tx] oneshot channel. Catches panics.
pub fn spawn_oneshot_with_catch_unwind<F, T, E>(
    f: F,
    tx: futures::channel::oneshot::Sender<Result<T, E>>,
) -> JoinHandle<Result<(), Result<T, E>>>
where
    F: Future<Output = Result<T, E>> + Send + 'static,
    T: Send + 'static,
    E: From<PanicError> + Send + 'static,
{
    let task = async move {
        match async_try_with_catch_unwind(f).await {
            Ok(result) => tx.send(result),
            Err(panic) => tx.send(Err(E::from(panic))),
        }
    };
    spawn(task)
}

/// Executes future [f] in a new tokio thread. Catches panics and feeds them into a [tx] mpsc channel
pub fn spawn_mpsc_with_catch_unwind<F, T, E>(
    f: F,
    mut tx: futures::channel::mpsc::Sender<Result<T, E>>,
) -> JoinHandle<()>
where
    F: Future<Output = ()> + Send + 'static,
    T: Send + 'static,
    E: From<PanicError> + Send + 'static,
{
    let task = async move {
        match async_try_with_catch_unwind(f).await {
            Ok(_) => (),
            Err(panic) => {
                tx.send(Err(E::from(panic))).await.ok();
            }
        }
    };
    spawn(task)
}

/// Executes fn [f] in a new tokio thread. Catches panics and feeds them into a [tx] mpsc channel.
pub fn spawn_blocking_mpsc_with_catch_unwind<F, R, T, E>(
    f: F,
    tx: tokio::sync::mpsc::Sender<Result<T, E>>,
) -> JoinHandle<()>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
    T: Send + 'static,
    E: From<PanicError> + Send + 'static,
{
    let task = move || match try_with_catch_unwind(f) {
        Ok(_) => (),
        Err(panic) => {
            tx.blocking_send(Err(E::from(panic))).ok();
        }
    };
    spawn_blocking(task)
}
