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

use crate::physical_plan::RecordBatchStream;
use arrow::datatypes::SchemaRef;
use arrow::error::Result;
use arrow::record_batch::RecordBatch;
use futures::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Implements [RecordBatchStream] by exposing a predefined schema.
/// Useful for wrapping stream adapters.
pub struct StreamWithSchema<S> {
    stream: S,
    schema: SchemaRef,
}

impl<S> StreamWithSchema<S> {
    fn stream(self: Pin<&mut Self>) -> Pin<&mut S> {
        unsafe { self.map_unchecked_mut(|s| &mut s.stream) }
    }
}

impl<S> StreamWithSchema<S>
where
    S: Stream<Item = Result<RecordBatch>> + Send,
{
    pub fn wrap(schema: SchemaRef, stream: S) -> Self {
        StreamWithSchema { stream, schema }
    }
}

impl<S> Stream for StreamWithSchema<S>
where
    S: Stream<Item = Result<RecordBatch>> + Send,
{
    type Item = S::Item;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.stream().poll_next(cx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.stream.size_hint()
    }
}

impl<S> RecordBatchStream for StreamWithSchema<S>
where
    S: Stream<Item = Result<RecordBatch>> + Send,
{
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}