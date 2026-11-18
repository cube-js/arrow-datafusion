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

//! Defines the projection execution plan. A projection determines which columns or expressions
//! are returned from a query. The SQL statement `SELECT a, b, a+b FROM t1` is an example
//! of a projection on table `t1` where the expressions `a`, `b`, and `a+b` are the
//! projection expressions. `SELECT` without `FROM` will only evaluate expressions.

use arrow::compute::concat;
use std::any::Any;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use crate::error::{DataFusionError, Result};
use crate::logical_plan::{Subquery, SubqueryType};
use crate::physical_plan::{DisplayFormatType, ExecutionPlan, Partitioning};
use arrow::array::{new_null_array, ArrayRef, BooleanArray};
use arrow::datatypes::{FieldRef, Schema, SchemaRef};
use arrow::error::{ArrowError, Result as ArrowResult};
use arrow::record_batch::RecordBatch;

use super::expressions::PhysicalSortExpr;
use super::{RecordBatchStream, SendableRecordBatchStream, Statistics};
use crate::execution::context::TaskContext;
use async_trait::async_trait;
use datafusion_common::{OuterQueryCursor, ScalarValue};
use futures::stream::Stream;
use futures::stream::StreamExt;
use tokio::sync::Mutex;

/// Execution plan for a sub query
#[derive(Debug)]
pub struct SubqueryExec {
    /// The input plan
    input: Arc<dyn ExecutionPlan>,
    /// Sub queries
    subqueries: Vec<Arc<dyn ExecutionPlan>>,
    /// Subquery types
    types: Vec<SubqueryType>,
    /// Merged schema
    schema: SchemaRef,
    /// Cursor used to send outer query column values to sub queries
    cursor: Arc<OuterQueryCursor>,
    /// Serializes use of `cursor` across the partitions of this plan
    cursor_lock: Arc<Mutex<()>>,
}

impl SubqueryExec {
    /// Create a projection on an input
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        subqueries: Vec<Arc<dyn ExecutionPlan>>,
        types: Vec<SubqueryType>,
        cursor: Arc<OuterQueryCursor>,
    ) -> Result<Self> {
        Self::try_new_with_lock(input, subqueries, types, cursor, Arc::default())
    }

    fn try_new_with_lock(
        input: Arc<dyn ExecutionPlan>,
        subqueries: Vec<Arc<dyn ExecutionPlan>>,
        types: Vec<SubqueryType>,
        cursor: Arc<OuterQueryCursor>,
        cursor_lock: Arc<Mutex<()>>,
    ) -> Result<Self> {
        let input_schema = input.schema();

        let mut total_fields: Vec<FieldRef> =
            input_schema.fields().iter().cloned().collect();
        for (q, t) in subqueries.iter().zip(types.iter()) {
            total_fields.extend(q.schema().fields().iter().map(|f| {
                let field = Subquery::transform_field(f, *t);
                // a scalar subquery yields NULL when it returns no rows, and
                // arrow rejects NULLs in a non-nullable RecordBatch column
                Arc::new(if matches!(t, SubqueryType::Scalar) {
                    field.with_nullable(true)
                } else {
                    field
                })
            }));
        }

        let merged_schema = Schema::new_with_metadata(total_fields, HashMap::new());

        if merged_schema.fields().len()
            != input.schema().fields().len() + subqueries.len()
        {
            return Err(DataFusionError::Plan("One or more correlated sub queries use same column names which is not supported".to_string()));
        }

        Ok(Self {
            input,
            subqueries,
            types,
            schema: Arc::new(merged_schema),
            cursor,
            cursor_lock,
        })
    }
}

#[async_trait]
impl ExecutionPlan for SubqueryExec {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Get the schema for this execution plan
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        let mut res = vec![self.input.clone()];
        res.extend(self.subqueries.iter().cloned());
        res
    }

    /// Get the output partitioning of this plan
    fn output_partitioning(&self) -> Partitioning {
        self.input.output_partitioning()
    }

    fn output_ordering(&self) -> Option<&[PhysicalSortExpr]> {
        self.input.output_ordering()
    }

    fn maintains_input_order(&self) -> bool {
        // tell optimizer this operator doesn't reorder its input
        true
    }

    fn relies_on_input_order(&self) -> bool {
        true
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() < 2 {
            return Err(DataFusionError::Internal(format!(
                "SubQueryExec should have not less than 2 children but got {}",
                children.len()
            )));
        }

        Ok(Arc::new(SubqueryExec::try_new_with_lock(
            children[0].clone(),
            children.iter().skip(1).cloned().collect(),
            self.types.clone(),
            self.cursor.clone(),
            self.cursor_lock.clone(),
        )?))
    }

    async fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let stream = self.input.execute(partition, context.clone()).await?;
        let cursor = self.cursor.clone();
        let subqueries = self.subqueries.clone();
        let types = self.types.clone();
        let context = context.clone();
        let size_hint = stream.size_hint();
        let schema = self.schema.clone();
        let cursor_lock = self.cursor_lock.clone();
        let res_stream = stream.then(move |batch| {
            let cursor = cursor.clone();
            let cursor_lock = cursor_lock.clone();
            let context = context.clone();
            let subqueries = subqueries.clone();
            let types = types.clone();
            let schema = schema.clone();
            async move {
                let batch = batch?;
                // Sub queries read the outer row from the shared cursor while
                // they run; hold the lock so concurrent partitions don't move it.
                let _cursor_guard = cursor_lock.lock().await;
                cursor.set_batch(Arc::new(batch.clone()))?;
                let mut subquery_arrays = vec![Vec::new(); subqueries.len()];
                for i in 0..batch.num_rows() {
                    cursor.set_position(i)?;
                    for (subquery_i, (subquery, subquery_type)) in
                        subqueries.iter().zip(types.iter()).enumerate()
                    {
                        subquery_arrays[subquery_i].push(
                            evaluate_subquery(subquery, *subquery_type, context.clone())
                                .await?,
                        );
                    }
                }
                let mut new_columns = batch.columns().to_vec();
                for subquery_array in subquery_arrays {
                    new_columns.push(concat(
                        subquery_array
                            .iter()
                            .map(|a| a.as_ref())
                            .collect::<Vec<_>>()
                            .as_slice(),
                    )?);
                }
                RecordBatch::try_new(schema.clone(), new_columns)
            }
        });
        Ok(Box::pin(SubQueryStream {
            schema: self.schema.clone(),
            stream: Box::pin(res_stream),
            size_hint,
        }))
    }

    fn fmt_as(
        &self,
        _t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        write!(f, "SubQueryExec")
    }

    fn statistics(&self) -> Statistics {
        // TODO
        self.input.statistics()
    }
}

/// Runs `subquery` for the outer row the cursor currently points at and returns
/// its result as a one-row array. Reads every batch the subquery produces, since
/// its rows may come in any of them (and some batches may be empty).
async fn evaluate_subquery(
    subquery: &Arc<dyn ExecutionPlan>,
    subquery_type: SubqueryType,
    context: Arc<TaskContext>,
) -> ArrowResult<ArrayRef> {
    let schema = subquery.schema();
    let fields = schema.fields();
    if fields.len() != 1 {
        return Err(ArrowError::ComputeError(format!(
            "Sub query should have only one column but got {}",
            fields.len()
        )));
    }
    let data_type = fields.first().unwrap().data_type();

    if subquery.output_partitioning().partition_count() != 1 {
        return Err(ArrowError::ComputeError(format!(
            "Sub query should have only one partition but got {}",
            subquery.output_partitioning().partition_count()
        )));
    }
    let mut stream = subquery.execute(0, context).await?;

    match subquery_type {
        SubqueryType::Scalar => {
            let mut value: Option<ArrayRef> = None;
            while let Some(subquery_batch) = stream.next().await {
                let column = subquery_batch?.column(0).clone();
                match (column.len(), &value) {
                    (0, _) => {}
                    (1, None) => value = Some(column),
                    _ => {
                        return Err(ArrowError::ComputeError(
                            "Sub query should return no more than one row".to_string(),
                        ))
                    }
                }
            }
            Ok(value.unwrap_or_else(|| new_null_array(data_type, 1)))
        }
        SubqueryType::Exists => {
            let mut exists = false;
            while let Some(subquery_batch) = stream.next().await {
                if subquery_batch?.num_rows() > 0 {
                    exists = true;
                    break;
                }
            }
            Ok(Arc::new(BooleanArray::from(vec![exists])))
        }
        SubqueryType::AnyAll => {
            // TODO: optimize?
            let mut scalars = vec![];
            while let Some(subquery_batch) = stream.next().await {
                let array_ref = subquery_batch?.column(0).clone();
                for i in 0..array_ref.len() {
                    scalars.push(ScalarValue::try_from_array(&array_ref, i)?);
                }
            }
            let list =
                ScalarValue::List(Some(Box::new(scalars)), Box::new(data_type.clone()));
            Ok(list.to_array())
        }
    }
}

/// SubQuery iterator
struct SubQueryStream {
    stream: Pin<Box<dyn Stream<Item = ArrowResult<RecordBatch>> + Send>>,
    schema: SchemaRef,
    size_hint: (usize, Option<usize>),
}

impl Stream for SubQueryStream {
    type Item = ArrowResult<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        self.stream.poll_next_unpin(cx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.size_hint
    }
}

impl RecordBatchStream for SubQueryStream {
    /// Get the schema
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
