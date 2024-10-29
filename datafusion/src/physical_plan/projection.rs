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

use std::any::Any;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use crate::error::{DataFusionError, Result};
use crate::physical_plan::{
    DisplayFormatType, ExecutionPlan, OptimizerHints, Partitioning, PhysicalExpr,
};
use arrow::datatypes::{Field, Schema, SchemaRef};
use arrow::error::Result as ArrowResult;
use arrow::record_batch::RecordBatch;

use super::{RecordBatchStream, SendableRecordBatchStream};
use async_trait::async_trait;

use crate::physical_plan::expressions::Column;
use futures::stream::Stream;
use futures::stream::StreamExt;

/// Execution plan for a projection
#[derive(Debug)]
pub struct ProjectionExec {
    /// The projection expressions stored as tuples of (expression, output column name)
    expr: Vec<(Arc<dyn PhysicalExpr>, String)>,
    /// The schema once the projection has been applied to the input
    schema: SchemaRef,
    /// The input plan
    input: Arc<dyn ExecutionPlan>,
}

impl ProjectionExec {
    /// Create a projection on an input
    pub fn try_new(
        expr: Vec<(Arc<dyn PhysicalExpr>, String)>,
        input: Arc<dyn ExecutionPlan>,
    ) -> Result<Self> {
        let input_schema = input.schema();

        let fields: Result<Vec<_>> = expr
            .iter()
            .map(|(e, name)| {
                Ok(Field::new(
                    name,
                    e.data_type(&input_schema)?,
                    e.nullable(&input_schema)?,
                ))
            })
            .collect();

        let schema = Arc::new(Schema::new(fields?));

        Ok(Self {
            expr,
            schema,
            input: input.clone(),
        })
    }

    /// The projection expressions stored as tuples of (expression, output column name)
    pub fn expr(&self) -> &[(Arc<dyn PhysicalExpr>, String)] {
        &self.expr
    }

    /// The input plan
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }
}

#[async_trait]
impl ExecutionPlan for ProjectionExec {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Get the schema for this execution plan
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.input.clone()]
    }

    /// Get the output partitioning of this plan
    fn output_partitioning(&self) -> Partitioning {
        self.input.output_partitioning()
    }

    fn with_new_children(
        &self,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match children.len() {
            1 => Ok(Arc::new(ProjectionExec::try_new(
                self.expr.clone(),
                children[0].clone(),
            )?)),
            _ => Err(DataFusionError::Internal(
                "ProjectionExec wrong number of children".to_string(),
            )),
        }
    }

    async fn execute(&self, partition: usize) -> Result<SendableRecordBatchStream> {
        Ok(Box::pin(ProjectionStream {
            schema: self.schema.clone(),
            expr: self.expr.iter().map(|x| x.0.clone()).collect(),
            input: self.input.execute(partition).await?,
        }))
    }

    fn output_hints(&self) -> OptimizerHints {
        let input_hints = self.input.output_hints();
        if input_hints == OptimizerHints::default() {
            return OptimizerHints::default();
        }

        let input_schema = self.input.schema();
        let mut input_to_output = vec![None; input_schema.fields().len()];
        for out_i in 0..self.expr.len() {
            let column;
            if let Some(c) = self.expr[out_i].0.as_any().downcast_ref::<Column>() {
                column = c;
            } else {
                continue;
            }
            // If we project input to two output columns, we just end up picking one (and have incomplete analysis).
            input_to_output[column.index()] = Some(out_i);
        }

        let single_value_columns = input_hints
            .single_value_columns
            .iter()
            .filter_map(|i| input_to_output[*i])
            .collect();
        let mut sort_order = Vec::new();
        if let Some(in_so) = input_hints.sort_order {
            for in_col in in_so {
                if let Some(out_col) = input_to_output[in_col] {
                    sort_order.push(out_col);
                } else if input_hints.single_value_columns.contains(&in_col) {
                    continue;
                } else {
                    break;
                }
            }
        };

        // Becomes Some(true) if the first column of the first segment is mapped.
        let mut prefix_maintained = None::<bool>;
        let mut approximate_sort_order = Vec::new();
        for in_segment in input_hints.approximate_sort_order {
            let mut out_segment = Vec::new();
            for in_col in in_segment {
                if let Some(out_col) = input_to_output[in_col] {
                    if prefix_maintained.is_none() {
                        prefix_maintained = Some(true);
                    }
                    out_segment.push(out_col);
                } else if input_hints.single_value_columns.contains(&in_col) {
                    continue;
                } else {
                    // Some column is missing.  Note that handling this case right here --
                    // projections missing columns, and splitting up the sort order into multiple
                    // segments -- is the main purpose of approximate_sort_order.
                    if !out_segment.is_empty() {
                        approximate_sort_order.push(out_segment);
                        out_segment = Vec::new();
                    }
                    if prefix_maintained.is_none() {
                        prefix_maintained = Some(false);
                    }

                    break;
                }
            }
            if prefix_maintained.is_none() {
                // The whole first segment was single-value columns and it's gone now.
                prefix_maintained = Some(false);
            }

            if !out_segment.is_empty() {
                approximate_sort_order.push(out_segment);
            }
        }
        let approximate_sort_order_is_strict = input_hints.approximate_sort_order_is_strict;
        let approximate_sort_order_is_prefix = input_hints.approximate_sort_order_is_prefix && prefix_maintained == Some(true);

        OptimizerHints {
            sort_order: if sort_order.is_empty() { None } else { Some(sort_order) },
            approximate_sort_order,
            approximate_sort_order_is_prefix,
            approximate_sort_order_is_strict,
            single_value_columns,
        }
    }

    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => {
                let expr: Vec<String> = self
                    .expr
                    .iter()
                    .map(|(e, alias)| {
                        let e = e.to_string();
                        if &e != alias {
                            format!("{} as {}", e, alias)
                        } else {
                            e
                        }
                    })
                    .collect();

                write!(f, "ProjectionExec: expr=[{}]", expr.join(", "))
            }
        }
    }
}

fn batch_project(
    batch: &RecordBatch,
    expressions: &[Arc<dyn PhysicalExpr>],
    schema: &SchemaRef,
) -> ArrowResult<RecordBatch> {
    expressions
        .iter()
        .map(|expr| expr.evaluate(batch))
        .map(|r| r.map(|v| v.into_array(batch.num_rows())))
        .collect::<Result<Vec<_>>>()
        .map_or_else(
            |e| Err(DataFusionError::into_arrow_external_error(e)),
            |arrays| RecordBatch::try_new(schema.clone(), arrays),
        )
}

/// Projection iterator
struct ProjectionStream {
    schema: SchemaRef,
    expr: Vec<Arc<dyn PhysicalExpr>>,
    input: SendableRecordBatchStream,
}

impl Stream for ProjectionStream {
    type Item = ArrowResult<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        self.input.poll_next_unpin(cx).map(|x| match x {
            Some(Ok(batch)) => Some(batch_project(&batch, &self.expr, &self.schema)),
            other => other,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // same number of record batches
        self.input.size_hint()
    }
}

impl RecordBatchStream for ProjectionStream {
    /// Get the schema
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::physical_plan::csv::{CsvExec, CsvReadOptions};
    use crate::physical_plan::expressions::col;
    use crate::test;
    use futures::future;

    #[tokio::test]
    async fn project_first_column() -> Result<()> {
        let schema = test::aggr_test_schema();

        let partitions = 4;
        let path = test::create_partitioned_csv("aggregate_test_100.csv", partitions)?;

        let csv = CsvExec::try_new(
            &path,
            CsvReadOptions::new().schema(&schema),
            None,
            1024,
            None,
        )?;

        // pick column c1 and name it column c1 in the output schema
        let projection = ProjectionExec::try_new(
            vec![(col("c1", &schema)?, "c1".to_string())],
            Arc::new(csv),
        )?;

        let mut partition_count = 0;
        let mut row_count = 0;
        for partition in 0..projection.output_partitioning().partition_count() {
            partition_count += 1;
            let stream = projection.execute(partition).await?;

            row_count += stream
                .map(|batch| {
                    let batch = batch.unwrap();
                    assert_eq!(1, batch.num_columns());
                    batch.num_rows()
                })
                .fold(0, |acc, x| future::ready(acc + x))
                .await;
        }
        assert_eq!(partitions, partition_count);
        assert_eq!(100, row_count);

        Ok(())
    }
}
