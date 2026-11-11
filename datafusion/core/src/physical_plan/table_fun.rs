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

//! TableFun (UDTFs) Physical Node

use std::any::Any;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use crate::error::{DataFusionError, Result};
use crate::physical_plan::{
    ColumnStatistics, DisplayFormatType, ExecutionPlan, Partitioning, PhysicalExpr,
};
use arrow::array::{new_null_array, Array, ArrayRef, NullArray, UInt64Array};
use arrow::compute::kernels;
use arrow::datatypes::{Field, FieldRef, Metadata, Schema, SchemaRef};
use arrow::error::Result as ArrowResult;
use arrow::record_batch::RecordBatch;
use datafusion_expr::TableFunctionRows;
use datafusion_physical_expr::TableFunctionExpr;

use super::expressions::{Column, PhysicalSortExpr};
use super::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use super::{RecordBatchStream, SendableRecordBatchStream, Statistics};
use crate::execution::context::TaskContext;
use async_trait::async_trait;
use futures::stream::Stream;
use futures::stream::StreamExt;

/// Execution plan for a table_fun
#[derive(Debug)]
pub struct TableFunExec {
    //
    // schema
    // input
    /// The expressions stored as tuples of (expression, output column name)
    expr: Vec<(Arc<dyn PhysicalExpr>, String)>,
    /// The schema once the node has been applied to the input
    schema: SchemaRef,
    /// The input plan
    input: Arc<dyn ExecutionPlan>,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
}

impl TableFunExec {
    /// Create a TableFun
    pub fn try_new(
        expr: Vec<(Arc<dyn PhysicalExpr>, String)>,
        input: Arc<dyn ExecutionPlan>,
    ) -> Result<Self> {
        let input_schema = input.schema();

        let fields: Result<Vec<Field>> = expr
            .iter()
            .map(|(e, name)| {
                let mut field = Field::new(
                    name,
                    e.data_type(&input_schema)?,
                    e.nullable(&input_schema)?,
                );
                field.set_metadata(get_field_metadata(e, &input_schema));

                Ok(field)
            })
            .collect();

        let mut total_fields: Vec<FieldRef> =
            input_schema.fields().iter().cloned().collect();
        total_fields.extend(fields.unwrap().into_iter().map(Arc::new));

        let schema = Schema::new_with_metadata(total_fields, HashMap::new());

        Ok(Self {
            expr,
            schema: Arc::new(schema),
            input: input.clone(),
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    /// The expressions stored as tuples of (expression, output column name)
    pub fn expr(&self) -> &[(Arc<dyn PhysicalExpr>, String)] {
        &self.expr
    }

    /// The input plan
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }
}

#[async_trait]
impl ExecutionPlan for TableFunExec {
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

    fn output_ordering(&self) -> Option<&[PhysicalSortExpr]> {
        self.input.output_ordering()
    }

    fn maintains_input_order(&self) -> bool {
        // tell optimizer this operator doesn't reorder its input
        true
    }

    fn relies_on_input_order(&self) -> bool {
        false
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match children.len() {
            1 => Ok(Arc::new(TableFunExec::try_new(
                self.expr.clone(),
                children[0].clone(),
            )?)),
            _ => Err(DataFusionError::Internal(
                "TableFunExec wrong number of children".to_string(),
            )),
        }
    }

    async fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let batch_size = context.session_config().batch_size.max(1);

        Ok(Box::pin(TableFunStream {
            schema: self.schema.clone(),
            expr: self.expr.iter().map(|x| x.0.clone()).collect(),
            input: self.input.execute(partition, context).await?,
            baseline_metrics: BaselineMetrics::new(&self.metrics, partition),
            batch_size,
            current: None,
            batches_without_yield: 0,
        }))
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

                write!(f, "TableFunExec: expr=[{}]", expr.join(", "))
            }
        }
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> Statistics {
        stats_table_fun(
            self.input.statistics(),
            self.expr.iter().map(|(e, _)| Arc::clone(e)),
        )
    }
}

/// If e is a direct column reference, returns the field level
/// metadata for that field, if any. Otherwise returns None
fn get_field_metadata(e: &Arc<dyn PhysicalExpr>, input_schema: &Schema) -> Metadata {
    let name = if let Some(column) = e.as_any().downcast_ref::<Column>() {
        column.name()
    } else {
        return Metadata::new();
    };

    input_schema
        .field_with_name(name)
        .ok()
        .map(|f| f.metadata().clone())
        .unwrap_or_default()
}

fn stats_table_fun(
    stats: Statistics,
    exprs: impl Iterator<Item = Arc<dyn PhysicalExpr>>,
) -> Statistics {
    let column_statistics = stats.column_statistics.map(|input_col_stats| {
        exprs
            .map(|e| {
                if let Some(col) = e.as_any().downcast_ref::<Column>() {
                    input_col_stats[col.index()].clone()
                } else {
                    // TODO stats: estimate more statistics from expressions
                    // (expressions should compute their statistics themselves)
                    ColumnStatistics::default()
                }
            })
            .collect()
    });

    Statistics {
        is_exact: stats.is_exact,
        num_rows: stats.num_rows,
        column_statistics,
        // TODO stats: knowing the type of the new columns we can guess the output size
        total_byte_size: None,
    }
}

/// Number of batches a stream returns in a row before it yields to the executor, so that a
/// table function with an endless output does not starve other tasks and can be cancelled
const YIELD_AFTER_BATCHES: usize = 64;

/// Input batch that is being expanded by table functions
struct InputBatch {
    /// Input columns, without the placeholder column of EmptyRelation
    columns: Vec<ArrayRef>,
    /// Values of every table function for the rows of the batch
    functions: Vec<Box<dyn TableFunctionRows>>,
    num_rows: usize,
    /// First row that still has values to return
    row: usize,
    /// Whether an output batch was returned for this input batch
    returned: bool,
}

/// TableFun iterator
struct TableFunStream {
    schema: SchemaRef,
    expr: Vec<Arc<dyn PhysicalExpr>>,
    input: SendableRecordBatchStream,
    baseline_metrics: BaselineMetrics,
    /// Maximum number of rows in an output batch
    batch_size: usize,
    current: Option<InputBatch>,
    /// Batches returned since the stream was pending for the last time
    batches_without_yield: usize,
}

impl TableFunStream {
    fn start_batch(&self, batch: &RecordBatch) -> ArrowResult<InputBatch> {
        let input_columns = self.schema.fields().len() - self.expr.len();
        let columns = if batch.num_columns() == input_columns {
            batch.columns().to_vec()
        } else {
            batch
                .columns()
                .iter()
                // Filtering out placeholder column of EmptyRelation
                .filter(|a| a.as_any().downcast_ref::<NullArray>().is_none())
                .cloned()
                .collect()
        };

        let functions = self
            .expr
            .iter()
            .map(|expr| {
                let t_expr = expr.as_any().downcast_ref::<TableFunctionExpr>().unwrap();
                t_expr.evaluate_table(batch)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(InputBatch {
            columns,
            functions,
            num_rows: batch.num_rows(),
            row: 0,
            returned: false,
        })
    }

    /// Returns the next output batch of the current input batch, at most `batch_size` rows.
    /// `None` when the input batch has no rows left. An input batch without any output rows
    /// still returns one empty batch.
    fn next_batch(&mut self) -> ArrowResult<Option<RecordBatch>> {
        let Some(input) = self.current.as_mut() else {
            return Ok(None);
        };
        let function_fields = &self.schema.fields()[input.columns.len()..];

        // Input row of every output row
        let mut rows: Vec<u64> = Vec::new();
        let mut sections: Vec<Vec<ArrayRef>> = vec![Vec::new(); input.functions.len()];

        while rows.len() < self.batch_size && input.row < input.num_rows {
            let wanted = self.batch_size - rows.len();
            let values = input
                .functions
                .iter_mut()
                .map(|function| function.next_values(input.row, wanted))
                .collect::<Result<Vec<_>>>()?;

            // A row is as long as its longest function, shorter ones are padded with nulls
            let len = values.iter().map(|v| v.len()).max().unwrap_or(0);
            if len > wanted {
                return Err(DataFusionError::Internal(format!(
                    "Table function returned {} values when at most {} were requested",
                    len, wanted
                ))
                .into());
            }

            if len > 0 {
                for ((values, sections), field) in values
                    .into_iter()
                    .zip(sections.iter_mut())
                    .zip(function_fields)
                {
                    let padding = len - values.len();
                    sections.push(values);
                    if padding > 0 {
                        sections.push(new_null_array(field.data_type(), padding));
                    }
                }
                rows.extend(std::iter::repeat(input.row as u64).take(len));
            }

            // Fewer values than requested: every function is done with the row
            if len < wanted {
                input.row += 1;
            }
        }

        if rows.is_empty() {
            if input.returned {
                return Ok(None);
            }
            input.returned = true;
            return Ok(Some(RecordBatch::new_empty(self.schema.clone())));
        }
        input.returned = true;

        let rows = UInt64Array::from(rows);
        let mut columns: Vec<ArrayRef> =
            Vec::with_capacity(input.columns.len() + sections.len());
        for column in &input.columns {
            columns.push(kernels::take::take(column.as_ref(), &rows, None)?);
        }
        for sections in sections {
            columns.push(kernels::concat::concat(
                sections
                    .iter()
                    .map(|a| a.as_ref())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )?);
        }

        RecordBatch::try_new(self.schema.clone(), columns).map(Some)
    }
}

impl Stream for TableFunStream {
    type Item = ArrowResult<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        loop {
            if self.current.is_some() {
                if self.batches_without_yield >= YIELD_AFTER_BATCHES {
                    self.batches_without_yield = 0;
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }

                match self.next_batch() {
                    Ok(Some(batch)) => {
                        self.batches_without_yield += 1;
                        let poll = Poll::Ready(Some(Ok(batch)));
                        return self.baseline_metrics.record_poll(poll);
                    }
                    Ok(None) => self.current = None,
                    Err(e) => {
                        self.current = None;
                        return self
                            .baseline_metrics
                            .record_poll(Poll::Ready(Some(Err(e))));
                    }
                }
            }

            match self.input.poll_next_unpin(cx) {
                Poll::Pending => {
                    self.batches_without_yield = 0;
                    return Poll::Pending;
                }
                Poll::Ready(Some(Ok(batch))) => match self.start_batch(&batch) {
                    Ok(input) => self.current = Some(input),
                    Err(e) => {
                        return self
                            .baseline_metrics
                            .record_poll(Poll::Ready(Some(Err(e))))
                    }
                },
                Poll::Ready(other) => {
                    return self.baseline_metrics.record_poll(Poll::Ready(other))
                }
            }
        }
    }
}

impl RecordBatchStream for TableFunStream {
    /// Get the schema
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
