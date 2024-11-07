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

//! Optimized plan for CrossJoin followed by Aggregate.
use crate::cube_ext::join::{
    left_cross_join, plan_cross_join, CrossJoinExec, SkewedLeftCrossJoin,
};
use crate::cube_ext::stream::StreamWithSchema;
use crate::error::Result;
use crate::execution::context::{ExecutionContextState, ExecutionProps};
use crate::logical_plan::{DFSchemaRef, Expr, LogicalPlan, UserDefinedLogicalNode};
use crate::optimizer::optimizer::OptimizerRule;
use crate::optimizer::utils::from_plan;
use crate::physical_plan::hash_aggregate::{create_accumulation_state, AggregateMode};
use crate::physical_plan::planner::{physical_name, ExtensionPlanner};
use crate::physical_plan::{hash_aggregate, PhysicalPlanner};
use crate::physical_plan::{
    AggregateExpr, ExecutionPlan, Partitioning, PhysicalExpr, SendableRecordBatchStream,
};
use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use futures::{stream, StreamExt};
use itertools::Itertools;
use std::any::Any;
use std::fmt::Formatter;
use std::sync::Arc;

#[derive(Debug)]
pub struct CrossJoinAgg {
    pub join: SkewedLeftCrossJoin,
    pub group_expr: Vec<Expr>,
    pub agg_expr: Vec<Expr>,
    pub schema: DFSchemaRef,
}

impl UserDefinedLogicalNode for CrossJoinAgg {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        self.join.inputs()
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    fn expressions(&self) -> Vec<Expr> {
        let mut es = self.join.expressions();
        es.extend_from_slice(&self.group_expr);
        es.extend_from_slice(&self.agg_expr);
        es
    }

    fn fmt_for_explain(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(
            f,
            "CrossJoinAgg: on {:?}, group_by = {:?}, aggregate = {:?} ",
            self.join.on, self.group_expr, self.agg_expr
        )
    }

    fn from_template(
        &self,
        exprs: &[Expr],
        inputs: &[LogicalPlan],
    ) -> Arc<dyn UserDefinedLogicalNode + Send + Sync> {
        assert!(self.agg_expr.len() + self.group_expr.len() <= exprs.len());
        let (exprs, agg_expr) = exprs.split_at(exprs.len() - self.agg_expr.len());
        let (exprs, group_expr) = exprs.split_at(exprs.len() - self.group_expr.len());
        let join = self.join.from_template_typed(exprs, inputs);
        Arc::new(CrossJoinAgg {
            join,
            group_expr: group_expr.to_vec(),
            agg_expr: agg_expr.to_vec(),
            schema: self.schema.clone(),
        })
    }
}

pub struct FoldCrossJoinAggregate;
impl OptimizerRule for FoldCrossJoinAggregate {
    fn optimize(
        &self,
        plan: &LogicalPlan,
        execution_props: &ExecutionProps,
    ) -> Result<LogicalPlan> {
        let inputs = plan
            .inputs()
            .into_iter()
            .map(|i| self.optimize(i, execution_props))
            .collect::<Result<Vec<_>>>()?;
        let exprs = plan.expressions();

        match plan {
            LogicalPlan::Aggregate {
                group_expr,
                aggr_expr,
                schema,
                input: join,
            } => match join.as_ref() {
                LogicalPlan::Extension { node } => {
                    if let Some(join) =
                        node.as_any().downcast_ref::<SkewedLeftCrossJoin>()
                    {
                        return Ok(LogicalPlan::Extension {
                            node: Arc::new(CrossJoinAgg {
                                join: join.clone(),
                                group_expr: group_expr.clone(),
                                agg_expr: aggr_expr.clone(),
                                schema: schema.clone(),
                            }),
                        });
                    }
                }
                _ => {}
            },
            _ => {}
        }

        from_plan(plan, &exprs, &inputs)
    }

    fn name(&self) -> &str {
        "fold_join_aggregate"
    }
}

pub struct CrossJoinAggPlanner;
impl ExtensionPlanner for CrossJoinAggPlanner {
    fn plan_extension(
        &self,
        planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        ctx_state: &ExecutionContextState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let node = match node.as_any().downcast_ref::<CrossJoinAgg>() {
            None => return Ok(None),
            Some(j) => j,
        };

        let inputs = physical_inputs;
        let join = plan_cross_join(planner, &node.join, inputs, ctx_state)?;
        let logical_join_schema = &node.join.schema;
        let physical_join_schema = join.schema();

        let mut group_expr = Vec::new();
        for e in &node.group_expr {
            let expr = planner.create_physical_expr(
                e,
                logical_join_schema,
                &physical_join_schema,
                ctx_state,
            )?;
            let name = physical_name(e, logical_join_schema)?;
            group_expr.push((expr, name));
        }
        let mut agg_expr = Vec::new();
        for e in &node.agg_expr {
            agg_expr.push(planner.create_aggregate_expr(
                e,
                logical_join_schema,
                &physical_join_schema,
                ctx_state,
            )?);
        }
        let schema = hash_aggregate::create_schema(
            &physical_join_schema,
            &group_expr,
            &agg_expr,
            AggregateMode::Full,
        )?;

        Ok(Some(Arc::new(CrossJoinAggExec {
            schema: Arc::new(schema),
            join,
            group_expr,
            agg_expr,
        })))
    }
}

#[derive(Debug)]
pub struct CrossJoinAggExec {
    pub schema: SchemaRef,
    pub join: CrossJoinExec,
    pub group_expr: Vec<(Arc<dyn PhysicalExpr>, String)>,
    pub agg_expr: Vec<Arc<dyn AggregateExpr>>,
}

#[async_trait]
impl ExecutionPlan for CrossJoinAggExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(1)
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        self.join.children()
    }

    fn with_new_children(
        &self,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(CrossJoinAggExec {
            schema: self.schema.clone(),
            join: self.join.with_new_children_typed(children),
            group_expr: self.group_expr.clone(),
            agg_expr: self.agg_expr.clone(),
        }))
    }

    async fn execute(&self, partition: usize) -> Result<SendableRecordBatchStream> {
        assert_eq!(partition, 0);
        let group_expr = self.group_expr.iter().map(|g| g.0.clone()).collect_vec();
        let left = self.join.compute_left().await?;

        let aggs = hash_aggregate::aggregate_expressions(
            &self.agg_expr,
            &AggregateMode::Full,
            self.group_expr.len(),
        )?;
        let mut accumulators = create_accumulation_state(&self.agg_expr)?;
        for partition in 0..self.join.right.output_partitioning().partition_count() {
            let mut batches = self.join.right.execute(partition).await?;
            while let Some(right) = batches.next().await {
                let right = right?;
                left_cross_join(
                    &left,
                    &right,
                    &self.join.schema,
                    self.join.on.as_ref(),
                    |joined, included| {
                        accumulators = hash_aggregate::group_aggregate_batch(
                            &AggregateMode::Full,
                            &group_expr,
                            &self.agg_expr,
                            joined,
                            std::mem::take(&mut accumulators),
                            &aggs,
                            |_, row| !included.value(row),
                        )?;
                        Ok(())
                    },
                )?;
            }
        }

        let out_schema = self.schema.clone();
        let r = hash_aggregate::create_batch_from_map(
            &AggregateMode::Full,
            &accumulators,
            self.group_expr.len(),
            &out_schema,
        )?;
        Ok(Box::pin(StreamWithSchema::wrap(
            out_schema,
            stream::once(async move { Ok(r) }),
        )))
    }
}
