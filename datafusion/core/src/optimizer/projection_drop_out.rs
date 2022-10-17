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

//! Projection Drop Out optimizer rule removes useless
//! projections

use crate::error::{DataFusionError, Result};
use crate::logical_plan::plan::{Aggregate, Projection, Sort, Subquery};
use crate::logical_plan::{
    normalize_col, replace_col_to_expr, unnormalize_col, Column, DFField, DFSchema,
    Filter, LogicalPlan,
};
use crate::optimizer::optimizer::OptimizerConfig;
use crate::optimizer::optimizer::OptimizerRule;
use crate::optimizer::utils;
use datafusion_expr::Expr;
use std::collections::HashMap;
use std::sync::Arc;

/// Optimizer that removes useless projections
#[derive(Default)]
pub struct ProjectionDropOut {}

impl OptimizerRule for ProjectionDropOut {
    fn optimize(
        &self,
        plan: &LogicalPlan,
        optimizer_config: &OptimizerConfig,
    ) -> Result<LogicalPlan> {
        optimize_plan(plan, optimizer_config, false).map(|(p, _)| p)
    }

    fn name(&self) -> &str {
        "projection_drop_out"
    }
}

impl ProjectionDropOut {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Self {}
    }
}

/// Recursively transverses the logical plan removing projections that are not needed.
fn optimize_plan(
    plan: &LogicalPlan,
    _optimizer_config: &OptimizerConfig,
    projection_child: bool,
) -> Result<(LogicalPlan, Option<HashMap<Column, Expr>>)> {
    match plan {
        LogicalPlan::Projection(Projection {
            input,
            expr,
            schema,
            alias,
        }) => {
            let (new_input, rewritten_exprs) =
                optimize_plan(input, _optimizer_config, true)?;

            if let Some(rewritten_exprs) = rewritten_exprs {
                if !rewritten_exprs.is_empty() {
                    return Ok((
                        LogicalPlan::Projection(Projection {
                            expr: rewrite_projection_expr(
                                expr,
                                schema,
                                &new_input,
                                rewritten_exprs.iter().collect(),
                            )?,
                            input: Arc::new(new_input),
                            schema: schema.clone(),
                            alias: alias.clone(),
                        }),
                        None,
                    ));
                }
            }

            if let LogicalPlan::Projection(projection) = &new_input {
                match (rewrite_expr_map(projection)?, alias) {
                    (Some(rewritten_exprs), Some(_)) => {
                        let new_input = projection.input.clone();
                        return Ok((
                            LogicalPlan::Projection(Projection {
                                expr: rewrite_projection_expr(
                                    expr,
                                    schema,
                                    &new_input,
                                    rewritten_exprs.iter().collect(),
                                )?,
                                input: new_input,
                                schema: schema.clone(),
                                alias: alias.clone(),
                            }),
                            None,
                        ));
                    }
                    _ => (),
                }
            }

            Ok((
                LogicalPlan::Projection(Projection {
                    expr: expr.clone(),
                    input: Arc::new(new_input),
                    schema: schema.clone(),
                    alias: alias.clone(),
                }),
                None,
            ))
        }
        LogicalPlan::Aggregate(Aggregate {
            group_expr,
            aggr_expr,
            schema,
            input,
        }) => {
            let new_input = optimize_plan(input, _optimizer_config, false)?.0;
            if projection_child {
                if let LogicalPlan::Projection(projection) = &new_input {
                    let rewritten_exprs = rewrite_expr_map(projection)?;

                    if let Some(rewritten_exprs) = rewritten_exprs {
                        let new_input = projection.input.as_ref().clone();
                        let mut rewritten_expr = HashMap::new();
                        let rewrite_map = rewritten_exprs.iter().collect();

                        let res = rewrite_aggregate_expr(
                            group_expr,
                            schema,
                            &new_input,
                            &rewrite_map,
                        )?;
                        let new_group_expr = res.0;

                        rewritten_expr.extend(res.1.into_iter());

                        let res = rewrite_aggregate_expr(
                            aggr_expr,
                            schema,
                            &new_input,
                            &rewrite_map,
                        )?;
                        let new_aggr_expr = res.0;

                        rewritten_expr.extend(res.1.into_iter());

                        let schema = Arc::new(DFSchema::new_with_metadata(
                            schema
                                .fields()
                                .iter()
                                .map(|f| {
                                    if let Some(Expr::Column(value)) =
                                        rewritten_expr.get(&Column::from_qualified_name(
                                            f.qualified_name().as_str(),
                                        ))
                                    {
                                        DFField::new(
                                            value.relation.as_ref().map(|s| s.as_str()),
                                            value.name.as_str(),
                                            f.data_type().clone(),
                                            f.is_nullable(),
                                        )
                                    } else {
                                        f.clone()
                                    }
                                })
                                .collect::<Vec<DFField>>(),
                            HashMap::new(),
                        )?);

                        return Ok((
                            LogicalPlan::Aggregate(Aggregate {
                                group_expr: new_group_expr,
                                aggr_expr: new_aggr_expr,
                                schema: schema,
                                input: Arc::new(new_input),
                            }),
                            Some(rewritten_expr),
                        ));
                    }
                }
            }

            Ok((
                LogicalPlan::Aggregate(Aggregate {
                    group_expr: group_expr.clone(),
                    aggr_expr: aggr_expr.clone(),
                    schema: schema.clone(),
                    input: Arc::new(new_input),
                }),
                None,
            ))
        }
        LogicalPlan::Filter(Filter { predicate, input }) => {
            let (new_input, rewrite_map) = optimize_plan(input, _optimizer_config, true)?;

            let new_predicate = match (&new_input, &rewrite_map) {
                (LogicalPlan::Aggregate(_), Some(rewrite_map)) => {
                    replace_col_to_expr(predicate.clone(), &rewrite_map.iter().collect())?
                }
                _ => predicate.clone(),
            };

            Ok((
                LogicalPlan::Filter(Filter {
                    predicate: new_predicate,
                    input: Arc::new(new_input),
                }),
                rewrite_map,
            ))
        }
        LogicalPlan::Sort(Sort { expr, input }) => {
            let (new_input, _) = optimize_plan(input, _optimizer_config, true)?;

            Ok((
                LogicalPlan::Sort(Sort {
                    expr: expr.clone(),
                    input: Arc::new(new_input),
                }),
                None,
            ))
        }
        LogicalPlan::Subquery(Subquery {
            input,
            subqueries,
            schema,
        }) => {
            // TODO: subqueries are not optimized
            Ok((
                LogicalPlan::Subquery(Subquery {
                    input: Arc::new(
                        optimize_plan(input, _optimizer_config, false).map(|(p, _)| p)?,
                    ),
                    subqueries: subqueries.clone(),
                    schema: schema.clone(),
                }),
                None,
            ))
        }
        LogicalPlan::Join(_)
        | LogicalPlan::Window(_)
        | LogicalPlan::Analyze(_)
        | LogicalPlan::Union(_)
        | LogicalPlan::Limit(_)
        | LogicalPlan::CrossJoin(_)
        | LogicalPlan::TableUDFs(_)
        | LogicalPlan::Distinct(_) => {
            let expr = plan.expressions();
            let inputs = plan
                .inputs()
                .into_iter()
                .map(|i| optimize_plan(i, _optimizer_config, false).map(|(p, _)| p))
                .collect::<Result<Vec<LogicalPlan>>>()?;

            utils::from_plan(plan, &expr, &inputs).map(|p| (p, None))
        }
        LogicalPlan::TableScan(_)
        | LogicalPlan::Repartition(_)
        | LogicalPlan::EmptyRelation(_)
        | LogicalPlan::Values(_)
        | LogicalPlan::CreateExternalTable(_)
        | LogicalPlan::CreateMemoryTable(_)
        | LogicalPlan::CreateCatalogSchema(_)
        | LogicalPlan::DropTable(_)
        | LogicalPlan::Extension { .. } => Ok((plan.clone(), None)),
        LogicalPlan::Explain { .. } => Err(DataFusionError::Internal(
            "Unsupported logical plan: Explain must be root of the plan".to_string(),
        )),
    }
}

fn rewrite_expr_map(projection: &Projection) -> Result<Option<HashMap<Column, Expr>>> {
    let mut rewritten_exprs = HashMap::new();

    for e in projection.expr.iter() {
        match e {
            // TODO: refactor
            Expr::Column(col) => {
                let column = Column {
                    relation: projection.alias.clone(),
                    name: col.name.clone(),
                };
                rewritten_exprs.insert(column, e.clone());
            }
            Expr::Alias(expr, _) => match &**expr {
                Expr::Column(_) | Expr::ScalarVariable(_, _) | Expr::Literal(_) => {
                    let column = Column {
                        relation: projection.alias.clone(),
                        name: e.name(projection.schema.as_ref())?,
                    };
                    rewritten_exprs.insert(column, *expr.clone());
                }
                _ => return Ok(None),
            },
            Expr::ScalarVariable(_, _) | Expr::Literal(_) => {
                let column = Column {
                    relation: projection.alias.clone(),
                    name: e.name(projection.schema.as_ref())?,
                };
                rewritten_exprs.insert(column, e.clone());
            }
            _ => return Ok(None),
        }
    }

    return Ok(Some(rewritten_exprs));
}

fn rewrite_projection_expr(
    expr: &Vec<Expr>,
    schema: &Arc<DFSchema>,
    input: &LogicalPlan,
    rewrite_map: HashMap<&Column, &Expr>,
) -> Result<Vec<Expr>> {
    expr.iter()
        .map(|e| match replace_col_to_expr(e.clone(), &rewrite_map) {
            Ok(expr) => {
                let old_name = expr_name(e, schema)?;
                let new_e = normalize_col(unnormalize_col(expr), &input)?;

                return Ok(if old_name != expr_name(&new_e, schema)? {
                    Expr::Alias(Box::new(new_e), old_name)
                } else {
                    new_e
                });
            }
            Err(e) => Err(e),
        })
        .collect::<Result<Vec<Expr>>>()
}

fn rewrite_aggregate_expr(
    expr: &Vec<Expr>,
    schema: &Arc<DFSchema>,
    input: &LogicalPlan,
    rewrite_map: &HashMap<&Column, &Expr>,
) -> Result<(Vec<Expr>, HashMap<Column, Expr>)> {
    let mut rewritten_map: HashMap<Column, Expr> = HashMap::new();
    let exprs = expr
        .iter()
        .map(|e| {
            let expr = replace_col_to_expr(e.clone(), rewrite_map);
            if let Ok(expr) = &expr {
                let old_name = e.name(schema)?;
                let new_name =
                    normalize_col(unnormalize_col(expr.clone()), input)?.name(schema)?;

                if old_name != new_name {
                    rewritten_map.insert(
                        Column::from_qualified_name(old_name.as_str()),
                        Expr::Column(Column::from_qualified_name(new_name.as_str())),
                    );
                }
            }

            expr
        })
        .collect::<Result<Vec<Expr>>>()?;

    return Ok((exprs, rewritten_map));
}

fn expr_name(e: &Expr, schema: &Arc<DFSchema>) -> Result<String> {
    match e {
        Expr::Column(col) => Ok(col.name.clone()),
        _ => e.name(schema),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logical_plan::{
        col, count, lit, round, Expr, JoinType, LogicalPlanBuilder,
    };
    use crate::test::*;

    #[test]
    fn simple_projections() -> Result<()> {
        let table_scan = test_table_scan()?;

        // select * from (select * from (select * from table) a) x;
        let plan = LogicalPlanBuilder::from(table_scan)
            .project_with_alias(
                vec![col("a"), col("b"), col("c")],
                Some("a".to_string()),
            )?
            .project_with_alias(
                vec![col("a"), col("b"), col("c")],
                Some("b".to_string()),
            )?
            .project(vec![col("a"), col("b"), col("c")])?
            .build()?;

        let expected = "Projection: #b.a, #b.b, #b.c\
        \n  Projection: #test.a, #test.b, #test.c, alias=b\
        \n    TableScan: test projection=None";

        assert_optimized_plan_eq(&plan, expected);

        Ok(())
    }

    #[test]
    fn simple_projections_with_literals() -> Result<()> {
        let table_scan = test_table_scan()?;

        // select * from (select *, 2 from (select a, 1 from table) a) x;
        let plan = LogicalPlanBuilder::from(table_scan)
            .project_with_alias(vec![col("a"), lit(1)], Some("a".to_string()))?
            .project_with_alias(
                vec![col("a"), col("Int32(1)"), lit(2)],
                Some("b".to_string()),
            )?
            .project(vec![col("a"), col("Int32(1)"), col("Int32(2)")])?
            .build()?;

        let expected = "Projection: #b.a, #b.Int32(1), #b.Int32(2)\
        \n  Projection: #test.a, Int32(1), Int32(2), alias=b\
        \n    TableScan: test projection=None";

        assert_optimized_plan_eq(&plan, expected);

        Ok(())
    }

    #[test]
    fn simple_projections_skip_aliases() -> Result<()> {
        let table_scan = test_table_scan()?;

        // select * from (select id first, a second, 2 third from (select a id, 1 num from table) a) x;
        let plan = LogicalPlanBuilder::from(table_scan)
            .project_with_alias(
                vec![col("a").alias("id"), lit(1).alias("n")],
                Some("a".to_string()),
            )?
            .project_with_alias(
                vec![
                    col("id").alias("first"),
                    col("n").alias("second"),
                    lit(2).alias("third"),
                ],
                Some("b".to_string()),
            )?
            .project(vec![col("first"), col("second"), col("third")])?
            .build()?;

        let expected = "Projection: #b.first, #b.second, #b.third\
        \n  Projection: #test.a AS first, Int32(1) AS second, Int32(2) AS third, alias=b\
        \n    TableScan: test projection=None";

        assert_optimized_plan_eq(&plan, expected);

        Ok(())
    }

    #[test]
    fn simple_projections_with_functions() -> Result<()> {
        let table_scan = test_table_scan()?;

        // select * from (select round(id) first, num second, 2 third from (select a id, 1 num from table) a) x;
        let plan = LogicalPlanBuilder::from(table_scan.clone())
            .project_with_alias(
                vec![col("a").alias("id"), lit(1).alias("n")],
                Some("a".to_string()),
            )?
            .project_with_alias(
                vec![
                    round(col("id")).alias("first"),
                    col("n").alias("second"),
                    lit(2).alias("third"),
                ],
                Some("b".to_string()),
            )?
            .project(vec![col("first"), col("second"), col("third")])?
            .build()?;

        let expected = "Projection: #b.first, #b.second, #b.third\
        \n  Projection: round(#test.a) AS first, Int32(1) AS second, Int32(2) AS third, alias=b\
        \n    TableScan: test projection=None";

        assert_optimized_plan_eq(&plan, expected);

        // select * from (select id first, a second, 2 third from (select round(a) id, 1 num from table) a) x;
        let plan = LogicalPlanBuilder::from(table_scan)
            .project_with_alias(
                vec![round(col("a")).alias("id"), lit(1).alias("n")],
                Some("a".to_string()),
            )?
            .project_with_alias(
                vec![
                    col("id").alias("first"),
                    col("n").alias("second"),
                    lit(2).alias("third"),
                ],
                Some("b".to_string()),
            )?
            .project(vec![col("first"), col("second"), col("third")])?
            .build()?;

        let expected = "Projection: #b.first, #b.second, #b.third\
        \n  Projection: #a.id AS first, #a.n AS second, Int32(2) AS third, alias=b\
        \n    Projection: round(#test.a) AS id, Int32(1) AS n, alias=a\
        \n      TableScan: test projection=None";

        assert_optimized_plan_eq(&plan, expected);

        Ok(())
    }

    #[test]
    fn simple_projections_with_limits() -> Result<()> {
        let table_scan = test_table_scan()?;

        // select * from (select id first, a second, 2 third from (select a id, 1 anum from table limit 10) a limit 10) x;
        let plan = LogicalPlanBuilder::from(table_scan)
            .project_with_alias(
                vec![col("a").alias("id"), lit(1).alias("n")],
                Some("a".to_string()),
            )?
            .limit(None, Some(10))?
            .project_with_alias(
                vec![
                    col("id").alias("first"),
                    col("n").alias("second"),
                    lit(2).alias("third"),
                ],
                Some("b".to_string()),
            )?
            .limit(None, Some(5))?
            .project(vec![col("first"), col("second"), col("third")])?
            .build()?;

        let expected = "Projection: #b.first, #b.second, #b.third\
        \n  Limit: skip=None, fetch=5\
        \n    Projection: #a.id AS first, #a.n AS second, Int32(2) AS third, alias=b\
        \n      Limit: skip=None, fetch=10\
        \n        Projection: #test.a AS id, Int32(1) AS n, alias=a\
        \n          TableScan: test projection=None";

        assert_optimized_plan_eq(&plan, expected);

        Ok(())
    }

    #[test]
    fn simple_projections_with_filters() -> Result<()> {
        let table_scan = test_table_scan()?;

        // select a from (select a from (select a from table) a where a > 0) x;
        let plan = LogicalPlanBuilder::from(table_scan)
            .project_with_alias(
                vec![col("a"), col("b"), col("c")],
                Some("a".to_string()),
            )?
            .project_with_alias(
                vec![col("a"), col("b"), col("c")],
                Some("b".to_string()),
            )?
            .filter(col("a").gt(lit(0)))?
            .project(vec![col("a"), col("b"), col("c")])?
            .build()?;

        let expected = "Projection: #b.a, #b.b, #b.c\
        \n  Filter: #b.a > Int32(0)\
        \n    Projection: #test.a, #test.b, #test.c, alias=b\
        \n      TableScan: test projection=None";

        assert_optimized_plan_eq(&plan, expected);

        Ok(())
    }

    #[test]
    fn simple_projections_with_functions_and_filters() -> Result<()> {
        let table_scan = test_table_scan()?;

        // select * from (select round(id) first, num second, 2 third from (select a id, 1 num from table) a where id > 0) x;
        let plan = LogicalPlanBuilder::from(table_scan.clone())
            .project_with_alias(
                vec![col("a").alias("id"), lit(1).alias("n")],
                Some("a".to_string()),
            )?
            .project_with_alias(
                vec![
                    round(col("id")).alias("first"),
                    col("n").alias("second"),
                    lit(2).alias("third"),
                ],
                Some("b".to_string()),
            )?
            .filter(col("first").gt(lit(0)))?
            .project(vec![col("first"), col("second"), col("third")])?
            .build()?;

        let expected = "Projection: #b.first, #b.second, #b.third\
        \n  Filter: #b.first > Int32(0)\
        \n    Projection: round(#test.a) AS first, Int32(1) AS second, Int32(2) AS third, alias=b\
        \n      TableScan: test projection=None";

        assert_optimized_plan_eq(&plan, expected);

        Ok(())
    }

    #[test]
    fn simple_projections_with_sort() -> Result<()> {
        let table_scan = test_table_scan()?;

        // select a from (select a from (select a from table order by a) a) x;
        let plan = LogicalPlanBuilder::from(table_scan.clone())
            .project_with_alias(
                vec![col("a"), col("b"), col("c")],
                Some("a".to_string()),
            )?
            .sort(vec![col("a")])?
            .project_with_alias(
                vec![col("a"), col("b"), col("c")],
                Some("b".to_string()),
            )?
            .project(vec![col("a"), col("b"), col("c")])?
            .build()?;

        let expected = "Projection: #b.a, #b.b, #b.c\
        \n  Projection: #a.a, #a.b, #a.c, alias=b\
        \n    Sort: #a.a\
        \n      Projection: #test.a, #test.b, #test.c, alias=a\
        \n        TableScan: test projection=None";

        assert_optimized_plan_eq(&plan, expected);

        // select a from (select a from (select a from table) a) x order by a;
        let plan = LogicalPlanBuilder::from(table_scan)
            .project_with_alias(
                vec![col("a"), col("b"), col("c")],
                Some("a".to_string()),
            )?
            .project_with_alias(
                vec![col("a"), col("b"), col("c")],
                Some("b".to_string()),
            )?
            .project(vec![col("a"), col("b"), col("c")])?
            .sort(vec![col("a")])?
            .build()?;

        let expected = "Sort: #b.a\
        \n  Projection: #b.a, #b.b, #b.c\
        \n    Projection: #test.a, #test.b, #test.c, alias=b\
        \n      TableScan: test projection=None";

        assert_optimized_plan_eq(&plan, expected);

        Ok(())
    }

    #[test]
    fn simple_projections_with_functions_and_sort() -> Result<()> {
        let table_scan = test_table_scan()?;

        // select * from (select round(id) first, 2 second from (select a id from pg_class) a order by round(id)) x;
        let plan = LogicalPlanBuilder::from(table_scan.clone())
            .project_with_alias(vec![col("a").alias("id")], Some("a".to_string()))?
            .project_with_alias(
                vec![round(col("id")).alias("first"), lit(2).alias("second")],
                Some("b".to_string()),
            )?
            .sort(vec![col("first")])?
            .project(vec![col("first"), col("second")])?
            .build()?;

        let expected = "Projection: #b.first, #b.second\
        \n  Sort: #b.first\
        \n    Projection: round(#test.a) AS first, Int32(2) AS second, alias=b\
        \n      TableScan: test projection=None";

        assert_optimized_plan_eq(&plan, expected);

        Ok(())
    }

    #[test]
    fn projections_with_aggr() -> Result<()> {
        let table_scan = test_table_scan()?;

        // select a from (select a from (select a from table) a) x group by 1;
        let plan = LogicalPlanBuilder::from(table_scan.clone())
            .project_with_alias(vec![col("a")], Some("a".to_string()))?
            .project_with_alias(vec![col("a")], Some("b".to_string()))?
            .aggregate(vec![col("a")], Vec::<Expr>::new())?
            .project(vec![col("a")])?
            .build()?;

        let expected = "Projection: #test.a\
        \n  Aggregate: groupBy=[[#test.a]], aggr=[[]]\
        \n    TableScan: test projection=None";

        assert_optimized_plan_eq(&plan, expected);

        // select a, count(a) from (select a from (select a from table) a group by 1) x group by 1;
        let plan = LogicalPlanBuilder::from(table_scan)
            .project_with_alias(vec![col("a")], Some("a".to_string()))?
            .aggregate(vec![col("a")], Vec::<Expr>::new())?
            .project_with_alias(vec![col("a")], Some("b".to_string()))?
            .aggregate(vec![col("a")], vec![count(col("a"))])?
            .project(vec![col("a"), col("COUNT(b.a)")])?
            .build()?;

        let expected = "Projection: #test.a, #COUNT(test.a) AS COUNT(b.a)\
        \n  Aggregate: groupBy=[[#test.a]], aggr=[[COUNT(#test.a)]]\
        \n    Aggregate: groupBy=[[#test.a]], aggr=[[]]\
        \n      TableScan: test projection=None";

        assert_optimized_plan_eq(&plan, expected);

        Ok(())
    }

    #[test]
    fn projections_with_aggr_and_literals() -> Result<()> {
        let table_scan = test_table_scan()?;

        // select * from (select *, 2 from (select a, 1, count(a) from table group by 1, 2) a group by 1, 2, 3) x group by 1, 2, 3, 4;
        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![col("a"), lit(1)], vec![count(col("a"))])?
            .project_with_alias(
                vec![col("a"), col("Int32(1)"), col("COUNT(test.a)")],
                Some("a".to_string()),
            )?
            .aggregate(
                vec![col("a"), col("Int32(1)"), col("COUNT(test.a)")],
                Vec::<Expr>::new(),
            )?
            .project_with_alias(
                vec![col("a"), col("Int32(1)"), col("COUNT(test.a)"), lit(2)],
                Some("b".to_string()),
            )?
            .aggregate(
                vec![
                    col("a"),
                    col("Int32(1)"),
                    col("COUNT(test.a)"),
                    col("Int32(2)"),
                ],
                Vec::<Expr>::new(),
            )?
            .project(vec![
                col("a"),
                col("Int32(1)"),
                col("COUNT(test.a)"),
                col("Int32(2)"),
            ])?
            .build()?;

        let expected = "Projection: #test.a, #Int32(1), #COUNT(test.a), #Int32(2)\
        \n  Aggregate: groupBy=[[#test.a, #Int32(1), #COUNT(test.a), Int32(2)]], aggr=[[]]\
        \n    Aggregate: groupBy=[[#test.a, #Int32(1), #COUNT(test.a)]], aggr=[[]]\
        \n      Aggregate: groupBy=[[#test.a, Int32(1)]], aggr=[[COUNT(#test.a)]]\
        \n        TableScan: test projection=None";

        assert_optimized_plan_eq(&plan, expected);

        Ok(())
    }

    #[test]
    fn projections_with_aggr_skip_aliases() -> Result<()> {
        let table_scan = test_table_scan()?;

        // select * from (select id first, num second, 2 third, count(id) fourth from (select a id, 1 num from table) a group by 1, 2, 3) x;
        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(
                vec![col("a").alias("id"), lit(1).alias("num")],
                Vec::<Expr>::new(),
            )?
            .project_with_alias(vec![col("id"), col("num")], Some("a".to_string()))?
            .aggregate(vec![col("id"), col("num"), lit(2)], vec![count(col("id"))])?
            .project_with_alias(
                vec![
                    col("id").alias("first"),
                    col("num").alias("second"),
                    col("Int32(2)").alias("third"),
                    col("COUNT(a.id)"),
                ],
                Some("b".to_string()),
            )?
            .project(vec![
                col("first"),
                col("second"),
                col("third"),
                col("COUNT(a.id)"),
            ])?
            .build()?;

        let expected = "Projection: #b.first, #b.second, #b.third, #b.COUNT(a.id)\
        \n  Projection: #id AS first, #num AS second, #Int32(2) AS third, #COUNT(id) AS COUNT(a.id), alias=b\
        \n    Aggregate: groupBy=[[#id, #num, Int32(2)]], aggr=[[COUNT(#id)]]\
        \n      Aggregate: groupBy=[[#test.a AS id, Int32(1) AS num]], aggr=[[]]\
        \n        TableScan: test projection=None";

        assert_optimized_plan_eq(&plan, expected);

        Ok(())
    }

    #[test]
    fn projections_with_aggr_and_limits() -> Result<()> {
        let table_scan = test_table_scan()?;

        // select * from (select id first, num second, 2 third from (select a id, 1 num from table group by 1, 2 limit 10) a group by 1, 2, 3 limit 10) x;
        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(
                vec![col("a").alias("id"), lit(1).alias("num")],
                Vec::<Expr>::new(),
            )?
            .project_with_alias(vec![col("id"), col("num")], Some("a".to_string()))?
            .limit(None, Some(10))?
            .aggregate(vec![col("id"), col("num"), lit(2)], Vec::<Expr>::new())?
            .project_with_alias(
                vec![
                    col("id").alias("first"),
                    col("num").alias("second"),
                    col("Int32(2)").alias("third"),
                ],
                Some("b".to_string()),
            )?
            .limit(None, Some(10))?
            .project(vec![col("first"), col("second"), col("third")])?
            .build()?;

        let expected = "Projection: #b.first, #b.second, #b.third\
        \n  Limit: skip=None, fetch=10\
        \n    Projection: #a.id AS first, #a.num AS second, #Int32(2) AS third, alias=b\
        \n      Aggregate: groupBy=[[#a.id, #a.num, Int32(2)]], aggr=[[]]\
        \n        Limit: skip=None, fetch=10\
        \n          Projection: #id, #num, alias=a\
        \n            Aggregate: groupBy=[[#test.a AS id, Int32(1) AS num]], aggr=[[]]\
        \n              TableScan: test projection=None";

        assert_optimized_plan_eq(&plan, expected);

        Ok(())
    }

    #[test]
    fn projections_with_aggr_and_filter() -> Result<()> {
        let table_scan = test_table_scan()?;

        // select round(num) from (select a num from table) x where num > 0 group by 1 order by round(num);
        let plan = LogicalPlanBuilder::from(table_scan)
            .project_with_alias(vec![col("a").alias("num")], Some("a".to_string()))?
            .project_with_alias(vec![col("num")], Some("b".to_string()))?
            .filter(col("num").gt(lit(0)))?
            .aggregate(vec![round(col("num"))], Vec::<Expr>::new())?
            .project(vec![col("Round(b.num)")])?
            .sort(vec![col("Round(b.num)")])?
            .build()?;

        let expected = "Sort: #round(b.num)\
        \n  Projection: #round(b.num)\
        \n    Aggregate: groupBy=[[round(#b.num)]], aggr=[[]]\
        \n      Filter: #b.num > Int32(0)\
        \n        Projection: #test.a AS num, alias=b\
        \n          TableScan: test projection=None";

        assert_optimized_plan_eq(&plan, expected);

        Ok(())
    }

    #[test]
    fn projections_with_aggr_and_sort() -> Result<()> {
        let table_scan = test_table_scan()?;

        // select round(num) from (select a num from table) x group by 1 order by round(num);
        let plan = LogicalPlanBuilder::from(table_scan.clone())
            .project_with_alias(vec![col("a").alias("num")], Some("a".to_string()))?
            .project_with_alias(vec![col("num")], Some("b".to_string()))?
            .aggregate(vec![round(col("num"))], Vec::<Expr>::new())?
            .project(vec![col("Round(b.num)")])?
            .sort(vec![col("Round(b.num)")])?
            .build()?;

        let expected = "Sort: #round(b.num)\
        \n  Projection: #round(test.a) AS round(b.num)\
        \n    Aggregate: groupBy=[[round(#test.a)]], aggr=[[]]\
        \n      TableScan: test projection=None";

        assert_optimized_plan_eq(&plan, expected);

        // select * from (select round(num) from (select a num from table) x group by 1 order by round(num)) a;
        let plan = LogicalPlanBuilder::from(table_scan)
            .project_with_alias(vec![col("a").alias("num")], Some("a".to_string()))?
            .project_with_alias(vec![col("num")], Some("b".to_string()))?
            .aggregate(vec![round(col("num"))], Vec::<Expr>::new())?
            .project(vec![col("Round(b.num)")])?
            .sort(vec![col("Round(b.num)")])?
            .project_with_alias(vec![col("Round(b.num)")], Some("x".to_string()))?
            .project(vec![col("Round(b.num)")])?
            .build()?;

        let expected = "Projection: #x.round(b.num)\
        \n  Projection: #round(b.num), alias=x\
        \n    Sort: #round(b.num)\
        \n      Projection: #round(test.a) AS round(b.num)\
        \n        Aggregate: groupBy=[[round(#test.a)]], aggr=[[]]\
        \n          TableScan: test projection=None";

        assert_optimized_plan_eq(&plan, expected);

        Ok(())
    }

    #[test]
    fn left_join() -> Result<()> {
        let table_scan = test_table_scan()?;

        // select *, round(num) from (select a num from table) x left join (select a from (select a from (select a from table) a where a > 0) x) n on num = a;
        let right = LogicalPlanBuilder::from(table_scan.clone())
            .project_with_alias(vec![col("a")], Some("a".to_string()))?
            .project_with_alias(vec![col("a")], Some("a".to_string()))?
            .project_with_alias(vec![col("a")], Some("x".to_string()))?
            .project_with_alias(vec![col("a")], Some("x".to_string()))?
            .project_with_alias(vec![col("a")], Some("n".to_string()))?
            .project_with_alias(vec![col("a")], Some("n".to_string()))?
            .build()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .project_with_alias(vec![col("a").alias("num")], Some("x".to_string()))?
            .project_with_alias(vec![col("num")], Some("x".to_string()))?
            .join(&right, JoinType::Left, (vec!["num"], vec!["a"]))?
            .project_with_alias(
                vec![col("num"), col("a"), round(col("num"))],
                Some("b".to_string()),
            )?
            .build()?;

        let expected = "Projection: #x.num, #n.a, round(#x.num), alias=b\
        \n  Left Join: #x.num = #n.a\
        \n    Projection: #test.a AS num, alias=x\
        \n      TableScan: test projection=None\
        \n    Projection: #test.a, alias=n\
        \n      TableScan: test projection=None";

        assert_optimized_plan_eq(&plan, expected);

        Ok(())
    }

    #[test]
    fn union_all() -> Result<()> {
        let table_scan = test_table_scan()?;

        // select a num from table union all select a from (select a from (select a from table) a where a > 0) x;
        let plan = LogicalPlanBuilder::from(table_scan.clone())
            .project_with_alias(vec![col("a").alias("num")], Some("x".to_string()))?
            .union(
                LogicalPlanBuilder::from(table_scan)
                    .project_with_alias(vec![col("a")], Some("a".to_string()))?
                    .project_with_alias(vec![col("a")], Some("a".to_string()))?
                    .project_with_alias(vec![col("a")], Some("x".to_string()))?
                    .project_with_alias(vec![col("a")], Some("x".to_string()))?
                    .project(vec![col("a")])?
                    .build()?,
            )?
            .build()?;

        let expected = "Union\
        \n  Projection: #test.a AS num, alias=x\
        \n    TableScan: test projection=None\
        \n  Projection: #x.a\
        \n    Projection: #test.a, alias=x\
        \n      TableScan: test projection=None";

        assert_optimized_plan_eq(&plan, expected);

        Ok(())
    }

    fn assert_optimized_plan_eq(plan: &LogicalPlan, expected: &str) {
        let optimized_plan = optimize(plan).expect("failed to optimize plan");
        let formatted_plan = format!("{:?}", optimized_plan);
        assert_eq!(formatted_plan, expected);
    }

    fn optimize(plan: &LogicalPlan) -> Result<LogicalPlan> {
        let rule = ProjectionDropOut::new();
        rule.optimize(plan, &OptimizerConfig::new())
    }
}
