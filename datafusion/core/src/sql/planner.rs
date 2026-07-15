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

//! SQL Query Planner (produces logical plan from SQL AST)

use std::collections::HashSet;
use std::iter;
use std::ops::RangeFrom;
use std::slice;
use std::str::FromStr;
use std::sync::{Arc, Mutex, RwLock};
use std::{convert::TryInto, vec};

use crate::catalog::TableReference;
use crate::datasource::TableProvider;
use crate::logical_plan::window_frames::{WindowFrame, WindowFrameUnits};
use crate::logical_plan::EmptyRelation;
use crate::logical_plan::Expr::Alias;
use crate::logical_plan::{
    and, builder::expand_qualified_wildcard, builder::expand_wildcard, col, lit,
    normalize_col, rewrite_udtfs_to_columns, Column, CreateMemoryTable, DFSchema,
    DFSchemaRef, DropTable, Expr, ExprSchemable, Like, LogicalPlan, LogicalPlanBuilder,
    Operator, PlanType, SubqueryType, ToDFSchema, ToStringifiedPlan,
};
use crate::optimizer::utils::exprlist_to_columns;
use crate::prelude::JoinType;
use crate::scalar::ScalarValue;
use crate::sql::utils::{
    find_udtf_exprs, make_decimal_type, normalize_ident, realias_duplicate_expr_aliases,
};
use crate::{
    error::{DataFusionError, Result},
    physical_plan::aggregates,
    physical_plan::udaf::AggregateUDF,
    physical_plan::udf::ScalarUDF,
    physical_plan::udtf::TableUDF,
    sql::parser::{CreateExternalTable, FileType, Statement as DFStatement},
};
use arrow::datatypes::*;
use datafusion_expr::{
    window_function::{BuiltInWindowFunction, WindowFunction},
    BuiltinScalarFunction,
};
use hashbrown::HashMap;

use datafusion_expr::expr::GroupingSet;
use sqlparser::ast::{
    AccessExpr, BinaryOperator, CaseWhen, CastKind, DataType as SQLDataType,
    DateTimeField, Distinct, ExactNumberInfo, Expr as SQLExpr, Fetch, FunctionArg,
    FunctionArgExpr, FunctionArgumentClause, FunctionArguments, GroupByExpr, Ident, Join,
    JoinConstraint, JoinOperator, LimitClause, ObjectName, ObjectNamePart, OrderBy,
    OrderByKind, OrderByOptions, Query, Select, SelectItem,
    SelectItemQualifiedWildcardKind, SetExpr, SetOperator, SetQuantifier, Subscript,
    TableFactor, TableWithJoins, TrimWhereField, UnaryOperator, Value, ValueWithSpan,
    Values as SQLValues, WindowType,
};
use sqlparser::ast::{ColumnDef as SQLColumnDef, ColumnOption};
use sqlparser::ast::{ObjectType, OrderByExpr, Statement};
use sqlparser::parser::ParserError::ParserError;

use super::{
    parser::DFParser,
    utils::{
        can_columns_satisfy_exprs, expr_as_column_expr, extract_aliases,
        find_aggregate_exprs, find_column_exprs, find_window_exprs, rebase_expr,
        resolve_aliases_to_exprs, resolve_positions_to_exprs,
    },
};
use crate::logical_plan::builder::{project_with_alias, table_udfs};
use crate::logical_plan::plan::{
    Analyze, CreateCatalogSchema, CreateExternalTable as PlanCreateExternalTable, Explain,
};

/// The ContextProvider trait allows the query planner to obtain meta-data about tables and
/// functions referenced in SQL statements
pub trait ContextProvider {
    /// Getter for a datasource
    fn get_table_provider(&self, name: TableReference) -> Option<Arc<dyn TableProvider>>;
    /// Getter for a UDF description
    fn get_function_meta(&self, name: &str) -> Option<Arc<ScalarUDF>>;
    /// Getter for a UDTF description
    fn get_table_function_meta(&self, name: &str) -> Option<Arc<TableUDF>>;
    /// Getter for a UDAF description
    fn get_aggregate_meta(&self, name: &str) -> Option<Arc<AggregateUDF>>;
    /// Getter for system/user-defined variable type
    fn get_variable_type(&self, variable_names: &[String]) -> Option<DataType>;
}

/// SQL query planner
pub struct SqlToRel<'a, S: ContextProvider> {
    schema_provider: &'a S,
    table_columns_precedence_over_projection: bool,
    context: SqlToRelContext,
    subquery_alias_iter: Arc<Mutex<RangeFrom<u32>>>,
}

/// Planning context
#[derive(Default)]
pub struct SqlToRelContext {
    outer_query_context_schema: Vec<DFSchemaRef>,
    subqueries_plans: Option<RwLock<Vec<(LogicalPlan, SubqueryType)>>>,
    ctes: HashMap<String, LogicalPlan>,
}

impl SqlToRelContext {
    /// Used to copy new version context based on the current one
    pub fn fork(&self) -> Self {
        Self {
            outer_query_context_schema: self.outer_query_context_schema.clone(),
            subqueries_plans: None,
            ctes: self.ctes.clone(),
        }
    }

    fn add_subquery_plan(
        &self,
        plan: LogicalPlan,
        subquery_type: SubqueryType,
    ) -> Result<()> {
        self.subqueries_plans.as_ref().ok_or_else(|| DataFusionError::Plan(format!("Sub query {:?} planned outside of sub query context. This type of sub query isn't supported", plan)))?.write().unwrap().push((plan, subquery_type));
        Ok(())
    }

    fn subqueries_plans(&self) -> Result<Option<Vec<(LogicalPlan, SubqueryType)>>> {
        Ok(if let Some(subqueries) = self.subqueries_plans.as_ref() {
            Some(
                subqueries
                    .read()
                    .map_err(|e| DataFusionError::Plan(e.to_string()))?
                    .iter()
                    .cloned()
                    .collect(),
            )
        } else {
            None
        })
    }
}

impl<'a, S: ContextProvider> SqlToRel<'a, S> {
    /// Create a new query planner
    pub fn new(schema_provider: &'a S) -> Self {
        Self::new_with_options(schema_provider, false)
    }

    /// Create a new query planner
    pub fn new_with_options(
        schema_provider: &'a S,
        table_columns_precedence_over_projection: bool,
    ) -> Self {
        SqlToRel {
            schema_provider,
            table_columns_precedence_over_projection,
            context: SqlToRelContext::default(),
            subquery_alias_iter: Arc::new(Mutex::new(0..)),
        }
    }

    /// Creates new version of SqlToRel with forked planning context
    pub fn with_context(&self, f: impl FnOnce(&mut SqlToRelContext)) -> Self {
        let mut context = self.context.fork();
        f(&mut context);
        SqlToRel {
            schema_provider: self.schema_provider,
            table_columns_precedence_over_projection: self
                .table_columns_precedence_over_projection,
            context,
            subquery_alias_iter: Arc::clone(&self.subquery_alias_iter),
        }
    }

    /// Generate a logical plan from an DataFusion SQL statement
    pub fn statement_to_plan(&self, statement: DFStatement) -> Result<LogicalPlan> {
        match statement {
            DFStatement::CreateExternalTable(s) => self.external_table_to_plan(s),
            DFStatement::Statement(s) => self.sql_statement_to_plan(*s),
        }
    }

    /// Generate a logical plan from an SQL statement
    pub fn sql_statement_to_plan(&self, sql: Statement) -> Result<LogicalPlan> {
        match sql {
            Statement::Explain {
                verbose,
                statement,
                analyze,
                ..
            } => self.explain_statement_to_plan(verbose, analyze, *statement),
            Statement::Query(query) => self.query_to_plan(*query),
            Statement::ShowVariable { variable } => self.show_variable_to_plan(&variable),
            Statement::CreateTable(sqlparser::ast::CreateTable {
                query: Some(query),
                name,
                columns,
                constraints,
                ..
            }) if columns.is_empty() && constraints.is_empty() => {
                let plan = self.query_to_plan(*query)?;

                Ok(LogicalPlan::CreateMemoryTable(CreateMemoryTable {
                    name: name.to_string(),
                    input: Arc::new(plan),
                }))
            }
            Statement::CreateTable(_) => Err(DataFusionError::NotImplemented(
                "Only `CREATE TABLE table_name AS SELECT ...` statement is supported"
                    .to_string(),
            )),
            Statement::CreateSchema {
                schema_name,
                if_not_exists,
                ..
            } => Ok(LogicalPlan::CreateCatalogSchema(CreateCatalogSchema {
                schema_name: schema_name.to_string(),
                if_not_exists,
                schema: Arc::new(DFSchema::empty()),
            })),
            Statement::Drop {
                object_type: ObjectType::Table,
                if_exists,
                names,
                ..
            } =>
            // We don't support cascade and purge for now.
            {
                Ok(LogicalPlan::DropTable(DropTable {
                    name: names.first().unwrap().to_string(),
                    if_exists,
                    schema: DFSchemaRef::new(DFSchema::empty()),
                }))
            }

            Statement::ShowTables {
                extended,
                full,
                show_options,
                ..
            } => {
                let has_extra = show_options.show_in.is_some()
                    || show_options.filter_position.is_some();
                self.show_tables_to_plan(extended, full, has_extra)
            }

            Statement::ShowColumns {
                extended,
                full,
                show_options,
            } => {
                let table_name = show_options
                    .show_in
                    .and_then(|s| s.parent_name)
                    .ok_or_else(|| {
                        DataFusionError::Plan(
                            "SHOW COLUMNS requires a table name".to_string(),
                        )
                    })?;
                let has_filter = show_options.filter_position.is_some();
                self.show_columns_to_plan(extended, full, &table_name, has_filter)
            }
            _ => Err(DataFusionError::NotImplemented(format!(
                "Unsupported SQL statement: {:?}",
                sql
            ))),
        }
    }

    /// Generate a logical plan from a "SHOW TABLES" query
    fn show_tables_to_plan(
        &self,
        extended: bool,
        full: bool,
        has_extra: bool,
    ) -> Result<LogicalPlan> {
        if self.has_table("information_schema", "tables") {
            // we only support the basic "SHOW TABLES"
            // https://github.com/apache/arrow-datafusion/issues/3188
            if has_extra || full || extended {
                Err(DataFusionError::Plan(
                    "Unsupported parameters to SHOW TABLES".to_string(),
                ))
            } else {
                let query = "SELECT * FROM information_schema.tables;";
                let mut rewrite = DFParser::parse_sql(query)?;
                assert_eq!(rewrite.len(), 1);
                self.statement_to_plan(rewrite.pop_front().unwrap())
            }
        } else {
            Err(DataFusionError::Plan(
                "SHOW TABLES is not supported unless information_schema is enabled"
                    .to_string(),
            ))
        }
    }

    /// Generate a logic plan from an SQL query
    pub fn query_to_plan(&self, query: Query) -> Result<LogicalPlan> {
        self.query_to_plan_with_alias(query, None)
    }

    /// Generate a logic plan from an SQL query with optional alias
    pub fn query_to_plan_with_alias(
        &self,
        query: Query,
        alias: Option<String>,
    ) -> Result<LogicalPlan> {
        let set_expr = *query.body;

        let mut ctes = self.context.ctes.clone();
        if let Some(with) = query.with {
            // Process CTEs from top to bottom
            // do not allow self-references
            for cte in with.cte_tables {
                // A `WITH` block can't use the same name for many times
                let cte_name: &str = cte.alias.name.value.as_ref();
                if ctes.contains_key(cte_name) {
                    return Err(DataFusionError::SQL(ParserError(format!(
                        "WITH query name {:?} specified more than once",
                        cte_name
                    ))));
                }

                let with_cte_context = self.with_context(|c| c.ctes = ctes.clone());
                // create logical plan & pass backreferencing CTEs
                let logical_plan = with_cte_context.query_to_plan_with_alias(
                    *cte.query,
                    Some(cte.alias.name.value.clone()),
                )?;
                ctes.insert(cte.alias.name.value, logical_plan);
            }
        }
        let with_cte_context = self.with_context(|c| c.ctes = ctes);

        let order_by_exprs = match query.order_by {
            Some(OrderBy {
                kind: OrderByKind::Expressions(exprs),
                ..
            }) => exprs,
            _ => vec![],
        };
        // ORDER BY expressions are passed down because `SELECT DISTINCT ON`
        // dedupes by the first row per group as defined by the query's ORDER BY
        let plan = with_cte_context.set_expr_to_plan(set_expr, alias, &order_by_exprs)?;
        let plan = with_cte_context.order_by(plan, order_by_exprs)?;

        let (skip, limit) = match query.limit_clause {
            Some(LimitClause::LimitOffset { limit, offset, .. }) => {
                (offset.map(|o| o.value), limit)
            }
            Some(LimitClause::OffsetCommaLimit { offset, limit }) => {
                (Some(offset), Some(limit))
            }
            None => (None, None),
        };
        with_cte_context.limit(plan, skip, limit, query.fetch)
    }

    fn set_expr_to_plan(
        &self,
        set_expr: SetExpr,
        alias: Option<String>,
        order_by: &[OrderByExpr],
    ) -> Result<LogicalPlan> {
        match set_expr {
            SetExpr::Select(s) => self.select_to_plan(*s, alias, order_by),
            SetExpr::Values(v) => self.sql_values_to_plan(v),
            SetExpr::SetOperation {
                op,
                left,
                right,
                set_quantifier,
            } => {
                let left_plan = self.set_expr_to_plan(*left, None, &[])?;
                let right_plan = self.set_expr_to_plan(*right, None, &[])?;
                let all = matches!(
                    set_quantifier,
                    SetQuantifier::All | SetQuantifier::AllByName
                );
                match op {
                    SetOperator::Union if all => LogicalPlanBuilder::from(left_plan)
                        .union(right_plan)?
                        .build(),
                    SetOperator::Union => LogicalPlanBuilder::from(left_plan)
                        .union_distinct(right_plan)?
                        .build(),
                    SetOperator::Intersect => {
                        LogicalPlanBuilder::intersect(left_plan, right_plan, all)
                    }
                    SetOperator::Except => {
                        LogicalPlanBuilder::except(left_plan, right_plan, all)
                    }
                    _ => Err(DataFusionError::NotImplemented(format!(
                        "Set operator {:?} is not implemented",
                        op
                    ))),
                }
            }
            SetExpr::Query(q) => self.query_to_plan_with_alias(*q, None),
            _ => Err(DataFusionError::NotImplemented(format!(
                "Query {} not implemented yet",
                set_expr
            ))),
        }
    }

    /// Generate a logical plan from a CREATE EXTERNAL TABLE statement
    pub fn external_table_to_plan(
        &self,
        statement: CreateExternalTable,
    ) -> Result<LogicalPlan> {
        let CreateExternalTable {
            name,
            columns,
            file_type,
            has_header,
            location,
            table_partition_cols,
        } = statement;

        // semantic checks
        match file_type {
            FileType::CSV => {}
            FileType::Parquet => {
                if !columns.is_empty() {
                    return Err(DataFusionError::Plan(
                        "Column definitions can not be specified for PARQUET files."
                            .into(),
                    ));
                }
            }
            FileType::NdJson => {}
            FileType::Avro => {}
        };

        let schema = self.build_schema(columns)?;

        Ok(LogicalPlan::CreateExternalTable(PlanCreateExternalTable {
            schema: schema.to_dfschema_ref()?,
            name,
            location,
            file_type,
            has_header,
            table_partition_cols,
        }))
    }

    /// Generate a plan for EXPLAIN ... that will print out a plan
    ///
    pub fn explain_statement_to_plan(
        &self,
        verbose: bool,
        analyze: bool,
        statement: Statement,
    ) -> Result<LogicalPlan> {
        let plan = self.sql_statement_to_plan(statement)?;
        let plan = Arc::new(plan);
        let schema = LogicalPlan::explain_schema();
        let schema = schema.to_dfschema_ref()?;

        if analyze {
            Ok(LogicalPlan::Analyze(Analyze {
                verbose,
                input: plan,
                schema,
            }))
        } else {
            let stringified_plans =
                vec![plan.to_stringified(PlanType::InitialLogicalPlan)];
            Ok(LogicalPlan::Explain(Explain {
                verbose,
                plan,
                stringified_plans,
                schema,
            }))
        }
    }

    fn build_schema(&self, columns: Vec<SQLColumnDef>) -> Result<Schema> {
        let mut fields = Vec::with_capacity(columns.len());

        for column in columns {
            let data_type = self.make_data_type(&column.data_type)?;
            let allow_null = column
                .options
                .iter()
                .any(|x| x.option == ColumnOption::Null);
            fields.push(Field::new(&column.name.value, data_type, allow_null));
        }

        Ok(Schema::new(fields))
    }

    /// Maps the SQL type to the corresponding Arrow `DataType`
    fn make_data_type(&self, sql_type: &SQLDataType) -> Result<DataType> {
        match sql_type {
            SQLDataType::BigInt(_) => Ok(DataType::Int64),
            SQLDataType::Int(_) | SQLDataType::Integer(_) => Ok(DataType::Int32),
            SQLDataType::SmallInt(_) => Ok(DataType::Int16),
            SQLDataType::Char(_) | SQLDataType::Varchar(_) | SQLDataType::Text => {
                Ok(DataType::Utf8)
            }
            SQLDataType::Decimal(info) => {
                let (precision, scale) = exact_number_info_to_precision_scale(info);
                make_decimal_type(precision, scale)
            }
            SQLDataType::Float(_) => Ok(DataType::Float32),
            SQLDataType::Real => Ok(DataType::Float32),
            SQLDataType::Double(_) | SQLDataType::DoublePrecision => {
                Ok(DataType::Float64)
            }
            SQLDataType::Boolean => Ok(DataType::Boolean),
            SQLDataType::Date => Ok(DataType::Date32),
            SQLDataType::Time(..) => Ok(DataType::Time64(TimeUnit::Millisecond)),
            SQLDataType::Timestamp(..) => {
                Ok(DataType::Timestamp(TimeUnit::Nanosecond, None))
            }
            _ => Err(DataFusionError::NotImplemented(format!(
                "The SQL data type {:?} is not implemented",
                sql_type
            ))),
        }
    }

    fn plan_from_tables(&self, from: Vec<TableWithJoins>) -> Result<Vec<LogicalPlan>> {
        match from.len() {
            0 => Ok(vec![LogicalPlanBuilder::empty(true).build()?]),
            _ => from
                .into_iter()
                .map(|t| self.plan_table_with_joins(t))
                .collect::<Result<Vec<_>>>(),
        }
    }

    fn plan_table_with_joins(&self, t: TableWithJoins) -> Result<LogicalPlan> {
        let left = self.create_relation(t.relation)?;
        match t.joins.len() {
            0 => Ok(left),
            _ => {
                let mut joins = t.joins.into_iter();
                let mut left = self.parse_relation_join(left, joins.next().unwrap())?;
                for join in joins {
                    left = self.parse_relation_join(left, join)?;
                }
                Ok(left)
            }
        }
    }

    fn parse_relation_join(&self, left: LogicalPlan, join: Join) -> Result<LogicalPlan> {
        let right = self.create_relation(join.relation)?;
        match join.join_operator {
            JoinOperator::Left(constraint) | JoinOperator::LeftOuter(constraint) => {
                self.parse_join(left, right, constraint, JoinType::Left)
            }
            JoinOperator::Right(constraint) | JoinOperator::RightOuter(constraint) => {
                self.parse_join(left, right, constraint, JoinType::Right)
            }
            JoinOperator::Inner(constraint) | JoinOperator::Join(constraint) => {
                self.parse_join(left, right, constraint, JoinType::Inner)
            }
            JoinOperator::FullOuter(constraint) => {
                self.parse_join(left, right, constraint, JoinType::Full)
            }
            JoinOperator::CrossJoin(_) => self.parse_cross_join(left, &right),
            other => Err(DataFusionError::NotImplemented(format!(
                "Unsupported JOIN operator {:?}",
                other
            ))),
        }
    }

    fn parse_cross_join(
        &self,
        left: LogicalPlan,
        right: &LogicalPlan,
    ) -> Result<LogicalPlan> {
        LogicalPlanBuilder::from(left).cross_join(right)?.build()
    }

    fn parse_join(
        &self,
        left: LogicalPlan,
        right: LogicalPlan,
        constraint: JoinConstraint,
        join_type: JoinType,
    ) -> Result<LogicalPlan> {
        match constraint {
            JoinConstraint::On(sql_expr) => {
                let mut keys: Vec<(Column, Column)> = vec![];
                let join_schema = left.schema().join(right.schema())?;

                // parse ON expression
                let expr = self.sql_to_rex(sql_expr, &join_schema)?;

                // expression that didn't match equi-join pattern
                let mut filter = vec![];

                // extract join keys
                extract_join_keys(expr, &mut keys, &mut filter);

                let mut cols = HashSet::new();
                exprlist_to_columns(&filter, &mut cols)?;

                let (left_keys, right_keys): (Vec<Column>, Vec<Column>) =
                    keys.into_iter().unzip();

                // return the logical plan representing the join
                if left_keys.is_empty() {
                    // When we don't have join keys, use cross join
                    let join = LogicalPlanBuilder::from(left).cross_join(&right)?;

                    join.filter(filter.into_iter().reduce(Expr::and).unwrap())?
                        .build()
                } else if filter.is_empty() {
                    let join = LogicalPlanBuilder::from(left).join(
                        &right,
                        join_type,
                        (left_keys, right_keys),
                    )?;
                    join.build()
                } else if join_type == JoinType::Inner {
                    let join = LogicalPlanBuilder::from(left).join(
                        &right,
                        join_type,
                        (left_keys, right_keys),
                    )?;
                    join.filter(filter.into_iter().reduce(Expr::and).unwrap())?
                        .build()
                }
                // Left join with all non-equijoin expressions from the right
                // l left join r
                // on l1=r1 and r2 > [..]
                else if join_type == JoinType::Left
                    && cols.iter().all(
                        |Column {
                             relation: qualifier,
                             name,
                         }| {
                            right
                                .schema()
                                .field_with_name(qualifier.as_deref(), name)
                                .is_ok()
                        },
                    )
                {
                    let join_filter_init = filter.remove(0);
                    LogicalPlanBuilder::from(left)
                        .join(
                            &LogicalPlanBuilder::from(right)
                                .filter(
                                    filter
                                        .into_iter()
                                        .fold(join_filter_init, |acc, e| acc.and(e)),
                                )?
                                .build()?,
                            join_type,
                            (left_keys, right_keys),
                        )?
                        .build()
                }
                // Right join with all non-equijoin expressions from the left
                // l right join r
                // on l1=r1 and l2 > [..]
                else if join_type == JoinType::Right
                    && cols.iter().all(
                        |Column {
                             relation: qualifier,
                             name,
                         }| {
                            left.schema()
                                .field_with_name(qualifier.as_deref(), name)
                                .is_ok()
                        },
                    )
                {
                    let join_filter_init = filter.remove(0);
                    LogicalPlanBuilder::from(left)
                        .filter(
                            filter
                                .into_iter()
                                .fold(join_filter_init, |acc, e| acc.and(e)),
                        )?
                        .join(&right, join_type, (left_keys, right_keys))?
                        .build()
                } else {
                    Err(DataFusionError::NotImplemented(format!(
                        "Unsupported expressions in {:?} JOIN: {:?}",
                        join_type, filter
                    )))
                }
            }
            JoinConstraint::Using(idents) => {
                let keys: Vec<Column> = idents
                    .iter()
                    .map(|x| Column::from_name(normalize_sql_object_name(x)))
                    .collect();
                LogicalPlanBuilder::from(left)
                    .join_using(&right, join_type, keys)?
                    .build()
            }
            JoinConstraint::Natural => {
                // https://issues.apache.org/jira/browse/ARROW-10727
                Err(DataFusionError::NotImplemented(
                    "NATURAL JOIN is not supported (https://issues.apache.org/jira/browse/ARROW-10727)".to_string(),
                ))
            }
            JoinConstraint::None => Err(DataFusionError::NotImplemented(
                "NONE constraint is not supported".to_string(),
            )),
        }
    }

    fn create_relation(&self, relation: TableFactor) -> Result<LogicalPlan> {
        let (plan, alias) = match relation {
            TableFactor::Table {
                ref name,
                alias,
                args,
                ..
            } => {
                // sqlparser now wraps table-function args in `Option<TableFunctionArgs>`
                let args = args.map(|a| a.args).unwrap_or_default();
                let table_name = normalize_sql_object_name(name);
                let table_ref: TableReference = table_name.as_str().into();
                let table_alias = alias.as_ref().map(|i| i.name.value.to_string());
                let default_table_alias = name
                    .0
                    .iter()
                    .last()
                    .map(object_name_part_to_string)
                    .unwrap();

                let cte = self.context.ctes.get(&table_name);
                let plan = match (cte, self.schema_provider.get_table_provider(table_ref))
                {
                    (Some(cte_plan), _) => match table_alias {
                        Some(cte_alias) => project_with_alias(
                            cte_plan.clone(),
                            vec![Expr::Wildcard],
                            Some(cte_alias),
                        ),
                        _ => Ok(cte_plan.clone()),
                    },
                    (_, Some(provider)) => LogicalPlanBuilder::scan(
                        // take alias into account to support `JOIN table1 as table2`
                        table_alias.unwrap_or(default_table_alias),
                        provider,
                        None,
                    )?
                    .build(),
                    (None, None) => {
                        let table_udf =
                            self.schema_provider.get_table_function_meta(&table_name);
                        if let Some(table_udf) = table_udf {
                            let udtf = Expr::TableUDF {
                                fun: table_udf,
                                args: self
                                    .function_args_to_expr(args, &DFSchema::empty(), None)
                                    .unwrap(),
                            };

                            let udtf_plan = table_udfs(
                                LogicalPlan::EmptyRelation(EmptyRelation {
                                    produce_one_row: true,
                                    schema: Arc::new(DFSchema::empty()),
                                }),
                                vec![udtf.clone()],
                            )
                            .unwrap();

                            if alias.is_none() {
                                return Ok(udtf_plan);
                            }

                            let mut select_exprs = rewrite_udtfs_to_columns(
                                vec![udtf],
                                udtf_plan.schema().clone().as_ref().to_owned(),
                            );

                            let alias = alias.unwrap();

                            if !alias.columns.is_empty() {
                                select_exprs = select_exprs
                                    .iter()
                                    .enumerate()
                                    .map(|(i, e)| {
                                        if alias.columns.len() > i {
                                            Expr::Alias(
                                                Box::new(e.clone()),
                                                alias.columns[i].to_string(),
                                            )
                                        } else {
                                            e.clone()
                                        }
                                    })
                                    .collect();
                            }

                            return project_with_alias(
                                udtf_plan,
                                select_exprs,
                                Some(alias.name.value),
                            );
                        }

                        Err(DataFusionError::Plan(format!(
                            "Table or CTE with name '{}' not found",
                            name
                        )))
                    }
                }?;

                (plan, alias)
            }
            TableFactor::Derived {
                subquery, alias, ..
            } => {
                // if alias is None, return Err
                if alias.is_none() {
                    return Err(DataFusionError::Plan(
                        "subquery in FROM must have an alias".to_string(),
                    ));
                }
                let logical_plan = self.query_to_plan_with_alias(
                    *subquery,
                    alias.as_ref().map(|a| a.name.value.to_string()),
                )?;
                (
                    project_with_alias(
                        logical_plan.clone(),
                        logical_plan.schema().fields().iter().map(|field| {
                            Expr::Column(Column {
                                relation: None,
                                name: field.name().clone(),
                            })
                        }),
                        alias.as_ref().map(|a| a.name.value.to_string()),
                    )?,
                    alias,
                )
            }
            TableFactor::NestedJoin {
                table_with_joins, ..
            } => (self.plan_table_with_joins(*table_with_joins)?, None),
            // @todo Support TableFactory::TableFunction?
            _ => {
                return Err(DataFusionError::NotImplemented(format!(
                    "Unsupported ast node {:?} in create_relation",
                    relation
                )));
            }
        };
        if let Some(alias) = alias {
            let columns_alias = alias.clone().columns;
            if columns_alias.is_empty() {
                // sqlparser-rs encodes AS t as an empty list of column alias
                Ok(plan)
            } else if columns_alias.len() != plan.schema().fields().len() {
                Err(DataFusionError::Plan(format!(
                    "Source table contains {} columns but only {} names given as column alias",
                    plan.schema().fields().len(),
                    columns_alias.len(),
                )))
            } else {
                Ok(LogicalPlanBuilder::from(plan.clone())
                    .project_with_alias(
                        plan.schema().fields().iter().zip(columns_alias.iter()).map(
                            |(field, ident)| col(field.name()).alias(&ident.name.value),
                        ),
                        Some(alias.name.value),
                    )?
                    .build()?)
            }
        } else {
            Ok(plan)
        }
    }

    /// Generate a logic plan from selection clause, the function contain optimization for cross join to inner join
    /// Related PR: <https://github.com/apache/arrow-datafusion/pull/1566>
    fn plan_selection(
        &self,
        selection: Option<SQLExpr>,
        plans: Vec<LogicalPlan>,
    ) -> Result<LogicalPlan> {
        // TODO: enable subqueries for joins
        let plan = match selection {
            Some(predicate_expr) => {
                // build join schema
                let mut fields = vec![];
                let mut metadata = std::collections::HashMap::new();
                for plan in &plans {
                    fields.extend_from_slice(plan.schema().fields());
                    metadata.extend(plan.schema().metadata().clone());
                }
                let join_schema = DFSchema::new_with_metadata(fields, metadata)?;

                let filter_expr = self.sql_to_rex(predicate_expr, &join_schema)?;

                // look for expressions of the form `<column> = <column>`
                let mut possible_join_keys = vec![];
                extract_possible_join_keys(&filter_expr, &mut possible_join_keys)?;

                let mut all_join_keys = HashSet::new();

                let mut plans = plans.into_iter();
                let mut left = plans.next().unwrap(); // have at least one plan

                // List of the plans that have not yet been joined
                let mut remaining_plans: Vec<Option<LogicalPlan>> =
                    plans.map(Some).collect();

                // Take from the list of remaining plans,
                loop {
                    let mut join_keys = vec![];

                    // Search all remaining plans for the next to
                    // join. Prefer the first one that has a join
                    // predicate in the predicate lists
                    let plan_with_idx =
                        remaining_plans.iter().enumerate().find(|(_idx, plan)| {
                            // skip plans that have been joined already
                            let plan = if let Some(plan) = plan {
                                plan
                            } else {
                                return false;
                            };

                            // can we find a match?
                            let left_schema = left.schema();
                            let right_schema = plan.schema();
                            for (l, r) in &possible_join_keys {
                                if left_schema.field_from_column(l).is_ok()
                                    && right_schema.field_from_column(r).is_ok()
                                {
                                    join_keys.push((l.clone(), r.clone()));
                                } else if left_schema.field_from_column(r).is_ok()
                                    && right_schema.field_from_column(l).is_ok()
                                {
                                    join_keys.push((r.clone(), l.clone()));
                                }
                            }
                            // stop if we found join keys
                            !join_keys.is_empty()
                        });

                    // If we did not find join keys, either there are
                    // no more plans, or we can't find any plans that
                    // can be joined with predicates
                    if join_keys.is_empty() {
                        assert!(plan_with_idx.is_none());

                        // pick the first non null plan to join
                        let plan_with_idx = remaining_plans
                            .iter()
                            .enumerate()
                            .find(|(_idx, plan)| plan.is_some());
                        if let Some((idx, _)) = plan_with_idx {
                            let plan = std::mem::take(&mut remaining_plans[idx]).unwrap();
                            left = LogicalPlanBuilder::from(left)
                                .cross_join(&plan)?
                                .build()?;
                        } else {
                            // no more plans to join
                            break;
                        }
                    } else {
                        // have a plan
                        let (idx, _) = plan_with_idx.expect("found plan node");
                        let plan = std::mem::take(&mut remaining_plans[idx]).unwrap();

                        let left_keys: Vec<Column> =
                            join_keys.iter().map(|(l, _)| l.clone()).collect();
                        let right_keys: Vec<Column> =
                            join_keys.iter().map(|(_, r)| r.clone()).collect();
                        let builder = LogicalPlanBuilder::from(left);
                        left = builder
                            .join(&plan, JoinType::Inner, (left_keys, right_keys))?
                            .build()?;
                    }

                    all_join_keys.extend(join_keys);
                }

                // remove join expressions from filter
                match remove_join_expressions(&filter_expr, &all_join_keys)? {
                    Some(filter_expr) => {
                        let left = self.wrap_with_subquery_plan_if_necessary(left)?;
                        LogicalPlanBuilder::from(left).filter(filter_expr)?.build()
                    }
                    _ => Ok(left),
                }
            }
            None => {
                if plans.len() == 1 {
                    Ok(plans[0].clone())
                } else {
                    let mut left = plans[0].clone();
                    for right in plans.iter().skip(1) {
                        left =
                            LogicalPlanBuilder::from(left).cross_join(right)?.build()?;
                    }
                    Ok(left)
                }
            }
        };
        plan
    }

    /// Generate a logic plan from an SQL select
    ///
    /// `order_by` is the enclosing query's ORDER BY clause; it only affects
    /// `SELECT DISTINCT ON`, which keeps the first row per group as defined
    /// by the query's ORDER BY. The actual sort is still applied by the caller.
    fn select_to_plan(
        &self,
        select: Select,
        alias: Option<String>,
        order_by: &[OrderByExpr],
    ) -> Result<LogicalPlan> {
        // process `from` clause
        let plans = self.plan_from_tables(select.from)?;
        let empty_from = matches!(plans.first(), Some(LogicalPlan::EmptyRelation(_)));

        // process `where` clause
        let with_where_outer_query_context =
            self.with_context(|c| c.subqueries_plans = Some(RwLock::new(Vec::new())));
        let plan =
            with_where_outer_query_context.plan_selection(select.selection, plans)?;

        // process the SELECT expressions, with wildcards expanded.
        let with_outer_query_context =
            self.with_context(|c| c.subqueries_plans = Some(RwLock::new(Vec::new())));
        let mut select_exprs = with_outer_query_context.prepare_select_exprs(
            &plan,
            select.projection,
            empty_from,
        )?;

        let mut plan =
            with_outer_query_context.wrap_with_subquery_plan_if_necessary(plan)?;

        // create proxy node to handle udtfs and rewrite udtfs to columns (returning by TableUDFs Node)
        let udtf_exprs = find_udtf_exprs(select_exprs.as_slice());
        if !udtf_exprs.is_empty() {
            plan = table_udfs(plan, udtf_exprs)?;
            select_exprs = rewrite_udtfs_to_columns(
                select_exprs,
                plan.schema().clone().as_ref().to_owned(),
            );
        }

        // NOTE (cubesql): realias expressions that have the same name and qualifier
        let select_exprs =
            realias_duplicate_expr_aliases(select_exprs, plan.schema(), None)?;

        // having and group by clause may reference aliases defined in select projection
        let projected_plan = self.project(plan.clone(), select_exprs.clone())?;

        let combined_schema = if self.table_columns_precedence_over_projection {
            let mut combined_schema = (**plan.schema()).clone();
            combined_schema.merge(projected_plan.schema());
            combined_schema
        } else {
            let mut combined_schema = (**projected_plan.schema()).clone();
            combined_schema.merge(plan.schema());
            combined_schema
        };

        // this alias map is resolved and looked up in both having exprs and group by exprs
        let mut alias_map = extract_aliases(&select_exprs);
        if self.table_columns_precedence_over_projection {
            alias_map.retain(|alias, _| {
                plan.schema().field_with_unqualified_name(alias).is_err()
            });
        }

        // Optionally the HAVING expression.
        let having_expr_opt = select
            .having
            .map::<Result<Expr>, _>(|having_expr| {
                let having_expr = *self.sql_expr_to_logical_expr(
                    having_expr,
                    &combined_schema,
                    None,
                )?;
                // This step "dereferences" any aliases in the HAVING clause.
                //
                // This is how we support queries with HAVING expressions that
                // refer to aliased columns.
                //
                // For example:
                //
                //   SELECT c1 AS m FROM t HAVING m > 10;
                //   SELECT c1, MAX(c2) AS m FROM t GROUP BY c1 HAVING m > 10;
                //
                // are rewritten as, respectively:
                //
                //   SELECT c1 AS m FROM t HAVING c1 > 10;
                //   SELECT c1, MAX(c2) AS m FROM t GROUP BY c1 HAVING MAX(c2) > 10;
                //
                let having_expr = resolve_aliases_to_exprs(&having_expr, &alias_map)?;
                if self.table_columns_precedence_over_projection {
                    normalize_col(having_expr, &plan)
                } else {
                    normalize_col(having_expr, &projected_plan)
                }
            })
            .transpose()?;

        // The outer expressions we will search through for
        // aggregates. Aggregates may be sourced from the SELECT...
        let mut aggr_expr_haystack = select_exprs.clone();
        // ... or from the HAVING.
        if let Some(having_expr) = &having_expr_opt {
            aggr_expr_haystack.push(having_expr.clone());
        }

        // All of the aggregate expressions (deduplicated).
        let aggr_exprs = find_aggregate_exprs(&aggr_expr_haystack);

        // All of the group by expressions
        let group_by_sql_exprs = match select.group_by {
            GroupByExpr::Expressions(exprs, _) => exprs,
            GroupByExpr::All(_) => {
                return Err(DataFusionError::NotImplemented(
                    "GROUP BY ALL is not supported".to_string(),
                ))
            }
        };
        let group_by_exprs = group_by_sql_exprs
            .into_iter()
            .map(|e| {
                let group_by_expr =
                    *self.sql_expr_to_logical_expr(e, &combined_schema, None)?;
                let group_by_expr = resolve_aliases_to_exprs(&group_by_expr, &alias_map)?;
                let group_by_expr =
                    resolve_positions_to_exprs(&group_by_expr, &select_exprs)
                        .unwrap_or(group_by_expr);
                let group_by_expr = if self.table_columns_precedence_over_projection {
                    normalize_col(group_by_expr, &plan)?
                } else {
                    normalize_col(group_by_expr, &projected_plan)?
                };
                self.validate_schema_satisfies_exprs(
                    plan.schema(),
                    slice::from_ref(&group_by_expr),
                )?;
                Ok(group_by_expr)
            })
            .collect::<Result<Vec<Expr>>>()?;

        // process group by, aggregation or having
        let (plan, select_exprs_post_aggr, having_expr_post_aggr_opt) =
            if !group_by_exprs.is_empty() || !aggr_exprs.is_empty() {
                self.aggregate(
                    plan,
                    &select_exprs,
                    &having_expr_opt,
                    group_by_exprs,
                    aggr_exprs,
                )?
            } else {
                if let Some(having_expr) = &having_expr_opt {
                    let available_columns = select_exprs
                        .iter()
                        .map(|expr| expr_as_column_expr(expr, &plan))
                        .collect::<Result<Vec<Expr>>>()?;

                    // Ensure the HAVING expression is using only columns
                    // provided by the SELECT.
                    if !can_columns_satisfy_exprs(
                        &available_columns,
                        slice::from_ref(having_expr),
                    )? {
                        return Err(DataFusionError::Plan(
                            "Having references column(s) not provided by the select"
                                .to_owned(),
                        ));
                    }
                }

                (plan, select_exprs, having_expr_opt)
            };

        let plan = if let Some(having_expr_post_aggr) = having_expr_post_aggr_opt {
            LogicalPlanBuilder::from(plan)
                .filter(having_expr_post_aggr)?
                .build()?
        } else {
            plan
        };

        // process window function
        let window_func_exprs = find_window_exprs(&select_exprs_post_aggr);

        let (plan, select_exprs_post_aggr) = if window_func_exprs.is_empty() {
            (plan, select_exprs_post_aggr)
        } else {
            let select_exprs_post_aggr_and_window = select_exprs_post_aggr
                .iter()
                .map(|expr| rebase_expr(expr, &window_func_exprs, &plan))
                .collect::<Result<Vec<Expr>>>()?;
            (
                LogicalPlanBuilder::window_plan(plan, window_func_exprs)?,
                select_exprs_post_aggr_and_window,
            )
        };

        // process `DISTINCT ON (...)` (Postgres extension): keep only the first
        // row of each set of rows sharing the ON expression values. "First" is
        // defined by the query's ORDER BY (arbitrary when there's no ORDER BY).
        // Planned as a `ROW_NUMBER()` window partitioned by the ON expressions
        // and ordered by the ORDER BY expressions, followed by a filter keeping
        // the first row of each partition. The enclosing query's ORDER BY is
        // applied on top by the caller, reordering already-deduplicated rows.
        let plan = if let Some(Distinct::On(on_sql_exprs)) = select.distinct.clone() {
            let plan_distinct_expr = |e: SQLExpr| -> Result<Expr> {
                let expr = *self.sql_expr_to_logical_expr(e, &combined_schema, None)?;
                let expr = resolve_aliases_to_exprs(&expr, &alias_map)?;
                let expr = resolve_positions_to_exprs(&expr, &select_exprs_post_aggr)
                    .unwrap_or(expr);
                normalize_col(expr, &plan)
            };

            let on_exprs = on_sql_exprs
                .into_iter()
                .map(plan_distinct_expr)
                .collect::<Result<Vec<Expr>>>()?;

            let sort_exprs = order_by
                .iter()
                .map(|e| {
                    let OrderByExpr {
                        expr,
                        options: OrderByOptions { asc, nulls_first },
                        ..
                    } = e.clone();
                    let expr = plan_distinct_expr(expr)?;
                    let asc = asc.unwrap_or(true);
                    Ok(Expr::Sort {
                        expr: Box::new(expr),
                        asc,
                        nulls_first: nulls_first.unwrap_or(!asc),
                    })
                })
                .collect::<Result<Vec<Expr>>>()?;

            // Postgres rule: ORDER BY (if present) must start with a run of
            // expressions from the ON list, and no ON expression may appear in
            // ORDER BY after that run.
            let is_on_expr = |s: &Expr| match s {
                Expr::Sort { expr, .. } => on_exprs.contains(expr.as_ref()),
                _ => false,
            };
            let leading = sort_exprs.iter().take_while(|s| is_on_expr(s)).count();
            if !sort_exprs.is_empty()
                && (leading == 0 || sort_exprs[leading..].iter().any(is_on_expr))
            {
                return Err(DataFusionError::Plan(
                    "SELECT DISTINCT ON expressions must match initial ORDER BY expressions"
                        .to_string(),
                ));
            }

            let row_number = Expr::WindowFunction {
                fun: WindowFunction::BuiltInWindowFunction(
                    BuiltInWindowFunction::RowNumber,
                ),
                args: vec![],
                partition_by: on_exprs,
                order_by: sort_exprs,
                window_frame: None,
            };
            let plan = LogicalPlanBuilder::window_plan(plan, vec![row_number.clone()])?;
            let row_number_col =
                Expr::Column(Column::from_name(row_number.name(plan.schema())?));
            LogicalPlanBuilder::from(plan)
                .filter(row_number_col.eq(lit(1_u64)))?
                .build()?
        } else {
            plan
        };

        // final projection
        let plan = project_with_alias(plan, select_exprs_post_aggr, alias)?;

        // process distinct clause (`SELECT ALL` keeps duplicates, so only dedupe on
        // an explicit `DISTINCT`)
        if matches!(select.distinct, Some(Distinct::Distinct)) {
            LogicalPlanBuilder::from(plan).distinct()?.build()
        } else {
            Ok(plan)
        }
    }

    fn wrap_with_subquery_plan_if_necessary(
        &self,
        plan: LogicalPlan,
    ) -> Result<LogicalPlan> {
        Ok(if let Some(subqueries) = &self.context.subqueries_plans {
            let subqueries = subqueries
                .read()
                .map_err(|e| DataFusionError::Plan(e.to_string()))?;
            if !subqueries.is_empty() {
                let (subqueries, types): (Vec<_>, Vec<_>) =
                    subqueries.clone().into_iter().unzip();
                LogicalPlanBuilder::from(plan)
                    .subquery(subqueries, types)?
                    .build()?
            } else {
                plan
            }
        } else {
            plan
        })
    }

    /// Returns the `Expr`'s corresponding to a SQL query's SELECT expressions.
    ///
    /// Wildcards are expanded into the concrete list of columns.
    fn prepare_select_exprs(
        &self,
        plan: &LogicalPlan,
        projection: Vec<SelectItem>,
        empty_from: bool,
    ) -> Result<Vec<Expr>> {
        let input_schema = plan.schema();
        let iter = projection
            .into_iter()
            .map(|expr| self.sql_select_to_rex(expr, input_schema))
            .collect::<Result<Vec<Expr>>>()?
            .into_iter();
        let plan = self.wrap_with_subquery_plan_if_necessary(plan.clone())?;
        iter.map(|expr| {
            Ok(match expr {
                Expr::Wildcard => {
                    if empty_from {
                        return Err(DataFusionError::Plan(
                            "SELECT * with no tables specified is not valid".to_string(),
                        ));
                    }
                    expand_wildcard(input_schema, &plan)?
                }
                Expr::QualifiedWildcard { ref qualifier } => {
                    expand_qualified_wildcard(qualifier, input_schema, &plan)?
                }
                _ => vec![normalize_col(expr, &plan)?],
            })
        })
        .flat_map(|res| match res {
            Ok(v) => v.into_iter().map(Ok).collect(),
            Err(e) => vec![Err(e)],
        })
        .collect::<Result<Vec<Expr>>>()
    }

    /// Wrap a plan in a projection
    fn project(&self, input: LogicalPlan, expr: Vec<Expr>) -> Result<LogicalPlan> {
        self.validate_schema_satisfies_exprs(input.schema(), &expr)?;
        LogicalPlanBuilder::from(input).project(expr)?.build()
    }

    /// Wrap a plan in an aggregate
    fn aggregate(
        &self,
        input: LogicalPlan,
        select_exprs: &[Expr],
        having_expr_opt: &Option<Expr>,
        group_by_exprs: Vec<Expr>,
        aggr_exprs: Vec<Expr>,
    ) -> Result<(LogicalPlan, Vec<Expr>, Option<Expr>)> {
        // create the aggregate plan

        // in this next section of code we are re-writing the projection to refer to columns
        // output by the aggregate plan. For example, if the projection contains the expression
        // `SUM(a)` then we replace that with a reference to a column `#SUM(a)` produced by
        // the aggregate plan.

        // combine the original grouping and aggregate expressions into one list (note that
        // we do not add the "having" expression since that is not part of the projection)
        let mut aggr_projection_exprs = vec![];
        for expr in &group_by_exprs {
            match expr {
                Expr::GroupingSet(GroupingSet::Rollup(exprs)) => {
                    aggr_projection_exprs.extend_from_slice(exprs)
                }
                Expr::GroupingSet(GroupingSet::Cube(exprs)) => {
                    aggr_projection_exprs.extend_from_slice(exprs)
                }
                Expr::GroupingSet(GroupingSet::GroupingSets(lists_of_exprs)) => {
                    for exprs in lists_of_exprs {
                        aggr_projection_exprs.extend_from_slice(exprs)
                    }
                }
                _ => aggr_projection_exprs.push(expr.clone()),
            }
        }
        aggr_projection_exprs.extend_from_slice(&aggr_exprs);

        let plan = LogicalPlanBuilder::from(input.clone())
            .aggregate(group_by_exprs, aggr_exprs)?
            .build()?;

        // After aggregation, these are all of the columns that will be
        // available to next phases of planning.
        let column_exprs_post_aggr = aggr_projection_exprs
            .iter()
            .map(|expr| expr_as_column_expr(expr, &input))
            .collect::<Result<Vec<Expr>>>()?;

        // Rewrite the SELECT expression to use the columns produced by the
        // aggregation.
        let select_exprs_post_aggr = select_exprs
            .iter()
            .map(|expr| rebase_expr(expr, &aggr_projection_exprs, &input))
            .collect::<Result<Vec<Expr>>>()?;

        if !can_columns_satisfy_exprs(&column_exprs_post_aggr, &select_exprs_post_aggr)? {
            return Err(DataFusionError::Plan(
                "Projection references non-aggregate values".to_owned(),
            ));
        }

        // Rewrite the HAVING expression to use the columns produced by the
        // aggregation.
        let having_expr_post_aggr_opt = if let Some(having_expr) = having_expr_opt {
            let having_expr_post_aggr =
                rebase_expr(having_expr, &aggr_projection_exprs, &input)?;

            if !can_columns_satisfy_exprs(
                &column_exprs_post_aggr,
                slice::from_ref(&having_expr_post_aggr),
            )? {
                return Err(DataFusionError::Plan(
                    "Having references non-aggregate values".to_owned(),
                ));
            }

            Some(having_expr_post_aggr)
        } else {
            None
        };

        Ok((plan, select_exprs_post_aggr, having_expr_post_aggr_opt))
    }

    /// Wrap a plan in a limit
    fn limit(
        &self,
        input: LogicalPlan,
        skip: Option<SQLExpr>,
        limit: Option<SQLExpr>,
        fetch: Option<Fetch>,
    ) -> Result<LogicalPlan> {
        if skip.is_none() && limit.is_none() && fetch.is_none() {
            return Ok(input);
        }

        let skip = match skip {
            Some(skip_expr) => {
                let skip = match self.sql_to_rex(skip_expr, input.schema())? {
                    Expr::Literal(ScalarValue::Int64(Some(s))) => {
                        if s < 0 {
                            return Err(DataFusionError::Plan(format!(
                                "Offset must be >= 0, '{}' was provided.",
                                s
                            )));
                        }
                        Ok(s as usize)
                    }
                    _ => Err(DataFusionError::Plan(
                        "Unexpected expression in OFFSET clause".to_string(),
                    )),
                }?;
                Some(skip)
            }
            _ => None,
        };

        let fetch = match (limit, fetch) {
            (Some(limit_expr), None) => {
                let n = match self.sql_to_rex(limit_expr, input.schema())? {
                    Expr::Literal(ScalarValue::Int64(Some(n))) => {
                        if n < 0 {
                            return Err(DataFusionError::Plan(
                                "LIMIT must not be negative".to_string(),
                            ));
                        }
                        Ok(n as usize)
                    }
                    _ => Err(DataFusionError::Plan(
                        "Unexpected expression for LIMIT clause".to_string(),
                    )),
                }?;
                Some(n)
            }
            (None, Some(fetch_expr)) => {
                if fetch_expr.with_ties {
                    return Err(DataFusionError::Plan(
                        "FETCH ... WITH TIES is not supported".to_string(),
                    ));
                }
                if fetch_expr.percent {
                    return Err(DataFusionError::Plan(
                        "FETCH ... n PERCENT ROWS is not supported".to_string(),
                    ));
                }
                let n = match fetch_expr.quantity {
                    Some(quantity) => match self.sql_to_rex(quantity, input.schema())? {
                        Expr::Literal(ScalarValue::Int64(Some(n))) => {
                            if n < 0 {
                                return Err(DataFusionError::Plan(
                                    "LIMIT must not be negative".to_string(),
                                ));
                            }
                            Ok(n as usize)
                        }
                        _ => Err(DataFusionError::Plan(
                            "Unexpected expression for LIMIT clause".to_string(),
                        )),
                    },
                    None => Ok(1),
                }?;
                Some(n)
            }
            (Some(_), Some(_)) => {
                return Err(DataFusionError::Plan(
                    "Only LIMIT or FETCH must be provided".to_string(),
                ))
            }
            _ => None,
        };

        LogicalPlanBuilder::from(input).limit(skip, fetch)?.build()
    }

    /// Wrap the logical in a sort
    fn order_by(
        &self,
        plan: LogicalPlan,
        order_by: Vec<OrderByExpr>,
    ) -> Result<LogicalPlan> {
        if order_by.is_empty() {
            return Ok(plan);
        }

        // NOTE(cubesql): ORDER BY may reference aggregate functions that reference columns
        // outside the schema but this is valid if we can add those columns
        // to the plan context downstream, so let's merge all schemas to avoid picking up columns
        // from outer context when evaluating subqueries
        let mut all_schemas = plan.all_schemas().into_iter();
        let combined_schema = if let Some(first_schema) = all_schemas.next() {
            let mut schema = first_schema.as_ref().clone();
            for next_schema in all_schemas {
                schema.merge(next_schema);
            }
            Some(schema)
        } else {
            None
        };

        let order_by_rex = order_by
            .into_iter()
            .map(|e| {
                self.order_by_to_sort_expr(
                    e,
                    plan.schema(),
                    combined_schema.as_ref(),
                    true,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        LogicalPlanBuilder::from(plan).sort(order_by_rex)?.build()
    }

    /// convert sql OrderByExpr to Expr::Sort
    fn order_by_to_sort_expr(
        &self,
        e: OrderByExpr,
        schema: &DFSchema,
        extended_schema: Option<&DFSchema>,
        parse_indexes: bool,
    ) -> Result<Expr> {
        let OrderByExpr {
            expr,
            options: OrderByOptions { asc, nulls_first },
            ..
        } = e;

        let expr = match expr {
            SQLExpr::Value(ValueWithSpan {
                value: Value::Number(v, _),
                ..
            }) if parse_indexes => {
                let field_index = v
                    .parse::<usize>()
                    .map_err(|err| DataFusionError::Plan(err.to_string()))?;

                if field_index == 0 {
                    return Err(DataFusionError::Plan(
                        "Order by index starts at 1 for column indexes".to_string(),
                    ));
                } else if schema.fields().len() < field_index {
                    return Err(DataFusionError::Plan(format!(
                        "Order by column out of bounds, specified: {}, max: {}",
                        field_index,
                        schema.fields().len()
                    )));
                }

                let field = schema.field(field_index - 1);
                Expr::Column(field.qualified_column())
            }
            e => *self.sql_expr_to_logical_expr(e, schema, extended_schema)?,
        };
        Ok({
            let asc = asc.unwrap_or(true);
            Expr::Sort {
                expr: Box::new(expr),
                asc,
                // when asc is true, by default nulls last to be consistent with postgres
                // postgres rule: https://www.postgresql.org/docs/current/queries-order.html
                nulls_first: nulls_first.unwrap_or(!asc),
            }
        })
    }

    /// Validate the schema provides all of the columns referenced in the expressions.
    fn validate_schema_satisfies_exprs(
        &self,
        schema: &DFSchema,
        exprs: &[Expr],
    ) -> Result<()> {
        find_column_exprs(exprs)
            .iter()
            .try_for_each(|col| match col {
                Expr::Column(col) => match &col.relation {
                    Some(r) => {
                        if let Some(plans) = self.context.subqueries_plans()? {
                            if plans.into_iter().any(|(p, _)| {
                                p.schema().field_with_qualified_name(r, &col.name).is_ok()
                            }) {
                                return Ok(());
                            }
                        }
                        schema.field_with_qualified_name(r, &col.name)?;
                        Ok(())
                    }
                    None => {
                        if let Some(plans) = self.context.subqueries_plans()? {
                            if plans.into_iter().any(|(p, _)| {
                                !p.schema()
                                    .fields_with_unqualified_name(&col.name)
                                    .is_empty()
                            }) {
                                return Ok(());
                            }
                        }
                        if !schema.fields_with_unqualified_name(&col.name).is_empty() {
                            Ok(())
                        } else {
                            Err(DataFusionError::Plan(format!(
                                "No field with unqualified name '{}'",
                                &col.name
                            )))
                        }
                    }
                }
                .map_err(|_: DataFusionError| {
                    DataFusionError::Plan(format!(
                        "Invalid identifier '{}' for schema {}",
                        col, schema
                    ))
                }),
                _ => Err(DataFusionError::Internal("Not a column".to_string())),
            })
    }

    /// Generate a relational expression from a select SQL expression
    fn sql_select_to_rex(&self, sql: SelectItem, schema: &DFSchema) -> Result<Expr> {
        match sql {
            SelectItem::UnnamedExpr(expr) => self.sql_to_rex(expr, schema),
            SelectItem::ExprWithAlias { expr, alias } => Ok(Alias(
                Box::new(self.sql_to_rex(expr, schema)?),
                // Hacky solution for compatibility with MySQL
                alias.value,
            )),
            SelectItem::ExprWithAliases { .. } => Err(DataFusionError::NotImplemented(
                "SELECT expression with multiple aliases is not supported".to_string(),
            )),
            SelectItem::Wildcard(_) => Ok(Expr::Wildcard),
            SelectItem::QualifiedWildcard(kind, _) => {
                let qualifier = match kind {
                    SelectItemQualifiedWildcardKind::ObjectName(object_name) => {
                        format!("{}", object_name)
                    }
                    SelectItemQualifiedWildcardKind::Expr(expr) => {
                        format!("{}", expr)
                    }
                };
                Ok(Expr::QualifiedWildcard { qualifier })
            }
        }
    }

    /// Generate a relational expression from a SQL expression
    pub fn sql_to_rex(&self, sql: SQLExpr, schema: &DFSchema) -> Result<Expr> {
        let mut expr = *self.sql_expr_to_logical_expr(sql, schema, None)?;
        expr = self.rewrite_partial_qualifier(expr, schema);
        self.validate_schema_satisfies_exprs(schema, &[expr.clone()])?;
        Ok(expr)
    }

    /// Rewrite aliases which are not-complete (e.g. ones that only include only table qualifier in a schema.table qualified relation)
    fn rewrite_partial_qualifier(&self, expr: Expr, schema: &DFSchema) -> Expr {
        match expr {
            Expr::Column(col) => match &col.relation {
                Some(q) => {
                    match schema
                        .fields()
                        .iter()
                        .find(|field| match field.qualifier() {
                            Some(field_q) => {
                                field.name() == &col.name
                                    && field_q.ends_with(&format!(".{}", q))
                            }
                            _ => false,
                        }) {
                        Some(df_field) => Expr::Column(Column {
                            relation: df_field.qualifier().cloned(),
                            name: df_field.name().clone(),
                        }),
                        None => Expr::Column(col),
                    }
                }
                None => Expr::Column(col),
            },
            _ => expr,
        }
    }

    fn sql_fn_arg_to_logical_expr(
        &self,
        sql: FunctionArg,
        schema: &DFSchema,
        extended_schema: Option<&DFSchema>,
    ) -> Result<Expr> {
        match sql {
            FunctionArg::Named {
                name: _,
                arg: FunctionArgExpr::Expr(arg),
                ..
            } => self
                .sql_expr_to_logical_expr(arg, schema, extended_schema)
                .map(|b| *b),
            FunctionArg::Named {
                name: _,
                arg: FunctionArgExpr::Wildcard,
                ..
            } => Ok(Expr::Wildcard),
            FunctionArg::Unnamed(FunctionArgExpr::Expr(arg)) => self
                .sql_expr_to_logical_expr(arg, schema, extended_schema)
                .map(|b| *b),
            FunctionArg::Unnamed(FunctionArgExpr::Wildcard) => Ok(Expr::Wildcard),
            _ => Err(DataFusionError::NotImplemented(format!(
                "Unsupported qualified wildcard argument: {:?}",
                sql
            ))),
        }
    }

    fn parse_sql_binary_any(
        &self,
        left: SQLExpr,
        op: BinaryOperator,
        right: SQLExpr,
        all: bool,
        schema: &DFSchema,
        extended_schema: Option<&DFSchema>,
    ) -> Result<Box<Expr>> {
        let operator = match op {
            BinaryOperator::Eq => Ok(Operator::Eq),
            BinaryOperator::NotEq => Ok(Operator::NotEq),
            BinaryOperator::Lt => Ok(Operator::Lt),
            BinaryOperator::LtEq => Ok(Operator::LtEq),
            BinaryOperator::Gt => Ok(Operator::Gt),
            BinaryOperator::GtEq => Ok(Operator::GtEq),
            _ => Err(DataFusionError::NotImplemented(format!(
                "Unsupported SQL ANY operator {:?}",
                op
            ))),
        }?;

        // sqlparser 0.62 represents `x op ANY/ALL (<subquery>)` with the subquery as the
        // right operand; it must be planned as an ANY/ALL subquery (not a scalar one).
        let right_expr = match right {
            SQLExpr::Subquery(q) => {
                Box::new(self.subquery_to_plan(q, SubqueryType::AnyAll, schema)?)
            }
            other => self.sql_expr_to_logical_expr(other, schema, extended_schema)?,
        };

        Ok(Box::new(Expr::AnyExpr {
            left: self.sql_expr_to_logical_expr(left, schema, extended_schema)?,
            op: operator,
            right: right_expr,
            all,
        }))
    }

    fn parse_sql_binary_op(
        &self,
        left: SQLExpr,
        op: BinaryOperator,
        right: SQLExpr,
        schema: &DFSchema,
        _extended_schema: Option<&DFSchema>,
    ) -> Result<Box<Expr>> {
        let operator = parse_sql_binary_operator(&op)?;

        Ok(Box::new(Expr::BinaryExpr {
            left: self.sql_expr_to_logical_expr(left, schema, None)?,
            op: operator,
            right: self.sql_expr_to_logical_expr(right, schema, None)?,
        }))
    }

    fn parse_sql_unary_op(
        &self,
        op: UnaryOperator,
        expr: SQLExpr,
        schema: &DFSchema,
        extended_schema: Option<&DFSchema>,
    ) -> Result<Box<Expr>> {
        match op {
            UnaryOperator::Not => Ok(Box::new(Expr::Not(
                self.sql_expr_to_logical_expr(expr, schema, extended_schema)?,
            ))),
            UnaryOperator::Plus => {
                self.sql_expr_to_logical_expr(expr, schema, extended_schema)
            }
            UnaryOperator::Minus => {
                match expr {
                    // optimization: if it's a number literal, we apply the negative operator
                    // here directly to calculate the new literal.
                    SQLExpr::Value(ValueWithSpan { value: Value::Number(n, _), .. }) => match n.parse::<i64>() {
                        Ok(n) => Ok(Box::new(lit(-n))),
                        Err(_) => Ok(Box::new(lit(-n
                            .parse::<f64>()
                            .map_err(|_e| {
                                DataFusionError::Internal(format!(
                                    "negative operator can be only applied to integer and float operands, got: {}",
                                    n))
                            })?))),
                    },
                    // not a literal, apply negative operator on expression
                    _ => Ok(Box::new(Expr::Negative(self.sql_expr_to_logical_expr(expr, schema, extended_schema)?))),
                }
            }
            _ => Err(DataFusionError::NotImplemented(format!(
                "Unsupported SQL unary operator {:?}",
                op
            ))),
        }
    }

    fn sql_values_to_plan(&self, values: SQLValues) -> Result<LogicalPlan> {
        // values should not be based on any other schema
        let schema = DFSchema::empty();
        let values = values
            .rows
            .into_iter()
            .map(|row| {
                row.content
                    .into_iter()
                    .map(|v| match v {
                        SQLExpr::Value(ValueWithSpan {
                            value: Value::Number(n, _),
                            ..
                        }) => parse_sql_number(&n),
                        SQLExpr::Value(ValueWithSpan {
                            value: Value::SingleQuotedString(s),
                            ..
                        }) => Ok(lit(s)),
                        SQLExpr::Value(ValueWithSpan {
                            value: Value::Null, ..
                        }) => Ok(Expr::Literal(ScalarValue::Null)),
                        SQLExpr::Value(ValueWithSpan {
                            value: Value::Boolean(n),
                            ..
                        }) => Ok(lit(n)),
                        SQLExpr::UnaryOp { op, expr } => self
                            .parse_sql_unary_op(op, *expr, &schema, None)
                            .map(|b| *b),
                        SQLExpr::BinaryOp { left, op, right } => self
                            .parse_sql_binary_op(*left, op, *right, &schema, None)
                            .map(|b| *b),
                        other => Err(DataFusionError::NotImplemented(format!(
                            "Unsupported value {:?} in a values list expression",
                            other
                        ))),
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        LogicalPlanBuilder::values(values)?.build()
    }

    pub(super) fn sql_rollup_to_expr(
        &self,
        exprs: Vec<Vec<SQLExpr>>,
        schema: &DFSchema,
        extended_schema: Option<&DFSchema>,
    ) -> Result<Box<Expr>> {
        let args: Result<Vec<_>> = exprs
            .into_iter()
            .map(|v| {
                if v.len() != 1 {
                    Err(DataFusionError::NotImplemented(
                        "Tuple expressions are not supported for Rollup expressions"
                            .to_string(),
                    ))
                } else {
                    self.sql_expr_to_logical_expr(v[0].clone(), schema, extended_schema)
                        .map(|b| *b)
                }
            })
            .collect();
        Ok(Box::new(Expr::GroupingSet(GroupingSet::Rollup(args?))))
    }

    pub(super) fn sql_cube_to_expr(
        &self,
        exprs: Vec<Vec<SQLExpr>>,
        schema: &DFSchema,
        extended_schema: Option<&DFSchema>,
    ) -> Result<Box<Expr>> {
        let args: Result<Vec<_>> = exprs
            .into_iter()
            .map(|v| {
                if v.len() != 1 {
                    Err(DataFusionError::NotImplemented(
                        "Tuple expressions are not supported for Cube expressions"
                            .to_string(),
                    ))
                } else {
                    self.sql_expr_to_logical_expr(v[0].clone(), schema, extended_schema)
                        .map(|b| *b)
                }
            })
            .collect();
        Ok(Box::new(Expr::GroupingSet(GroupingSet::Cube(args?))))
    }

    // Extended schema is used to look for columns down the plan
    // to avoid picking up columns from outer query context
    /// Generate a logical expression from a SQL expression.
    ///
    /// Chains of binary operators (e.g. `a OR b OR c OR …`) are the most common source of deep
    /// nesting in real queries. To avoid one stack frame per operator (which overflows on large
    /// chains), the binary-operator spine is walked iteratively with an explicit stack here, in
    /// postfix order, rather than recursively. Every non-binary node is delegated to
    /// [`Self::sql_expr_to_logical_expr_internal`]; nested binary ops inside those nodes are
    /// flattened the same way when their subtree is visited.
    fn sql_expr_to_logical_expr(
        &self,
        sql: SQLExpr,
        schema: &DFSchema,
        extended_schema: Option<&DFSchema>,
    ) -> Result<Box<Expr>> {
        // Non-binary top-level expressions keep the original `extended_schema` behaviour.
        if !matches!(sql, SQLExpr::BinaryOp { .. }) {
            return self.sql_expr_to_logical_expr_internal(sql, schema, extended_schema);
        }

        // Iteratively flatten the binary-operator spine. Operands are planned with
        // `extended_schema = None`, matching the historical `parse_sql_binary_op` behaviour
        // (binary operands never see the extended schema).
        let mut stack = vec![StackEntry::SQLExpr(Box::new(sql))];
        let mut eval_stack: Vec<Box<Expr>> = vec![];

        while let Some(entry) = stack.pop() {
            match entry {
                StackEntry::SQLExpr(sql_expr) => match *sql_expr {
                    SQLExpr::BinaryOp { left, op, right } => {
                        // Push in reverse so `left` is processed first and operands end up
                        // on `eval_stack` as [.., left, right] before the operator combines them.
                        stack.push(StackEntry::Operator(op));
                        stack.push(StackEntry::SQLExpr(right));
                        stack.push(StackEntry::SQLExpr(left));
                    }
                    other => {
                        eval_stack.push(
                            self.sql_expr_to_logical_expr_internal(other, schema, None)?,
                        );
                    }
                },
                StackEntry::Operator(op) => {
                    let operator = parse_sql_binary_operator(&op)?;
                    let right = eval_stack.pop().ok_or_else(|| {
                        DataFusionError::Internal(
                            "binary operator stack underflow (right)".to_string(),
                        )
                    })?;
                    let left = eval_stack.pop().ok_or_else(|| {
                        DataFusionError::Internal(
                            "binary operator stack underflow (left)".to_string(),
                        )
                    })?;
                    eval_stack.push(Box::new(Expr::BinaryExpr {
                        left,
                        op: operator,
                        right,
                    }));
                }
            }
        }

        eval_stack.pop().ok_or_else(|| {
            DataFusionError::Internal(
                "binary operator evaluation produced no result".to_string(),
            )
        })
    }

    fn sql_expr_to_logical_expr_internal(
        &self,
        sql: SQLExpr,
        schema: &DFSchema,
        extended_schema: Option<&DFSchema>,
    ) -> Result<Box<Expr>> {
        match sql {
            SQLExpr::Value(ValueWithSpan { value: Value::Number(n, _), .. }) => parse_sql_number(&n).map(Box::new),
            SQLExpr::Value(ValueWithSpan { value: Value::SingleQuotedString(ref s), .. }) => Ok(Box::new(lit(s.clone()))),
            SQLExpr::Value(ValueWithSpan { value: Value::EscapedStringLiteral(ref s), .. }) => Ok(Box::new(lit(s.clone()))),
            SQLExpr::Value(ValueWithSpan { value: Value::UnicodeStringLiteral(ref s), .. }) => parse_unicode_escaped_string(s, '\\').map(Box::new),
            SQLExpr::Value(ValueWithSpan { value: Value::Boolean(n), .. }) => Ok(Box::new(lit(n))),
            SQLExpr::Value(ValueWithSpan { value: Value::Null, .. }) => Ok(Box::new(Expr::Literal(ScalarValue::Null))),
            SQLExpr::Extract { field, expr, .. } => Ok(Box::new(Expr::ScalarFunction {
                fun: BuiltinScalarFunction::DatePart,
                args: vec![
                    Expr::Literal(ScalarValue::Utf8(Some(format!("{}", field)))),
                    *self.sql_expr_to_logical_expr(*expr, schema, extended_schema)?,
                ],
            })),
            /* CubeSQL */
            SQLExpr::Position { expr, r#in, .. } => {
                let args = vec![
                    FunctionArg::Unnamed(FunctionArgExpr::Expr(*expr)),
                    FunctionArg::Unnamed(FunctionArgExpr::Expr(*r#in)),
                ];
                match self.schema_provider.get_function_meta("position") {
                    Some(fm) => {
                        let args = self.function_args_to_expr(args, schema, extended_schema)?;

                        Ok(Box::new(Expr::ScalarUDF { fun: fm, args }))
                    }
                    _ => Err(DataFusionError::Plan("Invalid function 'position'".to_string()))
                }
            },

            SQLExpr::Interval(interval) => {
                self.sql_interval_to_expr(interval, schema, extended_schema)
            }

            // @todo Support
            SQLExpr::Collate { expr, .. } => self.sql_expr_to_logical_expr(*expr, schema, extended_schema),

            SQLExpr::Array(arr) => self.sql_array_literal(arr.elem, schema, extended_schema).map(Box::new),

            SQLExpr::Identifier(id) => {
                if id.value.starts_with('@') {
                    // TODO: figure out if ScalarVariables should be insensitive.
                    let var_names = vec![id.value];
                    let ty = self
                        .schema_provider
                        .get_variable_type(&var_names)
                        .ok_or_else(|| {
                            DataFusionError::Execution(format!(
                                "variable {:?} has no type information",
                                var_names
                            ))
                        })?;
                    Ok(Box::new(Expr::ScalarVariable(ty, var_names)))
                } else {
                    // Don't use `col()` here because it will try to
                    // interpret names with '.' as if they were
                    // compound indenfiers, but this is not a compound
                    // identifier. (e.g. it is "foo.bar" not foo.bar)

                    // Rules for finding the column:
                    // - try the current schema first
                    // - if available, try extended schema to allow adding
                    //   missing columns and aggregate expressions down the plan
                    // - try to get column from outer query context last
                    // - finally use the column as-is
                    if schema.field_with_unqualified_name(&id.value).is_ok() {
                        return Ok(Box::new(Expr::Column(Column {
                            relation: None,
                            name: id.value,
                        })))
                    }

                    if let Some(extended_schema) = extended_schema {
                        if extended_schema.field_with_unqualified_name(&id.value).is_ok() {
                            return Ok(Box::new(Expr::Column(Column {
                                relation: None,
                                name: id.value,
                            })))
                        }
                    }

                    if let Some(f) = self.context.outer_query_context_schema.iter().find_map(|s| s.field_with_unqualified_name(&id.value).ok()) {
                        return Ok(Box::new(Expr::OuterColumn(f.data_type().clone(), Column {
                            relation: None,
                            name: id.value,
                        })))
                    }

                    Ok(Box::new(Expr::Column(Column {
                        relation: None,
                        name: id.value,
                    })))
                }
            }

            SQLExpr::CompoundFieldAccess { root, access_chain } => {
                self.sql_compound_field_access_to_expr(
                    root,
                    access_chain,
                    schema,
                    extended_schema,
                )
            }

            SQLExpr::CompoundIdentifier(ids) => {
                self.sql_compound_identifier_to_expr(ids, schema, extended_schema)
            }

            SQLExpr::Case {
                operand,
                conditions,
                else_result,
                ..
            } => self.sql_case_to_expr(
                operand,
                conditions,
                else_result,
                schema,
                extended_schema,
            ),

            SQLExpr::Cast {
                kind,
                expr,
                data_type,
                ..
            } => {
                let expr =
                    self.sql_expr_to_logical_expr(*expr, schema, extended_schema)?;
                let data_type = convert_data_type(&data_type)?;
                match kind {
                    CastKind::TryCast | CastKind::SafeCast => {
                        Ok(Box::new(Expr::TryCast { expr, data_type }))
                    }
                    _ => Ok(Box::new(Expr::Cast { expr, data_type })),
                }
            }

            SQLExpr::TypedString(sqlparser::ast::TypedString {
                data_type,
                value,
                ..
            }) => Ok(Box::new(Expr::Cast {
                expr: Box::new(lit(value.into_string().unwrap_or_default())),
                data_type: convert_data_type(&data_type)?,
            })),

            SQLExpr::IsNull(expr) => Ok(Box::new(Expr::IsNull(
                self.sql_expr_to_logical_expr(*expr, schema, extended_schema)?,
            ))),

            SQLExpr::IsNotNull(expr) => Ok(Box::new(Expr::IsNotNull(
                self.sql_expr_to_logical_expr(*expr, schema, extended_schema)?,
            ))),

            // `x IS [NOT] TRUE/FALSE` is translated to a boolean equality comparison.
            SQLExpr::IsTrue(expr) => Ok(Box::new(Expr::BinaryExpr {
                left: self.sql_expr_to_logical_expr(*expr, schema, extended_schema)?,
                op: Operator::Eq,
                right: Box::new(lit(true)),
            })),
            SQLExpr::IsNotTrue(expr) => Ok(Box::new(Expr::BinaryExpr {
                left: self.sql_expr_to_logical_expr(*expr, schema, extended_schema)?,
                op: Operator::NotEq,
                right: Box::new(lit(true)),
            })),
            SQLExpr::IsFalse(expr) => Ok(Box::new(Expr::BinaryExpr {
                left: self.sql_expr_to_logical_expr(*expr, schema, extended_schema)?,
                op: Operator::Eq,
                right: Box::new(lit(false)),
            })),
            SQLExpr::IsNotFalse(expr) => Ok(Box::new(Expr::BinaryExpr {
                left: self.sql_expr_to_logical_expr(*expr, schema, extended_schema)?,
                op: Operator::NotEq,
                right: Box::new(lit(false)),
            })),

            SQLExpr::IsDistinctFrom(left, right) => Ok(Box::new(Expr::BinaryExpr {
                left: self.sql_expr_to_logical_expr(*left, schema, extended_schema)?,
                op: Operator::IsDistinctFrom,
                right: self.sql_expr_to_logical_expr(*right, schema, extended_schema)?,
            })),

            SQLExpr::IsNotDistinctFrom(left, right) => Ok(Box::new(Expr::BinaryExpr {
                left: self.sql_expr_to_logical_expr(*left, schema, extended_schema)?,
                op: Operator::IsNotDistinctFrom,
                right: self.sql_expr_to_logical_expr(*right, schema, extended_schema)?,
            })),

            SQLExpr::UnaryOp { op, expr } => {
                self.parse_sql_unary_op(op, *expr, schema, extended_schema)
            }

            SQLExpr::Between {
                expr,
                negated,
                low,
                high,
            } => Ok(Box::new(Expr::Between {
                expr: self.sql_expr_to_logical_expr(*expr, schema, extended_schema)?,
                negated,
                low: self.sql_expr_to_logical_expr(*low, schema, extended_schema)?,
                high: self.sql_expr_to_logical_expr(*high, schema, extended_schema)?,
            })),

            SQLExpr::InList {
                expr,
                list,
                negated,
            } => {
                let list_expr = list
                    .into_iter()
                    .map(|e| self.sql_expr_to_logical_expr(e, schema, extended_schema).map(|b| *b))
                    .collect::<Result<Vec<_>>>()?;

                Ok(Box::new(Expr::InList {
                    expr: self.sql_expr_to_logical_expr(*expr, schema, extended_schema)?,
                    list: list_expr,
                    negated,
                }))
            }

            SQLExpr::Like { negated, expr, pattern, escape_char, .. } => {
                let pattern = self.sql_expr_to_logical_expr(*pattern, schema, extended_schema)?;
                let pattern_type = pattern.get_type(schema)?;
                if pattern_type != DataType::Utf8 && pattern_type != DataType::Null {
                    return Err(DataFusionError::Plan(
                        "Invalid pattern in LIKE expression".to_string(),
                    ));
                }
                Ok(Box::new(Expr::Like(Like::new(
                    negated,
                    self.sql_expr_to_logical_expr(*expr, schema, extended_schema)?,
                    pattern,
                    escape_char_to_char(escape_char),
                ))))
            }

            SQLExpr::ILike { negated, expr, pattern, escape_char, .. } => {
                let pattern = self.sql_expr_to_logical_expr(*pattern, schema, extended_schema)?;
                let pattern_type = pattern.get_type(schema)?;
                if pattern_type != DataType::Utf8 && pattern_type != DataType::Null {
                    return Err(DataFusionError::Plan(
                        "Invalid pattern in ILIKE expression".to_string(),
                    ));
                }
                Ok(Box::new(Expr::ILike(Like::new(
                    negated,
                    self.sql_expr_to_logical_expr(*expr, schema, extended_schema)?,
                    pattern,
                    escape_char_to_char(escape_char),
                ))))
            }

            SQLExpr::SimilarTo { negated, expr, pattern, escape_char, .. } => {
                let pattern = self.sql_expr_to_logical_expr(*pattern, schema, extended_schema)?;
                let pattern_type = pattern.get_type(schema)?;
                if pattern_type != DataType::Utf8 && pattern_type != DataType::Null {
                    return Err(DataFusionError::Plan(
                        "Invalid pattern in SIMILAR TO expression".to_string(),
                    ));
                }
                Ok(Box::new(Expr::SimilarTo(Like::new(
                    negated,
                    self.sql_expr_to_logical_expr(*expr, schema, extended_schema)?,
                    pattern,
                    escape_char_to_char(escape_char),
                ))))
            }

            SQLExpr::BinaryOp {
                left,
                op,
                right,
            } => self.parse_sql_binary_op(*left, op, *right, schema, extended_schema),

            SQLExpr::AnyOp {
                left,
                compare_op,
                right,
                ..
            } => self.parse_sql_binary_any(
                *left,
                compare_op,
                *right,
                false,
                schema,
                extended_schema,
            ),

            SQLExpr::AllOp {
                left,
                compare_op,
                right,
            } => self.parse_sql_binary_any(
                *left,
                compare_op,
                *right,
                true,
                schema,
                extended_schema,
            ),

            #[cfg(feature = "unicode_expressions")]
            SQLExpr::Substring {
                expr,
                substring_from,
                substring_for,
                ..
            } => self.sql_substring_to_expr(
                expr,
                substring_from,
                substring_for,
                schema,
                extended_schema,
            ),

            #[cfg(not(feature = "unicode_expressions"))]
            SQLExpr::Substring {
                ..
            } => {
                Err(DataFusionError::Internal(
                    "statement substring requires compilation with feature flag: unicode_expressions.".to_string()
                ))
            }

            SQLExpr::Trim { expr, trim_where, trim_what, .. } => {
                self.sql_trim_to_expr(expr, trim_where, trim_what, schema, extended_schema)
            }
            SQLExpr::Rollup(exprs) => {
                self.sql_rollup_to_expr(exprs, schema, extended_schema)
            }
            SQLExpr::Cube(exprs) => {
                self.sql_cube_to_expr(exprs, schema, extended_schema)
            }

            SQLExpr::Function(function) => {
                self.sql_function_to_expr(function, schema, extended_schema)
            }

            SQLExpr::Nested(e) => self.sql_expr_to_logical_expr(*e, schema, extended_schema),

            SQLExpr::Subquery(q) => self.subquery_to_plan(q, SubqueryType::Scalar, schema).map(Box::new),

            // InSubquery uses `AnyAll` since it's expected to be replaced
            SQLExpr::InSubquery { expr, subquery, negated } => Ok(Box::new(Expr::InSubquery {
                expr: self.sql_expr_to_logical_expr(*expr, schema, extended_schema)?,
                subquery: Box::new(self.subquery_to_plan(subquery, SubqueryType::AnyAll, schema)?),
                negated,
            })),

            SQLExpr::Exists { subquery, .. } => {
                self.subquery_to_plan(subquery, SubqueryType::Exists, schema).map(Box::new)
            }

            // TODO: To support AtTimeZone when DF supports timezones
            SQLExpr::AtTimeZone { timestamp, .. } => {
                self.sql_expr_to_logical_expr(*timestamp, schema, extended_schema)
            }

            _ => Err(DataFusionError::NotImplemented(format!(
                "Unsupported ast node {:?} in sqltorel",
                sql
            ))),
        }
    }

    /// Plan a `CompoundFieldAccess` (e.g. `a.b`, `arr[i]`, `struct.field`).
    ///
    /// Extracted out of [`Self::sql_expr_to_logical_expr`] so its locals do not inflate the
    /// stack frame of the deeply-recursive dispatcher. `#[inline(never)]` keeps the frames
    /// separate in optimized builds.
    #[inline(never)]
    fn sql_compound_field_access_to_expr(
        &self,
        root: Box<SQLExpr>,
        access_chain: Vec<AccessExpr>,
        schema: &DFSchema,
        extended_schema: Option<&DFSchema>,
    ) -> Result<Box<Expr>> {
        // A leading chain of identifier dot-accesses (e.g. `r.value`) is a
        // (possibly qualified) column reference; collapse it into a single
        // identifier before applying any remaining subscript/field accesses.
        let mut chain = access_chain.into_iter().peekable();
        let base_expr = match *root {
            SQLExpr::Identifier(id) => {
                let mut idents = vec![id];
                while let Some(AccessExpr::Dot(SQLExpr::Identifier(_))) = chain.peek() {
                    if let Some(AccessExpr::Dot(SQLExpr::Identifier(field))) =
                        chain.next()
                    {
                        idents.push(field);
                    }
                }
                let base_sql = if idents.len() == 1 {
                    SQLExpr::Identifier(idents.pop().unwrap())
                } else {
                    SQLExpr::CompoundIdentifier(idents)
                };
                *self.sql_expr_to_logical_expr(base_sql, schema, extended_schema)?
            }
            other => *self.sql_expr_to_logical_expr(other, schema, extended_schema)?,
        };
        let mut expr = base_expr;
        for access in chain {
            expr = match access {
                AccessExpr::Subscript(Subscript::Index { index }) => {
                    let key =
                        self.sql_expr_to_logical_expr(index, schema, extended_schema)?;
                    Expr::GetIndexedField {
                        expr: Box::new(expr),
                        key,
                    }
                }
                AccessExpr::Dot(SQLExpr::Identifier(field)) => Expr::GetIndexedField {
                    expr: Box::new(expr),
                    key: Box::new(Expr::Literal(ScalarValue::Utf8(Some(field.value)))),
                },
                other => {
                    return Err(DataFusionError::NotImplemented(format!(
                        "Unsupported compound field access: {:?}",
                        other
                    )))
                }
            };
        }
        Ok(Box::new(expr))
    }

    /// Plan a `CompoundIdentifier` (e.g. `t.col`, `schema.t.col`, `@@var`).
    ///
    /// Extracted out of [`Self::sql_expr_to_logical_expr`] to keep the dispatcher's stack frame
    /// small on the deep recursion path. `#[inline(never)]` keeps the frames separate.
    #[inline(never)]
    fn sql_compound_identifier_to_expr(
        &self,
        ids: Vec<Ident>,
        schema: &DFSchema,
        extended_schema: Option<&DFSchema>,
    ) -> Result<Box<Expr>> {
        let mut var_names: Vec<_> = ids.into_iter().map(normalize_ident).collect();

        if &var_names[0][0..1] == "@" {
            let ty = self
                .schema_provider
                .get_variable_type(&var_names)
                .ok_or_else(|| {
                    DataFusionError::Execution(format!(
                        "variable {:?} has no type information",
                        var_names
                    ))
                })?;
            Ok(Box::new(Expr::ScalarVariable(ty, var_names)))
        } else {
            match (var_names.pop(), var_names.pop()) {
                (Some(name), Some(relation)) => {
                    if let Some(schema) = var_names.pop() {
                        if !var_names.is_empty() {
                            return Err(DataFusionError::NotImplemented(format!(
                                "Unsupported compound identifier '{:?}'",
                                var_names,
                            )));
                        }
                        let schema = schema.to_lowercase();
                        if !["public", "pg_catalog"].contains(&schema.as_str()) {
                            return Err(DataFusionError::NotImplemented(format!(
                                "Unsupported compound identifier '{:?}'",
                                schema,
                            )));
                        }
                    }

                    // Rules for finding the column:
                    // - try the current schema first
                    // - if available, try extended schema to allow adding
                    //   missing columns and aggregate expressions down the plan
                    // - try to get column from outer query context last
                    // - finally use the column as-is
                    for schema in [schema].iter().chain(extended_schema.iter()) {
                        if schema.field_with_qualified_name(&relation, &name).is_ok() {
                            return Ok(Box::new(Expr::Column(Column {
                                relation: Some(relation),
                                name,
                            })));
                        }

                        let search_term = format!(".{}.{}", relation, name);
                        if schema
                            .fields()
                            .iter()
                            .any(|f| f.qualified_name().as_str().ends_with(&search_term))
                        {
                            // this could probably be improved but here we handle the case
                            // where the qualifier is only a partial qualifier such as when
                            // referencing "t1.foo" when the available field is "public.t1.foo"
                            return Ok(Box::new(Expr::Column(Column {
                                relation: Some(relation),
                                name,
                            })));
                        }

                        if let Some(field) =
                            schema.fields().iter().find(|f| f.name().eq(&relation))
                        {
                            // Access to a field of a column which is a structure, example: SELECT my_struct.key
                            return Ok(Box::new(Expr::GetIndexedField {
                                expr: Box::new(Expr::Column(field.qualified_column())),
                                key: Box::new(Expr::Literal(ScalarValue::Utf8(Some(
                                    name,
                                )))),
                            }));
                        }
                    }

                    if let Some(f) = self
                        .context
                        .outer_query_context_schema
                        .iter()
                        .find_map(|s| s.field_with_qualified_name(&relation, &name).ok())
                    {
                        // Access to an outer column from a subquery
                        return Ok(Box::new(Expr::OuterColumn(
                            f.data_type().clone(),
                            Column {
                                relation: Some(relation),
                                name,
                            },
                        )));
                    }

                    // This is a fix for Sort with relation. See filter_idents_test test for more information.
                    Ok(Box::new(Expr::Column(Column {
                        relation: Some(relation),
                        name,
                    })))
                }
                _ => Err(DataFusionError::NotImplemented(format!(
                    "Unsupported compound identifier '{:?}'",
                    var_names,
                ))),
            }
        }
    }

    /// Plan an `INTERVAL` literal or expression.
    ///
    /// Extracted out of [`Self::sql_expr_to_logical_expr`] to keep the deeply-recursive
    /// dispatcher's stack frame small. `#[inline(never)]` keeps the frames separate.
    #[inline(never)]
    fn sql_interval_to_expr(
        &self,
        interval: sqlparser::ast::Interval,
        schema: &DFSchema,
        extended_schema: Option<&DFSchema>,
    ) -> Result<Box<Expr>> {
        let sqlparser::ast::Interval {
            value,
            leading_field,
            leading_precision,
            last_field,
            fractional_seconds_precision,
        } = interval;
        match *value {
            SQLExpr::Value(ValueWithSpan {
                value: Value::Number(value, _),
                ..
            }) => self
                .sql_interval_to_literal(
                    value,
                    leading_field,
                    leading_precision,
                    last_field,
                    fractional_seconds_precision,
                )
                .map(Box::new),
            SQLExpr::Value(ValueWithSpan {
                value: Value::SingleQuotedString(value),
                ..
            }) => self
                .sql_interval_to_literal(
                    value,
                    leading_field,
                    leading_precision,
                    last_field,
                    fractional_seconds_precision,
                )
                .map(Box::new),
            expr => {
                let unit = leading_field
                    .as_ref()
                    .map(|dt| dt.to_string())
                    .unwrap_or_else(|| "second".to_string());

                let fun = if let Some(leading_field) = leading_field {
                    match leading_field {
                        DateTimeField::Year => BuiltinScalarFunction::ToMonthInterval,
                        DateTimeField::Month => BuiltinScalarFunction::ToMonthInterval,
                        DateTimeField::Quarter => BuiltinScalarFunction::ToMonthInterval,
                        _ => BuiltinScalarFunction::ToDayInterval,
                    }
                } else {
                    BuiltinScalarFunction::ToDayInterval
                };

                Ok(Box::new(Expr::ScalarFunction {
                    fun,
                    args: vec![
                        *self.sql_expr_to_logical_expr(expr, schema, extended_schema)?,
                        Expr::Literal(ScalarValue::Utf8(Some(unit.to_lowercase()))),
                    ],
                }))
            }
        }
    }

    /// Plan a `CASE` expression.
    ///
    /// Extracted out of [`Self::sql_expr_to_logical_expr`] to keep the dispatcher's stack frame
    /// small on the deep recursion path. `#[inline(never)]` keeps the frames separate.
    #[inline(never)]
    fn sql_case_to_expr(
        &self,
        operand: Option<Box<SQLExpr>>,
        conditions: Vec<CaseWhen>,
        else_result: Option<Box<SQLExpr>>,
        schema: &DFSchema,
        extended_schema: Option<&DFSchema>,
    ) -> Result<Box<Expr>> {
        let expr = if let Some(e) = operand {
            Some(self.sql_expr_to_logical_expr(*e, schema, extended_schema)?)
        } else {
            None
        };
        let when_then_expr = conditions
            .into_iter()
            .map(|CaseWhen { condition, result }| {
                Ok((
                    self.sql_expr_to_logical_expr(condition, schema, extended_schema)?,
                    self.sql_expr_to_logical_expr(result, schema, extended_schema)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let else_expr = if let Some(e) = else_result {
            Some(self.sql_expr_to_logical_expr(*e, schema, extended_schema)?)
        } else {
            None
        };

        Ok(Box::new(Expr::Case {
            expr,
            when_then_expr,
            else_expr,
        }))
    }

    /// Plan a `SUBSTRING(expr FROM .. FOR ..)` expression.
    ///
    /// Extracted out of [`Self::sql_expr_to_logical_expr`] to keep the dispatcher's stack frame
    /// small on the deep recursion path. `#[inline(never)]` keeps the frames separate.
    #[cfg(feature = "unicode_expressions")]
    #[inline(never)]
    fn sql_substring_to_expr(
        &self,
        expr: Box<SQLExpr>,
        substring_from: Option<Box<SQLExpr>>,
        substring_for: Option<Box<SQLExpr>>,
        schema: &DFSchema,
        extended_schema: Option<&DFSchema>,
    ) -> Result<Box<Expr>> {
        let args = match (substring_from, substring_for) {
            (Some(from_expr), Some(for_expr)) => {
                let arg =
                    *self.sql_expr_to_logical_expr(*expr, schema, extended_schema)?;
                let from_logic = *self.sql_expr_to_logical_expr(
                    *from_expr,
                    schema,
                    extended_schema,
                )?;
                let for_logic =
                    *self.sql_expr_to_logical_expr(*for_expr, schema, extended_schema)?;
                vec![arg, from_logic, for_logic]
            }
            (Some(from_expr), None) => {
                let arg =
                    *self.sql_expr_to_logical_expr(*expr, schema, extended_schema)?;
                let from_logic = *self.sql_expr_to_logical_expr(
                    *from_expr,
                    schema,
                    extended_schema,
                )?;
                vec![arg, from_logic]
            }
            (None, Some(for_expr)) => {
                let arg =
                    *self.sql_expr_to_logical_expr(*expr, schema, extended_schema)?;
                let from_logic = Expr::Literal(ScalarValue::Int64(Some(1)));
                let for_logic =
                    *self.sql_expr_to_logical_expr(*for_expr, schema, extended_schema)?;
                vec![arg, from_logic, for_logic]
            }
            (None, None) => {
                return Err(DataFusionError::Plan(format!(
                    "Substring without for/from is not valid {:?}",
                    expr
                )));
            }
        };

        Ok(Box::new(Expr::ScalarFunction {
            fun: BuiltinScalarFunction::Substr,
            args,
        }))
    }

    /// Plan a `TRIM([LEADING|TRAILING|BOTH] [chars] FROM expr)` expression.
    ///
    /// Extracted out of [`Self::sql_expr_to_logical_expr`] to keep the dispatcher's stack frame
    /// small on the deep recursion path. `#[inline(never)]` keeps the frames separate.
    #[inline(never)]
    fn sql_trim_to_expr(
        &self,
        expr: Box<SQLExpr>,
        trim_where: Option<TrimWhereField>,
        trim_what: Option<Box<SQLExpr>>,
        schema: &DFSchema,
        extended_schema: Option<&DFSchema>,
    ) -> Result<Box<Expr>> {
        let fun = match trim_where {
            Some(TrimWhereField::Leading) => BuiltinScalarFunction::Ltrim,
            Some(TrimWhereField::Trailing) => BuiltinScalarFunction::Rtrim,
            Some(TrimWhereField::Both) => BuiltinScalarFunction::Btrim,
            None => BuiltinScalarFunction::Trim,
        };
        let where_expr = trim_what;
        let arg = *self.sql_expr_to_logical_expr(*expr, schema, extended_schema)?;
        let args = match where_expr {
            Some(to_trim) => {
                let to_trim =
                    *self.sql_expr_to_logical_expr(*to_trim, schema, extended_schema)?;
                vec![arg, to_trim]
            }
            None => vec![arg],
        };
        Ok(Box::new(Expr::ScalarFunction { fun, args }))
    }

    /// Plan a function call: scalar/aggregate/window built-ins, `ROLLUP`/`CUBE`, and UDF/UDAF/UDTF.
    ///
    /// Extracted out of [`Self::sql_expr_to_logical_expr`] because it is the heaviest arm (many
    /// locals: window spec, partition/order-by, frame, resolved function); keeping it out of the
    /// deeply-recursive dispatcher shrinks the dispatcher's stack frame. `#[inline(never)]` keeps
    /// the frames separate in optimized builds.
    #[inline(never)]
    fn sql_function_to_expr(
        &self,
        function: sqlparser::ast::Function,
        schema: &DFSchema,
        extended_schema: Option<&DFSchema>,
    ) -> Result<Box<Expr>> {
        let name = if function.name.0.len() > 1 {
            // Postgres allows catalog functions to be called with an explicit schema
            // qualifier (e.g. `pg_catalog.array_agg(...)`, `pg_catalog.pg_get_expr(...)`).
            // DataFusion resolves functions by their bare name, so drop a leading
            // `pg_catalog`/`public` qualifier and dispatch on the final identifier.
            // (sqlparser used to special-case `pg_catalog.array_agg` into a dedicated
            // `ArrayAgg` node; since that node was removed it now arrives as a regular
            // compound-named function and must be normalized here.)
            let qualifier =
                object_name_part_to_string(&function.name.0[0]).to_ascii_lowercase();
            if function.name.0.len() == 2
                && (qualifier == "pg_catalog" || qualifier == "public")
            {
                object_name_part_to_string(&function.name.0[1]).to_ascii_lowercase()
            } else {
                // DF doesn't handle compound identifiers
                // (e.g. "foo.bar") for function names yet
                function.name.to_string().to_ascii_lowercase()
            }
        } else {
            object_name_part_to_string(&function.name.0[0]).to_ascii_lowercase()
        };

        // `ARRAY(<subquery>)` parses as a function call since sqlparser 0.62 (it used to be
        // `Expr::ArraySubquery`). Array-subqueries are not executed; preserve the historic
        // behaviour of substituting an empty array literal.
        if name == "array" && matches!(function.args, FunctionArguments::Subquery(_)) {
            log::warn!("ARRAY(<subquery>) is not supported yet. Replacing with scalar empty array.");
            return Ok(Box::new(Expr::Literal(ScalarValue::List(
                Some(Box::new(vec![])),
                Box::new(DataType::Utf8),
            ))));
        }

        let over = function.over;
        let within_group = function.within_group;
        let (arg_list, distinct, clauses) = function_arguments_into_args(function.args);

        // DataFusion does not support an in-argument `LIMIT` clause such as
        // `array_agg(expr LIMIT n)`. (An in-argument `ORDER BY` is ignored, as before.)
        for clause in &clauses {
            if let FunctionArgumentClause::Limit(expr) = clause {
                return Err(DataFusionError::NotImplemented(format!(
                    "LIMIT not supported in {}: {}",
                    name.to_ascii_uppercase(),
                    expr
                )));
            }
        }

        // first, check SQL reserved words
        if name == "rollup" {
            let args = self.function_args_to_expr(arg_list, schema, extended_schema)?;
            return Ok(Box::new(Expr::GroupingSet(GroupingSet::Rollup(args))));
        } else if name == "cube" {
            let args = self.function_args_to_expr(arg_list, schema, extended_schema)?;
            return Ok(Box::new(Expr::GroupingSet(GroupingSet::Cube(args))));
        }

        // next, scalar built-in
        if let Ok(fun) = BuiltinScalarFunction::from_str(&name) {
            let args = self.function_args_to_expr(arg_list, schema, extended_schema)?;
            return Ok(Box::new(Expr::ScalarFunction { fun, args }));
        };

        // then, window function
        if let Some(window) = over {
            let window = match window {
                WindowType::WindowSpec(spec) => spec,
                WindowType::NamedWindow(name) => {
                    return Err(DataFusionError::NotImplemented(format!(
                        "Named window reference {} is not supported",
                        name
                    )))
                }
            };
            let partition_by = window
                .partition_by
                .into_iter()
                .map(|e| {
                    self.sql_expr_to_logical_expr(e, schema, extended_schema)
                        .map(|b| *b)
                })
                .collect::<Result<Vec<_>>>()?;
            let order_by = window
                .order_by
                .into_iter()
                .map(|e| self.order_by_to_sort_expr(e, schema, extended_schema, true))
                .collect::<Result<Vec<_>>>()?;
            let window_frame = window
                .window_frame
                .as_ref()
                .map(|window_frame| {
                    let window_frame: WindowFrame = window_frame.clone().try_into()?;
                    if WindowFrameUnits::Range == window_frame.units
                        && order_by.len() != 1
                    {
                        Err(DataFusionError::Plan(format!(
                            "With window frame of type RANGE, the order by expression must be of length 1, got {}", order_by.len())))
                    } else {
                        Ok(window_frame)
                    }
                })
                .transpose()?;
            let fun = WindowFunction::from_str(&name)?;
            match fun {
                WindowFunction::AggregateFunction(aggregate_fun) => {
                    let (aggregate_fun, args) = self.aggregate_fn_to_expr(
                        aggregate_fun,
                        arg_list,
                        schema,
                        extended_schema,
                    )?;

                    return Ok(Box::new(Expr::WindowFunction {
                        fun: WindowFunction::AggregateFunction(aggregate_fun),
                        args,
                        partition_by,
                        order_by,
                        window_frame,
                    }));
                }
                WindowFunction::BuiltInWindowFunction(window_fun) => {
                    return Ok(Box::new(Expr::WindowFunction {
                        fun: WindowFunction::BuiltInWindowFunction(window_fun),
                        args: self.function_args_to_expr(
                            arg_list,
                            schema,
                            extended_schema,
                        )?,
                        partition_by,
                        order_by,
                        window_frame,
                    }));
                }
            }
        }

        // next, aggregate built-ins
        if let Ok(fun) = aggregates::AggregateFunction::from_str(&name) {
            let (fun, args) =
                self.aggregate_fn_to_expr(fun, arg_list, schema, extended_schema)?;
            let agg = Expr::AggregateFunction {
                fun,
                distinct,
                args,
                within_group: None,
            };
            return self.apply_within_group(agg, within_group, schema, extended_schema);
        };

        // finally, user-defined functions (UDF) and UDAF
        match self.schema_provider.get_function_meta(&name) {
            Some(fm) => {
                let args =
                    self.function_args_to_expr(arg_list, schema, extended_schema)?;

                Ok(Box::new(Expr::ScalarUDF { fun: fm, args }))
            }
            None => match self.schema_provider.get_aggregate_meta(&name) {
                Some(fm) => {
                    let args =
                        self.function_args_to_expr(arg_list, schema, extended_schema)?;
                    Ok(Box::new(Expr::AggregateUDF {
                        fun: fm,
                        args,
                        distinct,
                    }))
                }
                None => match self.schema_provider.get_table_function_meta(&name) {
                    Some(fm) => {
                        let args = self.function_args_to_expr(
                            arg_list,
                            schema,
                            extended_schema,
                        )?;
                        Ok(Box::new(Expr::TableUDF { fun: fm, args }))
                    }
                    _ => Err(DataFusionError::Plan(format!(
                        "Invalid function '{}'",
                        name
                    ))),
                },
            },
        }
    }

    /// Attach a `WITHIN GROUP (ORDER BY ...)` clause (now carried on sqlparser's `Function`)
    /// to an aggregate expression. Only built-in aggregate functions are supported.
    fn apply_within_group(
        &self,
        mut expr: Expr,
        within_group: Vec<OrderByExpr>,
        input_schema: &DFSchema,
        extended_schema: Option<&DFSchema>,
    ) -> Result<Box<Expr>> {
        if within_group.is_empty() {
            return Ok(Box::new(expr));
        }
        if let Expr::AggregateFunction {
            within_group: agg_within_group,
            ..
        } = &mut expr
        {
            let order_by = within_group
                .into_iter()
                .map(|e| {
                    self.order_by_to_sort_expr(e, input_schema, extended_schema, false)
                })
                .collect::<Result<Vec<_>>>()?;
            *agg_within_group = Some(order_by);
            return Ok(Box::new(expr));
        }
        Err(DataFusionError::NotImplemented(
            "WITHIN GROUP is only supported with built-in aggregate functions"
                .to_string(),
        ))
    }

    fn function_args_to_expr(
        &self,
        args: Vec<FunctionArg>,
        schema: &DFSchema,
        extended_schema: Option<&DFSchema>,
    ) -> Result<Vec<Expr>> {
        args.into_iter()
            .map(|a| self.sql_fn_arg_to_logical_expr(a, schema, extended_schema))
            .collect::<Result<Vec<Expr>>>()
    }

    fn aggregate_fn_to_expr(
        &self,
        fun: aggregates::AggregateFunction,
        fn_args: Vec<FunctionArg>,
        schema: &DFSchema,
        extended_schema: Option<&DFSchema>,
    ) -> Result<(aggregates::AggregateFunction, Vec<Expr>)> {
        let args = match fun {
            aggregates::AggregateFunction::Count => fn_args
                .into_iter()
                .map(|a| match a {
                    FunctionArg::Unnamed(FunctionArgExpr::Expr(SQLExpr::Value(
                        ValueWithSpan {
                            value: Value::Number(_, _),
                            ..
                        },
                    ))) => Ok(lit(1_u8)),
                    FunctionArg::Unnamed(FunctionArgExpr::Wildcard) => Ok(lit(1_u8)),
                    _ => self.sql_fn_arg_to_logical_expr(a, schema, extended_schema),
                })
                .collect::<Result<Vec<Expr>>>()?,
            aggregates::AggregateFunction::ApproxMedian => fn_args
                .into_iter()
                .map(|a| self.sql_fn_arg_to_logical_expr(a, schema, extended_schema))
                .chain(iter::once(Ok(lit(0.5_f64))))
                .collect::<Result<Vec<Expr>>>()?,
            _ => self.function_args_to_expr(fn_args, schema, extended_schema)?,
        };

        let fun = match fun {
            aggregates::AggregateFunction::ApproxMedian => {
                aggregates::AggregateFunction::ApproxPercentileCont
            }
            _ => fun,
        };

        Ok((fun, args))
    }

    fn sql_interval_to_literal(
        &self,
        value: String,
        leading_field: Option<DateTimeField>,
        leading_precision: Option<u64>,
        last_field: Option<DateTimeField>,
        fractional_seconds_precision: Option<u64>,
    ) -> Result<Expr> {
        if leading_precision.is_some() {
            return Err(DataFusionError::NotImplemented(format!(
                "Unsupported Interval Expression with leading_precision {:?}",
                leading_precision
            )));
        }

        if last_field.is_some() {
            return Err(DataFusionError::NotImplemented(format!(
                "Unsupported Interval Expression with last_field {:?}",
                last_field
            )));
        }

        if fractional_seconds_precision.is_some() {
            return Err(DataFusionError::NotImplemented(format!(
                "Unsupported Interval Expression with fractional_seconds_precision {:?}",
                fractional_seconds_precision
            )));
        }

        const SECONDS_PER_HOUR: f32 = 3_600_f32;
        const MILLIS_PER_SECOND: f32 = 1_000_f32;

        // We are storing parts as integers, it's why we need to align parts fractional
        // INTERVAL '0.5 MONTH' = 15 days, INTERVAL '1.5 MONTH' = 1 month 15 days
        // INTERVAL '0.5 DAY' = 12 hours, INTERVAL '1.5 DAY' = 1 day 12 hours
        let align_interval_parts = |month_part: f32,
                                    mut day_part: f32,
                                    mut milles_part: f32|
         -> (i32, i32, f32) {
            // Convert fractional month to days, It's not supported by Arrow types, but anyway
            day_part += (month_part - (month_part as i32) as f32) * 30_f32;

            // Convert fractional days to hours
            milles_part += (day_part - ((day_part as i32) as f32))
                * 24_f32
                * SECONDS_PER_HOUR
                * MILLIS_PER_SECOND;

            (month_part as i32, day_part as i32, milles_part)
        };

        let calculate_from_part = |interval_period_str: &str,
                                   interval_type: &str|
         -> Result<(i32, i32, f32)> {
            // @todo It's better to use Decimal in order to protect rounding errors
            // Wait https://github.com/apache/arrow/pull/9232
            let interval_period = match f32::from_str(interval_period_str) {
                Ok(n) => n,
                Err(_) => {
                    return Err(DataFusionError::SQL(ParserError(format!(
                        "Unsupported Interval Expression with value {:?}",
                        value
                    ))));
                }
            };

            if interval_period > (i32::MAX as f32) {
                return Err(DataFusionError::NotImplemented(format!(
                    "Interval field value out of range: {:?}",
                    value
                )));
            }

            if interval_period < (i32::MIN as f32) {
                return Err(DataFusionError::NotImplemented(format!(
                    "Interval field value out of range: {:?}",
                    value
                )));
            }

            match interval_type.to_lowercase().as_str() {
                "years" | "year" | "y" => {
                    Ok(align_interval_parts(interval_period * 12_f32, 0.0, 0.0))
                }
                "quarter" | "qtr" => {
                    Ok(align_interval_parts(interval_period * 3_f32, 0.0, 0.0))
                }
                "months" | "month" | "mons" | "mon" => {
                    Ok(align_interval_parts(interval_period, 0.0, 0.0))
                }
                "weeks" | "week" | "w" => {
                    Ok(align_interval_parts(0.0, interval_period * 7_f32, 0.0))
                }
                "days" | "day" | "d" => {
                    Ok(align_interval_parts(0.0, interval_period, 0.0))
                }
                "hours" | "hour" | "h" => {
                    Ok((0, 0, interval_period * SECONDS_PER_HOUR * MILLIS_PER_SECOND))
                }
                "minutes" | "minute" | "mins" | "min" | "m" => {
                    Ok((0, 0, interval_period * 60_f32 * MILLIS_PER_SECOND))
                }
                "seconds" | "second" | "secs" | "sec" | "s" => {
                    Ok((0, 0, interval_period * MILLIS_PER_SECOND))
                }
                "milliseconds" | "millisecond" | "msecs" | "msec" | "ms" => {
                    Ok((0, 0, interval_period))
                }
                _ => Err(DataFusionError::NotImplemented(format!(
                    "Invalid input syntax for type interval: {:?}",
                    value
                ))),
            }
        };

        let mut result_month: i32 = 0;
        let mut result_days: i32 = 0;
        let mut result_millis: i32 = 0;

        let mut parts = value.split_whitespace();

        let out_of_range = || {
            DataFusionError::NotImplemented(format!(
                "Interval field value out of range: {:?}",
                value
            ))
        };

        loop {
            let interval_period_str = parts.next();
            if interval_period_str.is_none() {
                break;
            }

            let leading_field = leading_field
                .as_ref()
                .map(|dt| dt.to_string())
                .unwrap_or_else(|| "second".to_string());

            let unit = parts
                .next()
                .map(|part| part.to_string())
                .unwrap_or(leading_field);

            let (diff_month, diff_days, diff_millis) =
                calculate_from_part(interval_period_str.unwrap(), &unit)?;

            result_month = result_month
                .checked_add(diff_month)
                .ok_or_else(out_of_range)?;

            result_days = result_days
                .checked_add(diff_days)
                .ok_or_else(out_of_range)?;

            result_millis = result_millis
                .checked_add(diff_millis as i32)
                .ok_or_else(out_of_range)?;
        }

        // Interval is tricky thing
        // 1 day is not 24 hours because timezones, 1 year != 365/364! 30 days != 1 month
        // The true way to store and calculate intervals is to store it as it defined
        // It's why we there are 3 different interval types in Arrow
        if result_month != 0 && (result_days != 0 || result_millis != 0) {
            let result = IntervalMonthDayNanoType::make_value(
                result_month,
                result_days,
                // IntervalMonthDayNano uses nanos, but IntervalDayTime uses millis
                result_millis as i64 * 1_000_000_i64,
            );
            return Ok(Expr::Literal(ScalarValue::IntervalMonthDayNano(Some(
                result,
            ))));
        }

        // Month interval
        if result_month != 0 {
            return Ok(Expr::Literal(ScalarValue::IntervalYearMonth(Some(
                result_month,
            ))));
        }

        let result = IntervalDayTimeType::make_value(result_days, result_millis);
        Ok(Expr::Literal(ScalarValue::IntervalDayTime(Some(result))))
    }

    fn show_variable_to_plan(&self, variable: &[Ident]) -> Result<LogicalPlan> {
        let variable = variable
            .iter()
            .map(|i| i.value.clone())
            .collect::<Vec<_>>()
            .join(".");
        Err(DataFusionError::NotImplemented(format!(
            "SHOW {} not implemented. Supported syntax: SHOW <TABLES>",
            variable
        )))
    }

    fn show_columns_to_plan(
        &self,
        extended: bool,
        full: bool,
        table_name: &ObjectName,
        has_filter: bool,
    ) -> Result<LogicalPlan> {
        if has_filter {
            return Err(DataFusionError::Plan(
                "SHOW COLUMNS with WHERE or LIKE is not supported".to_string(),
            ));
        }

        if !self.has_table("information_schema", "columns") {
            return Err(DataFusionError::Plan(
                "SHOW COLUMNS is not supported unless information_schema is enabled"
                    .to_string(),
            ));
        }

        if self
            .schema_provider
            .get_table_provider(table_name.try_into()?)
            .is_none()
        {
            return Err(DataFusionError::Plan(format!(
                "Unknown relation for SHOW COLUMNS: {}",
                table_name
            )));
        }

        // Figure out the where clause
        let columns = vec!["table_name", "table_schema", "table_catalog"].into_iter();
        let where_clause = table_name
            .0
            .iter()
            .rev()
            .zip(columns)
            .map(|(ident, column_name)| format!(r#"{} = '{}'"#, column_name, ident))
            .collect::<Vec<_>>()
            .join(" AND ");

        // treat both FULL and EXTENDED as the same
        let select_list = if full || extended {
            "*"
        } else {
            "table_catalog, table_schema, table_name, column_name, data_type, is_nullable"
        };

        let query = format!(
            "SELECT {} FROM information_schema.columns WHERE {}",
            select_list, where_clause
        );

        let mut rewrite = DFParser::parse_sql(&query)?;
        assert_eq!(rewrite.len(), 1);
        self.statement_to_plan(rewrite.pop_front().unwrap())
    }

    /// Return true if there is a table provider available for "schema.table"
    fn has_table(&self, schema: &str, table: &str) -> bool {
        let tables_reference = TableReference::Partial { schema, table };
        self.schema_provider
            .get_table_provider(tables_reference)
            .is_some()
    }

    fn sql_array_literal(
        &self,
        elements: Vec<SQLExpr>,
        schema: &DFSchema,
        extended_schema: Option<&DFSchema>,
    ) -> Result<Expr> {
        let mut values = Vec::with_capacity(elements.len());

        for element in elements {
            let value =
                *self.sql_expr_to_logical_expr(element, schema, extended_schema)?;
            match value {
                Expr::Literal(scalar) => {
                    values.push(scalar);
                }
                _ => {
                    return Err(DataFusionError::NotImplemented(format!(
                        "Arrays with elements other than literal are not supported: {}",
                        value
                    )));
                }
            }
        }

        let data_types: HashSet<DataType> = values
            .iter()
            .filter_map(|e| match e.get_datatype() {
                DataType::Null => None,
                _ => Some(e.get_datatype()),
            })
            .collect();

        if data_types.is_empty() {
            Ok(Expr::Literal(ScalarValue::List(
                None,
                Box::new(DataType::Utf8),
            )))
        } else if data_types.len() > 1 {
            Err(DataFusionError::NotImplemented(format!(
                "Arrays with different types are not supported: {:?}",
                data_types,
            )))
        } else {
            let data_type = data_types.iter().next().unwrap().clone();

            Ok(Expr::Literal(ScalarValue::List(
                Some(Box::new(values)),
                Box::new(data_type),
            )))
        }
    }

    fn subquery_to_plan(
        &self,
        query: Box<Query>,
        subquery_type: SubqueryType,
        schema: &DFSchema,
    ) -> Result<Expr> {
        let with_outer_query_context = self.with_context(|c| {
            c.outer_query_context_schema.push(Arc::new(schema.clone()))
        });
        let alias_name = {
            let mut subquery_alias_iter = with_outer_query_context
                .subquery_alias_iter
                .lock()
                .map_err(|_| {
                    DataFusionError::Plan(
                        "Unable to lock subquery alias iterator".to_string(),
                    )
                })?;
            let alias_index = subquery_alias_iter.next().ok_or_else(|| {
                DataFusionError::Plan(
                    "Unable to assign an alias to a subquery".to_string(),
                )
            })?;
            format!("__subquery-{}", alias_index)
        };
        let plan = with_outer_query_context
            .query_to_plan_with_alias(*query, Some(alias_name))?;

        let fields = plan.schema().fields();
        if fields.len() != 1 {
            return Err(DataFusionError::Plan(format!("Correlated sub query requires only one column in result set but found: {:?}", fields)));
        }
        let column = fields.iter().next().unwrap().qualified_column();
        self.context.add_subquery_plan(plan, subquery_type)?;
        Ok(Expr::Column(column))
    }
}

/// Normalize a SQL object name
fn normalize_sql_object_name(sql_object_name: &ObjectName) -> String {
    sql_object_name
        .0
        .iter()
        .map(object_name_part_to_string)
        .collect::<Vec<String>>()
        .join(".")
}

/// Extract the identifier value of an [`ObjectNamePart`]. Function-style name parts
/// (dialect-specific) are rendered via their `Display` implementation.
fn object_name_part_to_string(part: &ObjectNamePart) -> String {
    match part.as_ident() {
        Some(ident) => ident.value.clone(),
        None => part.to_string(),
    }
}
/// Remove join expressions from a filter expression
fn remove_join_expressions(
    expr: &Expr,
    join_columns: &HashSet<(Column, Column)>,
) -> Result<Option<Expr>> {
    match expr {
        Expr::BinaryExpr { left, op, right } => match op {
            Operator::Eq => match (left.as_ref(), right.as_ref()) {
                (Expr::Column(l), Expr::Column(r)) => {
                    if join_columns.contains(&(l.clone(), r.clone()))
                        || join_columns.contains(&(r.clone(), l.clone()))
                    {
                        Ok(None)
                    } else {
                        Ok(Some(expr.clone()))
                    }
                }
                _ => Ok(Some(expr.clone())),
            },
            Operator::And => {
                let l = remove_join_expressions(left, join_columns)?;
                let r = remove_join_expressions(right, join_columns)?;
                match (l, r) {
                    (Some(ll), Some(rr)) => Ok(Some(and(ll, rr))),
                    (Some(ll), _) => Ok(Some(ll)),
                    (_, Some(rr)) => Ok(Some(rr)),
                    _ => Ok(None),
                }
            }
            _ => Ok(Some(expr.clone())),
        },
        _ => Ok(Some(expr.clone())),
    }
}

/// Extracts equijoin ON condition be a single Eq or multiple conjunctive Eqs
/// Filters matching this pattern are added to `accum`
/// Filters that don't match this pattern are added to `accum_filter`
/// Examples:
/// ```text
/// foo = bar => accum=[(foo, bar)] accum_filter=[]
/// foo = bar AND bar = baz => accum=[(foo, bar), (bar, baz)] accum_filter=[]
/// foo = bar AND baz > 1 => accum=[(foo, bar)] accum_filter=[baz > 1]
/// ```
fn extract_join_keys(
    expr: Expr,
    accum: &mut Vec<(Column, Column)>,
    accum_filter: &mut Vec<Expr>,
) {
    match &expr {
        Expr::BinaryExpr { left, op, right } => match op {
            Operator::Eq => match (left.as_ref(), right.as_ref()) {
                (Expr::Column(l), Expr::Column(r)) => {
                    accum.push((l.clone(), r.clone()));
                }
                _other => {
                    accum_filter.push(expr);
                }
            },
            Operator::And => {
                if let Expr::BinaryExpr { left, op: _, right } = expr {
                    extract_join_keys(*left, accum, accum_filter);
                    extract_join_keys(*right, accum, accum_filter);
                }
            }
            _other => {
                accum_filter.push(expr);
            }
        },
        _other => {
            accum_filter.push(expr);
        }
    }
}

/// Extract join keys from a WHERE clause
fn extract_possible_join_keys(
    expr: &Expr,
    accum: &mut Vec<(Column, Column)>,
) -> Result<()> {
    match expr {
        Expr::BinaryExpr { left, op, right } => match op {
            Operator::Eq => match (left.as_ref(), right.as_ref()) {
                (Expr::Column(l), Expr::Column(r)) => {
                    accum.push((l.clone(), r.clone()));
                    Ok(())
                }
                _ => Ok(()),
            },
            Operator::And => {
                extract_possible_join_keys(left, accum)?;
                extract_possible_join_keys(right, accum)
            }
            _ => Ok(()),
        },
        _ => Ok(()),
    }
}

/// Convert SQL data type to relational representation of data type
pub fn convert_data_type(sql_type: &SQLDataType) -> Result<DataType> {
    match sql_type {
        SQLDataType::Boolean => Ok(DataType::Boolean),
        SQLDataType::TinyInt(_) => Ok(DataType::Int8),
        SQLDataType::TinyIntUnsigned(_) => Ok(DataType::UInt8),
        SQLDataType::SmallInt(_) => Ok(DataType::Int16),
        SQLDataType::SmallIntUnsigned(_) => Ok(DataType::UInt16),
        SQLDataType::Int(_) | SQLDataType::Integer(_) => Ok(DataType::Int32),
        SQLDataType::IntUnsigned(_) => Ok(DataType::UInt32),
        SQLDataType::BigInt(_) => Ok(DataType::Int64),
        SQLDataType::BigIntUnsigned(_) => Ok(DataType::UInt64),
        SQLDataType::Float(_) => Ok(DataType::Float32),
        SQLDataType::Real => Ok(DataType::Float32),
        SQLDataType::Double(_) | SQLDataType::DoublePrecision => Ok(DataType::Float64),
        SQLDataType::Char(_) | SQLDataType::Varchar(_) | SQLDataType::Text => {
            Ok(DataType::Utf8)
        }
        SQLDataType::Timestamp(..) => Ok(DataType::Timestamp(TimeUnit::Nanosecond, None)),
        SQLDataType::Date => Ok(DataType::Date32),
        SQLDataType::Decimal(info) => {
            let (precision, scale) = exact_number_info_to_precision_scale(info);
            make_decimal_type(precision, scale)
        }
        SQLDataType::Interval { .. } => {
            Ok(DataType::Interval(IntervalUnit::MonthDayNano))
        }
        other => Err(DataFusionError::NotImplemented(format!(
            "Unsupported SQL type {:?}",
            other
        ))),
    }
}

/// sqlparser now represents `ESCAPE '<c>'` clauses as an optional `ValueWithSpan`; this
/// codebase only supports a single-character escape, so extract the first char if present.
fn escape_char_to_char(escape_char: Option<ValueWithSpan>) -> Option<char> {
    escape_char
        .and_then(|v| v.into_string())
        .and_then(|s| s.chars().next())
}

/// Flatten sqlparser's `FunctionArguments` into a plain argument list plus the `DISTINCT`
/// flag that older sqlparser versions exposed directly as `Function::distinct`.
fn function_arguments_into_args(
    args: FunctionArguments,
) -> (Vec<FunctionArg>, bool, Vec<FunctionArgumentClause>) {
    match args {
        FunctionArguments::List(list) => {
            let distinct = matches!(
                list.duplicate_treatment,
                Some(sqlparser::ast::DuplicateTreatment::Distinct)
            );
            (list.args, distinct, list.clauses)
        }
        FunctionArguments::None | FunctionArguments::Subquery(_) => {
            (vec![], false, vec![])
        }
    }
}

/// Decompose sqlparser's `ExactNumberInfo` (the precision/scale carried by `DECIMAL`,
/// `NUMERIC`, etc.) into the `(precision, scale)` pair expected by `make_decimal_type`.
fn exact_number_info_to_precision_scale(
    info: &ExactNumberInfo,
) -> (Option<u64>, Option<u64>) {
    match info {
        ExactNumberInfo::None => (None, None),
        ExactNumberInfo::Precision(p) => (Some(*p), None),
        ExactNumberInfo::PrecisionAndScale(p, s) => (Some(*p), Some(*s as u64)),
    }
}

// Parse number in sql string, convert to Expr::Literal
fn parse_sql_number(n: &str) -> Result<Expr> {
    match n.parse::<i64>() {
        Ok(n) => Ok(lit(n)),
        Err(_) => Ok(lit(n.parse::<f64>().unwrap())),
    }
}

/// Work item for the iterative binary-operator evaluation in
/// [`SqlToRel::sql_expr_to_logical_expr`]. A binary-operator spine is flattened onto a stack of
/// these entries and evaluated in postfix order, so deeply chained operators (`a OR b OR c …`)
/// don't consume one native call frame per operator.
enum StackEntry {
    SQLExpr(Box<SQLExpr>),
    Operator(BinaryOperator),
}

/// Translate a sqlparser [`BinaryOperator`] into a DataFusion [`Operator`].
///
/// Shared by [`SqlToRel::parse_sql_binary_op`] and the iterative binary-operator
/// evaluation in [`SqlToRel::sql_expr_to_logical_expr`] so both stay in sync.
fn parse_sql_binary_operator(op: &BinaryOperator) -> Result<Operator> {
    match op {
        BinaryOperator::Gt => Ok(Operator::Gt),
        BinaryOperator::GtEq => Ok(Operator::GtEq),
        BinaryOperator::Lt => Ok(Operator::Lt),
        BinaryOperator::LtEq => Ok(Operator::LtEq),
        BinaryOperator::Eq => Ok(Operator::Eq),
        BinaryOperator::NotEq => Ok(Operator::NotEq),
        BinaryOperator::Plus => Ok(Operator::Plus),
        BinaryOperator::Minus => Ok(Operator::Minus),
        BinaryOperator::Multiply => Ok(Operator::Multiply),
        BinaryOperator::Divide => Ok(Operator::Divide),
        BinaryOperator::Modulo => Ok(Operator::Modulo),
        BinaryOperator::And => Ok(Operator::And),
        BinaryOperator::Or => Ok(Operator::Or),
        BinaryOperator::PGRegexMatch => Ok(Operator::RegexMatch),
        BinaryOperator::PGRegexIMatch => Ok(Operator::RegexIMatch),
        BinaryOperator::PGRegexNotMatch => Ok(Operator::RegexNotMatch),
        BinaryOperator::PGRegexNotIMatch => Ok(Operator::RegexNotIMatch),
        BinaryOperator::BitwiseAnd => Ok(Operator::BitwiseAnd),
        BinaryOperator::BitwiseOr => Ok(Operator::BitwiseOr),
        BinaryOperator::PGBitwiseShiftRight => Ok(Operator::BitwiseShiftRight),
        BinaryOperator::PGBitwiseShiftLeft => Ok(Operator::BitwiseShiftLeft),
        BinaryOperator::StringConcat => Ok(Operator::StringConcat),
        // TODO: PGExponentiation needs to be introduced, but DF doesn't pass dialect
        // so using BitwiseXor is safe for now since it's not implemented anyway
        BinaryOperator::BitwiseXor => Ok(Operator::Exponentiate),
        _ => Err(DataFusionError::NotImplemented(format!(
            "Unsupported SQL binary operator {:?}",
            op
        ))),
    }
}

fn parse_unicode_escaped_string(s: &str, delimiter: char) -> Result<Expr> {
    let mut result = String::new();
    let mut chars = s.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if c == delimiter {
            if let Some((_, next)) = chars.peek() {
                if next == &delimiter {
                    result.push(delimiter);
                    chars.next();
                } else {
                    let (parsed, len) =
                        parse_unicode_escaped_point(&s[i + 1..], delimiter)?;
                    result.push(parsed);
                    chars.nth(len - 1);
                }
            } else {
                return Err(invalid_unicode_escape_error(s, delimiter));
            }
        } else {
            result.push(c)
        }
    }
    Ok(lit(result))
}

fn parse_unicode_escaped_point(s: &str, delimiter: char) -> Result<(char, usize)> {
    let (point_start, point_end) = if s.starts_with('+') { (1, 7) } else { (0, 4) };
    if point_end <= s.len() {
        let byte = u32::from_str_radix(&s[point_start..point_end], 16)
            .map_err(|_| invalid_unicode_escape_error(s, delimiter))?;
        if let Some(c) = char::from_u32(byte) {
            Ok((c, point_end))
        } else {
            Err(invalid_unicode_escape_error(s, delimiter))
        }
    } else {
        Err(invalid_unicode_escape_error(s, delimiter))
    }
}

fn invalid_unicode_escape_error(s: &str, delimiter: char) -> DataFusionError {
    DataFusionError::SQL(ParserError(format!(
        "Invalid Unicode escape in {}. Unicode escapes must be {}XXXX or {}+XXXXXX",
        s, delimiter, delimiter,
    )))
}

#[cfg(test)]
mod tests {
    use crate::datasource::empty::EmptyTable;
    use crate::{assert_contains, logical_plan::create_udf, sql::parser::DFParser};
    use datafusion_expr::{ScalarFunctionImplementation, Volatility};

    use super::*;

    #[test]
    fn test_parse_unicode_escaped_string() {
        assert_eq!(
            parse_unicode_escaped_string("pppp", '\\').unwrap(),
            Expr::Literal(ScalarValue::Utf8(Some("pppp".to_string())))
        );
        assert_eq!(
            parse_unicode_escaped_string("d\\0061t\\+000061", '\\').unwrap(),
            Expr::Literal(ScalarValue::Utf8(Some("data".to_string())))
        );
        assert_eq!(
            parse_unicode_escaped_string("d\\0061\\\\t\\+000061", '\\').unwrap(),
            Expr::Literal(ScalarValue::Utf8(Some("da\\ta".to_string())))
        );
        assert_eq!(
            parse_unicode_escaped_string("d!0061t\\!+000061\\", '!').unwrap(),
            Expr::Literal(ScalarValue::Utf8(Some("dat\\a\\".to_string())))
        );
        assert_eq!(
            parse_unicode_escaped_string("!!d!0061!!t\\!+000061\\", '!').unwrap(),
            Expr::Literal(ScalarValue::Utf8(Some("!da!t\\a\\".to_string())))
        );
        assert_eq!(
            parse_unicode_escaped_string("d!0061t\\!+000061\\", '!').unwrap(),
            Expr::Literal(ScalarValue::Utf8(Some("dat\\a\\".to_string())))
        );
        assert!(parse_unicode_escaped_string("d\\0061t\\+000061\\", '\\').is_err());
        assert!(parse_unicode_escaped_string("d\\0061t\\+061", '\\').is_err());
        assert!(parse_unicode_escaped_string("d\\H061t\\+061", '\\').is_err());
    }
    #[test]
    fn select_no_relation() {
        quick_test(
            "SELECT 1",
            "Projection: Int64(1)\
             \n  EmptyRelation",
        );
    }

    #[test]
    fn test_real_f32() {
        quick_test(
            "SELECT CAST(1.1 AS REAL)",
            "Projection: CAST(Float64(1.1) AS Float32)\
             \n  EmptyRelation",
        );
    }

    #[test]
    fn test_int_decimal_default() {
        quick_test(
            "SELECT CAST(10 AS DECIMAL)",
            "Projection: CAST(Int64(10) AS Decimal(38, 10))\
             \n  EmptyRelation",
        );
    }

    #[test]
    fn select_column_does_not_exist() {
        let sql = "SELECT doesnotexist FROM person";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert!(matches!(
            err,
            DataFusionError::Plan(msg) if msg.contains("Invalid identifier '#doesnotexist' for schema "),
        ));
    }

    #[test]
    fn select_repeated_column() {
        // let sql = "SELECT age, age FROM person";
        // let err = logical_plan(sql).expect_err("query should have failed");
        // assert_eq!(
        //     r##"Plan("Projections require unique expression names but the expression \"#person.age\" at position 0 and \"#person.age\" at position 1 have the same name. Consider aliasing (\"AS\") one of them.")"##,
        //     format!("{:?}", err)
        // );

        // NOTE: this is supported with cubesql patches
        quick_test(
            "SELECT age, age FROM person",
            "Projection: #person.age, #person.age AS age__1\
            \n  TableScan: person projection=None",
        );
    }

    #[test]
    fn select_wildcard_with_repeated_column() {
        // let sql = "SELECT *, age FROM person";
        // let err = logical_plan(sql).expect_err("query should have failed");
        // assert_eq!(
        //     r##"Plan("Projections require unique expression names but the expression \"#person.age\" at position 3 and \"#person.age\" at position 8 have the same name. Consider aliasing (\"AS\") one of them.")"##,
        //     format!("{:?}", err)
        // );

        // NOTE: this is supported with cubesql patches
        quick_test(
            "SELECT *, age FROM person",
            "Projection: #person.id, #person.first_name, #person.last_name, #person.age, #person.state, #person.salary, #person.birth_date, #person.😀, #person.age AS age__1\
            \n  TableScan: person projection=None",
        );
    }

    #[test]
    fn select_wildcard_with_repeated_column_but_is_aliased() {
        quick_test(
            "SELECT *, first_name AS fn from person",
            "Projection: #person.id, #person.first_name, #person.last_name, #person.age, #person.state, #person.salary, #person.birth_date, #person.😀, #person.first_name AS fn\
            \n  TableScan: person projection=None",
        );
    }

    #[test]
    fn select_scalar_func_with_literal_no_relation() {
        quick_test(
            "SELECT sqrt(9)",
            "Projection: sqrt(Int64(9))\
             \n  EmptyRelation",
        );
    }

    #[test]
    fn select_simple_filter() {
        let sql = "SELECT id, first_name, last_name \
                   FROM person WHERE state = 'CO'";
        let expected = "Projection: #person.id, #person.first_name, #person.last_name\
                        \n  Filter: #person.state = Utf8(\"CO\")\
                        \n    TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_filter_column_does_not_exist() {
        let sql = "SELECT first_name FROM person WHERE doesnotexist = 'A'";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert!(matches!(
            err,
            DataFusionError::Plan(msg) if msg.contains("Invalid identifier '#doesnotexist' for schema "),
        ));
    }

    #[test]
    fn select_filter_cannot_use_alias() {
        let sql = "SELECT first_name AS x FROM person WHERE x = 'A'";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert!(matches!(
            err,
            DataFusionError::Plan(msg) if msg.contains("Invalid identifier '#x' for schema "),
        ));
    }

    #[test]
    fn select_neg_filter() {
        let sql = "SELECT id, first_name, last_name \
                   FROM person WHERE NOT state";
        let expected = "Projection: #person.id, #person.first_name, #person.last_name\
                        \n  Filter: NOT #person.state\
                        \n    TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_compound_filter() {
        let sql = "SELECT id, first_name, last_name \
                   FROM person WHERE state = 'CO' AND age >= 21 AND age <= 65";
        let expected = "Projection: #person.id, #person.first_name, #person.last_name\
            \n  Filter: #person.state = Utf8(\"CO\") AND #person.age >= Int64(21) AND #person.age <= Int64(65)\
            \n    TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn test_timestamp_filter() {
        let sql =
            "SELECT state FROM person WHERE birth_date < CAST (158412331400600000 as timestamp)";

        let expected = "Projection: #person.state\
            \n  Filter: #person.birth_date < CAST(Int64(158412331400600000) AS Timestamp(Nanosecond, None))\
            \n    TableScan: person projection=None";

        quick_test(sql, expected);
    }

    #[test]
    fn test_date_filter() {
        let sql =
            "SELECT state FROM person WHERE birth_date < CAST ('2020-01-01' as date)";

        let expected = "Projection: #person.state\
            \n  Filter: #person.birth_date < CAST(Utf8(\"2020-01-01\") AS Date32)\
            \n    TableScan: person projection=None";

        quick_test(sql, expected);
    }

    #[test]
    fn select_all_boolean_operators() {
        let sql = "SELECT age, first_name, last_name \
                   FROM person \
                   WHERE age = 21 \
                   AND age != 21 \
                   AND age > 21 \
                   AND age >= 21 \
                   AND age < 65 \
                   AND age <= 65";
        let expected = "Projection: #person.age, #person.first_name, #person.last_name\
                        \n  Filter: #person.age = Int64(21) \
                        AND #person.age != Int64(21) \
                        AND #person.age > Int64(21) \
                        AND #person.age >= Int64(21) \
                        AND #person.age < Int64(65) \
                        AND #person.age <= Int64(65)\
                        \n    TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_between() {
        let sql = "SELECT state FROM person WHERE age BETWEEN 21 AND 65";
        let expected = "Projection: #person.state\
            \n  Filter: #person.age BETWEEN Int64(21) AND Int64(65)\
            \n    TableScan: person projection=None";

        quick_test(sql, expected);
    }

    #[test]
    fn select_between_negated() {
        let sql = "SELECT state FROM person WHERE age NOT BETWEEN 21 AND 65";
        let expected = "Projection: #person.state\
            \n  Filter: #person.age NOT BETWEEN Int64(21) AND Int64(65)\
            \n    TableScan: person projection=None";

        quick_test(sql, expected);
    }

    #[test]
    fn select_nested() {
        let sql = "SELECT fn2, last_name
                   FROM (
                     SELECT fn1 as fn2, last_name, birth_date
                     FROM (
                       SELECT first_name AS fn1, last_name, birth_date, age
                       FROM person
                     ) AS a
                   ) AS b";
        let expected = "Projection: #b.fn2, #b.last_name\
                        \n  Projection: #b.fn2, #b.last_name, #b.birth_date, alias=b\
                        \n    Projection: #a.fn1 AS fn2, #a.last_name, #a.birth_date, alias=b\
                        \n      Projection: #a.fn1, #a.last_name, #a.birth_date, #a.age, alias=a\
                        \n        Projection: #person.first_name AS fn1, #person.last_name, #person.birth_date, #person.age, alias=a\
                        \n          TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_nested_with_filters() {
        let sql = "SELECT fn1, age
                   FROM (
                     SELECT first_name AS fn1, age
                     FROM person
                     WHERE age > 20
                   ) AS a
                   WHERE fn1 = 'X' AND age < 30";

        let expected = "Projection: #a.fn1, #a.age\
                        \n  Filter: #a.fn1 = Utf8(\"X\") AND #a.age < Int64(30)\
                        \n    Projection: #a.fn1, #a.age, alias=a\
                        \n      Projection: #person.first_name AS fn1, #person.age, alias=a\
                        \n        Filter: #person.age > Int64(20)\
                        \n          TableScan: person projection=None";

        quick_test(sql, expected);
    }

    #[test]
    fn select_nested_with_dotted_columns() {
        let sql = "SELECT \"a.b\"
                   FROM (
                       SELECT 1 \"a.b\"
                   ) AS t";
        let expected = "Projection: #t.a.b\
                      \n  Projection: #t.a.b, alias=t\
                      \n    Projection: Int64(1) AS a.b, alias=t\
                      \n      EmptyRelation";
        quick_test(sql, expected);
    }

    #[test]
    fn table_with_column_alias() {
        let sql = "SELECT a, b, c
                   FROM lineitem l (a, b, c)";
        let expected = "Projection: #l.a, #l.b, #l.c\
                        \n  Projection: #l.l_item_id AS a, #l.l_description AS b, #l.price AS c, alias=l\
                        \n    TableScan: l projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn table_with_column_alias_number_cols() {
        let sql = "SELECT a, b, c
                   FROM lineitem l (a, b)";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert_eq!(
            "Plan(\"Source table contains 3 columns but only 2 names given as column alias\")",
            format!("{:?}", err)
        );
    }

    #[test]
    fn select_with_having() {
        let sql = "SELECT id, age
                   FROM person
                   HAVING age > 100 AND age < 200";
        let expected = "Projection: #person.id, #person.age\
                        \n  Filter: #person.age > Int64(100) AND #person.age < Int64(200)\
                        \n    TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_with_having_referencing_column_not_in_select() {
        let sql = "SELECT id, age
                   FROM person
                   HAVING first_name = 'M'";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert_eq!(
            "Plan(\"Expression #person.first_name could not be resolved from available columns: #person.id, #person.age\")",
            format!("{:?}", err)
        );
    }

    #[test]
    fn select_with_having_referencing_column_nested_in_select_expression() {
        let sql = "SELECT id, age + 1
                   FROM person
                   HAVING age > 100";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert_eq!(
            "Plan(\"Expression #person.age could not be resolved from available columns: #person.id, #person.age + Int64(1)\")",
            format!("{:?}", err)
        );
    }

    #[test]
    fn select_with_having_with_aggregate_not_in_select() {
        let sql = "SELECT first_name
                   FROM person
                   HAVING MAX(age) > 100";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert_eq!(
            "Plan(\"Expression #person.first_name could not be resolved from available columns: #MAX(person.age)\")",
            format!("{:?}", err)
        );
    }

    #[test]
    fn select_aggregate_with_having_that_reuses_aggregate() {
        let sql = "SELECT MAX(age)
                   FROM person
                   HAVING MAX(age) < 30";
        let expected = "Projection: #MAX(person.age)\
                        \n  Filter: #MAX(person.age) < Int64(30)\
                        \n    Aggregate: groupBy=[[]], aggr=[[MAX(#person.age)]]\
                        \n      TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_aggregate_with_group_by_having_and_order_by_aggregate() {
        let sql = "SELECT first_name, MAX(age)
                   FROM person
                   GROUP BY first_name
                   HAVING MAX(age) < 30
                   ORDER BY MAX(age) DESC";
        let expected = "Sort: #MAX(person.age) DESC NULLS FIRST\
                        \n  Projection: #person.first_name, #MAX(person.age)\
                        \n    Filter: #MAX(person.age) < Int64(30)\
                        \n      Aggregate: groupBy=[[#person.first_name]], aggr=[[MAX(#person.age)]]\
                        \n        TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_aggregate_with_having_with_aggregate_not_in_select() {
        let sql = "SELECT MAX(age)
                   FROM person
                   HAVING MAX(first_name) > 'M'";
        let expected = "Projection: #MAX(person.age)\
                        \n  Filter: #MAX(person.first_name) > Utf8(\"M\")\
                        \n    Aggregate: groupBy=[[]], aggr=[[MAX(#person.age), MAX(#person.first_name)]]\
                        \n      TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_aggregate_with_having_referencing_column_not_in_select() {
        let sql = "SELECT COUNT(*)
                   FROM person
                   HAVING first_name = 'M'";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert_eq!(
            "Plan(\"Expression #person.first_name could not be resolved from available columns: #COUNT(UInt8(1))\")",
            format!("{:?}", err)
        );
    }

    #[test]
    fn select_aggregate_aliased_with_having_referencing_aggregate_by_its_alias() {
        let sql = "SELECT MAX(age) as max_age
                   FROM person
                   HAVING max_age < 30";
        // FIXME: add test for having in execution
        let expected = "Projection: #MAX(person.age) AS max_age\
                        \n  Filter: #MAX(person.age) < Int64(30)\
                        \n    Aggregate: groupBy=[[]], aggr=[[MAX(#person.age)]]\
                        \n      TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_aggregate_aliased_with_having_that_reuses_aggregate_but_not_by_its_alias() {
        let sql = "SELECT MAX(age) as max_age
                   FROM person
                   HAVING MAX(age) < 30";
        let expected = "Projection: #MAX(person.age) AS max_age\
                        \n  Filter: #MAX(person.age) < Int64(30)\
                        \n    Aggregate: groupBy=[[]], aggr=[[MAX(#person.age)]]\
                        \n      TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_aggregate_with_group_by_with_having() {
        let sql = "SELECT first_name, MAX(age)
                   FROM person
                   GROUP BY first_name
                   HAVING first_name = 'M'";
        let expected = "Projection: #person.first_name, #MAX(person.age)\
                        \n  Filter: #person.first_name = Utf8(\"M\")\
                        \n    Aggregate: groupBy=[[#person.first_name]], aggr=[[MAX(#person.age)]]\
                        \n      TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_aggregate_with_group_by_with_having_and_where() {
        let sql = "SELECT first_name, MAX(age)
                   FROM person
                   WHERE id > 5
                   GROUP BY first_name
                   HAVING MAX(age) < 100";
        let expected = "Projection: #person.first_name, #MAX(person.age)\
                        \n  Filter: #MAX(person.age) < Int64(100)\
                        \n    Aggregate: groupBy=[[#person.first_name]], aggr=[[MAX(#person.age)]]\
                        \n      Filter: #person.id > Int64(5)\
                        \n        TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_aggregate_with_group_by_with_having_and_where_filtering_on_aggregate_column(
    ) {
        let sql = "SELECT first_name, MAX(age)
                   FROM person
                   WHERE id > 5 AND age > 18
                   GROUP BY first_name
                   HAVING MAX(age) < 100";
        let expected = "Projection: #person.first_name, #MAX(person.age)\
                        \n  Filter: #MAX(person.age) < Int64(100)\
                        \n    Aggregate: groupBy=[[#person.first_name]], aggr=[[MAX(#person.age)]]\
                        \n      Filter: #person.id > Int64(5) AND #person.age > Int64(18)\
                        \n        TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_aggregate_with_group_by_with_having_using_column_by_alias() {
        let sql = "SELECT first_name AS fn, MAX(age)
                   FROM person
                   GROUP BY first_name
                   HAVING MAX(age) > 2 AND fn = 'M'";
        let expected = "Projection: #person.first_name AS fn, #MAX(person.age)\
                        \n  Filter: #MAX(person.age) > Int64(2) AND #person.first_name = Utf8(\"M\")\
                        \n    Aggregate: groupBy=[[#person.first_name]], aggr=[[MAX(#person.age)]]\
                        \n      TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_aggregate_with_group_by_with_having_using_columns_with_and_without_their_aliases(
    ) {
        let sql = "SELECT first_name AS fn, MAX(age) AS max_age
                   FROM person
                   GROUP BY first_name
                   HAVING MAX(age) > 2 AND max_age < 5 AND first_name = 'M' AND fn = 'N'";
        let expected = "Projection: #person.first_name AS fn, #MAX(person.age) AS max_age\
                        \n  Filter: #MAX(person.age) > Int64(2) AND #MAX(person.age) < Int64(5) AND #person.first_name = Utf8(\"M\") AND #person.first_name = Utf8(\"N\")\
                        \n    Aggregate: groupBy=[[#person.first_name]], aggr=[[MAX(#person.age)]]\
                        \n      TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_aggregate_with_group_by_with_having_that_reuses_aggregate() {
        let sql = "SELECT first_name, MAX(age)
                   FROM person
                   GROUP BY first_name
                   HAVING MAX(age) > 100";
        let expected = "Projection: #person.first_name, #MAX(person.age)\
                        \n  Filter: #MAX(person.age) > Int64(100)\
                        \n    Aggregate: groupBy=[[#person.first_name]], aggr=[[MAX(#person.age)]]\
                        \n      TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_aggregate_with_group_by_with_having_referencing_column_not_in_group_by() {
        let sql = "SELECT first_name, MAX(age)
                   FROM person
                   GROUP BY first_name
                   HAVING MAX(age) > 10 AND last_name = 'M'";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert_eq!(
            "Plan(\"Expression #person.last_name could not be resolved from available columns: #person.first_name, #MAX(person.age)\")",
            format!("{:?}", err)
        );
    }

    #[test]
    fn select_aggregate_with_group_by_with_having_that_reuses_aggregate_multiple_times() {
        let sql = "SELECT first_name, MAX(age)
                   FROM person
                   GROUP BY first_name
                   HAVING MAX(age) > 100 AND MAX(age) < 200";
        let expected = "Projection: #person.first_name, #MAX(person.age)\
                        \n  Filter: #MAX(person.age) > Int64(100) AND #MAX(person.age) < Int64(200)\
                        \n    Aggregate: groupBy=[[#person.first_name]], aggr=[[MAX(#person.age)]]\
                        \n      TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_aggregate_with_group_by_with_having_using_aggreagate_not_in_select() {
        let sql = "SELECT first_name, MAX(age)
                   FROM person
                   GROUP BY first_name
                   HAVING MAX(age) > 100 AND MIN(id) < 50";
        let expected = "Projection: #person.first_name, #MAX(person.age)\
                        \n  Filter: #MAX(person.age) > Int64(100) AND #MIN(person.id) < Int64(50)\
                        \n    Aggregate: groupBy=[[#person.first_name]], aggr=[[MAX(#person.age), MIN(#person.id)]]\
                        \n      TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_aggregate_aliased_with_group_by_with_having_referencing_aggregate_by_its_alias(
    ) {
        let sql = "SELECT first_name, MAX(age) AS max_age
                   FROM person
                   GROUP BY first_name
                   HAVING max_age > 100";
        let expected = "Projection: #person.first_name, #MAX(person.age) AS max_age\
                        \n  Filter: #MAX(person.age) > Int64(100)\
                        \n    Aggregate: groupBy=[[#person.first_name]], aggr=[[MAX(#person.age)]]\
                        \n      TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_aggregate_compound_aliased_with_group_by_with_having_referencing_compound_aggregate_by_its_alias(
    ) {
        let sql = "SELECT first_name, MAX(age) + 1 AS max_age_plus_one
                   FROM person
                   GROUP BY first_name
                   HAVING max_age_plus_one > 100";
        let expected =
            "Projection: #person.first_name, #MAX(person.age) + Int64(1) AS max_age_plus_one\
                        \n  Filter: #MAX(person.age) + Int64(1) > Int64(100)\
                        \n    Aggregate: groupBy=[[#person.first_name]], aggr=[[MAX(#person.age)]]\
                        \n      TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_aggregate_with_group_by_with_having_using_derived_column_aggreagate_not_in_select(
    ) {
        let sql = "SELECT first_name, MAX(age)
                   FROM person
                   GROUP BY first_name
                   HAVING MAX(age) > 100 AND MIN(id - 2) < 50";
        let expected = "Projection: #person.first_name, #MAX(person.age)\
                        \n  Filter: #MAX(person.age) > Int64(100) AND #MIN(person.id - Int64(2)) < Int64(50)\
                        \n    Aggregate: groupBy=[[#person.first_name]], aggr=[[MAX(#person.age), MIN(#person.id - Int64(2))]]\
                        \n      TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_aggregate_with_group_by_with_having_using_count_star_not_in_select() {
        let sql = "SELECT first_name, MAX(age)
                   FROM person
                   GROUP BY first_name
                   HAVING MAX(age) > 100 AND COUNT(*) < 50";
        let expected = "Projection: #person.first_name, #MAX(person.age)\
                        \n  Filter: #MAX(person.age) > Int64(100) AND #COUNT(UInt8(1)) < Int64(50)\
                        \n    Aggregate: groupBy=[[#person.first_name]], aggr=[[MAX(#person.age), COUNT(UInt8(1))]]\
                        \n      TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_aggregate_with_group_by_with_two_same_exprs() {
        let sql = "SELECT first_name as name, first_name as nickname, 1 as first_num, 1 as second_num
                   FROM person
                   GROUP BY 1, 2, 3, 4";
        let expected = "Projection: #person.first_name AS name, #person.first_name AS nickname, #Int64(1) AS first_num, #Int64(1) AS second_num\
                        \n  Aggregate: groupBy=[[#person.first_name, Int64(1)]], aggr=[[]]\
                        \n    TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_binary_expr() {
        let sql = "SELECT age + salary from person";
        let expected = "Projection: #person.age + #person.salary\
                        \n  TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_binary_expr_nested() {
        let sql = "SELECT (age + salary)/2 from person";
        let expected = "Projection: #person.age + #person.salary / Int64(2)\
                        \n  TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_wildcard_with_groupby() {
        quick_test(
            r#"SELECT * FROM person GROUP BY id, first_name, last_name, age, state, salary, birth_date, "😀""#,
            "Projection: #person.id, #person.first_name, #person.last_name, #person.age, #person.state, #person.salary, #person.birth_date, #person.😀\
             \n  Aggregate: groupBy=[[#person.id, #person.first_name, #person.last_name, #person.age, #person.state, #person.salary, #person.birth_date, #person.😀]], aggr=[[]]\
             \n    TableScan: person projection=None",
        );
        quick_test(
            "SELECT * FROM (SELECT first_name, last_name FROM person) AS a GROUP BY first_name, last_name",
            "Projection: #a.first_name, #a.last_name\
             \n  Aggregate: groupBy=[[#a.first_name, #a.last_name]], aggr=[[]]\
             \n    Projection: #a.first_name, #a.last_name, alias=a\
             \n      Projection: #person.first_name, #person.last_name, alias=a\
             \n        TableScan: person projection=None",
        );
    }

    #[test]
    fn select_simple_aggregate() {
        quick_test(
            "SELECT MIN(age) FROM person",
            "Projection: #MIN(person.age)\
            \n  Aggregate: groupBy=[[]], aggr=[[MIN(#person.age)]]\
            \n    TableScan: person projection=None",
        );
    }

    #[test]
    fn test_sum_aggregate() {
        quick_test(
            "SELECT SUM(age) from person",
            "Projection: #SUM(person.age)\
            \n  Aggregate: groupBy=[[]], aggr=[[SUM(#person.age)]]\
            \n    TableScan: person projection=None",
        );
    }

    #[test]
    fn select_simple_aggregate_column_does_not_exist() {
        let sql = "SELECT MIN(doesnotexist) FROM person";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert!(matches!(
            err,
            DataFusionError::Plan(msg) if msg.contains("Invalid identifier '#doesnotexist' for schema "),
        ));
    }

    #[test]
    fn select_simple_aggregate_repeated_aggregate() {
        // let sql = "SELECT MIN(age), MIN(age) FROM person";
        // let err = logical_plan(sql).expect_err("query should have failed");
        // assert_eq!(
        //     r##"Plan("Projections require unique expression names but the expression \"MIN(#person.age)\" at position 0 and \"MIN(#person.age)\" at position 1 have the same name. Consider aliasing (\"AS\") one of them.")"##,
        //     format!("{:?}", err)
        // );

        // NOTE: this is supported with cubesql patches
        quick_test(
            "SELECT MIN(age), MIN(age) FROM person",
            "Projection: #MIN(person.age), #MIN(person.age) AS MIN(person.age)__1\
             \n  Aggregate: groupBy=[[]], aggr=[[MIN(#person.age)]]\
             \n    TableScan: person projection=None",
        );
    }

    #[test]
    fn select_simple_aggregate_repeated_aggregate_with_single_alias() {
        quick_test(
            "SELECT MIN(age), MIN(age) AS a FROM person",
            "Projection: #MIN(person.age), #MIN(person.age) AS a\
             \n  Aggregate: groupBy=[[]], aggr=[[MIN(#person.age)]]\
             \n    TableScan: person projection=None",
        );
    }

    #[test]
    fn select_simple_aggregate_repeated_aggregate_with_unique_aliases() {
        quick_test(
            "SELECT MIN(age) AS a, MIN(age) AS b FROM person",
            "Projection: #MIN(person.age) AS a, #MIN(person.age) AS b\
             \n  Aggregate: groupBy=[[]], aggr=[[MIN(#person.age)]]\
             \n    TableScan: person projection=None",
        );
    }

    #[test]
    fn select_simple_aggregate_repeated_aggregate_with_repeated_aliases() {
        // let sql = "SELECT MIN(age) AS a, MIN(age) AS a FROM person";
        // let err = logical_plan(sql).expect_err("query should have failed");
        // assert_eq!(
        //     r##"Plan("Projections require unique expression names but the expression \"MIN(#person.age) AS a\" at position 0 and \"MIN(#person.age) AS a\" at position 1 have the same name. Consider aliasing (\"AS\") one of them.")"##,
        //     format!("{:?}", err)
        // );

        // NOTE: this is supported with cubesql patches
        quick_test(
            "SELECT MIN(age) AS a, MIN(age) AS a FROM person",
            "Projection: #MIN(person.age) AS a, #MIN(person.age) AS a__1\
             \n  Aggregate: groupBy=[[]], aggr=[[MIN(#person.age)]]\
             \n    TableScan: person projection=None",
        );
    }

    #[test]
    fn select_simple_aggregate_with_groupby() {
        quick_test(
            "SELECT state, MIN(age), MAX(age) FROM person GROUP BY state",
            "Projection: #person.state, #MIN(person.age), #MAX(person.age)\
            \n  Aggregate: groupBy=[[#person.state]], aggr=[[MIN(#person.age), MAX(#person.age)]]\
            \n    TableScan: person projection=None",
        );
    }

    #[test]
    fn select_simple_aggregate_with_groupby_with_aliases() {
        quick_test(
            "SELECT state AS a, MIN(age) AS b FROM person GROUP BY state",
            "Projection: #person.state AS a, #MIN(person.age) AS b\
             \n  Aggregate: groupBy=[[#person.state]], aggr=[[MIN(#person.age)]]\
             \n    TableScan: person projection=None",
        );
    }

    #[test]
    fn select_simple_aggregate_with_groupby_with_aliases_repeated() {
        // let sql = "SELECT state AS a, MIN(age) AS a FROM person GROUP BY state";
        // let err = logical_plan(sql).expect_err("query should have failed");
        // assert_eq!(
        //     r##"Plan("Projections require unique expression names but the expression \"#person.state AS a\" at position 0 and \"MIN(#person.age) AS a\" at position 1 have the same name. Consider aliasing (\"AS\") one of them.")"##,
        //     format!("{:?}", err)
        // );

        // NOTE: this is supported with cubesql patches
        quick_test(
            "SELECT state AS a, MIN(age) AS a FROM person GROUP BY state",
            "Projection: #person.state AS a, #MIN(person.age) AS a__1\
             \n  Aggregate: groupBy=[[#person.state]], aggr=[[MIN(#person.age)]]\
             \n    TableScan: person projection=None",
        );
    }

    #[test]
    fn select_simple_aggregate_with_groupby_column_unselected() {
        quick_test(
            "SELECT MIN(age), MAX(age) FROM person GROUP BY state",
            "Projection: #MIN(person.age), #MAX(person.age)\
             \n  Aggregate: groupBy=[[#person.state]], aggr=[[MIN(#person.age), MAX(#person.age)]]\
             \n    TableScan: person projection=None",
        );
    }

    #[test]
    fn select_simple_aggregate_with_groupby_and_column_in_group_by_does_not_exist() {
        let sql = "SELECT SUM(age) FROM person GROUP BY doesnotexist";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert!(matches!(
            err,
            DataFusionError::Plan(msg) if msg.contains("Column #doesnotexist not found in provided schemas"),
        ));
    }

    #[test]
    fn select_simple_aggregate_with_groupby_and_column_in_aggregate_does_not_exist() {
        let sql = "SELECT SUM(doesnotexist) FROM person GROUP BY first_name";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert!(matches!(
            err,
            DataFusionError::Plan(msg) if msg.contains("Invalid identifier '#doesnotexist' for schema "),
        ));
    }

    #[test]
    fn select_interval_out_of_range() {
        let sql = "SELECT INTERVAL '100000000000000000 day'";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert_eq!(
            r#"NotImplemented("Interval field value out of range: \"100000000000000000 day\"")"#,
            format!("{:?}", err)
        );
    }

    #[test]
    fn select_compound_interval_out_of_range() {
        // First test single interval, it should be parseable
        quick_test(
            "SELECT INTERVAL '2000000000 day'",
            "Projection: IntervalDayTime(\"8589934592000000000\")\n  EmptyRelation",
        );

        // Then double that interval, so each separate part is parsable, but their sum is not
        let sql = "SELECT INTERVAL '2000000000 day 2000000000 day'";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert_eq!(
            r#"NotImplemented("Interval field value out of range: \"2000000000 day 2000000000 day\"")"#,
            format!("{:?}", err)
        );
    }

    #[test]
    fn select_neg_interval_out_of_range() {
        let sql = "SELECT INTERVAL '-100000000000000000 day'";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert_eq!(
            r#"NotImplemented("Interval field value out of range: \"-100000000000000000 day\"")"#,
            format!("{:?}", err)
        );
    }

    #[test]
    fn select_neg_compound_interval_out_of_range() {
        // First test single interval, it should be parseable
        quick_test(
            "SELECT INTERVAL '-2000000000 day'",
            "Projection: IntervalDayTime(\"-8589934592000000000\")\n  EmptyRelation",
        );

        // Then double that interval, so each separate part is parsable, but their sum is not
        let sql = "SELECT INTERVAL '-2000000000 day -2000000000 day'";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert_eq!(
            r#"NotImplemented("Interval field value out of range: \"-2000000000 day -2000000000 day\"")"#,
            format!("{:?}", err)
        );
    }

    #[test]
    fn select_array_no_common_type() {
        let sql = "SELECT [1, true, null]";
        let err = logical_plan(sql).expect_err("query should have failed");

        // HashSet doesnt guaranty order
        assert_contains!(
            format!("{:?}", err),
            r#"NotImplemented("Arrays with different types are not supported: "#
        );
    }

    #[test]
    fn select_array_non_literal_type() {
        let sql = "SELECT [now()]";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert_eq!(
            r#"NotImplemented("Arrays with elements other than literal are not supported: now()")"#,
            format!("{:?}", err)
        );
    }

    #[test]
    fn select_simple_aggregate_with_groupby_and_column_is_in_aggregate_and_groupby() {
        quick_test(
            "SELECT MAX(first_name) FROM person GROUP BY first_name",
            "Projection: #MAX(person.first_name)\
             \n  Aggregate: groupBy=[[#person.first_name]], aggr=[[MAX(#person.first_name)]]\
             \n    TableScan: person projection=None",
        );
    }

    #[test]
    fn select_simple_aggregate_with_groupby_can_use_positions() {
        quick_test(
            "SELECT state, age AS b, COUNT(1) FROM person GROUP BY 1, 2",
            "Projection: #person.state, #person.age AS b, #COUNT(UInt8(1))\
             \n  Aggregate: groupBy=[[#person.state, #person.age]], aggr=[[COUNT(UInt8(1))]]\
             \n    TableScan: person projection=None",
        );
        quick_test(
            "SELECT state, age AS b, COUNT(1) FROM person GROUP BY 2, 1",
            "Projection: #person.state, #person.age AS b, #COUNT(UInt8(1))\
             \n  Aggregate: groupBy=[[#person.age, #person.state]], aggr=[[COUNT(UInt8(1))]]\
             \n    TableScan: person projection=None",
        );
    }

    #[test]
    fn select_simple_aggregate_with_groupby_position_out_of_range() {
        let sql = "SELECT state, MIN(age) FROM person GROUP BY 0";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert_eq!(
            "Plan(\"Expression #person.state could not be resolved from available columns: #Int64(0), #MIN(person.age)\")",
            format!("{:?}", err)
        );

        let sql2 = "SELECT state, MIN(age) FROM person GROUP BY 5";
        let err2 = logical_plan(sql2).expect_err("query should have failed");
        assert_eq!(
            "Plan(\"Expression #person.state could not be resolved from available columns: #Int64(5), #MIN(person.age)\")",
            format!("{:?}", err2)
        );
    }

    #[test]
    fn select_simple_aggregate_with_groupby_can_use_alias() {
        quick_test(
            "SELECT state AS a, MIN(age) AS b FROM person GROUP BY a",
            "Projection: #person.state AS a, #MIN(person.age) AS b\
             \n  Aggregate: groupBy=[[#person.state]], aggr=[[MIN(#person.age)]]\
             \n    TableScan: person projection=None",
        );
    }

    #[test]
    fn select_simple_aggregate_with_groupby_aggregate_repeated() {
        // let sql = "SELECT state, MIN(age), MIN(age) FROM person GROUP BY state";
        // let err = logical_plan(sql).expect_err("query should have failed");
        // assert_eq!(
        //     r##"Plan("Projections require unique expression names but the expression \"MIN(#person.age)\" at position 1 and \"MIN(#person.age)\" at position 2 have the same name. Consider aliasing (\"AS\") one of them.")"##,
        //     format!("{:?}", err)
        // );

        // NOTE: this is supported with cubesql patches
        quick_test(
            "SELECT state, MIN(age), MIN(age) FROM person GROUP BY state",
            "Projection: #person.state, #MIN(person.age), #MIN(person.age) AS MIN(person.age)__1\
             \n  Aggregate: groupBy=[[#person.state]], aggr=[[MIN(#person.age)]]\
             \n    TableScan: person projection=None",
        )
    }

    #[test]
    fn select_simple_aggregate_with_groupby_aggregate_repeated_and_one_has_alias() {
        quick_test(
            "SELECT state, MIN(age), MIN(age) AS ma FROM person GROUP BY state",
            "Projection: #person.state, #MIN(person.age), #MIN(person.age) AS ma\
             \n  Aggregate: groupBy=[[#person.state]], aggr=[[MIN(#person.age)]]\
             \n    TableScan: person projection=None",
        )
    }

    #[test]
    fn select_simple_aggregate_with_groupby_non_column_expression_unselected() {
        quick_test(
            "SELECT MIN(first_name) FROM person GROUP BY age + 1",
            "Projection: #MIN(person.first_name)\
             \n  Aggregate: groupBy=[[#person.age + Int64(1)]], aggr=[[MIN(#person.first_name)]]\
             \n    TableScan: person projection=None",
        );
    }

    #[test]
    fn select_simple_aggregate_with_groupby_non_column_expression_selected_and_resolvable(
    ) {
        quick_test(
            "SELECT age + 1, MIN(first_name) FROM person GROUP BY age + 1",
            "Projection: #person.age + Int64(1), #MIN(person.first_name)\
             \n  Aggregate: groupBy=[[#person.age + Int64(1)]], aggr=[[MIN(#person.first_name)]]\
             \n    TableScan: person projection=None",
        );
        quick_test(
            "SELECT MIN(first_name), age + 1 FROM person GROUP BY age + 1",
            "Projection: #MIN(person.first_name), #person.age + Int64(1)\
             \n  Aggregate: groupBy=[[#person.age + Int64(1)]], aggr=[[MIN(#person.first_name)]]\
             \n    TableScan: person projection=None",
        );
    }

    #[test]
    fn select_simple_aggregate_with_groupby_non_column_expression_nested_and_resolvable()
    {
        quick_test(
            "SELECT ((age + 1) / 2) * (age + 1), MIN(first_name) FROM person GROUP BY age + 1",
            "Projection: #person.age + Int64(1) / Int64(2) * #person.age + Int64(1), #MIN(person.first_name)\
             \n  Aggregate: groupBy=[[#person.age + Int64(1)]], aggr=[[MIN(#person.first_name)]]\
             \n    TableScan: person projection=None",
        );
    }

    #[test]
    fn select_simple_aggregate_with_groupby_non_column_expression_nested_and_not_resolvable(
    ) {
        // The query should fail, because age + 9 is not in the group by.
        let sql =
            "SELECT ((age + 1) / 2) * (age + 9), MIN(first_name) FROM person GROUP BY age + 1";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert_eq!(
            r#"Plan("Expression #person.age could not be resolved from available columns: #person.age + Int64(1), #MIN(person.first_name)")"#,
            format!("{:?}", err)
        );
    }

    #[test]
    fn select_simple_aggregate_with_groupby_non_column_expression_and_its_column_selected(
    ) {
        let sql = "SELECT age, MIN(first_name) FROM person GROUP BY age + 1";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert_eq!(
            r#"Plan("Expression #person.age could not be resolved from available columns: #person.age + Int64(1), #MIN(person.first_name)")"#,
            format!("{:?}", err)
        );
    }

    #[test]
    fn select_simple_aggregate_nested_in_binary_expr_with_groupby() {
        quick_test(
            "SELECT state, MIN(age) < 10 FROM person GROUP BY state",
            "Projection: #person.state, #MIN(person.age) < Int64(10)\
             \n  Aggregate: groupBy=[[#person.state]], aggr=[[MIN(#person.age)]]\
             \n    TableScan: person projection=None",
        );
    }

    #[test]
    fn select_simple_aggregate_and_nested_groupby_column() {
        quick_test(
            "SELECT age + 1, MAX(first_name) FROM person GROUP BY age",
            "Projection: #person.age + Int64(1), #MAX(person.first_name)\
             \n  Aggregate: groupBy=[[#person.age]], aggr=[[MAX(#person.first_name)]]\
             \n    TableScan: person projection=None",
        );
    }

    #[test]
    fn select_aggregate_compounded_with_groupby_column() {
        quick_test(
            "SELECT age + MIN(salary) FROM person GROUP BY age",
            "Projection: #person.age + #MIN(person.salary)\
             \n  Aggregate: groupBy=[[#person.age]], aggr=[[MIN(#person.salary)]]\
             \n    TableScan: person projection=None",
        );
    }

    #[test]
    fn select_aggregate_with_non_column_inner_expression_with_groupby() {
        quick_test(
            "SELECT state, MIN(age + 1) FROM person GROUP BY state",
            "Projection: #person.state, #MIN(person.age + Int64(1))\
            \n  Aggregate: groupBy=[[#person.state]], aggr=[[MIN(#person.age + Int64(1))]]\
            \n    TableScan: person projection=None",
        );
    }

    #[test]
    fn test_wildcard() {
        quick_test(
            "SELECT * from person",
            "Projection: #person.id, #person.first_name, #person.last_name, #person.age, #person.state, #person.salary, #person.birth_date, #person.😀\
            \n  TableScan: person projection=None",
        );
    }

    #[test]
    fn select_count_one() {
        let sql = "SELECT COUNT(1) FROM person";
        let expected = "Projection: #COUNT(UInt8(1))\
                        \n  Aggregate: groupBy=[[]], aggr=[[COUNT(UInt8(1))]]\
                        \n    TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_count_column() {
        let sql = "SELECT COUNT(id) FROM person";
        let expected = "Projection: #COUNT(person.id)\
                        \n  Aggregate: groupBy=[[]], aggr=[[COUNT(#person.id)]]\
                        \n    TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_approx_median() {
        let sql = "SELECT approx_median(age) FROM person";
        let expected = "Projection: #APPROXPERCENTILECONT(person.age,Float64(0.5))\
                        \n  Aggregate: groupBy=[[]], aggr=[[APPROXPERCENTILECONT(#person.age, Float64(0.5))]]\
                        \n    TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_percentile_cont() {
        let sql = "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY age) FROM person";
        let expected = "Projection: #PERCENTILECONT(Float64(0.5)) WITHIN GROUP (ORDER BY [#person.age ASC NULLS LAST])\
                        \n  Aggregate: groupBy=[[]], aggr=[[PERCENTILECONT(Float64(0.5)) WITHIN GROUP (ORDER BY #person.age ASC NULLS LAST)]]\
                        \n    TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_scalar_func() {
        let sql = "SELECT sqrt(age) FROM person";
        let expected = "Projection: sqrt(#person.age)\
                        \n  TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_aliased_scalar_func() {
        let sql = "SELECT sqrt(person.age) AS square_people FROM person";
        let expected = "Projection: sqrt(#person.age) AS square_people\
                        \n  TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_where_nullif_division() {
        let sql = "SELECT c3/(c4+c5) \
                   FROM aggregate_test_100 WHERE c3/nullif(c4+c5, 0) > 0.1";
        let expected = "Projection: #aggregate_test_100.c3 / #aggregate_test_100.c4 + #aggregate_test_100.c5\
            \n  Filter: #aggregate_test_100.c3 / nullif(#aggregate_test_100.c4 + #aggregate_test_100.c5, Int64(0)) > Float64(0.1)\
            \n    TableScan: aggregate_test_100 projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_where_with_negative_operator() {
        let sql = "SELECT c3 FROM aggregate_test_100 WHERE c3 > -0.1 AND -c4 > 0";
        let expected = "Projection: #aggregate_test_100.c3\
            \n  Filter: #aggregate_test_100.c3 > Float64(-0.1) AND (- #aggregate_test_100.c4) > Int64(0)\
            \n    TableScan: aggregate_test_100 projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_where_with_positive_operator() {
        let sql = "SELECT c3 FROM aggregate_test_100 WHERE c3 > +0.1 AND +c4 > 0";
        let expected = "Projection: #aggregate_test_100.c3\
            \n  Filter: #aggregate_test_100.c3 > Float64(0.1) AND #aggregate_test_100.c4 > Int64(0)\
            \n    TableScan: aggregate_test_100 projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_order_by_index() {
        let sql = "SELECT id FROM person ORDER BY 1";
        let expected = "Sort: #person.id ASC NULLS LAST\
                        \n  Projection: #person.id\
                        \n    TableScan: person projection=None";

        quick_test(sql, expected);
    }

    #[test]
    fn select_order_by_multiple_index() {
        let sql = "SELECT id, state, age FROM person ORDER BY 1, 3";
        let expected = "Sort: #person.id ASC NULLS LAST, #person.age ASC NULLS LAST\
                        \n  Projection: #person.id, #person.state, #person.age\
                        \n    TableScan: person projection=None";

        quick_test(sql, expected);
    }

    #[test]
    fn select_order_by_index_of_0() {
        let sql = "SELECT id FROM person ORDER BY 0";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert_eq!(
            "Plan(\"Order by index starts at 1 for column indexes\")",
            format!("{:?}", err)
        );
    }

    #[test]
    fn select_order_by_index_oob() {
        let sql = "SELECT id FROM person ORDER BY 2";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert_eq!(
            "Plan(\"Order by column out of bounds, specified: 2, max: 1\")",
            format!("{:?}", err)
        );
    }

    #[test]
    fn select_order_by() {
        let sql = "SELECT id FROM person ORDER BY id";
        let expected = "Sort: #person.id ASC NULLS LAST\
                        \n  Projection: #person.id\
                        \n    TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_order_by_desc() {
        let sql = "SELECT id FROM person ORDER BY id DESC";
        let expected = "Sort: #person.id DESC NULLS FIRST\
                        \n  Projection: #person.id\
                        \n    TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_order_by_nulls_last() {
        quick_test(
            "SELECT id FROM person ORDER BY id DESC NULLS LAST",
            "Sort: #person.id DESC NULLS LAST\
            \n  Projection: #person.id\
            \n    TableScan: person projection=None",
        );

        quick_test(
            "SELECT id FROM person ORDER BY id NULLS LAST",
            "Sort: #person.id ASC NULLS LAST\
            \n  Projection: #person.id\
            \n    TableScan: person projection=None",
        );
    }

    #[test]
    fn select_order_by_missing_aggr() {
        let sql = "SELECT state FROM person GROUP BY state ORDER BY SUM(age)";
        let expected = "Projection: #person.state\
                        \n  Sort: #SUM(person.age) ASC NULLS LAST\
                        \n    Projection: #person.state, #SUM(person.age)\
                        \n      Aggregate: groupBy=[[#person.state]], aggr=[[SUM(#person.age)]]\
                        \n        TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_order_by_ungrouped_column() {
        let sql = "SELECT state FROM person GROUP BY state ORDER BY age";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert!(matches!(
            err,
            DataFusionError::Plan(msg) if msg.contains(
                "column \"person.age\" must appear in the GROUP BY clause or be used in an aggregate function"
            ),
        ));
    }

    #[test]
    fn select_order_by_case_over_ungrouped_column() {
        // ORDER BY expression references a column consumed by the Aggregate, even
        // though its CASE conditions match the GROUP BY expression
        let sql = "SELECT CASE WHEN age > 30 THEN 'old' ELSE 'young' END, COUNT(*) \
            FROM person \
            GROUP BY 1 \
            ORDER BY CASE WHEN age > 30 THEN 1 ELSE 2 END";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert!(matches!(
            err,
            DataFusionError::Plan(msg) if msg.contains(
                "column \"person.age\" must appear in the GROUP BY clause or be used in an aggregate function"
            ),
        ));
    }

    #[test]
    fn select_order_by_ungrouped_column_no_group_by() {
        // Implicit aggregation without GROUP BY: ORDER BY cannot reference a raw column
        let sql = "SELECT COUNT(*) FROM person ORDER BY age";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert!(matches!(
            err,
            DataFusionError::Plan(msg) if msg.contains(
                "column \"person.age\" must appear in the GROUP BY clause or be used in an aggregate function"
            ),
        ));
    }

    #[test]
    fn select_group_by() {
        let sql = "SELECT state FROM person GROUP BY state";
        let expected = "Projection: #person.state\
                        \n  Aggregate: groupBy=[[#person.state]], aggr=[[]]\
                        \n    TableScan: person projection=None";

        quick_test(sql, expected);
    }

    #[test]
    fn select_group_by_columns_not_in_select() {
        let sql = "SELECT MAX(age) FROM person GROUP BY state";
        let expected = "Projection: #MAX(person.age)\
                        \n  Aggregate: groupBy=[[#person.state]], aggr=[[MAX(#person.age)]]\
                        \n    TableScan: person projection=None";

        quick_test(sql, expected);
    }

    #[test]
    fn select_group_by_count_star() {
        let sql = "SELECT state, COUNT(*) FROM person GROUP BY state";
        let expected = "Projection: #person.state, #COUNT(UInt8(1))\
                        \n  Aggregate: groupBy=[[#person.state]], aggr=[[COUNT(UInt8(1))]]\
                        \n    TableScan: person projection=None";

        quick_test(sql, expected);
    }

    #[test]
    fn select_group_by_needs_projection() {
        let sql = "SELECT COUNT(state), state FROM person GROUP BY state";
        let expected = "\
        Projection: #COUNT(person.state), #person.state\
        \n  Aggregate: groupBy=[[#person.state]], aggr=[[COUNT(#person.state)]]\
        \n    TableScan: person projection=None";

        quick_test(sql, expected);
    }

    #[test]
    fn select_7480_1() {
        let sql = "SELECT c1, MIN(c12) FROM aggregate_test_100 GROUP BY c1, c13";
        let expected = "Projection: #aggregate_test_100.c1, #MIN(aggregate_test_100.c12)\
                       \n  Aggregate: groupBy=[[#aggregate_test_100.c1, #aggregate_test_100.c13]], aggr=[[MIN(#aggregate_test_100.c12)]]\
                       \n    TableScan: aggregate_test_100 projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_7480_2() {
        let sql = "SELECT c1, c13, MIN(c12) FROM aggregate_test_100 GROUP BY c1";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert_eq!(
            "Plan(\"Expression #aggregate_test_100.c13 could not be resolved from available columns: #aggregate_test_100.c1, #MIN(aggregate_test_100.c12)\")",
            format!("{:?}", err)
        );
    }

    #[test]
    fn create_external_table_csv() {
        let sql = "CREATE EXTERNAL TABLE t(c1 int) STORED AS CSV LOCATION 'foo.csv'";
        let expected = "CreateExternalTable: \"t\"";
        quick_test(sql, expected);
    }

    #[test]
    fn create_external_table_csv_no_schema() {
        let sql = "CREATE EXTERNAL TABLE t STORED AS CSV LOCATION 'foo.csv'";
        let expected = "CreateExternalTable: \"t\"";
        quick_test(sql, expected);
    }

    #[test]
    fn create_external_table_parquet() {
        let sql =
            "CREATE EXTERNAL TABLE t(c1 int) STORED AS PARQUET LOCATION 'foo.parquet'";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert_eq!(
            "Plan(\"Column definitions can not be specified for PARQUET files.\")",
            format!("{:?}", err)
        );
    }

    #[test]
    fn create_external_table_parquet_no_schema() {
        let sql = "CREATE EXTERNAL TABLE t STORED AS PARQUET LOCATION 'foo.parquet'";
        let expected = "CreateExternalTable: \"t\"";
        quick_test(sql, expected);
    }

    #[test]
    fn equijoin_explicit_syntax() {
        let sql = "SELECT id, order_id \
            FROM person \
            JOIN orders \
            ON id = customer_id";
        let expected = "Projection: #person.id, #orders.order_id\
        \n  Inner Join: #person.id = #orders.customer_id\
        \n    TableScan: person projection=None\
        \n    TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn equijoin_unsupported_expression() {
        let sql = "SELECT id, order_id \
            FROM person \
            JOIN orders \
            ON id = customer_id AND order_id > 1 ";
        let expected = "Projection: #person.id, #orders.order_id\
        \n  Filter: #orders.order_id > Int64(1)\
        \n    Inner Join: #person.id = #orders.customer_id\
        \n      TableScan: person projection=None\
        \n      TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn left_equijoin_unsupported_expression() {
        let sql = "SELECT id, order_id \
            FROM person \
            LEFT JOIN orders \
            ON id = customer_id AND order_id > 1";
        let expected = "Projection: #person.id, #orders.order_id\
        \n  Left Join: #person.id = #orders.customer_id\
        \n    TableScan: person projection=None\
        \n    Filter: #orders.order_id > Int64(1)\
        \n      TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn right_equijoin_unsupported_expression() {
        let sql = "SELECT id, order_id \
            FROM person \
            RIGHT JOIN orders \
            ON id = customer_id AND id > 1";
        let expected = "Projection: #person.id, #orders.order_id\
        \n  Right Join: #person.id = #orders.customer_id\
        \n    Filter: #person.id > Int64(1)\
        \n      TableScan: person projection=None\
        \n    TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn join_with_table_name() {
        let sql = "SELECT id, order_id \
            FROM person \
            JOIN orders \
            ON person.id = orders.customer_id";
        let expected = "Projection: #person.id, #orders.order_id\
        \n  Inner Join: #person.id = #orders.customer_id\
        \n    TableScan: person projection=None\
        \n    TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn join_with_using() {
        let sql = "SELECT person.first_name, id \
            FROM person \
            JOIN person as person2 \
            USING (id)";
        let expected = "Projection: #person.first_name, #person.id\
        \n  Inner Join: Using #person.id = #person2.id\
        \n    TableScan: person projection=None\
        \n    TableScan: person2 projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn project_wildcard_on_join_with_using() {
        let sql = "SELECT * \
            FROM lineitem \
            JOIN lineitem as lineitem2 \
            USING (l_item_id)";
        let expected = "Projection: #lineitem.l_item_id, #lineitem.l_description, #lineitem.price, #lineitem2.l_description, #lineitem2.price\
        \n  Inner Join: Using #lineitem.l_item_id = #lineitem2.l_item_id\
        \n    TableScan: lineitem projection=None\
        \n    TableScan: lineitem2 projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn equijoin_explicit_syntax_3_tables() {
        let sql = "SELECT id, order_id, l_description \
            FROM person \
            JOIN orders ON id = customer_id \
            JOIN lineitem ON o_item_id = l_item_id";
        let expected =
            "Projection: #person.id, #orders.order_id, #lineitem.l_description\
            \n  Inner Join: #orders.o_item_id = #lineitem.l_item_id\
            \n    Inner Join: #person.id = #orders.customer_id\
            \n      TableScan: person projection=None\
            \n      TableScan: orders projection=None\
            \n    TableScan: lineitem projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn boolean_literal_in_condition_expression() {
        let sql = "SELECT order_id \
        FROM orders \
        WHERE delivered = false OR delivered = true";
        let expected = "Projection: #orders.order_id\
            \n  Filter: #orders.delivered = Boolean(false) OR #orders.delivered = Boolean(true)\
            \n    TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn union() {
        let sql = "SELECT order_id from orders UNION SELECT order_id FROM orders";
        let expected = "\
        Distinct:\
        \n  Union\
        \n    Projection: #orders.order_id\
        \n      TableScan: orders projection=None\
        \n    Projection: #orders.order_id\
        \n      TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn union_all() {
        let sql = "SELECT order_id from orders UNION ALL SELECT order_id FROM orders";
        let expected = "Union\
            \n  Projection: #orders.order_id\
            \n    TableScan: orders projection=None\
            \n  Projection: #orders.order_id\
            \n    TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn union_4_combined_in_one() {
        let sql = "SELECT order_id from orders
                    UNION ALL SELECT order_id FROM orders
                    UNION ALL SELECT order_id FROM orders
                    UNION ALL SELECT order_id FROM orders";
        let expected = "Union\
            \n  Projection: #orders.order_id\
            \n    TableScan: orders projection=None\
            \n  Projection: #orders.order_id\
            \n    TableScan: orders projection=None\
            \n  Projection: #orders.order_id\
            \n    TableScan: orders projection=None\
            \n  Projection: #orders.order_id\
            \n    TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn union_with_different_column_names() {
        let sql = "SELECT order_id from orders UNION ALL SELECT customer_id FROM orders";
        let expected = "Union\
            \n  Projection: #orders.order_id\
            \n    TableScan: orders projection=None\
            \n  Projection: #orders.customer_id\
            \n    TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn union_values_with_no_alias() {
        let sql = "SELECT 1, 2 UNION ALL SELECT 3, 4";
        let expected = "Union\
            \n  Projection: Int64(1) AS Int64(1), Int64(2) AS Int64(2)\
            \n    EmptyRelation\
            \n  Projection: Int64(3) AS Int64(1), Int64(4) AS Int64(2)\
            \n    EmptyRelation";
        quick_test(sql, expected);
    }

    #[test]
    fn union_with_incompatible_data_type() {
        let sql = "SELECT interval '1 year 1 day' UNION ALL SELECT 1";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert_eq!(
            "Plan(\"UNION Column Int64(1) (type: Int64) is \
            not compatible with column IntervalMonthDayNano\
            (\\\"950737950189618795196236955648\\\") \
            (type: Interval(MonthDayNano))\")",
            format!("{:?}", err)
        );
    }

    #[test]
    fn union_with_different_decimal_data_types() {
        let sql = "SELECT 1 a UNION ALL SELECT 1.1 a";
        let expected = "Union\
            \n  Projection: CAST(Int64(1) AS Float64) AS a\
            \n    EmptyRelation\
            \n  Projection: Float64(1.1) AS a\
            \n    EmptyRelation";
        quick_test(sql, expected);
    }

    #[test]
    fn union_with_null() {
        let sql = "SELECT NULL a UNION ALL SELECT 1.1 a";
        let expected = "Union\
            \n  Projection: CAST(NULL AS Float64) AS a\
            \n    EmptyRelation\
            \n  Projection: Float64(1.1) AS a\
            \n    EmptyRelation";
        quick_test(sql, expected);
    }

    #[test]
    fn union_with_float_and_string() {
        let sql = "SELECT 'a' a UNION ALL SELECT 1.1 a";
        let expected = "Union\
            \n  Projection: Utf8(\"a\") AS a\
            \n    EmptyRelation\
            \n  Projection: CAST(Float64(1.1) AS Utf8) AS a\
            \n    EmptyRelation";
        quick_test(sql, expected);
    }

    #[test]
    fn union_with_multiply_cols() {
        let sql = "SELECT 'a' a, 1 b UNION ALL SELECT 1.1 a, 1.1 b";
        let expected = "Union\
            \n  Projection: Utf8(\"a\") AS a, CAST(Int64(1) AS Float64) AS b\
            \n    EmptyRelation\
            \n  Projection: CAST(Float64(1.1) AS Utf8) AS a, Float64(1.1) AS b\
            \n    EmptyRelation";
        quick_test(sql, expected);
    }

    #[test]
    fn sorted_union_with_different_types_and_group_by() {
        let sql = "SELECT a FROM (select 1 a) x GROUP BY 1 UNION ALL (SELECT a FROM (select 1.1 a) x GROUP BY 1) ORDER BY 1";
        let expected = "Sort: #a ASC NULLS LAST\
            \n  Union\
            \n    Projection: CAST(#x.a AS Float64) AS a\
            \n      Aggregate: groupBy=[[#x.a]], aggr=[[]]\
            \n        Projection: #x.a, alias=x\
            \n          Projection: Int64(1) AS a, alias=x\
            \n            EmptyRelation\
            \n    Projection: #x.a\
            \n      Aggregate: groupBy=[[#x.a]], aggr=[[]]\
            \n        Projection: #x.a, alias=x\
            \n          Projection: Float64(1.1) AS a, alias=x\
            \n            EmptyRelation";
        quick_test(sql, expected);
    }

    #[test]
    fn union_with_binary_expr_and_cast() {
        let sql = "SELECT cast(0.0 + a as integer) FROM (select 1 a) x GROUP BY 1 UNION ALL (SELECT 2.1 + a FROM (select 1 a) x GROUP BY 1)";
        let expected = "Union\
            \n  Projection: CAST(#CAST(Float64(0) + x.a AS Int32) AS Float64) AS CAST(Float64(0) + x.a AS Int32)\
            \n    Aggregate: groupBy=[[CAST(Float64(0) + #x.a AS Int32)]], aggr=[[]]\
            \n      Projection: #x.a, alias=x\
            \n        Projection: Int64(1) AS a, alias=x\
            \n          EmptyRelation\
            \n  Projection: #Float64(2.1) + x.a\
            \n    Aggregate: groupBy=[[Float64(2.1) + #x.a]], aggr=[[]]\
            \n      Projection: #x.a, alias=x\
            \n        Projection: Int64(1) AS a, alias=x\
            \n          EmptyRelation";
        quick_test(sql, expected);
    }

    #[test]
    fn union_with_aliases() {
        let sql = "SELECT a as a1 FROM (select 1 a) x GROUP BY 1 UNION ALL (SELECT a as a1 FROM (select 1.1 a) x GROUP BY 1)";
        let expected = "Union\
            \n  Projection: CAST(#x.a AS Float64) AS a1\
            \n    Aggregate: groupBy=[[#x.a]], aggr=[[]]\
            \n      Projection: #x.a, alias=x\
            \n        Projection: Int64(1) AS a, alias=x\
            \n          EmptyRelation\
            \n  Projection: #x.a AS a1\
            \n    Aggregate: groupBy=[[#x.a]], aggr=[[]]\
            \n      Projection: #x.a, alias=x\
            \n        Projection: Float64(1.1) AS a, alias=x\
            \n          EmptyRelation";
        quick_test(sql, expected);
    }

    #[test]
    fn empty_over() {
        let sql = "SELECT order_id, MAX(order_id) OVER () from orders";
        let expected = "\
        Projection: #orders.order_id, #MAX(orders.order_id)\
        \n  WindowAggr: windowExpr=[[MAX(#orders.order_id)]]\
        \n    TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn empty_over_with_alias() {
        let sql = "SELECT order_id oid, MAX(order_id) OVER () max_oid from orders";
        let expected = "\
        Projection: #orders.order_id AS oid, #MAX(orders.order_id) AS max_oid\
        \n  WindowAggr: windowExpr=[[MAX(#orders.order_id)]]\
        \n    TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn empty_over_dup_with_alias() {
        let sql = "SELECT order_id oid, MAX(order_id) OVER () max_oid, MAX(order_id) OVER () max_oid_dup from orders";
        let expected = "\
        Projection: #orders.order_id AS oid, #MAX(orders.order_id) AS max_oid, #MAX(orders.order_id) AS max_oid_dup\
        \n  WindowAggr: windowExpr=[[MAX(#orders.order_id)]]\
        \n    TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn empty_over_dup_with_different_sort() {
        let sql = "SELECT order_id oid, MAX(order_id) OVER (), MAX(order_id) OVER (ORDER BY order_id) from orders";
        let expected = "\
        Projection: #orders.order_id AS oid, #MAX(orders.order_id), #MAX(orders.order_id) ORDER BY [#orders.order_id ASC NULLS LAST]\
        \n  WindowAggr: windowExpr=[[MAX(#orders.order_id)]]\
        \n    WindowAggr: windowExpr=[[MAX(#orders.order_id) ORDER BY [#orders.order_id ASC NULLS LAST]]]\
        \n      TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn empty_over_plus() {
        let sql = "SELECT order_id, MAX(qty * 1.1) OVER () from orders";
        let expected = "\
        Projection: #orders.order_id, #MAX(orders.qty * Float64(1.1))\
        \n  WindowAggr: windowExpr=[[MAX(#orders.qty * Float64(1.1))]]\
        \n    TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn empty_over_multiple() {
        let sql =
            "SELECT order_id, MAX(qty) OVER (), min(qty) over (), aVg(qty) OVER () from orders";
        let expected = "\
        Projection: #orders.order_id, #MAX(orders.qty), #MIN(orders.qty), #AVG(orders.qty)\
        \n  WindowAggr: windowExpr=[[MAX(#orders.qty), MIN(#orders.qty), AVG(#orders.qty)]]\
        \n    TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    /// psql result
    /// ```
    ///                               QUERY PLAN
    /// ----------------------------------------------------------------------
    /// WindowAgg  (cost=69.83..87.33 rows=1000 width=8)
    ///   ->  Sort  (cost=69.83..72.33 rows=1000 width=8)
    ///         Sort Key: order_id
    ///         ->  Seq Scan on orders  (cost=0.00..20.00 rows=1000 width=8)
    /// ```
    #[test]
    fn over_partition_by() {
        let sql = "SELECT order_id, MAX(qty) OVER (PARTITION BY order_id) from orders";
        let expected = "\
        Projection: #orders.order_id, #MAX(orders.qty) PARTITION BY [#orders.order_id]\
        \n  WindowAggr: windowExpr=[[MAX(#orders.qty) PARTITION BY [#orders.order_id]]]\
        \n    TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    /// psql result
    /// ```
    ///                                     QUERY PLAN
    /// ----------------------------------------------------------------------------------
    /// WindowAgg  (cost=137.16..154.66 rows=1000 width=12)
    /// ->  Sort  (cost=137.16..139.66 rows=1000 width=12)
    ///         Sort Key: order_id
    ///         ->  WindowAgg  (cost=69.83..87.33 rows=1000 width=12)
    ///             ->  Sort  (cost=69.83..72.33 rows=1000 width=8)
    ///                     Sort Key: order_id DESC
    ///                     ->  Seq Scan on orders  (cost=0.00..20.00 rows=1000 width=8)
    /// ```
    #[test]
    fn over_order_by() {
        let sql = "SELECT order_id, MAX(qty) OVER (ORDER BY order_id), MIN(qty) OVER (ORDER BY order_id DESC) from orders";
        let expected = "\
        Projection: #orders.order_id, #MAX(orders.qty) ORDER BY [#orders.order_id ASC NULLS LAST], #MIN(orders.qty) ORDER BY [#orders.order_id DESC NULLS FIRST]\
        \n  WindowAggr: windowExpr=[[MAX(#orders.qty) ORDER BY [#orders.order_id ASC NULLS LAST]]]\
        \n    WindowAggr: windowExpr=[[MIN(#orders.qty) ORDER BY [#orders.order_id DESC NULLS FIRST]]]\
        \n      TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn over_order_by_with_window_frame_double_end() {
        let sql = "SELECT order_id, MAX(qty) OVER (ORDER BY order_id ROWS BETWEEN 3 PRECEDING and 3 FOLLOWING), MIN(qty) OVER (ORDER BY order_id DESC) from orders";
        let expected = "\
        Projection: #orders.order_id, #MAX(orders.qty) ORDER BY [#orders.order_id ASC NULLS LAST] ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING, #MIN(orders.qty) ORDER BY [#orders.order_id DESC NULLS FIRST]\
        \n  WindowAggr: windowExpr=[[MAX(#orders.qty) ORDER BY [#orders.order_id ASC NULLS LAST] ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING]]\
        \n    WindowAggr: windowExpr=[[MIN(#orders.qty) ORDER BY [#orders.order_id DESC NULLS FIRST]]]\
        \n      TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn over_order_by_with_window_frame_single_end() {
        let sql = "SELECT order_id, MAX(qty) OVER (ORDER BY order_id ROWS 3 PRECEDING), MIN(qty) OVER (ORDER BY order_id DESC) from orders";
        let expected = "\
        Projection: #orders.order_id, #MAX(orders.qty) ORDER BY [#orders.order_id ASC NULLS LAST] ROWS BETWEEN 3 PRECEDING AND CURRENT ROW, #MIN(orders.qty) ORDER BY [#orders.order_id DESC NULLS FIRST]\
        \n  WindowAggr: windowExpr=[[MAX(#orders.qty) ORDER BY [#orders.order_id ASC NULLS LAST] ROWS BETWEEN 3 PRECEDING AND CURRENT ROW]]\
        \n    WindowAggr: windowExpr=[[MIN(#orders.qty) ORDER BY [#orders.order_id DESC NULLS FIRST]]]\
        \n      TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn over_order_by_with_window_frame_range_value_check() {
        let sql = "SELECT order_id, MAX(qty) OVER (ORDER BY order_id RANGE 3 PRECEDING) from orders";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert_eq!(
            "NotImplemented(\"With WindowFrameUnits=RANGE, the bound cannot be 3 PRECEDING or FOLLOWING at the moment\")",
            format!("{:?}", err)
        );
    }

    #[test]
    fn over_order_by_with_window_frame_range_order_by_check() {
        let sql =
            "SELECT order_id, MAX(qty) OVER (RANGE UNBOUNDED PRECEDING) from orders";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert_eq!(
            "Plan(\"With window frame of type RANGE, the order by expression must be of length 1, got 0\")",
            format!("{:?}", err)
        );
    }

    #[test]
    fn over_order_by_with_window_frame_range_order_by_check_2() {
        let sql =
            "SELECT order_id, MAX(qty) OVER (ORDER BY order_id, qty RANGE UNBOUNDED PRECEDING) from orders";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert_eq!(
            "Plan(\"With window frame of type RANGE, the order by expression must be of length 1, got 2\")",
            format!("{:?}", err)
        );
    }

    #[test]
    fn over_order_by_with_window_frame_single_end_groups() {
        let sql = "SELECT order_id, MAX(qty) OVER (ORDER BY order_id GROUPS 3 PRECEDING), MIN(qty) OVER (ORDER BY order_id DESC) from orders";
        let expected = "\
        Projection: #orders.order_id, #MAX(orders.qty) ORDER BY [#orders.order_id ASC NULLS LAST] GROUPS BETWEEN 3 PRECEDING AND CURRENT ROW, #MIN(orders.qty) ORDER BY [#orders.order_id DESC NULLS FIRST]\
        \n  WindowAggr: windowExpr=[[MAX(#orders.qty) ORDER BY [#orders.order_id ASC NULLS LAST] GROUPS BETWEEN 3 PRECEDING AND CURRENT ROW]]\
        \n    WindowAggr: windowExpr=[[MIN(#orders.qty) ORDER BY [#orders.order_id DESC NULLS FIRST]]]\
        \n      TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    /// psql result
    /// ```
    ///                                     QUERY PLAN
    /// -----------------------------------------------------------------------------------
    /// WindowAgg  (cost=142.16..162.16 rows=1000 width=16)
    ///   ->  Sort  (cost=142.16..144.66 rows=1000 width=16)
    ///         Sort Key: order_id
    ///         ->  WindowAgg  (cost=72.33..92.33 rows=1000 width=16)
    ///               ->  Sort  (cost=72.33..74.83 rows=1000 width=12)
    ///                     Sort Key: ((order_id + 1))
    ///                     ->  Seq Scan on orders  (cost=0.00..22.50 rows=1000 width=12)
    /// ```
    #[test]
    fn over_order_by_two_sort_keys() {
        let sql = "SELECT order_id, MAX(qty) OVER (ORDER BY order_id), MIN(qty) OVER (ORDER BY (order_id + 1)) from orders";
        let expected = "\
        Projection: #orders.order_id, #MAX(orders.qty) ORDER BY [#orders.order_id ASC NULLS LAST], #MIN(orders.qty) ORDER BY [#orders.order_id + Int64(1) ASC NULLS LAST]\
        \n  WindowAggr: windowExpr=[[MAX(#orders.qty) ORDER BY [#orders.order_id ASC NULLS LAST]]]\
        \n    WindowAggr: windowExpr=[[MIN(#orders.qty) ORDER BY [#orders.order_id + Int64(1) ASC NULLS LAST]]]\
        \n      TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    /// psql result
    /// ```
    ///                                        QUERY PLAN
    /// ----------------------------------------------------------------------------------------
    /// WindowAgg  (cost=139.66..172.16 rows=1000 width=24)
    ///   ->  WindowAgg  (cost=139.66..159.66 rows=1000 width=16)
    ///         ->  Sort  (cost=139.66..142.16 rows=1000 width=12)
    ///               Sort Key: qty, order_id
    ///               ->  WindowAgg  (cost=69.83..89.83 rows=1000 width=12)
    ///                     ->  Sort  (cost=69.83..72.33 rows=1000 width=8)
    ///                           Sort Key: order_id, qty
    ///                           ->  Seq Scan on orders  (cost=0.00..20.00 rows=1000 width=8)
    /// ```
    #[test]
    fn over_order_by_sort_keys_sorting() {
        let sql = "SELECT order_id, MAX(qty) OVER (ORDER BY qty, order_id), SUM(qty) OVER (), MIN(qty) OVER (ORDER BY order_id, qty) from orders";
        let expected = "\
        Projection: #orders.order_id, #MAX(orders.qty) ORDER BY [#orders.qty ASC NULLS LAST, #orders.order_id ASC NULLS LAST], #SUM(orders.qty), #MIN(orders.qty) ORDER BY [#orders.order_id ASC NULLS LAST, #orders.qty ASC NULLS LAST]\
        \n  WindowAggr: windowExpr=[[SUM(#orders.qty)]]\
        \n    WindowAggr: windowExpr=[[MAX(#orders.qty) ORDER BY [#orders.qty ASC NULLS LAST, #orders.order_id ASC NULLS LAST]]]\
        \n      WindowAggr: windowExpr=[[MIN(#orders.qty) ORDER BY [#orders.order_id ASC NULLS LAST, #orders.qty ASC NULLS LAST]]]\
        \n        TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    /// psql result
    /// ```
    ///                                     QUERY PLAN
    /// ----------------------------------------------------------------------------------
    /// WindowAgg  (cost=69.83..117.33 rows=1000 width=24)
    ///   ->  WindowAgg  (cost=69.83..104.83 rows=1000 width=16)
    ///         ->  WindowAgg  (cost=69.83..89.83 rows=1000 width=12)
    ///               ->  Sort  (cost=69.83..72.33 rows=1000 width=8)
    ///                     Sort Key: order_id, qty
    ///                     ->  Seq Scan on orders  (cost=0.00..20.00 rows=1000 width=8)
    /// ```
    ///
    /// FIXME: for now we are not detecting prefix of sorting keys in order to save one sort exec phase
    #[test]
    fn over_order_by_sort_keys_sorting_prefix_compacting() {
        let sql = "SELECT order_id, MAX(qty) OVER (ORDER BY order_id), SUM(qty) OVER (), MIN(qty) OVER (ORDER BY order_id, qty) from orders";
        let expected = "\
        Projection: #orders.order_id, #MAX(orders.qty) ORDER BY [#orders.order_id ASC NULLS LAST], #SUM(orders.qty), #MIN(orders.qty) ORDER BY [#orders.order_id ASC NULLS LAST, #orders.qty ASC NULLS LAST]\
        \n  WindowAggr: windowExpr=[[SUM(#orders.qty)]]\
        \n    WindowAggr: windowExpr=[[MAX(#orders.qty) ORDER BY [#orders.order_id ASC NULLS LAST]]]\
        \n      WindowAggr: windowExpr=[[MIN(#orders.qty) ORDER BY [#orders.order_id ASC NULLS LAST, #orders.qty ASC NULLS LAST]]]\
        \n        TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    /// psql result
    /// ```
    ///                                        QUERY PLAN
    /// ----------------------------------------------------------------------------------------
    /// WindowAgg  (cost=139.66..172.16 rows=1000 width=24)
    ///   ->  WindowAgg  (cost=139.66..159.66 rows=1000 width=16)
    ///         ->  Sort  (cost=139.66..142.16 rows=1000 width=12)
    ///               Sort Key: order_id, qty
    ///               ->  WindowAgg  (cost=69.83..89.83 rows=1000 width=12)
    ///                     ->  Sort  (cost=69.83..72.33 rows=1000 width=8)
    ///                           Sort Key: qty, order_id
    ///                           ->  Seq Scan on orders  (cost=0.00..20.00 rows=1000 width=8)
    /// ```
    ///
    /// FIXME: for now we are not detecting prefix of sorting keys in order to re-arrange with global
    /// sort
    #[test]
    fn over_order_by_sort_keys_sorting_global_order_compacting() {
        let sql = "SELECT order_id, MAX(qty) OVER (ORDER BY qty, order_id), SUM(qty) OVER (), MIN(qty) OVER (ORDER BY order_id, qty) from orders ORDER BY order_id";
        let expected = "\
        Sort: #orders.order_id ASC NULLS LAST\
        \n  Projection: #orders.order_id, #MAX(orders.qty) ORDER BY [#orders.qty ASC NULLS LAST, #orders.order_id ASC NULLS LAST], #SUM(orders.qty), #MIN(orders.qty) ORDER BY [#orders.order_id ASC NULLS LAST, #orders.qty ASC NULLS LAST]\
        \n    WindowAggr: windowExpr=[[SUM(#orders.qty)]]\
        \n      WindowAggr: windowExpr=[[MAX(#orders.qty) ORDER BY [#orders.qty ASC NULLS LAST, #orders.order_id ASC NULLS LAST]]]\
        \n        WindowAggr: windowExpr=[[MIN(#orders.qty) ORDER BY [#orders.order_id ASC NULLS LAST, #orders.qty ASC NULLS LAST]]]\
        \n          TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    /// psql result
    /// ```
    ///                               QUERY PLAN
    /// ----------------------------------------------------------------------
    /// WindowAgg  (cost=69.83..89.83 rows=1000 width=12)
    ///   ->  Sort  (cost=69.83..72.33 rows=1000 width=8)
    ///         Sort Key: order_id, qty
    ///         ->  Seq Scan on orders  (cost=0.00..20.00 rows=1000 width=8)
    /// ```
    #[test]
    fn over_partition_by_order_by() {
        let sql =
            "SELECT order_id, MAX(qty) OVER (PARTITION BY order_id ORDER BY qty) from orders";
        let expected = "\
        Projection: #orders.order_id, #MAX(orders.qty) PARTITION BY [#orders.order_id] ORDER BY [#orders.qty ASC NULLS LAST]\
        \n  WindowAggr: windowExpr=[[MAX(#orders.qty) PARTITION BY [#orders.order_id] ORDER BY [#orders.qty ASC NULLS LAST]]]\
        \n    TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    /// psql result
    /// ```
    ///                               QUERY PLAN
    /// ----------------------------------------------------------------------
    /// WindowAgg  (cost=69.83..89.83 rows=1000 width=12)
    ///   ->  Sort  (cost=69.83..72.33 rows=1000 width=8)
    ///         Sort Key: order_id, qty
    ///         ->  Seq Scan on orders  (cost=0.00..20.00 rows=1000 width=8)
    /// ```
    #[test]
    fn over_partition_by_order_by_no_dup() {
        let sql =
            "SELECT order_id, MAX(qty) OVER (PARTITION BY order_id, qty ORDER BY qty) from orders";
        let expected = "\
        Projection: #orders.order_id, #MAX(orders.qty) PARTITION BY [#orders.order_id, #orders.qty] ORDER BY [#orders.qty ASC NULLS LAST]\
        \n  WindowAggr: windowExpr=[[MAX(#orders.qty) PARTITION BY [#orders.order_id, #orders.qty] ORDER BY [#orders.qty ASC NULLS LAST]]]\
        \n    TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    /// psql result
    /// ```
    ///                                     QUERY PLAN
    /// ----------------------------------------------------------------------------------
    /// WindowAgg  (cost=142.16..162.16 rows=1000 width=16)
    ///   ->  Sort  (cost=142.16..144.66 rows=1000 width=12)
    ///         Sort Key: qty, order_id
    ///         ->  WindowAgg  (cost=69.83..92.33 rows=1000 width=12)
    ///               ->  Sort  (cost=69.83..72.33 rows=1000 width=8)
    ///                     Sort Key: order_id, qty
    ///                     ->  Seq Scan on orders  (cost=0.00..20.00 rows=1000 width=8)
    /// ```
    #[test]
    fn over_partition_by_order_by_mix_up() {
        let sql =
            "SELECT order_id, MAX(qty) OVER (PARTITION BY order_id, qty ORDER BY qty), MIN(qty) OVER (PARTITION BY qty ORDER BY order_id) from orders";
        let expected = "\
        Projection: #orders.order_id, #MAX(orders.qty) PARTITION BY [#orders.order_id, #orders.qty] ORDER BY [#orders.qty ASC NULLS LAST], #MIN(orders.qty) PARTITION BY [#orders.qty] ORDER BY [#orders.order_id ASC NULLS LAST]\
        \n  WindowAggr: windowExpr=[[MIN(#orders.qty) PARTITION BY [#orders.qty] ORDER BY [#orders.order_id ASC NULLS LAST]]]\
        \n    WindowAggr: windowExpr=[[MAX(#orders.qty) PARTITION BY [#orders.order_id, #orders.qty] ORDER BY [#orders.qty ASC NULLS LAST]]]\
        \n      TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    /// psql result
    /// ```
    ///                                  QUERY PLAN
    /// -----------------------------------------------------------------------------
    /// WindowAgg  (cost=69.83..109.83 rows=1000 width=24)
    ///   ->  WindowAgg  (cost=69.83..92.33 rows=1000 width=20)
    ///         ->  Sort  (cost=69.83..72.33 rows=1000 width=16)
    ///               Sort Key: order_id, qty, price
    ///               ->  Seq Scan on orders  (cost=0.00..20.00 rows=1000 width=16)
    /// ```
    /// FIXME: for now we are not detecting prefix of sorting keys in order to save one sort exec phase
    #[test]
    fn over_partition_by_order_by_mix_up_prefix() {
        let sql =
            "SELECT order_id, MAX(qty) OVER (PARTITION BY order_id ORDER BY qty), MIN(qty) OVER (PARTITION BY order_id, qty ORDER BY price) from orders";
        let expected = "\
        Projection: #orders.order_id, #MAX(orders.qty) PARTITION BY [#orders.order_id] ORDER BY [#orders.qty ASC NULLS LAST], #MIN(orders.qty) PARTITION BY [#orders.order_id, #orders.qty] ORDER BY [#orders.price ASC NULLS LAST]\
        \n  WindowAggr: windowExpr=[[MAX(#orders.qty) PARTITION BY [#orders.order_id] ORDER BY [#orders.qty ASC NULLS LAST]]]\
        \n    WindowAggr: windowExpr=[[MIN(#orders.qty) PARTITION BY [#orders.order_id, #orders.qty] ORDER BY [#orders.price ASC NULLS LAST]]]\
        \n      TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn approx_median_window() {
        let sql =
            "SELECT order_id, APPROX_MEDIAN(qty) OVER(PARTITION BY order_id) from orders";
        let expected = "\
        Projection: #orders.order_id, #APPROXPERCENTILECONT(orders.qty,Float64(0.5)) PARTITION BY [#orders.order_id]\
        \n  WindowAggr: windowExpr=[[APPROXPERCENTILECONT(#orders.qty, Float64(0.5)) PARTITION BY [#orders.order_id]]]\
        \n    TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn window_function_with_window_frame() {
        let sql =
            "SELECT order_id, AVG(qty) OVER(PARTITION BY order_id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) from orders";
        let expected = "\
        Projection: #orders.order_id, #AVG(orders.qty) PARTITION BY [#orders.order_id] ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW\
        \n  WindowAggr: windowExpr=[[AVG(#orders.qty) PARTITION BY [#orders.order_id] ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW]]\
        \n    TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_typedstring() {
        let sql = "SELECT date '2020-12-10' AS date FROM person";
        let expected = "Projection: CAST(Utf8(\"2020-12-10\") AS Date32) AS date\
            \n  TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_multibyte_column() {
        let sql = r#"SELECT "😀" FROM person"#;
        let expected = "Projection: #person.😀\
            \n  TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn cte_order_by_unprojected_column_in_aliased_projection() {
        // The ORDER BY references `cnt`, which is not in the CTE's projection; the
        // sort machinery must push it into the aliased projection and reference it
        // through the projection alias
        let sql =
            "WITH t1 AS (SELECT state, age, COUNT(*) AS cnt FROM person GROUP BY 1, 2), \
            t2 AS (SELECT state, age FROM t1 ORDER BY state, cnt DESC) \
            SELECT * FROM t2";
        let expected = "Projection: #t2.state, #t2.age\
            \n  Projection: #t2.state, #t2.age\
            \n    Sort: #t2.state ASC NULLS LAST, #t2.t1.cnt DESC NULLS FIRST\
            \n      Projection: #t1.state, #t1.age, #t1.cnt AS t1.cnt, alias=t2\
            \n        Projection: #person.state, #person.age, #COUNT(UInt8(1)) AS cnt, alias=t1\
            \n          Aggregate: groupBy=[[#person.state, #person.age]], aggr=[[COUNT(UInt8(1))]]\
            \n            TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn distinct_on() {
        // DISTINCT ON plans into a ROW_NUMBER() window partitioned by the ON
        // expressions and ordered by the ORDER BY expressions, plus a filter
        // keeping the first row per partition
        let sql =
            "SELECT DISTINCT ON (state) state, age FROM person ORDER BY state, age DESC";
        let expected = "Sort: #person.state ASC NULLS LAST, #person.age DESC NULLS FIRST\
            \n  Projection: #person.state, #person.age\
            \n    Filter: #row_number PARTITION BY [#person.state] ORDER BY [#person.state ASC NULLS LAST, #person.age DESC NULLS FIRST] = UInt64(1)\
            \n      WindowAggr: windowExpr=[[ROW_NUMBER() PARTITION BY [#person.state] ORDER BY [#person.state ASC NULLS LAST, #person.age DESC NULLS FIRST]]]\
            \n        TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn cte_distinct_on() {
        // DISTINCT ON inside a CTE: dedupe happens below the CTE's aliased
        // projection, before the outer query consumes it
        let sql = "WITH t1 AS (SELECT state, age, COUNT(*) AS cnt FROM person GROUP BY 1, 2), \
            t2 AS (SELECT DISTINCT ON (state) state, cnt FROM t1 ORDER BY state, cnt DESC) \
            SELECT * FROM t2";
        let expected = "Projection: #t2.state, #t2.cnt\
            \n  Sort: #t2.state ASC NULLS LAST, #t2.cnt DESC NULLS FIRST\
            \n    Projection: #t1.state, #t1.cnt, alias=t2\
            \n      Filter: #row_number PARTITION BY [#t1.state] ORDER BY [#t1.state ASC NULLS LAST, #t1.cnt DESC NULLS FIRST] = UInt64(1)\
            \n        WindowAggr: windowExpr=[[ROW_NUMBER() PARTITION BY [#t1.state] ORDER BY [#t1.state ASC NULLS LAST, #t1.cnt DESC NULLS FIRST]]]\
            \n          Projection: #person.state, #person.age, #COUNT(UInt8(1)) AS cnt, alias=t1\
            \n            Aggregate: groupBy=[[#person.state, #person.age]], aggr=[[COUNT(UInt8(1))]]\
            \n              TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn distinct_on_order_by_mismatch() {
        let sql = "SELECT DISTINCT ON (state) state, age FROM person ORDER BY age";
        let err = logical_plan(sql).expect_err("query should have failed");
        assert!(matches!(
            err,
            DataFusionError::Plan(msg) if msg.contains(
                "SELECT DISTINCT ON expressions must match initial ORDER BY expressions"
            ),
        ));
    }

    #[test]
    fn distinct_on_without_order_by() {
        // Postgres allows DISTINCT ON with no ORDER BY: an arbitrary row is kept
        // per group
        let sql = "SELECT DISTINCT ON (state) state, age FROM person";
        let expected = "Projection: #person.state, #person.age\
            \n  Filter: #row_number PARTITION BY [#person.state] = UInt64(1)\
            \n    WindowAggr: windowExpr=[[ROW_NUMBER() PARTITION BY [#person.state]]]\
            \n      TableScan: person projection=None";
        quick_test(sql, expected);
    }

    fn logical_plan(sql: &str) -> Result<LogicalPlan> {
        let planner = SqlToRel::new(&MockContextProvider {});
        let result = DFParser::parse_sql(sql);
        let mut ast = result?;
        planner.statement_to_plan(ast.pop_front().unwrap())
    }

    /// Create logical plan, write with formatter, compare to expected output
    fn quick_test(sql: &str, expected: &str) {
        let plan = logical_plan(sql).unwrap();
        assert_eq!(format!("{:?}", plan), expected);
    }

    /// A long chain of binary operators (`1 OR 1 OR 1 OR …`) must plan without overflowing the
    /// stack. The AST is built directly (the recursive SQL parser would overflow on input this
    /// deep before planning is even reached), and the binary-operator spine is walked iteratively
    /// by `sql_expr_to_logical_expr`, so planning uses O(1) call depth regardless of chain length.
    #[test]
    fn deep_binary_op_chain_does_not_overflow_the_stack() {
        use sqlparser::ast::Value;

        // Deep enough that the previous one-frame-per-operator recursion would overflow a
        // typical (2 MiB) thread stack many times over.
        const DEPTH: usize = 50_000;
        let lit_one =
            || SQLExpr::Value(Value::Number("1".to_string(), false).with_empty_span());

        // Build a left-deep chain iteratively (no recursion, so the build itself is safe).
        let mut expr = lit_one();
        for _ in 0..DEPTH {
            expr = SQLExpr::BinaryOp {
                left: Box::new(expr),
                op: BinaryOperator::Or,
                right: Box::new(lit_one()),
            };
        }

        let planner = SqlToRel::new(&MockContextProvider {});
        let schema = DFSchema::empty();
        // The planner dismantles the input spine iteratively, so the input is not deep-dropped.
        let planned = planner
            .sql_expr_to_logical_expr(expr, &schema, None)
            .expect("deep OR chain should plan");
        assert!(matches!(planned.as_ref(), Expr::BinaryExpr { .. }));

        // The produced `Expr` tree is just as deep; dropping it would recurse `DEPTH` times and
        // overflow the stack. We're only testing the planner here, so leak it intentionally.
        std::mem::forget(planned);
    }

    struct MockContextProvider {}

    impl ContextProvider for MockContextProvider {
        fn get_table_provider(
            &self,
            name: TableReference,
        ) -> Option<Arc<dyn TableProvider>> {
            let schema = match name.table() {
                "test_decimal" => Some(Schema::new(vec![
                    Field::new("id", DataType::Int32, false),
                    Field::new("price", DataType::Decimal(10, 2), false),
                ])),
                "person" => Some(Schema::new(vec![
                    Field::new("id", DataType::UInt32, false),
                    Field::new("first_name", DataType::Utf8, false),
                    Field::new("last_name", DataType::Utf8, false),
                    Field::new("age", DataType::Int32, false),
                    Field::new("state", DataType::Utf8, false),
                    Field::new("salary", DataType::Float64, false),
                    Field::new(
                        "birth_date",
                        DataType::Timestamp(TimeUnit::Nanosecond, None),
                        false,
                    ),
                    Field::new("😀", DataType::Int32, false),
                ])),
                "orders" => Some(Schema::new(vec![
                    Field::new("order_id", DataType::UInt32, false),
                    Field::new("customer_id", DataType::UInt32, false),
                    Field::new("o_item_id", DataType::Utf8, false),
                    Field::new("qty", DataType::Int32, false),
                    Field::new("price", DataType::Float64, false),
                    Field::new("delivered", DataType::Boolean, false),
                ])),
                "lineitem" => Some(Schema::new(vec![
                    Field::new("l_item_id", DataType::UInt32, false),
                    Field::new("l_description", DataType::Utf8, false),
                    Field::new("price", DataType::Float64, false),
                ])),
                "aggregate_test_100" => Some(Schema::new(vec![
                    Field::new("c1", DataType::Utf8, false),
                    Field::new("c2", DataType::UInt32, false),
                    Field::new("c3", DataType::Int8, false),
                    Field::new("c4", DataType::Int16, false),
                    Field::new("c5", DataType::Int32, false),
                    Field::new("c6", DataType::Int64, false),
                    Field::new("c7", DataType::UInt8, false),
                    Field::new("c8", DataType::UInt16, false),
                    Field::new("c9", DataType::UInt32, false),
                    Field::new("c10", DataType::UInt64, false),
                    Field::new("c11", DataType::Float32, false),
                    Field::new("c12", DataType::Float64, false),
                    Field::new("c13", DataType::Utf8, false),
                ])),
                _ => None,
            };
            schema.map(|s| -> Arc<dyn TableProvider> {
                Arc::new(EmptyTable::new(Arc::new(s)))
            })
        }

        fn get_function_meta(&self, name: &str) -> Option<Arc<ScalarUDF>> {
            let f: ScalarFunctionImplementation =
                Arc::new(|_| Err(DataFusionError::NotImplemented("".to_string())));
            match name {
                "my_sqrt" => Some(Arc::new(create_udf(
                    "my_sqrt",
                    vec![DataType::Float64],
                    Arc::new(DataType::Float64),
                    Volatility::Immutable,
                    f,
                ))),
                _ => None,
            }
        }

        fn get_table_function_meta(&self, _name: &str) -> Option<Arc<TableUDF>> {
            unimplemented!()
        }

        fn get_aggregate_meta(&self, _name: &str) -> Option<Arc<AggregateUDF>> {
            unimplemented!()
        }

        fn get_variable_type(&self, _: &[String]) -> Option<DataType> {
            unimplemented!()
        }
    }

    #[test]
    fn select_partially_qualified_column() {
        let sql = r#"SELECT person.first_name FROM public.person"#;
        let expected = "Projection: #person.first_name\
            \n  TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn select_collate_cubesql() {
        let sql = r#"SELECT person.first_name FROM public.person WHERE first_name collate utf8_general_ci = 'dmitry'"#;
        let expected = "Projection: #person.first_name\
        \n  Filter: #person.first_name = Utf8(\"dmitry\")\
        \n    TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn cross_join_to_inner_join() {
        let sql = "select person.id from person, orders, lineitem where person.id = lineitem.l_item_id and orders.o_item_id = lineitem.l_description;";
        let expected = "Projection: #person.id\
                                 \n  Inner Join: #lineitem.l_description = #orders.o_item_id\
                                 \n    Inner Join: #person.id = #lineitem.l_item_id\
                                 \n      TableScan: person projection=None\
                                 \n      TableScan: lineitem projection=None\
                                 \n    TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn cross_join_not_to_inner_join() {
        let sql = "select person.id from person, orders, lineitem where person.id = person.age;";
        let expected = "Projection: #person.id\
                                    \n  Filter: #person.id = #person.age\
                                    \n    CrossJoin:\
                                    \n      CrossJoin:\
                                    \n        TableScan: person projection=None\
                                    \n        TableScan: orders projection=None\
                                    \n      TableScan: lineitem projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn cte_use_same_name_multiple_times() {
        let sql = "with a as (select * from person), a as (select * from orders) select * from a;";
        let expected = "SQL error: ParserError(\"WITH query name \\\"a\\\" specified more than once\")";
        let result = logical_plan(sql).err().unwrap();
        assert_eq!(expected, format!("{}", result));
    }

    #[test]
    fn subquery_select() {
        let sql = "select person.id, (select lineitem.l_item_id from lineitem where person.id = lineitem.l_item_id limit 1) from person";
        let expected = "Projection: #person.id, #__subquery-0.l_item_id\
                        \n  Subquery: types=[Scalar]\
                        \n    TableScan: person projection=None\
                        \n    Limit: skip=None, fetch=1\
                        \n      Projection: #lineitem.l_item_id, alias=__subquery-0\
                        \n        Filter: ^#person.id = #lineitem.l_item_id\
                        \n          TableScan: lineitem projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn subquery_select_without_from() {
        let sql = "select person.id, (select person.age + 1) from person";
        let expected = "Projection: #person.id, #__subquery-0.person.age + Int64(1)\
                        \n  Subquery: types=[Scalar]\
                        \n    TableScan: person projection=None\
                        \n    Projection: ^#person.age + Int64(1), alias=__subquery-0\
                        \n      EmptyRelation";
        quick_test(sql, expected);
    }

    #[test]
    fn subquery_where() {
        let sql = "select person.id from person where person.id > (select lineitem.l_item_id from lineitem limit 1)";
        let expected = "Projection: #person.id\
                        \n  Filter: #person.id > #__subquery-0.l_item_id\
                        \n    Subquery: types=[Scalar]\
                        \n      TableScan: person projection=None\
                        \n      Limit: skip=None, fetch=1\
                        \n        Projection: #lineitem.l_item_id, alias=__subquery-0\
                        \n          TableScan: lineitem projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn subquery_where_without_from() {
        let sql = "select person.id from person where person.id = (select person.id)";
        let expected = "Projection: #person.id\
                        \n  Filter: #person.id = #__subquery-0.person.id\
                        \n    Subquery: types=[Scalar]\
                        \n      TableScan: person projection=None\
                        \n      Projection: ^#person.id, alias=__subquery-0\
                        \n        EmptyRelation";
        quick_test(sql, expected);
    }

    #[test]
    fn subquery_select_and_where() {
        let sql = "select person.id, (select person.id) from person where person.id > (select lineitem.l_item_id from lineitem limit 1)";
        let expected = "Projection: #person.id, #__subquery-1.person.id\
                        \n  Subquery: types=[Scalar]\
                        \n    Filter: #person.id > #__subquery-0.l_item_id\
                        \n      Subquery: types=[Scalar]\
                        \n        TableScan: person projection=None\
                        \n        Limit: skip=None, fetch=1\
                        \n          Projection: #lineitem.l_item_id, alias=__subquery-0\
                        \n            TableScan: lineitem projection=None\
                        \n    Projection: ^#person.id, alias=__subquery-1\
                        \n      EmptyRelation";
        quick_test(sql, expected);
    }

    #[test]
    fn subquery_select_and_where_without_from() {
        let sql = "select person.id, (select person.id) from person where person.id = (select person.id)";
        let expected = "Projection: #person.id, #__subquery-1.person.id\
                        \n  Subquery: types=[Scalar]\
                        \n    Filter: #person.id = #__subquery-0.person.id\
                        \n      Subquery: types=[Scalar]\
                        \n        TableScan: person projection=None\
                        \n        Projection: ^#person.id, alias=__subquery-0\
                        \n          EmptyRelation\
                        \n    Projection: ^#person.id, alias=__subquery-1\
                        \n      EmptyRelation";
        quick_test(sql, expected);
    }

    #[test]
    fn subquery_exists() {
        let sql = "select person.id, exists(select person.id) from person where exists(select 1 where false)";
        let expected = "Projection: #person.id, #__subquery-1.person.id\
                        \n  Subquery: types=[Exists]\
                        \n    Filter: #__subquery-0.Int64(1)\
                        \n      Subquery: types=[Exists]\
                        \n        TableScan: person projection=None\
                        \n        Projection: Int64(1), alias=__subquery-0\
                        \n          Filter: Boolean(false)\
                        \n            EmptyRelation\
                        \n    Projection: ^#person.id, alias=__subquery-1\
                        \n      EmptyRelation";
        quick_test(sql, expected);
    }

    #[test]
    fn subquery_any() {
        let sql = "select person.id from person where person.id = any(select person.id)";
        let expected = "Projection: #person.id\
                        \n  Filter: #person.id = ANY(#__subquery-0.person.id)\
                        \n    Subquery: types=[AnyAll]\
                        \n      TableScan: person projection=None\
                        \n      Projection: ^#person.id, alias=__subquery-0\
                        \n        EmptyRelation";
        quick_test(sql, expected);
    }

    #[test]
    fn subquery_all() {
        let sql = "select person.id, person.id = all(select person.id) from person";
        let expected =
            "Projection: #person.id, #person.id = ALL(#__subquery-0.person.id)\
                        \n  Subquery: types=[AnyAll]\
                        \n    TableScan: person projection=None\
                        \n    Projection: ^#person.id, alias=__subquery-0\
                        \n      EmptyRelation";
        quick_test(sql, expected);
    }

    #[test]
    fn subquery_in() {
        let sql =
            "select person.id, person.id in (select person.id from person) from person";
        let expected = "Projection: #person.id, #person.id IN (#__subquery-0.id)\
                        \n  Subquery: types=[AnyAll]\
                        \n    TableScan: person projection=None\
                        \n    Projection: #person.id, alias=__subquery-0\
                        \n      TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn subquery_compound_identifier_self_reference() {
        let sql = "SELECT person.id \
            FROM person \
            WHERE person.id IN ( \
                SELECT person.id \
                FROM person \
                WHERE person.id > 10 \
            )";
        let expected = "\
              Projection: #person.id\
            \n  Filter: #person.id IN (#__subquery-0.id)\
            \n    Subquery: types=[AnyAll]\
            \n      TableScan: person projection=None\
            \n      Projection: #person.id, alias=__subquery-0\
            \n        Filter: #person.id > Int64(10)\
            \n          TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn join_on_disjunction_condition() {
        let sql = "SELECT id, order_id \
            FROM person \
            JOIN orders ON id = customer_id OR person.age > 30";
        let expected = "Projection: #person.id, #orders.order_id\
            \n  Filter: #person.id = #orders.customer_id OR #person.age > Int64(30)\
            \n    CrossJoin:\
            \n      TableScan: person projection=None\
            \n      TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn join_on_complex_condition() {
        let sql = "SELECT id, order_id \
            FROM person \
            JOIN orders ON id = customer_id AND (person.age > 30 OR person.last_name = 'X')";
        let expected = "Projection: #person.id, #orders.order_id\
            \n  Filter: #person.age > Int64(30) OR #person.last_name = Utf8(\"X\")\
            \n    Inner Join: #person.id = #orders.customer_id\
            \n      TableScan: person projection=None\
            \n      TableScan: orders projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn test_zero_offset_with_limit() {
        let sql = "select id from person where person.id > 100 LIMIT 5 OFFSET 0;";
        let expected = "Limit: skip=0, fetch=5\
                                    \n  Projection: #person.id\
                                    \n    Filter: #person.id > Int64(100)\
                                    \n      TableScan: person projection=None";
        quick_test(sql, expected);

        // Flip the order of LIMIT and OFFSET in the query. Plan should remain the same.
        let sql = "SELECT id FROM person WHERE person.id > 100 OFFSET 0 LIMIT 5;";
        quick_test(sql, expected);

        // Replace LIMIT with FETCH ROWS in the query. Plan should remain the same.
        let sql = "SELECT id FROM person WHERE person.id > 100 OFFSET 0 FETCH NEXT 5 ROWS ONLY;";
        quick_test(sql, expected);
    }

    #[test]
    fn test_offset_no_limit() {
        let sql = "SELECT id FROM person WHERE person.id > 100 OFFSET 5;";
        let expected = "Limit: skip=5, fetch=None\
        \n  Projection: #person.id\
        \n    Filter: #person.id > Int64(100)\
        \n      TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn test_offset_after_limit() {
        let sql = "select id from person where person.id > 100 LIMIT 5 OFFSET 3;";
        let expected = "Limit: skip=3, fetch=5\
        \n  Projection: #person.id\
        \n    Filter: #person.id > Int64(100)\
        \n      TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn test_offset_before_limit() {
        let sql = "select id from person where person.id > 100 OFFSET 3 LIMIT 5;";
        let expected = "Limit: skip=3, fetch=5\
        \n  Projection: #person.id\
        \n    Filter: #person.id > Int64(100)\
        \n      TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[test]
    fn test_at_time_zone() {
        let sql = "select CAST(158412331400600000 as timestamp) AT TIME ZONE 'Etc/UTC';";
        let expected = "Projection: CAST(Int64(158412331400600000) AS Timestamp(Nanosecond, None))\n  EmptyRelation";
        quick_test(sql, expected);
    }

    #[test]
    fn test_union_ctes() {
        let sql = "\
        \n  WITH w AS (SELECT 1 l)\
        \n  SELECT w.l\
        \n  FROM w\
        \n  UNION ALL (SELECT w.l FROM w)\
        \n;";
        let expected = "\
        Union\
        \n  Projection: #w.l\
        \n    Projection: Int64(1) AS l, alias=w\
        \n      EmptyRelation\
        \n  Projection: #w.l\
        \n    Projection: Int64(1) AS l, alias=w\
        \n      EmptyRelation";
        quick_test(sql, expected);
    }
    #[test]
    fn test_subquery_ctes() {
        let sql = "with w  as (select id  from person where id < 100) select id from person where person.id in (select id from w);";
        let expected = "\
        Projection: #person.id\
        \n  Filter: #person.id IN (#__subquery-0.id)\
        \n    Subquery: types=[AnyAll]\
        \n      TableScan: person projection=None\
        \n      Projection: #w.id, alias=__subquery-0\
        \n        Projection: #person.id, alias=w\
        \n          Filter: #person.id < Int64(100)\
        \n            TableScan: person projection=None";
        quick_test(sql, expected);
    }
    #[tokio::test]
    async fn aggregate_with_rollup() {
        let sql = "SELECT id, state, age, COUNT(*) FROM person GROUP BY id, ROLLUP (state, age)";
        let expected = "Projection: #person.id, #person.state, #person.age, #COUNT(UInt8(1))\
        \n  Aggregate: groupBy=[[#person.id, ROLLUP (#person.state, #person.age)]], aggr=[[COUNT(UInt8(1))]]\
        \n    TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[tokio::test]
    async fn aggregate_with_cube() {
        let sql =
            "SELECT id, state, age, COUNT(*) FROM person GROUP BY id, CUBE (state, age)";
        let expected = "Projection: #person.id, #person.state, #person.age, #COUNT(UInt8(1))\
        \n  Aggregate: groupBy=[[#person.id, CUBE (#person.state, #person.age)]], aggr=[[COUNT(UInt8(1))]]\
        \n    TableScan: person projection=None";
        quick_test(sql, expected);
    }

    #[tokio::test]
    async fn round_decimal() {
        let sql = "SELECT round(price/3, 2) FROM test_decimal";
        let expected = "Projection: round(#test_decimal.price / Int64(3), Int64(2))\
        \n  TableScan: test_decimal projection=None";
        quick_test(sql, expected);
    }

    #[ignore] // see https://github.com/apache/arrow-datafusion/issues/2469
    #[tokio::test]
    async fn aggregate_with_grouping_sets() {
        let sql = "SELECT id, state, age, COUNT(*) FROM person GROUP BY id, GROUPING SETS ((state), (state, age), (id, state))";
        let expected = "TBD";
        quick_test(sql, expected);
    }
}
