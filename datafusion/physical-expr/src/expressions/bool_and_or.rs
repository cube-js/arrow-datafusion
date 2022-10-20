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

//! Defines physical expressions that can evaluated at runtime during query execution

use std::{any::Any, sync::Arc};

use crate::{AggregateExpr, PhysicalExpr};
use arrow::{
    array::{Array, ArrayRef, BooleanArray},
    datatypes::{DataType, Field},
};
use datafusion_common::{DataFusionError, Result, ScalarValue};
use datafusion_expr::Accumulator;

use super::format_state_name;

/// BOOL_AND aggregate expression
/// Returns TRUE if all non-null values in the given expression were true.
#[derive(Debug)]
pub struct BoolAnd {
    name: String,
    expr: Arc<dyn PhysicalExpr>,
}

impl BoolAnd {
    pub fn new(expr: Arc<dyn PhysicalExpr>, name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            expr,
        }
    }
}

impl AggregateExpr for BoolAnd {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn field(&self) -> Result<Field> {
        Ok(Field::new(&self.name, DataType::Boolean, true))
    }

    fn state_fields(&self) -> Result<Vec<Field>> {
        Ok(vec![Field::new(
            &format_state_name(&self.name, "value"),
            DataType::Boolean,
            true,
        )])
    }

    fn expressions(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        vec![self.expr.clone()]
    }

    fn create_accumulator(&self) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(BoolAndAccumulator::new()))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Debug)]
struct BoolAndAccumulator {
    value: Option<bool>,
}

impl BoolAndAccumulator {
    pub fn new() -> Self {
        Self { value: None }
    }
}

impl Accumulator for BoolAndAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if self.value == Some(false) {
            return Ok(());
        }

        let array = &values[0];
        let values = array
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| {
                DataFusionError::Plan("BOOL_AND only supports boolean values".to_string())
            })?;

        if values.null_count() == values.len() {
            return Ok(());
        }

        for i in 0..values.len() {
            if array.is_null(i) {
                continue;
            }

            self.value = Some(values.value(i));
            if self.value == Some(false) {
                return Ok(());
            }
        }

        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        // bool_and(bool1, bool2, bool3, ...) = bool1 && bool2 && bool3 && ...
        self.update_batch(states)
    }

    fn state(&self) -> Result<Vec<ScalarValue>> {
        Ok(vec![ScalarValue::Boolean(self.value)])
    }

    fn evaluate(&self) -> Result<ScalarValue> {
        Ok(ScalarValue::Boolean(self.value))
    }
}

/// BOOL_OR aggregate expression
/// Returns TRUE if any non-null value in the given expression was true.
#[derive(Debug)]
pub struct BoolOr {
    name: String,
    expr: Arc<dyn PhysicalExpr>,
}

impl BoolOr {
    pub fn new(expr: Arc<dyn PhysicalExpr>, name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            expr,
        }
    }
}

impl AggregateExpr for BoolOr {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn field(&self) -> Result<Field> {
        Ok(Field::new(&self.name, DataType::Boolean, true))
    }

    fn state_fields(&self) -> Result<Vec<Field>> {
        Ok(vec![Field::new(
            &format_state_name(&self.name, "value"),
            DataType::Boolean,
            true,
        )])
    }

    fn expressions(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        vec![self.expr.clone()]
    }

    fn create_accumulator(&self) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(BoolOrAccumulator::new()))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Debug)]
struct BoolOrAccumulator {
    value: Option<bool>,
}

impl BoolOrAccumulator {
    pub fn new() -> Self {
        Self { value: None }
    }
}

impl Accumulator for BoolOrAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if self.value == Some(true) {
            return Ok(());
        }

        let array = &values[0];
        let values = array
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| {
                DataFusionError::Plan("BOOL_OR only supports boolean values".to_string())
            })?;

        if values.null_count() == values.len() {
            return Ok(());
        }

        for i in 0..values.len() {
            if array.is_null(i) {
                continue;
            }

            self.value = Some(values.value(i));
            if self.value == Some(true) {
                return Ok(());
            }
        }

        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        // bool_or(bool1, bool2, bool3, ...) = bool1 || bool2 || bool3 || ...
        self.update_batch(states)
    }

    fn state(&self) -> Result<Vec<ScalarValue>> {
        Ok(vec![ScalarValue::Boolean(self.value)])
    }

    fn evaluate(&self) -> Result<ScalarValue> {
        Ok(ScalarValue::Boolean(self.value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expressions::col;
    use crate::expressions::tests::aggregate;
    use arrow::datatypes::*;
    use arrow::record_batch::RecordBatch;
    use datafusion_common::Result;

    macro_rules! bool_and_or_test {
        ($ARRAY:expr, $OP:ident, $EXPECTED:expr) => {{
            let schema = Schema::new(vec![Field::new("a", DataType::Boolean, true)]);

            let batch = RecordBatch::try_new(
                Arc::new(schema.clone()),
                vec![Arc::new(BooleanArray::from($ARRAY))],
            )?;

            let agg = Arc::new(<$OP>::new(col("a", &schema)?, "bla".to_string()));
            let actual = aggregate(&batch, agg)?;
            let expected = ScalarValue::from($EXPECTED);

            assert_eq!(expected, actual);
        }};
    }

    #[test]
    fn bool_and() -> Result<()> {
        bool_and_or_test!(
            vec![Some(true), Some(true), Some(true), Some(true), Some(true)],
            BoolAnd,
            Some(true)
        );

        bool_and_or_test!(
            vec![Some(true), Some(false), Some(true), Some(true), Some(true)],
            BoolAnd,
            Some(false)
        );

        bool_and_or_test!(
            vec![
                Some(false),
                Some(false),
                Some(false),
                Some(false),
                Some(false)
            ],
            BoolAnd,
            Some(false)
        );

        Ok(())
    }

    #[test]
    fn bool_and_with_nulls() -> Result<()> {
        bool_and_or_test!(
            vec![Some(true), Some(true), None, Some(true), Some(true)],
            BoolAnd,
            Some(true)
        );

        bool_and_or_test!(
            vec![Some(true), Some(false), Some(true), Some(true), None],
            BoolAnd,
            Some(false)
        );

        bool_and_or_test!(
            vec![Some(false), Some(false), None, Some(false), Some(false)],
            BoolAnd,
            Some(false)
        );

        bool_and_or_test!(vec![None, None, None, None, None], BoolAnd, None::<bool>);

        Ok(())
    }

    #[test]
    fn bool_or() -> Result<()> {
        bool_and_or_test!(
            vec![Some(true), Some(true), Some(true), Some(true), Some(true)],
            BoolOr,
            Some(true)
        );

        bool_and_or_test!(
            vec![
                Some(false),
                Some(true),
                Some(false),
                Some(false),
                Some(false)
            ],
            BoolOr,
            Some(true)
        );

        bool_and_or_test!(
            vec![
                Some(false),
                Some(false),
                Some(false),
                Some(false),
                Some(false)
            ],
            BoolOr,
            Some(false)
        );

        Ok(())
    }

    #[test]
    fn bool_or_with_nulls() -> Result<()> {
        bool_and_or_test!(
            vec![Some(true), Some(true), None, Some(true), Some(true)],
            BoolOr,
            Some(true)
        );

        bool_and_or_test!(
            vec![Some(false), Some(true), Some(false), Some(false), None],
            BoolOr,
            Some(true)
        );

        bool_and_or_test!(
            vec![Some(false), Some(false), None, Some(false), Some(false)],
            BoolOr,
            Some(false)
        );

        bool_and_or_test!(vec![None, None, None, None, None], BoolOr, None::<bool>);

        Ok(())
    }
}
