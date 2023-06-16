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

//! Any/All expression

use std::any::Any;
use std::sync::Arc;

use arrow::array::{
    BooleanArray, Int16Array, Int32Array, Int64Array, Int8Array, ListArray,
    PrimitiveArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow::datatypes::ArrowPrimitiveType;
use arrow::{
    datatypes::{DataType, Schema},
    record_batch::RecordBatch,
};

use crate::expressions::try_cast;
use crate::PhysicalExpr;
use arrow::array::*;

use datafusion_common::{DataFusionError, Result};
use datafusion_expr::{ColumnarValue, Operator};

macro_rules! compare_op_scalar {
    ($LEFT:expr, $LIST_VALUES:expr, $OP:expr, $LIST_VALUES_TYPE:ty, $LIST_FROM_SCALAR:expr, $VALUE_FROM_SCALAR:expr) => {{
        let len = if $VALUE_FROM_SCALAR {
            $LIST_VALUES.len()
        } else {
            $LEFT.len()
        };
        let mut builder = BooleanBuilder::new(len);

        for i in 0..len {
            let left_i = if $VALUE_FROM_SCALAR { 0 } else { i };
            let list_i = if $LIST_FROM_SCALAR { 0 } else { i };

            if $LIST_VALUES.is_null(list_i) {
                builder.append_value(false)?;
            } else {
                let list_values = $LIST_VALUES.value(list_i);
                let list_values = list_values
                    .as_any()
                    .downcast_ref::<$LIST_VALUES_TYPE>()
                    .unwrap();
                if list_values.is_empty() {
                    builder.append_value(false)?;
                } else if $LEFT.is_null(left_i) {
                    builder.append_null()?;
                } else {
                    let result = $OP($LEFT.value(left_i), list_values);
                    builder.append_option(result)?;
                }
            }
        }

        Ok(builder.finish())
    }};
}

macro_rules! make_primitive {
    ($VALUES:expr, $IN_VALUES:expr, $OP:expr, $TYPE:ident, $LIST_FROM_SCALAR:expr, $VALUE_FROM_SCALAR:expr, $ALL:expr) => {{
        let left = $VALUES.as_any().downcast_ref::<$TYPE>().expect(&format!(
            "Unable to downcast values to {}",
            stringify!($TYPE)
        ));

        Ok(ColumnarValue::Array(Arc::new(compare_primitive(
            left,
            $IN_VALUES,
            $LIST_FROM_SCALAR,
            $VALUE_FROM_SCALAR,
            $OP,
            $ALL,
        )?)))
    }};
}

fn compare_primitive<T: ArrowPrimitiveType>(
    array: &PrimitiveArray<T>,
    list: &ListArray,
    list_from_scalar: bool,
    value_from_scalar: bool,
    op: Operator,
    all: bool,
) -> Result<BooleanArray> {
    macro_rules! comparator_primitive {
        ($($OP:pat = ($FN:tt),)*) => {
            match op {
                $(
                    $OP => if all {
                        |x, v: &PrimitiveArray<T>| wrap_option_primitive(v, true, v.values().iter().all(|y| &x $FN y))
                    } else {
                        |x, v: &PrimitiveArray<T>| wrap_option_primitive(v, false, v.values().iter().any(|y| &x $FN y))
                    },
                )*
                op => return unsupported_op(op),
            }
        };
    }
    let fun = comparator_primitive!(
        Operator::Eq = (==),
        Operator::NotEq = (!=),
        Operator::Lt = (<),
        Operator::LtEq = (<=),
        Operator::Gt = (>),
        Operator::GtEq = (>=),
    );
    compare_op_scalar!(
        array,
        list,
        fun,
        PrimitiveArray<T>,
        list_from_scalar,
        value_from_scalar
    )
}

fn wrap_option_primitive<T: ArrowPrimitiveType>(
    v: &PrimitiveArray<T>,
    all: bool,
    result: bool,
) -> Option<bool> {
    if result != all {
        return Some(!all);
    }
    if v.null_count() > 0 {
        return None;
    }
    Some(all)
}

fn compare_bool(
    array: &BooleanArray,
    list: &ListArray,
    list_from_scalar: bool,
    value_from_scalar: bool,
    op: Operator,
    all: bool,
) -> Result<BooleanArray> {
    macro_rules! comparator_bool {
        ($($OP:pat = ($FN:tt inverted $IFN:tt),)*) => {
            match op {
                $(
                    $OP => if all {
                        |x, v: &BooleanArray| {
                            for i in 0..v.len() {
                                if !v.is_null(i) && x $IFN v.value(i) {
                                    return Some(false)
                                }
                            }
                            wrap_option_bool(v, true)
                        }
                    } else {
                        |x, v: &BooleanArray| {
                            for i in 0..v.len() {
                                if !v.is_null(i) && x $FN v.value(i) {
                                    return Some(true)
                                }
                            }
                            wrap_option_bool(v, false)
                        }
                    }
                )*
                op => return unsupported_op(op),
            }
        };
    }
    let fun = comparator_bool!(
        Operator::Eq = (== inverted !=),
        Operator::NotEq = (!= inverted ==),
        Operator::Lt = (< inverted >=),
        Operator::LtEq = (<= inverted >),
        Operator::Gt = (> inverted <=),
        Operator::GtEq = (>= inverted <),
    );
    compare_op_scalar!(
        array,
        list,
        fun,
        BooleanArray,
        list_from_scalar,
        value_from_scalar
    )
}

fn wrap_option_bool(v: &BooleanArray, all: bool) -> Option<bool> {
    if v.null_count() > 0 {
        return None;
    }
    Some(all)
}

fn compare_utf8<OffsetSize: StringOffsetSizeTrait>(
    array: &GenericStringArray<OffsetSize>,
    list: &ListArray,
    list_from_scalar: bool,
    value_from_scalar: bool,
    op: Operator,
    all: bool,
) -> Result<BooleanArray> {
    macro_rules! comparator_utf8 {
        ($($OP:pat = ($FN:tt inverted $IFN:tt),)*) => {
            match op {
                $(
                    $OP => if all {
                        |x, v: &GenericStringArray<OffsetSize>| {
                            for i in 0..v.len() {
                                if !v.is_null(i) && x $IFN v.value(i) {
                                    return Some(false)
                                }
                            }
                            wrap_option_utf8(v, true)
                        }
                    } else {
                        |x, v: &GenericStringArray<OffsetSize>| {
                            for i in 0..v.len() {
                                if !v.is_null(i) && x $FN v.value(i) {
                                    return Some(true)
                                }
                            }
                            wrap_option_utf8(v, false)
                        }
                    }
                )*
                op => return unsupported_op(op),
            }
        };
    }
    let fun = comparator_utf8!(
        Operator::Eq = (== inverted !=),
        Operator::NotEq = (!= inverted ==),
        Operator::Lt = (< inverted >=),
        Operator::LtEq = (<= inverted >),
        Operator::Gt = (> inverted <=),
        Operator::GtEq = (>= inverted <),
    );
    compare_op_scalar!(
        array,
        list,
        fun,
        GenericStringArray<OffsetSize>,
        list_from_scalar,
        value_from_scalar
    )
}

fn wrap_option_utf8<OffsetSize: StringOffsetSizeTrait>(
    v: &GenericStringArray<OffsetSize>,
    all: bool,
) -> Option<bool> {
    if v.null_count() > 0 {
        return None;
    }
    Some(all)
}

fn unsupported_op<T>(op: Operator) -> Result<T> {
    Err(DataFusionError::Execution(format!(
        "ANY/ALL does not support operator {}",
        op
    )))
}

/// AnyExpr
#[derive(Debug)]
pub struct AnyExpr {
    value: Arc<dyn PhysicalExpr>,
    op: Operator,
    list: Arc<dyn PhysicalExpr>,
    all: bool,
}

impl AnyExpr {
    /// Create a new InList expression
    pub fn new(
        value: Arc<dyn PhysicalExpr>,
        op: Operator,
        list: Arc<dyn PhysicalExpr>,
        all: bool,
    ) -> Self {
        Self {
            value,
            op,
            list,
            all,
        }
    }

    /// Compare for specific utf8 types
    fn compare_utf8<T: StringOffsetSizeTrait>(
        &self,
        array: ArrayRef,
        list: &ListArray,
        list_from_scalar: bool,
        value_from_scalar: bool,
        op: Operator,
        all: bool,
    ) -> Result<ColumnarValue> {
        let array = array
            .as_any()
            .downcast_ref::<GenericStringArray<T>>()
            .unwrap();

        Ok(ColumnarValue::Array(Arc::new(compare_utf8(
            array,
            list,
            list_from_scalar,
            value_from_scalar,
            op,
            all,
        )?)))
    }

    /// Compare for specific utf8 types
    fn compare_bool(
        &self,
        array: ArrayRef,
        list: &ListArray,
        list_from_scalar: bool,
        value_from_scalar: bool,
        op: Operator,
        all: bool,
    ) -> Result<ColumnarValue> {
        let array = array.as_any().downcast_ref::<BooleanArray>().unwrap();

        Ok(ColumnarValue::Array(Arc::new(compare_bool(
            array,
            list,
            list_from_scalar,
            value_from_scalar,
            op,
            all,
        )?)))
    }
}

impl std::fmt::Display for AnyExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let keyword = if self.all { "ALL" } else { "ANY" };
        write!(f, "{} {} {}({})", self.value, self.op, keyword, self.list)
    }
}

impl PhysicalExpr for AnyExpr {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, _: &Schema) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn nullable(&self, _: &Schema) -> Result<bool> {
        Ok(true)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let (value, value_from_scalar) = match self.value.evaluate(batch)? {
            ColumnarValue::Array(array) => (array, false),
            ColumnarValue::Scalar(scalar) => (scalar.to_array(), true),
        };

        let (list, list_from_scalar) = match self.list.evaluate(batch)? {
            ColumnarValue::Array(array) => (array, false),
            ColumnarValue::Scalar(scalar) => (scalar.to_array(), true),
        };
        let as_list = list
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("Unable to downcast list to ListArray");

        match value.data_type() {
            DataType::Float16 => {
                make_primitive!(
                    value,
                    as_list,
                    self.op,
                    Float16Array,
                    list_from_scalar,
                    value_from_scalar,
                    self.all
                )
            }
            DataType::Float32 => {
                make_primitive!(
                    value,
                    as_list,
                    self.op,
                    Float32Array,
                    list_from_scalar,
                    value_from_scalar,
                    self.all
                )
            }
            DataType::Float64 => {
                make_primitive!(
                    value,
                    as_list,
                    self.op,
                    Float64Array,
                    list_from_scalar,
                    value_from_scalar,
                    self.all
                )
            }
            DataType::Int8 => {
                make_primitive!(
                    value,
                    as_list,
                    self.op,
                    Int8Array,
                    list_from_scalar,
                    value_from_scalar,
                    self.all
                )
            }
            DataType::Int16 => {
                make_primitive!(
                    value,
                    as_list,
                    self.op,
                    Int16Array,
                    list_from_scalar,
                    value_from_scalar,
                    self.all
                )
            }
            DataType::Int32 => {
                make_primitive!(
                    value,
                    as_list,
                    self.op,
                    Int32Array,
                    list_from_scalar,
                    value_from_scalar,
                    self.all
                )
            }
            DataType::Int64 => {
                make_primitive!(
                    value,
                    as_list,
                    self.op,
                    Int64Array,
                    list_from_scalar,
                    value_from_scalar,
                    self.all
                )
            }
            DataType::UInt8 => {
                make_primitive!(
                    value,
                    as_list,
                    self.op,
                    UInt8Array,
                    list_from_scalar,
                    value_from_scalar,
                    self.all
                )
            }
            DataType::UInt16 => {
                make_primitive!(
                    value,
                    as_list,
                    self.op,
                    UInt16Array,
                    list_from_scalar,
                    value_from_scalar,
                    self.all
                )
            }
            DataType::UInt32 => {
                make_primitive!(
                    value,
                    as_list,
                    self.op,
                    UInt32Array,
                    list_from_scalar,
                    value_from_scalar,
                    self.all
                )
            }
            DataType::UInt64 => {
                make_primitive!(
                    value,
                    as_list,
                    self.op,
                    UInt64Array,
                    list_from_scalar,
                    value_from_scalar,
                    self.all
                )
            }
            DataType::Boolean => self.compare_bool(
                value,
                as_list,
                list_from_scalar,
                value_from_scalar,
                self.op,
                self.all,
            ),
            DataType::Utf8 => self.compare_utf8::<i32>(
                value,
                as_list,
                list_from_scalar,
                value_from_scalar,
                self.op,
                self.all,
            ),
            DataType::LargeUtf8 => self.compare_utf8::<i64>(
                value,
                as_list,
                list_from_scalar,
                value_from_scalar,
                self.op,
                self.all,
            ),
            datatype => Result::Err(DataFusionError::NotImplemented(format!(
                "AnyExpr does not support datatype {:?}.",
                datatype
            ))),
        }
    }
}

/// return two physical expressions that are optionally coerced to a
/// common type that the binary operator supports.
fn any_cast(
    value: Arc<dyn PhysicalExpr>,
    _op: &Operator,
    list: Arc<dyn PhysicalExpr>,
    input_schema: &Schema,
) -> Result<(Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>)> {
    let tmp = list.data_type(input_schema)?;
    let list_type = match &tmp {
        DataType::List(f) => f.data_type(),
        _ => {
            return Err(DataFusionError::NotImplemented(
                "ANY/ALL supports only literal arrays or subqueries".to_string(),
            ))
        }
    };

    Ok((try_cast(value, input_schema, list_type.clone())?, list))
}

/// Creates an expression AnyExpr
pub fn any(
    value: Arc<dyn PhysicalExpr>,
    op: Operator,
    list: Arc<dyn PhysicalExpr>,
    all: bool,
    input_schema: &Schema,
) -> Result<Arc<dyn PhysicalExpr>> {
    let (l, r) = any_cast(value, &op, list, input_schema)?;
    Ok(Arc::new(AnyExpr::new(l, op, r, all)))
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::Field;

    use super::*;
    use crate::expressions::{col, lit};
    use datafusion_common::{Result, ScalarValue};

    // applies the any expr to an input batch
    macro_rules! execute_any {
        ($BATCH:expr, $OP:expr, $EXPECTED:expr, $COL_A:expr, $COL_B:expr, $ALL:expr, $SCHEMA:expr) => {{
            let expr = any($COL_A, $OP, $COL_B, $ALL, $SCHEMA).unwrap();
            let result = expr.evaluate(&$BATCH)?.into_array($BATCH.num_rows());
            let result = result
                .as_any()
                .downcast_ref::<BooleanArray>()
                .expect("failed to downcast to BooleanArray");
            let expected = &BooleanArray::from($EXPECTED);
            assert_eq!(expected, result);
        }};
    }

    #[test]
    fn any_int64_array_list() -> Result<()> {
        let field_a = Field::new("a", DataType::Int64, true);
        let field_b = Field::new(
            "b",
            DataType::List(Box::new(Field::new("item", DataType::Int64, true))),
            true,
        );

        let schema = Schema::new(vec![field_a, field_b]);
        let a = Int64Array::from(vec![Some(0), Some(3), None]);
        let col_a = col("a", &schema)?;

        let values_builder = Int64Builder::new(3 * 3);
        let mut builder = ListBuilder::new(values_builder);

        for _ in 0..3 {
            builder.values().append_value(0).unwrap();
            builder.values().append_value(1).unwrap();
            builder.values().append_value(2).unwrap();
            builder.append(true).unwrap();
        }

        let b = builder.finish();
        let col_b = col("b", &schema)?;

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(a), Arc::new(b)],
        )?;

        execute_any!(
            batch,
            Operator::Eq,
            vec![Some(true), Some(false), None],
            col_a.clone(),
            col_b.clone(),
            false,
            &schema
        );

        Ok(())
    }

    // applies the any expr to an input batch and list
    macro_rules! execute_any_with_list {
        ($BATCH:expr, $LIST:expr, $OP:expr, $EXPECTED:expr, $COL:expr, $ALL:expr, $SCHEMA:expr) => {{
            let expr = any($COL, $OP, $LIST, $ALL, $SCHEMA).unwrap();
            let result = expr.evaluate(&$BATCH)?.into_array($BATCH.num_rows());
            let result = result
                .as_any()
                .downcast_ref::<BooleanArray>()
                .expect("failed to downcast to BooleanArray");
            let expected = &BooleanArray::from($EXPECTED);
            assert_eq!(expected, result);
        }};
    }

    #[test]
    fn any_int64_scalar_list() -> Result<()> {
        let field_a = Field::new("a", DataType::Int64, true);
        let schema = Schema::new(vec![field_a.clone()]);
        let a = Int64Array::from(vec![Some(0), Some(3), None]);
        let col_a = col("a", &schema)?;
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(a)])?;

        // expression: "a = ANY (0, 1, 2)"
        let list = lit(ScalarValue::List(
            Some(Box::new(vec![
                ScalarValue::Int64(Some(0)),
                ScalarValue::Int64(Some(1)),
                ScalarValue::Int64(Some(2)),
            ])),
            Box::new(DataType::Int64),
        ));

        let schema = &Schema::new(vec![
            field_a,
            Field::new(
                "b",
                DataType::List(Box::new(Field::new("d", DataType::Int64, true))),
                true,
            ),
        ]);
        execute_any_with_list!(
            batch,
            list,
            Operator::Eq,
            vec![Some(true), Some(false), None],
            col_a.clone(),
            false,
            schema
        );

        Ok(())
    }

    #[test]
    fn any_utf8_scalar_list() -> Result<()> {
        let field_a = Field::new("a", DataType::Utf8, true);
        let schema = Schema::new(vec![field_a.clone()]);
        let a = StringArray::from(vec![Some("a"), Some("d"), None]);
        let col_a = col("a", &schema)?;
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(a)])?;

        // expression: "a = ANY ('a', 'b', 'c')"
        let list = lit(ScalarValue::List(
            Some(Box::new(vec![
                ScalarValue::Utf8(Some("a".to_string())),
                ScalarValue::Utf8(Some("b".to_string())),
                ScalarValue::Utf8(Some("c".to_string())),
            ])),
            Box::new(DataType::Utf8),
        ));

        let schema = &Schema::new(vec![
            field_a,
            Field::new(
                "b",
                DataType::List(Box::new(Field::new("d", DataType::Utf8, true))),
                true,
            ),
        ]);
        execute_any_with_list!(
            batch,
            list,
            Operator::Eq,
            vec![Some(true), Some(false), None],
            col_a.clone(),
            false,
            schema
        );

        Ok(())
    }

    #[test]
    fn any_bool_scalar_list() -> Result<()> {
        let field_a = Field::new("a", DataType::Boolean, true);
        let schema = Schema::new(vec![field_a.clone()]);
        let a = BooleanArray::from(vec![Some(true), Some(false), None]);
        let col_a = col("a", &schema)?;
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(a)])?;

        // expression: "a = ANY (true)"
        let list = lit(ScalarValue::List(
            Some(Box::new(vec![ScalarValue::Boolean(Some(true))])),
            Box::new(DataType::Boolean),
        ));

        let schema = &Schema::new(vec![
            field_a,
            Field::new(
                "b",
                DataType::List(Box::new(Field::new("d", DataType::Boolean, true))),
                true,
            ),
        ]);
        execute_any_with_list!(
            batch,
            list,
            Operator::Eq,
            vec![Some(true), Some(false), None],
            col_a.clone(),
            false,
            schema
        );

        Ok(())
    }
}
