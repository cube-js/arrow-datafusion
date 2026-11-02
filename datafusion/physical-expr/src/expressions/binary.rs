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

use std::convert::TryInto;
use std::{any::Any, sync::Arc};

use arrow::array::*;
use arrow::compute::kernels::arithmetic::powf;
use arrow::compute::kernels::boolean::{and_kleene, not, or_kleene};
use arrow::compute::kernels::cmp;
use arrow::compute::kernels::numeric;
use arrow::compute::kernels::regexp::{regexp_is_match, regexp_is_match_scalar};
use arrow::datatypes::{
    ArrowPrimitiveType, DataType, Decimal128Type, Float32Type, Float64Type,
    IntervalDayTimeType, IntervalMonthDayNanoType, IntervalUnit, Schema,
    MAX_DECIMAL128_FOR_EACH_PRECISION, MIN_DECIMAL128_FOR_EACH_PRECISION,
};
use arrow::error::ArrowError;
use arrow::error::ArrowError::DivideByZero;
use arrow::record_batch::RecordBatch;

use super::binary_distinct::{
    coerce_types_distinct, distinct_types_allowed, evaluate_distinct_with_resolved_args,
};
use super::like::{like_arrays, like_scalar};
use crate::expressions::try_cast;
use crate::string_expressions;
use crate::PhysicalExpr;
use datafusion_common::ScalarValue;
use datafusion_common::{
    interval_day_time_to_i64, interval_month_day_nano_to_i128, DataFusionError, Result,
};
use datafusion_expr::binary_rule::binary_operator_data_type;
use datafusion_expr::{binary_rule::coerce_types, ColumnarValue, Operator};

// TODO move to arrow_rs
// https://github.com/apache/arrow-rs/issues/1312
fn as_decimal_array(arr: &dyn Array) -> &Decimal128Array {
    arr.as_any()
        .downcast_ref::<Decimal128Array>()
        .expect("Unable to downcast to typed array to DecimalArray")
}

/// Comparison operators supported by [`compare_datum`]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ComparisonOp {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    IsDistinctFrom,
    IsNotDistinctFrom,
}

impl ComparisonOp {
    /// Returns the comparison operator matching `op`, if `op` is a comparison
    fn from_operator(op: &Operator) -> Option<Self> {
        Some(match op {
            Operator::Eq => ComparisonOp::Eq,
            Operator::NotEq => ComparisonOp::NotEq,
            Operator::Lt => ComparisonOp::Lt,
            Operator::LtEq => ComparisonOp::LtEq,
            Operator::Gt => ComparisonOp::Gt,
            Operator::GtEq => ComparisonOp::GtEq,
            Operator::IsDistinctFrom => ComparisonOp::IsDistinctFrom,
            Operator::IsNotDistinctFrom => ComparisonOp::IsNotDistinctFrom,
            _ => return None,
        })
    }

    /// Compares two non-null values
    fn compare<K: PartialOrd>(self, left: &K, right: &K) -> bool {
        match self {
            ComparisonOp::Eq | ComparisonOp::IsNotDistinctFrom => left == right,
            ComparisonOp::NotEq | ComparisonOp::IsDistinctFrom => left != right,
            ComparisonOp::Lt => left < right,
            ComparisonOp::LtEq => left <= right,
            ComparisonOp::Gt => left > right,
            ComparisonOp::GtEq => left >= right,
        }
    }
}

/// Compares two primitive arrays (each side may be a scalar, i.e. an array of length 1
/// whose value is broadcast) using `key` to map the native values before comparing them.
///
/// The result is null when either side is null, except for `IS [NOT] DISTINCT FROM`,
/// which never returns null.
fn compare_primitive<T, K, F>(
    op: ComparisonOp,
    left: &PrimitiveArray<T>,
    left_is_scalar: bool,
    right: &PrimitiveArray<T>,
    right_is_scalar: bool,
    key: F,
) -> Result<BooleanArray>
where
    T: ArrowPrimitiveType,
    K: PartialOrd,
    F: Fn(T::Native) -> K,
{
    if !left_is_scalar && !right_is_scalar && left.len() != right.len() {
        return Err(DataFusionError::ArrowError(ArrowError::ComputeError(
            "Cannot perform comparison operation on arrays of different length"
                .to_string(),
        )));
    }
    let len = if left_is_scalar {
        right.len()
    } else {
        left.len()
    };
    let value = |array: &PrimitiveArray<T>, is_scalar: bool, i: usize| {
        let i = if is_scalar { 0 } else { i };
        if array.is_valid(i) {
            Some(key(array.value(i)))
        } else {
            None
        }
    };
    Ok((0..len)
        .map(|i| {
            let l = value(left, left_is_scalar, i);
            let r = value(right, right_is_scalar, i);
            match op {
                ComparisonOp::IsDistinctFrom => Some(l != r),
                ComparisonOp::IsNotDistinctFrom => Some(l == r),
                _ => match (l, r) {
                    (Some(l), Some(r)) => Some(op.compare(&l, &r)),
                    _ => None,
                },
            }
        })
        .collect())
}

/// Evaluates the comparison `op` on two [`Datum`]s (arrays or scalars).
///
/// Floats are compared with IEEE 754 semantics (`NaN != NaN`, `-0.0 == 0.0`),
/// decimals by their raw (unscaled) values and day-time / month-day-nano intervals by
/// their packed integer representation, like the arrow 13 kernels did. Every other type
/// is delegated to the arrow `cmp` kernels.
pub(crate) fn compare_datum(
    op: ComparisonOp,
    left: &dyn Datum,
    right: &dyn Datum,
) -> Result<BooleanArray> {
    let (l, l_s) = left.get();
    let (r, r_s) = right.get();
    match (l.data_type(), r.data_type()) {
        (DataType::Float32, DataType::Float32) => compare_primitive(
            op,
            l.as_primitive::<Float32Type>(),
            l_s,
            r.as_primitive::<Float32Type>(),
            r_s,
            |v| v,
        ),
        (DataType::Float64, DataType::Float64) => compare_primitive(
            op,
            l.as_primitive::<Float64Type>(),
            l_s,
            r.as_primitive::<Float64Type>(),
            r_s,
            |v| v,
        ),
        (DataType::Decimal128(_, _), DataType::Decimal128(_, _)) => compare_primitive(
            op,
            l.as_primitive::<Decimal128Type>(),
            l_s,
            r.as_primitive::<Decimal128Type>(),
            r_s,
            |v| v,
        ),
        (
            DataType::Interval(IntervalUnit::DayTime),
            DataType::Interval(IntervalUnit::DayTime),
        ) => compare_primitive(
            op,
            l.as_primitive::<IntervalDayTimeType>(),
            l_s,
            r.as_primitive::<IntervalDayTimeType>(),
            r_s,
            interval_day_time_to_i64,
        ),
        (
            DataType::Interval(IntervalUnit::MonthDayNano),
            DataType::Interval(IntervalUnit::MonthDayNano),
        ) => compare_primitive(
            op,
            l.as_primitive::<IntervalMonthDayNanoType>(),
            l_s,
            r.as_primitive::<IntervalMonthDayNanoType>(),
            r_s,
            interval_month_day_nano_to_i128,
        ),
        _ => {
            let result = match op {
                ComparisonOp::Eq => cmp::eq(left, right),
                ComparisonOp::NotEq => cmp::neq(left, right),
                ComparisonOp::Lt => cmp::lt(left, right),
                ComparisonOp::LtEq => cmp::lt_eq(left, right),
                ComparisonOp::Gt => cmp::gt(left, right),
                ComparisonOp::GtEq => cmp::gt_eq(left, right),
                ComparisonOp::IsDistinctFrom => cmp::distinct(left, right),
                ComparisonOp::IsNotDistinctFrom => cmp::not_distinct(left, right),
            }?;
            Ok(result)
        }
    }
}

/// Evaluates the comparison `op` on an array and a scalar value
pub(crate) fn compare_array_scalar(
    op: ComparisonOp,
    array: &dyn Array,
    scalar: &ScalarValue,
) -> Result<BooleanArray> {
    let scalar = Scalar::new(scalar.to_array_of_size(1));
    compare_datum(op, &array, &scalar)
}

/// Evaluates the comparison `op` on a scalar value and an array
fn compare_scalar_array(
    op: ComparisonOp,
    scalar: &ScalarValue,
    array: &dyn Array,
) -> Result<BooleanArray> {
    let scalar = Scalar::new(scalar.to_array_of_size(1));
    compare_datum(op, &scalar, &array)
}

// TODO move decimal kernels to to arrow-rs
// https://github.com/apache/arrow-rs/issues/1200

#[cfg(test)]
pub(super) fn eq_decimal_scalar(
    left: &Decimal128Array,
    right: i128,
) -> Result<BooleanArray> {
    compare_decimal_scalar(ComparisonOp::Eq, left, right)
}

#[cfg(test)]
pub(super) fn eq_decimal(
    left: &Decimal128Array,
    right: &Decimal128Array,
) -> Result<BooleanArray> {
    compare_decimal(ComparisonOp::Eq, left, right)
}

#[cfg(test)]
fn neq_decimal_scalar(left: &Decimal128Array, right: i128) -> Result<BooleanArray> {
    compare_decimal_scalar(ComparisonOp::NotEq, left, right)
}

#[cfg(test)]
fn neq_decimal(left: &Decimal128Array, right: &Decimal128Array) -> Result<BooleanArray> {
    compare_decimal(ComparisonOp::NotEq, left, right)
}

#[cfg(test)]
fn lt_decimal_scalar(left: &Decimal128Array, right: i128) -> Result<BooleanArray> {
    compare_decimal_scalar(ComparisonOp::Lt, left, right)
}

#[cfg(test)]
fn lt_decimal(left: &Decimal128Array, right: &Decimal128Array) -> Result<BooleanArray> {
    compare_decimal(ComparisonOp::Lt, left, right)
}

#[cfg(test)]
fn lt_eq_decimal_scalar(left: &Decimal128Array, right: i128) -> Result<BooleanArray> {
    compare_decimal_scalar(ComparisonOp::LtEq, left, right)
}

#[cfg(test)]
fn lt_eq_decimal(
    left: &Decimal128Array,
    right: &Decimal128Array,
) -> Result<BooleanArray> {
    compare_decimal(ComparisonOp::LtEq, left, right)
}

#[cfg(test)]
fn gt_decimal_scalar(left: &Decimal128Array, right: i128) -> Result<BooleanArray> {
    compare_decimal_scalar(ComparisonOp::Gt, left, right)
}

#[cfg(test)]
fn gt_decimal(left: &Decimal128Array, right: &Decimal128Array) -> Result<BooleanArray> {
    compare_decimal(ComparisonOp::Gt, left, right)
}

#[cfg(test)]
fn gt_eq_decimal_scalar(left: &Decimal128Array, right: i128) -> Result<BooleanArray> {
    compare_decimal_scalar(ComparisonOp::GtEq, left, right)
}

#[cfg(test)]
fn gt_eq_decimal(
    left: &Decimal128Array,
    right: &Decimal128Array,
) -> Result<BooleanArray> {
    compare_decimal(ComparisonOp::GtEq, left, right)
}

#[cfg(test)]
fn is_distinct_from_decimal(
    left: &Decimal128Array,
    right: &Decimal128Array,
) -> Result<BooleanArray> {
    compare_decimal(ComparisonOp::IsDistinctFrom, left, right)
}

#[cfg(test)]
fn is_not_distinct_from_decimal(
    left: &Decimal128Array,
    right: &Decimal128Array,
) -> Result<BooleanArray> {
    compare_decimal(ComparisonOp::IsNotDistinctFrom, left, right)
}

#[cfg(test)]
fn compare_decimal_scalar(
    op: ComparisonOp,
    left: &Decimal128Array,
    right: i128,
) -> Result<BooleanArray> {
    let right = Decimal128Array::from(vec![right]);
    compare_primitive(op, left, false, &right, true, |v| v)
}

#[cfg(test)]
fn compare_decimal(
    op: ComparisonOp,
    left: &Decimal128Array,
    right: &Decimal128Array,
) -> Result<BooleanArray> {
    compare_primitive(op, left, false, right, false, |v| v)
}

/// Builds the result of a decimal arithmetic operation, validating every value against
/// the precision of the result like arrow 13's `DecimalBuilder` did.
struct DecimalResultBuilder {
    builder: Decimal128Builder,
    precision: u8,
    scale: i8,
}

impl DecimalResultBuilder {
    fn new(capacity: usize, precision: u8, scale: i8) -> Self {
        Self {
            builder: Decimal128Builder::with_capacity(capacity),
            precision,
            scale,
        }
    }

    fn append_value(&mut self, value: i128) -> Result<()> {
        let precision = self.precision as usize;
        let max = MAX_DECIMAL128_FOR_EACH_PRECISION[precision];
        let min = MIN_DECIMAL128_FOR_EACH_PRECISION[precision];
        if value > max {
            return Err(DataFusionError::ArrowError(
                ArrowError::InvalidArgumentError(format!(
                    "{} is too large to store in a Decimal of precision {}. Max is {}",
                    value, precision, max
                )),
            ));
        }
        if value < min {
            return Err(DataFusionError::ArrowError(
                ArrowError::InvalidArgumentError(format!(
                    "{} is too small to store in a Decimal of precision {}. Min is {}",
                    value, precision, min
                )),
            ));
        }
        self.builder.append_value(value);
        Ok(())
    }

    fn append_null(&mut self) -> Result<()> {
        self.builder.append_null();
        Ok(())
    }

    fn finish(&mut self) -> Result<Decimal128Array> {
        Ok(self
            .builder
            .finish()
            .with_precision_and_scale(self.precision, self.scale)?)
    }
}

fn add_decimal_scalar(left: &Decimal128Array, right: i128) -> Result<Decimal128Array> {
    let mut decimal_builder =
        DecimalResultBuilder::new(left.len(), left.precision(), left.scale());
    for i in 0..left.len() {
        if left.is_null(i) {
            decimal_builder.append_null()?;
        } else {
            decimal_builder.append_value(left.value(i) + right)?;
        }
    }
    decimal_builder.finish()
}

fn add_decimal(
    left: &Decimal128Array,
    right: &Decimal128Array,
) -> Result<Decimal128Array> {
    let mut decimal_builder =
        DecimalResultBuilder::new(left.len(), left.precision(), left.scale());
    for i in 0..left.len() {
        if left.is_null(i) || right.is_null(i) {
            decimal_builder.append_null()?;
        } else {
            decimal_builder.append_value(left.value(i) + right.value(i))?;
        }
    }
    decimal_builder.finish()
}

fn subtract_decimal_scalar(
    left: &Decimal128Array,
    right: i128,
) -> Result<Decimal128Array> {
    let mut decimal_builder =
        DecimalResultBuilder::new(left.len(), left.precision(), left.scale());
    for i in 0..left.len() {
        if left.is_null(i) {
            decimal_builder.append_null()?;
        } else {
            decimal_builder.append_value(left.value(i) - right)?;
        }
    }
    decimal_builder.finish()
}

fn subtract_decimal(
    left: &Decimal128Array,
    right: &Decimal128Array,
) -> Result<Decimal128Array> {
    let mut decimal_builder =
        DecimalResultBuilder::new(left.len(), left.precision(), left.scale());
    for i in 0..left.len() {
        if left.is_null(i) || right.is_null(i) {
            decimal_builder.append_null()?;
        } else {
            decimal_builder.append_value(left.value(i) - right.value(i))?;
        }
    }
    decimal_builder.finish()
}

fn multiply_decimal_scalar(
    left: &Decimal128Array,
    right: i128,
) -> Result<Decimal128Array> {
    let mut decimal_builder =
        DecimalResultBuilder::new(left.len(), left.precision(), left.scale());
    let divide = 10_i128.pow(left.scale() as u32);
    for i in 0..left.len() {
        if left.is_null(i) {
            decimal_builder.append_null()?;
        } else {
            decimal_builder.append_value(left.value(i) * right / divide)?;
        }
    }
    decimal_builder.finish()
}

fn multiply_decimal(
    left: &Decimal128Array,
    right: &Decimal128Array,
) -> Result<Decimal128Array> {
    let mut decimal_builder =
        DecimalResultBuilder::new(left.len(), left.precision(), left.scale());
    let divide = 10_i128.pow(left.scale() as u32);
    for i in 0..left.len() {
        if left.is_null(i) || right.is_null(i) {
            decimal_builder.append_null()?;
        } else {
            decimal_builder.append_value(left.value(i) * right.value(i) / divide)?;
        }
    }
    decimal_builder.finish()
}

fn divide_decimal_scalar(left: &Decimal128Array, right: i128) -> Result<Decimal128Array> {
    let mut decimal_builder =
        DecimalResultBuilder::new(left.len(), left.precision(), left.scale());
    let mul = 10_f64.powi(left.scale() as i32);
    for i in 0..left.len() {
        if left.is_null(i) {
            decimal_builder.append_null()?;
        } else if right == 0 {
            return Err(DataFusionError::ArrowError(DivideByZero));
        } else {
            let l_value = left.value(i) as f64;
            let result = ((l_value / right as f64) * mul) as i128;
            decimal_builder.append_value(result)?;
        }
    }
    decimal_builder.finish()
}

fn divide_decimal(
    left: &Decimal128Array,
    right: &Decimal128Array,
) -> Result<Decimal128Array> {
    let mut decimal_builder =
        DecimalResultBuilder::new(left.len(), left.precision(), left.scale());
    let mul = 10_f64.powi(left.scale() as i32);
    for i in 0..left.len() {
        if left.is_null(i) || right.is_null(i) {
            decimal_builder.append_null()?;
        } else if right.value(i) == 0 {
            return Err(DataFusionError::ArrowError(DivideByZero));
        } else {
            let l_value = left.value(i) as f64;
            let r_value = right.value(i) as f64;
            let result = ((l_value / r_value) * mul) as i128;
            decimal_builder.append_value(result)?;
        }
    }
    decimal_builder.finish()
}

fn modulus_decimal_scalar(
    left: &Decimal128Array,
    right: i128,
) -> Result<Decimal128Array> {
    let mut decimal_builder =
        DecimalResultBuilder::new(left.len(), left.precision(), left.scale());
    for i in 0..left.len() {
        if left.is_null(i) {
            decimal_builder.append_null()?;
        } else if right == 0 {
            return Err(DataFusionError::ArrowError(DivideByZero));
        } else {
            decimal_builder.append_value(left.value(i) % right)?;
        }
    }
    decimal_builder.finish()
}

fn modulus_decimal(
    left: &Decimal128Array,
    right: &Decimal128Array,
) -> Result<Decimal128Array> {
    let mut decimal_builder =
        DecimalResultBuilder::new(left.len(), left.precision(), left.scale());
    for i in 0..left.len() {
        if left.is_null(i) || right.is_null(i) {
            decimal_builder.append_null()?;
        } else if right.value(i) == 0 {
            return Err(DataFusionError::ArrowError(DivideByZero));
        } else {
            decimal_builder.append_value(left.value(i) % right.value(i))?;
        }
    }
    decimal_builder.finish()
}

/// Arithmetic operators evaluated by [`arithmetic_arrays`] / [`arithmetic_array_scalar`]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArithmeticOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulus,
}

impl ArithmeticOp {
    fn from_operator(op: &Operator) -> Option<Self> {
        Some(match op {
            Operator::Plus => ArithmeticOp::Add,
            Operator::Minus => ArithmeticOp::Subtract,
            Operator::Multiply => ArithmeticOp::Multiply,
            Operator::Divide => ArithmeticOp::Divide,
            Operator::Modulo => ArithmeticOp::Modulus,
            _ => return None,
        })
    }

    /// The name of the (arrow 13) kernel, used in error messages
    fn name(self) -> &'static str {
        match self {
            ArithmeticOp::Add => "add",
            ArithmeticOp::Subtract => "subtract",
            ArithmeticOp::Multiply => "multiply",
            ArithmeticOp::Divide => "divide",
            ArithmeticOp::Modulus => "modulus",
        }
    }

    fn is_division(self) -> bool {
        matches!(self, ArithmeticOp::Divide | ArithmeticOp::Modulus)
    }
}

/// Types supported by the primitive arithmetic kernels
fn is_primitive_arithmetic_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
    )
}

/// Invokes the arrow arithmetic kernel for `op`.
///
/// Integer addition, subtraction and multiplication wrap on overflow, division and modulo
/// return `DivideByZero` for a zero integer divisor.
fn arithmetic_kernel(
    op: ArithmeticOp,
    left: &dyn Datum,
    right: &dyn Datum,
) -> Result<ArrayRef> {
    let result = match op {
        ArithmeticOp::Add => numeric::add_wrapping(left, right),
        ArithmeticOp::Subtract => numeric::sub_wrapping(left, right),
        ArithmeticOp::Multiply => numeric::mul_wrapping(left, right),
        ArithmeticOp::Divide => numeric::div(left, right),
        ArithmeticOp::Modulus => numeric::rem(left, right),
    }?;
    Ok(result)
}

/// Returns `DivideByZero` if a valid value of `right` is zero where `left` is valid too
fn check_float_divisor<T: ArrowPrimitiveType>(
    left: &dyn Array,
    right: &dyn Array,
) -> Result<()> {
    let right = right.as_primitive::<T>();
    let zero = T::Native::default();
    let has_zero_divisor = (0..right.len())
        .any(|i| left.is_valid(i) && right.is_valid(i) && right.value(i) == zero);
    if has_zero_divisor {
        Err(DataFusionError::ArrowError(DivideByZero))
    } else {
        Ok(())
    }
}

/// Returns true if `scalar` is a non-null zero number
fn scalar_is_zero(scalar: &ScalarValue) -> bool {
    match scalar {
        ScalarValue::Int8(Some(v)) => *v == 0,
        ScalarValue::Int16(Some(v)) => *v == 0,
        ScalarValue::Int32(Some(v)) => *v == 0,
        ScalarValue::Int64(Some(v)) => *v == 0,
        ScalarValue::UInt8(Some(v)) => *v == 0,
        ScalarValue::UInt16(Some(v)) => *v == 0,
        ScalarValue::UInt32(Some(v)) => *v == 0,
        ScalarValue::UInt64(Some(v)) => *v == 0,
        ScalarValue::Float32(Some(v)) => *v == 0.0,
        ScalarValue::Float64(Some(v)) => *v == 0.0,
        _ => false,
    }
}

/// Evaluates the arithmetic `op` on two arrays of the same type
fn arithmetic_arrays(
    op: ArithmeticOp,
    left: &ArrayRef,
    right: &ArrayRef,
) -> Result<ArrayRef> {
    match left.data_type() {
        // TODO support decimal type
        // which is not the primitive type
        DataType::Decimal128(_, _) => {
            let left = as_decimal_array(left);
            let right = as_decimal_array(right);
            let result = match op {
                ArithmeticOp::Add => add_decimal(left, right),
                ArithmeticOp::Subtract => subtract_decimal(left, right),
                ArithmeticOp::Multiply => multiply_decimal(left, right),
                ArithmeticOp::Divide => divide_decimal(left, right),
                ArithmeticOp::Modulus => modulus_decimal(left, right),
            }?;
            Ok(Arc::new(result))
        }
        data_type if is_primitive_arithmetic_type(data_type) => {
            // arrow 13 returned `DivideByZero` for floats as well
            if op.is_division() {
                match data_type {
                    DataType::Float32 => {
                        check_float_divisor::<Float32Type>(left.as_ref(), right.as_ref())?
                    }
                    DataType::Float64 => {
                        check_float_divisor::<Float64Type>(left.as_ref(), right.as_ref())?
                    }
                    _ => {}
                }
            }
            arithmetic_kernel(op, left, right)
        }
        other => Err(DataFusionError::Internal(format!(
            "Data type {:?} not supported for binary operation '{}' on primitive arrays",
            other,
            op.name()
        ))),
    }
}

/// Evaluates the arithmetic `op` on an array and a scalar of the same type.
/// Types without a scalar kernel yield `None`, so the caller falls back
/// to the array implementation.
fn arithmetic_array_scalar(
    op: ArithmeticOp,
    array: &dyn Array,
    scalar: &ScalarValue,
) -> Option<Result<ArrayRef>> {
    match array.data_type() {
        // TODO support decimal type
        // which is not the primitive type
        DataType::Decimal128(_, _) => {
            Some(decimal_arithmetic_array_scalar(op, array, scalar))
        }
        data_type if is_primitive_arithmetic_type(data_type) => {
            // arrow 13 returned `DivideByZero` for a zero divisor even if the array is empty
            if op.is_division() && scalar_is_zero(scalar) {
                return Some(Err(DataFusionError::ArrowError(DivideByZero)));
            }
            let scalar = Scalar::new(scalar.to_array_of_size(1));
            Some(arithmetic_kernel(op, &array, &scalar))
        }
        // e.g. `Timestamp - Timestamp`, which only the array implementation supports
        _ => None,
    }
}

fn decimal_arithmetic_array_scalar(
    op: ArithmeticOp,
    array: &dyn Array,
    scalar: &ScalarValue,
) -> Result<ArrayRef> {
    if scalar.is_null() {
        return Ok(new_null_array(array.data_type(), array.len()));
    }
    let left = as_decimal_array(array);
    let right: i128 = scalar.clone().try_into()?;
    let result = match op {
        ArithmeticOp::Add => add_decimal_scalar(left, right),
        ArithmeticOp::Subtract => subtract_decimal_scalar(left, right),
        ArithmeticOp::Multiply => multiply_decimal_scalar(left, right),
        ArithmeticOp::Divide => divide_decimal_scalar(left, right),
        ArithmeticOp::Modulus => modulus_decimal_scalar(left, right),
    }?;
    Ok(Arc::new(result))
}

/// `LIKE` family operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LikeOp {
    Like,
    NotLike,
    ILike,
    NotILike,
}

impl LikeOp {
    fn from_operator(op: &Operator) -> Option<Self> {
        Some(match op {
            Operator::Like => LikeOp::Like,
            Operator::NotLike => LikeOp::NotLike,
            Operator::ILike => LikeOp::ILike,
            Operator::NotILike => LikeOp::NotILike,
            _ => return None,
        })
    }
}

impl LikeOp {
    fn case_insensitive(self) -> bool {
        matches!(self, LikeOp::ILike | LikeOp::NotILike)
    }

    fn negated(self) -> bool {
        matches!(self, LikeOp::NotLike | LikeOp::NotILike)
    }

    fn name(self) -> &'static str {
        match self {
            LikeOp::Like => "like",
            LikeOp::NotLike => "nlike",
            LikeOp::ILike => "ilike",
            LikeOp::NotILike => "nilike",
        }
    }
}

/// Evaluates the `LIKE` family operator `op` on a string array and a scalar pattern,
/// with the semantics of arrow 13 (see [`super::like`]). A NULL pattern yields NULL.
fn like_array_scalar(
    op: LikeOp,
    array: &dyn Array,
    pattern: &ScalarValue,
) -> Result<ArrayRef> {
    let (ci, negated) = (op.case_insensitive(), op.negated());
    let result = match (array.data_type(), pattern) {
        (DataType::Utf8, ScalarValue::Utf8(Some(pattern))) => {
            like_scalar(array.as_string::<i32>(), pattern, ci, negated)?
        }
        (DataType::LargeUtf8, ScalarValue::LargeUtf8(Some(pattern))) => {
            like_scalar(array.as_string::<i64>(), pattern, ci, negated)?
        }
        (
            DataType::Utf8 | DataType::LargeUtf8,
            ScalarValue::Utf8(None) | ScalarValue::LargeUtf8(None),
        ) => BooleanArray::new_null(array.len()),
        (other, _) => {
            return Err(DataFusionError::Internal(format!(
                "Data type {:?} not supported for scalar operation '{}' on string array",
                other,
                op.name()
            )))
        }
    };
    Ok(Arc::new(result))
}

/// Evaluates the `LIKE` family operator `op` on two string arrays, with the semantics of
/// arrow 13 (see [`super::like`])
fn like_array_array(op: LikeOp, left: &dyn Array, right: &dyn Array) -> Result<ArrayRef> {
    let (ci, negated) = (op.case_insensitive(), op.negated());
    let result = match left.data_type() {
        DataType::Utf8 => like_arrays(
            left.as_string::<i32>(),
            right.as_string::<i32>(),
            ci,
            negated,
        )?,
        DataType::LargeUtf8 => like_arrays(
            left.as_string::<i64>(),
            right.as_string::<i64>(),
            ci,
            negated,
        )?,
        other => {
            return Err(DataFusionError::Internal(format!(
                "Data type {:?} not supported for binary operation '{}' on string arrays",
                other,
                op.name()
            )))
        }
    };
    Ok(Arc::new(result))
}

/// Evaluates a regular expression match (`~`, `~*`, `!~`, `!~*`) on two string arrays
fn regex_match_arrays(
    left: &dyn Array,
    right: &dyn Array,
    negated: bool,
    case_insensitive: bool,
) -> Result<ArrayRef> {
    let result = match left.data_type() {
        DataType::Utf8 => {
            let left = left.as_string::<i32>();
            let right = right.as_string::<i32>();
            let flags = case_insensitive.then(|| StringArray::from(vec!["i"; left.len()]));
            regexp_is_match(left, right, flags.as_ref())?
        }
        DataType::LargeUtf8 => {
            let left = left.as_string::<i64>();
            let right = right.as_string::<i64>();
            let flags = case_insensitive
                .then(|| LargeStringArray::from(vec!["i"; left.len()]));
            regexp_is_match(left, right, flags.as_ref())?
        }
        other => {
            return Err(DataFusionError::Internal(format!(
                "Data type {:?} not supported for binary_string_array_flag_op operation '{}' on string array",
                other, "regexp_is_match"
            )))
        }
    };
    let result = if negated { not(&result)? } else { result };
    Ok(Arc::new(result))
}

/// Evaluates a regular expression match (`~`, `~*`, `!~`, `!~*`) on a string array
/// and a scalar pattern
fn regex_match_array_scalar(
    array: &dyn Array,
    scalar: &ScalarValue,
    negated: bool,
    case_insensitive: bool,
) -> Result<ArrayRef> {
    let pattern = match scalar {
        ScalarValue::Utf8(pattern) | ScalarValue::LargeUtf8(pattern) => pattern,
        other => {
            return Err(DataFusionError::Internal(format!(
                "compute_utf8_flag_op_scalar failed to cast literal value {} for operation '{}'",
                other, "regexp_is_match"
            )))
        }
    };
    let pattern = match pattern {
        Some(pattern) => pattern,
        // a NULL pattern never matches
        None => return Ok(new_null_array(&DataType::Boolean, array.len())),
    };
    let flag = if case_insensitive { Some("i") } else { None };
    let result = match array.data_type() {
        DataType::Utf8 => {
            regexp_is_match_scalar(array.as_string::<i32>(), pattern, flag)?
        }
        DataType::LargeUtf8 => {
            regexp_is_match_scalar(array.as_string::<i64>(), pattern, flag)?
        }
        other => {
            return Err(DataFusionError::Internal(format!(
                "Data type {:?} not supported for binary_string_array_flag_op_scalar operation '{}' on string array",
                other, "regexp_is_match"
            )))
        }
    };
    let result = if negated { not(&result)? } else { result };
    Ok(Arc::new(result))
}

/// The binary_bitwise_array_op macro only evaluates for integer types
/// like int64, int32.
/// It is used to do bitwise operation.
macro_rules! binary_bitwise_array_op {
    ($LEFT:expr, $RIGHT:expr, $METHOD:expr, $ARRAY_TYPE:ident) => {{
        let len = $LEFT.len();
        let left = $LEFT.as_any().downcast_ref::<$ARRAY_TYPE>().unwrap();
        let right = $RIGHT.as_any().downcast_ref::<$ARRAY_TYPE>().unwrap();
        let result = (0..len)
            .into_iter()
            .map(|i| {
                if left.is_null(i) || right.is_null(i) {
                    None
                } else {
                    Some($METHOD(left.value(i), right.value(i)))
                }
            })
            .collect::<$ARRAY_TYPE>();
        Ok(Arc::new(result))
    }};
}

/// The binary_bitwise_array_op macro only evaluates for integer types
/// like int64, int32.
/// It is used to do bitwise operation on an array with a scalar.
macro_rules! binary_bitwise_array_scalar {
    ($LEFT:expr, $RIGHT:expr, $METHOD:expr, $ARRAY_TYPE:ident, $TYPE:ty) => {{
        let len = $LEFT.len();
        let array = $LEFT.as_any().downcast_ref::<$ARRAY_TYPE>().unwrap();
        let scalar = $RIGHT;
        if scalar.is_null() {
            Ok(new_null_array(array.data_type(), len))
        } else {
            let right: $TYPE = scalar.try_into().unwrap();
            let result = (0..len)
                .into_iter()
                .map(|i| {
                    if array.is_null(i) {
                        None
                    } else {
                        Some($METHOD(array.value(i), right))
                    }
                })
                .collect::<$ARRAY_TYPE>();
            Ok(Arc::new(result) as ArrayRef)
        }
    }};
}

fn bitwise_and(left: ArrayRef, right: ArrayRef) -> Result<ArrayRef> {
    match &left.data_type() {
        DataType::Int8 => {
            binary_bitwise_array_op!(left, right, |a, b| a & b, Int8Array)
        }
        DataType::Int16 => {
            binary_bitwise_array_op!(left, right, |a, b| a & b, Int16Array)
        }
        DataType::Int32 => {
            binary_bitwise_array_op!(left, right, |a, b| a & b, Int32Array)
        }
        DataType::Int64 => {
            binary_bitwise_array_op!(left, right, |a, b| a & b, Int64Array)
        }
        other => Err(DataFusionError::Internal(format!(
            "Data type {:?} not supported for binary operation '{}' on dyn arrays",
            other,
            Operator::BitwiseAnd
        ))),
    }
}

fn bitwise_or(left: ArrayRef, right: ArrayRef) -> Result<ArrayRef> {
    match &left.data_type() {
        DataType::Int8 => {
            binary_bitwise_array_op!(left, right, |a, b| a | b, Int8Array)
        }
        DataType::Int16 => {
            binary_bitwise_array_op!(left, right, |a, b| a | b, Int16Array)
        }
        DataType::Int32 => {
            binary_bitwise_array_op!(left, right, |a, b| a | b, Int32Array)
        }
        DataType::Int64 => {
            binary_bitwise_array_op!(left, right, |a, b| a | b, Int64Array)
        }
        other => Err(DataFusionError::Internal(format!(
            "Data type {:?} not supported for binary operation '{}' on dyn arrays",
            other,
            Operator::BitwiseOr
        ))),
    }
}

fn bitwise_shift_right(left: ArrayRef, right: ArrayRef) -> Result<ArrayRef> {
    match &left.data_type() {
        DataType::Int8 => {
            binary_bitwise_array_op!(
                left,
                right,
                |a: i8, b: i8| a.wrapping_shr(b as u32),
                Int8Array
            )
        }
        DataType::Int16 => {
            binary_bitwise_array_op!(
                left,
                right,
                |a: i16, b: i16| a.wrapping_shr(b as u32),
                Int16Array
            )
        }
        DataType::Int32 => {
            binary_bitwise_array_op!(
                left,
                right,
                |a: i32, b: i32| a.wrapping_shr(b as u32),
                Int32Array
            )
        }
        DataType::Int64 => {
            binary_bitwise_array_op!(
                left,
                right,
                |a: i64, b: i64| a.wrapping_shr(b as u32),
                Int64Array
            )
        }
        other => Err(DataFusionError::Internal(format!(
            "Data type {:?} not supported for binary operation '{}' on dyn arrays",
            other,
            Operator::BitwiseShiftRight
        ))),
    }
}

fn bitwise_shift_left(left: ArrayRef, right: ArrayRef) -> Result<ArrayRef> {
    match &left.data_type() {
        DataType::Int8 => {
            binary_bitwise_array_op!(
                left,
                right,
                |a: i8, b: i8| a.wrapping_shl(b as u32),
                Int8Array
            )
        }
        DataType::Int16 => {
            binary_bitwise_array_op!(
                left,
                right,
                |a: i16, b: i16| a.wrapping_shl(b as u32),
                Int16Array
            )
        }
        DataType::Int32 => {
            binary_bitwise_array_op!(
                left,
                right,
                |a: i32, b: i32| a.wrapping_shl(b as u32),
                Int32Array
            )
        }
        DataType::Int64 => {
            binary_bitwise_array_op!(
                left,
                right,
                |a: i64, b: i64| a.wrapping_shl(b as u32),
                Int64Array
            )
        }
        other => Err(DataFusionError::Internal(format!(
            "Data type {:?} not supported for binary operation '{}' on dyn arrays",
            other,
            Operator::BitwiseShiftLeft
        ))),
    }
}

/// Use datafusion build-in expression `concat` to evaluate `StringConcat` operator.
/// Besides, any `NULL` exists on lhs or rhs will come out result `NULL`
/// 1. 'a' || 'b' || 32 = 'ab32'
/// 2. 'a' || NULL = NULL
fn string_concat(left: ArrayRef, right: ArrayRef) -> Result<ArrayRef> {
    let ignore_null = match string_expressions::concat(&[
        ColumnarValue::Array(left.clone()),
        ColumnarValue::Array(right.clone()),
    ])? {
        ColumnarValue::Array(array_ref) => array_ref,
        scalar_value => scalar_value.into_array(left.clone().len()),
    };
    let ignore_null_array = ignore_null.as_any().downcast_ref::<StringArray>().unwrap();
    let result = (0..ignore_null_array.len())
        .map(|index| {
            if left.is_null(index) || right.is_null(index) {
                None
            } else {
                Some(ignore_null_array.value(index))
            }
        })
        .collect::<StringArray>();

    Ok(Arc::new(result) as ArrayRef)
}

fn bitwise_and_scalar(
    array: &dyn Array,
    scalar: ScalarValue,
) -> Option<Result<ArrayRef>> {
    let result = match array.data_type() {
        DataType::Int8 => {
            binary_bitwise_array_scalar!(array, scalar, |a, b| a & b, Int8Array, i8)
        }
        DataType::Int16 => {
            binary_bitwise_array_scalar!(array, scalar, |a, b| a & b, Int16Array, i16)
        }
        DataType::Int32 => {
            binary_bitwise_array_scalar!(array, scalar, |a, b| a & b, Int32Array, i32)
        }
        DataType::Int64 => {
            binary_bitwise_array_scalar!(array, scalar, |a, b| a & b, Int64Array, i64)
        }
        other => Err(DataFusionError::Internal(format!(
            "Data type {:?} not supported for binary operation '{}' on dyn arrays",
            other,
            Operator::BitwiseAnd
        ))),
    };
    Some(result)
}

fn bitwise_or_scalar(array: &dyn Array, scalar: ScalarValue) -> Option<Result<ArrayRef>> {
    let result = match array.data_type() {
        DataType::Int8 => {
            binary_bitwise_array_scalar!(array, scalar, |a, b| a | b, Int8Array, i8)
        }
        DataType::Int16 => {
            binary_bitwise_array_scalar!(array, scalar, |a, b| a | b, Int16Array, i16)
        }
        DataType::Int32 => {
            binary_bitwise_array_scalar!(array, scalar, |a, b| a | b, Int32Array, i32)
        }
        DataType::Int64 => {
            binary_bitwise_array_scalar!(array, scalar, |a, b| a | b, Int64Array, i64)
        }
        other => Err(DataFusionError::Internal(format!(
            "Data type {:?} not supported for binary operation '{}' on dyn arrays",
            other,
            Operator::BitwiseOr
        ))),
    };
    Some(result)
}

fn bitwise_shift_left_scalar(
    array: &dyn Array,
    scalar: ScalarValue,
) -> Option<Result<ArrayRef>> {
    let result = match array.data_type() {
        DataType::Int8 => {
            binary_bitwise_array_scalar!(
                array,
                scalar,
                |a: i8, b: i8| a.wrapping_shl(b as u32),
                Int8Array,
                i8
            )
        }
        DataType::Int16 => {
            binary_bitwise_array_scalar!(
                array,
                scalar,
                |a: i16, b: i16| a.wrapping_shl(b as u32),
                Int16Array,
                i16
            )
        }
        DataType::Int32 => {
            binary_bitwise_array_scalar!(
                array,
                scalar,
                |a: i32, b: i32| a.wrapping_shl(b as u32),
                Int32Array,
                i32
            )
        }
        DataType::Int64 => {
            binary_bitwise_array_scalar!(
                array,
                scalar,
                |a: i64, b: i64| a.wrapping_shl(b as u32),
                Int64Array,
                i64
            )
        }
        other => Err(DataFusionError::Internal(format!(
            "Data type {:?} not supported for binary operation '{}' on dyn arrays",
            other,
            Operator::BitwiseShiftLeft
        ))),
    };
    Some(result)
}

fn bitwise_shift_right_scalar(
    array: &dyn Array,
    scalar: ScalarValue,
) -> Option<Result<ArrayRef>> {
    let result = match array.data_type() {
        DataType::Int8 => {
            binary_bitwise_array_scalar!(
                array,
                scalar,
                |a: i8, b: i8| a.wrapping_shr(b as u32),
                Int8Array,
                i8
            )
        }
        DataType::Int16 => {
            binary_bitwise_array_scalar!(
                array,
                scalar,
                |a: i16, b: i16| a.wrapping_shr(b as u32),
                Int16Array,
                i16
            )
        }
        DataType::Int32 => {
            binary_bitwise_array_scalar!(
                array,
                scalar,
                |a: i32, b: i32| a.wrapping_shr(b as u32),
                Int32Array,
                i32
            )
        }
        DataType::Int64 => {
            binary_bitwise_array_scalar!(
                array,
                scalar,
                |a: i64, b: i64| a.wrapping_shr(b as u32),
                Int64Array,
                i64
            )
        }
        other => Err(DataFusionError::Internal(format!(
            "Data type {:?} not supported for binary operation '{}' on dyn arrays",
            other,
            Operator::BitwiseShiftLeft
        ))),
    };
    Some(result)
}

/// Binary expression
#[derive(Debug)]
pub struct BinaryExpr {
    left: Arc<dyn PhysicalExpr>,
    op: Operator,
    right: Arc<dyn PhysicalExpr>,
}

impl BinaryExpr {
    /// Create new binary expression
    pub fn new(
        left: Arc<dyn PhysicalExpr>,
        op: Operator,
        right: Arc<dyn PhysicalExpr>,
    ) -> Self {
        Self { left, op, right }
    }

    /// Get the left side of the binary expression
    pub fn left(&self) -> &Arc<dyn PhysicalExpr> {
        &self.left
    }

    /// Get the right side of the binary expression
    pub fn right(&self) -> &Arc<dyn PhysicalExpr> {
        &self.right
    }

    /// Get the operator for this binary expression
    pub fn op(&self) -> &Operator {
        &self.op
    }
}

impl std::fmt::Display for BinaryExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} {} {}", self.left, self.op, self.right)
    }
}

/// Invoke a boolean kernel on a pair of arrays
macro_rules! boolean_op {
    ($LEFT:expr, $RIGHT:expr, $OP:ident) => {{
        let ll = $LEFT
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("boolean_op failed to downcast array");
        let rr = $RIGHT
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("boolean_op failed to downcast array");
        Ok(Arc::new($OP(&ll, &rr)?))
    }};
}

impl PhysicalExpr for BinaryExpr {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, input_schema: &Schema) -> Result<DataType> {
        binary_operator_data_type(
            &self.left.data_type(input_schema)?,
            &self.op,
            &self.right.data_type(input_schema)?,
        )
    }

    fn nullable(&self, input_schema: &Schema) -> Result<bool> {
        Ok(self.left.nullable(input_schema)? || self.right.nullable(input_schema)?)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let left_value = self.left.evaluate(batch)?;
        let right_value = self.right.evaluate(batch)?;
        let left_data_type = left_value.data_type();
        let right_data_type = right_value.data_type();

        if left_data_type != right_data_type
            && !distinct_types_allowed(&left_data_type, &self.op, &right_data_type)
        {
            return Err(DataFusionError::Internal(format!(
                "Cannot evaluate binary expression {:?} with types {:?} and {:?}",
                self.op, left_data_type, right_data_type
            )));
        }

        // Attempt to use special kernels if one input is scalar and the other is an array.
        // Distinct-type arithmetic (timestamp/interval) only has array kernels.
        let scalar_result = match (&left_value, &right_value) {
            _ if left_data_type != right_data_type => None,
            (ColumnarValue::Array(array), ColumnarValue::Scalar(scalar)) => {
                // if left is array and right is literal - use scalar operations
                self.evaluate_array_scalar(array.as_ref(), scalar)
            }
            (ColumnarValue::Scalar(scalar), ColumnarValue::Array(array)) => {
                // if left is literal and right is array - use scalar operations
                self.evaluate_scalar_array(scalar, array.as_ref())
            }
            (_, _) => None, // default to array implementation
        };

        if let Some(result) = scalar_result {
            return result.map(ColumnarValue::Array);
        }

        // if both arrays or both literals - extract arrays and continue execution
        let (left, right) = (
            left_value.into_array(batch.num_rows()),
            right_value.into_array(batch.num_rows()),
        );
        self.evaluate_with_resolved_args(left, &left_data_type, right, &right_data_type)
            .map(ColumnarValue::Array)
    }
}

impl BinaryExpr {
    /// Evaluate the expression of the left input is an array and
    /// right is literal - use scalar operations
    fn evaluate_array_scalar(
        &self,
        array: &dyn Array,
        scalar: &ScalarValue,
    ) -> Option<Result<ArrayRef>> {
        if let Some(op) = ComparisonOp::from_operator(&self.op) {
            return match op {
                // `IS [NOT] DISTINCT FROM` only has an array implementation
                ComparisonOp::IsDistinctFrom | ComparisonOp::IsNotDistinctFrom => None,
                op => Some(
                    compare_array_scalar(op, array, scalar)
                        .map(|a| Arc::new(a) as ArrayRef),
                ),
            };
        }
        if let Some(op) = LikeOp::from_operator(&self.op) {
            return Some(like_array_scalar(op, array, scalar));
        }
        if let Some(op) = ArithmeticOp::from_operator(&self.op) {
            return arithmetic_array_scalar(op, array, scalar);
        }
        match &self.op {
            Operator::RegexMatch => {
                Some(regex_match_array_scalar(array, scalar, false, false))
            }
            Operator::RegexIMatch => {
                Some(regex_match_array_scalar(array, scalar, false, true))
            }
            Operator::RegexNotMatch => {
                Some(regex_match_array_scalar(array, scalar, true, false))
            }
            Operator::RegexNotIMatch => {
                Some(regex_match_array_scalar(array, scalar, true, true))
            }
            Operator::BitwiseAnd => bitwise_and_scalar(array, scalar.clone()),
            Operator::BitwiseOr => bitwise_or_scalar(array, scalar.clone()),
            Operator::BitwiseShiftLeft => {
                bitwise_shift_left_scalar(array, scalar.clone())
            }
            Operator::BitwiseShiftRight => {
                bitwise_shift_right_scalar(array, scalar.clone())
            }
            // if scalar operation is not supported - fallback to array implementation
            _ => None,
        }
    }

    /// Evaluate the expression if the left input is a literal and the
    /// right is an array
    fn evaluate_scalar_array(
        &self,
        scalar: &ScalarValue,
        array: &dyn Array,
    ) -> Option<Result<ArrayRef>> {
        match ComparisonOp::from_operator(&self.op) {
            // `IS [NOT] DISTINCT FROM` only has an array implementation
            Some(ComparisonOp::IsDistinctFrom | ComparisonOp::IsNotDistinctFrom)
            | None => None,
            Some(op) => Some(
                compare_scalar_array(op, scalar, array).map(|a| Arc::new(a) as ArrayRef),
            ),
        }
    }

    fn evaluate_with_resolved_args(
        &self,
        left: Arc<dyn Array>,
        left_data_type: &DataType,
        right: Arc<dyn Array>,
        right_data_type: &DataType,
    ) -> Result<ArrayRef> {
        // first try to evaluate expressions with distinct types
        if let Some(result) = evaluate_distinct_with_resolved_args(
            Arc::clone(&left),
            left_data_type,
            &self.op,
            Arc::clone(&right),
            right_data_type,
        ) {
            return result;
        }

        match &self.op {
            Operator::Like | Operator::NotLike | Operator::ILike | Operator::NotILike => {
                // unwrap: the operator is a LIKE operator
                like_array_array(
                    LikeOp::from_operator(&self.op).unwrap(),
                    left.as_ref(),
                    right.as_ref(),
                )
            }
            Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq
            | Operator::Eq
            | Operator::NotEq
            | Operator::IsDistinctFrom
            | Operator::IsNotDistinctFrom => {
                // unwrap: the operator is a comparison operator
                let op = ComparisonOp::from_operator(&self.op).unwrap();
                Ok(Arc::new(compare_datum(op, &left, &right)?))
            }
            Operator::Plus
            | Operator::Minus
            | Operator::Multiply
            | Operator::Divide
            | Operator::Modulo => {
                // unwrap: the operator is an arithmetic operator
                let op = ArithmeticOp::from_operator(&self.op).unwrap();
                arithmetic_arrays(op, &left, &right)
            }
            Operator::Exponentiate => {
                let ll = left
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .expect("compute_op failed to downcast array");
                let rr = right
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .expect("compute_op failed to downcast array");
                Ok(Arc::new(powf(ll, rr)?))
            }
            Operator::And => {
                if left_data_type == &DataType::Boolean {
                    boolean_op!(left, right, and_kleene)
                } else {
                    Err(DataFusionError::Internal(format!(
                        "Cannot evaluate binary expression {:?} with types {:?} and {:?}",
                        self.op,
                        left.data_type(),
                        right.data_type()
                    )))
                }
            }
            Operator::Or => {
                if left_data_type == &DataType::Boolean {
                    boolean_op!(left, right, or_kleene)
                } else {
                    Err(DataFusionError::Internal(format!(
                        "Cannot evaluate binary expression {:?} with types {:?} and {:?}",
                        self.op, left_data_type, right_data_type
                    )))
                }
            }
            Operator::RegexMatch => {
                regex_match_arrays(left.as_ref(), right.as_ref(), false, false)
            }
            Operator::RegexIMatch => {
                regex_match_arrays(left.as_ref(), right.as_ref(), false, true)
            }
            Operator::RegexNotMatch => {
                regex_match_arrays(left.as_ref(), right.as_ref(), true, false)
            }
            Operator::RegexNotIMatch => {
                regex_match_arrays(left.as_ref(), right.as_ref(), true, true)
            }
            Operator::BitwiseAnd => bitwise_and(left, right),
            Operator::BitwiseOr => bitwise_or(left, right),
            Operator::BitwiseShiftRight => bitwise_shift_right(left, right),
            Operator::BitwiseShiftLeft => bitwise_shift_left(left, right),
            Operator::StringConcat => string_concat(left, right),
        }
    }
}

/// return two physical expressions that are optionally coerced to a
/// common type that the binary operator supports.
fn binary_cast(
    lhs: Arc<dyn PhysicalExpr>,
    op: &Operator,
    rhs: Arc<dyn PhysicalExpr>,
    input_schema: &Schema,
) -> Result<(Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>)> {
    let lhs_type = &lhs.data_type(input_schema)?;
    let rhs_type = &rhs.data_type(input_schema)?;

    // first try to coerce types for expressions that must have distinct
    // left and right types
    if let Some((lhs_result_type, rhs_result_type)) =
        coerce_types_distinct(lhs_type, op, rhs_type)
    {
        return Ok((
            try_cast(lhs, input_schema, lhs_result_type)?,
            try_cast(rhs, input_schema, rhs_result_type)?,
        ));
    }

    let result_type = coerce_types(lhs_type, op, rhs_type)?;

    Ok((
        try_cast(lhs, input_schema, result_type.clone())?,
        try_cast(rhs, input_schema, result_type)?,
    ))
}

/// Create a binary expression whose arguments are correctly coerced.
/// This function errors if it is not possible to coerce the arguments
/// to computational types supported by the operator.
pub fn binary(
    lhs: Arc<dyn PhysicalExpr>,
    op: Operator,
    rhs: Arc<dyn PhysicalExpr>,
    input_schema: &Schema,
) -> Result<Arc<dyn PhysicalExpr>> {
    let (l, r) = binary_cast(lhs, &op, rhs, input_schema)?;
    Ok(Arc::new(BinaryExpr::new(l, op, r)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expressions::{col, lit};
    use arrow::datatypes::{
        ArrowNumericType, Field, Float64Type, Int32Type, IntervalDayTime, SchemaRef,
        TimeUnit,
    };
    use arrow::util::display::array_value_to_string;
    use datafusion_common::Result;

    // Create a binary expression without coercion. Used here when we do not want to coerce the expressions
    // to valid types. Usage can result in an execution (after plan) error.
    fn binary_simple(
        l: Arc<dyn PhysicalExpr>,
        op: Operator,
        r: Arc<dyn PhysicalExpr>,
        input_schema: &Schema,
    ) -> Arc<dyn PhysicalExpr> {
        binary(l, op, r, input_schema).unwrap()
    }

    #[test]
    fn binary_comparison() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]);
        let a = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let b = Int32Array::from(vec![1, 2, 4, 8, 16]);

        // expression: "a < b"
        let lt = binary_simple(
            col("a", &schema)?,
            Operator::Lt,
            col("b", &schema)?,
            &schema,
        );
        let batch =
            RecordBatch::try_new(Arc::new(schema), vec![Arc::new(a), Arc::new(b)])?;

        let result = lt.evaluate(&batch)?.into_array(batch.num_rows());
        assert_eq!(result.len(), 5);

        let expected = [false, false, true, true, true];
        let result = result
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("failed to downcast to BooleanArray");
        for (i, &expected_item) in expected.iter().enumerate().take(5) {
            assert_eq!(result.value(i), expected_item);
        }

        Ok(())
    }

    #[test]
    fn like_keeps_arrow13_semantics() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Utf8, true),
            Field::new("b", DataType::Utf8, true),
        ]);
        let a = StringArray::from(vec![
            Some("a\\b"),
            Some("a\nb"),
            Some("C:\\temp"),
            Some("STRASSE"),
            None,
        ]);
        let b = StringArray::from(vec![
            Some("a\\b"),
            Some("a_b"),
            Some("C:\\\\%"),
            Some("straße%"),
            Some("%"),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(a), Arc::new(b)],
        )?;
        let evaluate = |op: Operator, right: Arc<dyn PhysicalExpr>| -> Result<ArrayRef> {
            let expr = BinaryExpr::new(col("a", &schema)?, op, right);
            Ok(expr.evaluate(&batch)?.into_array(batch.num_rows()))
        };
        let pattern = |p: &str| lit(ScalarValue::Utf8(Some(p.to_string())));

        // a backslash that does not escape a wildcard is literal
        let result = evaluate(Operator::Like, pattern("a\\b"))?;
        assert_eq!(
            result.as_boolean(),
            &BooleanArray::from(vec![
                Some(true),
                Some(false),
                Some(false),
                Some(false),
                None
            ])
        );
        // `\\%` is a literal backslash and a literal `%`
        let result = evaluate(Operator::Like, pattern("C:\\\\%"))?;
        assert_eq!(
            result.as_boolean(),
            &BooleanArray::from(vec![
                Some(false),
                Some(false),
                Some(false),
                Some(false),
                None
            ])
        );
        // `_` matches any character but a newline
        let result = evaluate(Operator::NotLike, pattern("a_b"))?;
        assert_eq!(
            result.as_boolean(),
            &BooleanArray::from(vec![
                Some(false),
                Some(true),
                Some(true),
                Some(true),
                None
            ])
        );
        // the prefix fast path of ILIKE compares upper case forms
        let result = evaluate(Operator::ILike, pattern("straße%"))?;
        assert_eq!(
            result.as_boolean(),
            &BooleanArray::from(vec![
                Some(false),
                Some(false),
                Some(false),
                Some(true),
                None
            ])
        );
        // a NULL pattern yields NULL
        let result = evaluate(Operator::Like, lit(ScalarValue::Utf8(None)))?;
        assert_eq!(result.null_count(), 5);

        // array patterns
        let result = evaluate(Operator::Like, col("b", &schema)?)?;
        assert_eq!(
            result.as_boolean(),
            &BooleanArray::from(vec![
                Some(true),
                Some(false),
                Some(false),
                Some(false),
                None
            ])
        );
        let result = evaluate(Operator::ILike, col("b", &schema)?)?;
        assert_eq!(
            result.as_boolean(),
            &BooleanArray::from(vec![
                Some(true),
                Some(false),
                Some(false),
                Some(false),
                None
            ])
        );
        Ok(())
    }

    #[test]
    fn timestamp_array_interval_scalar_arithmetic() -> Result<()> {
        const DAY: i64 = 86_400_000_000_000;
        let schema = Schema::new(vec![Field::new(
            "a",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        )]);
        let a = TimestampNanosecondArray::from(vec![0, DAY]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;
        let one_day = lit(ScalarValue::IntervalDayTime(Some(IntervalDayTime::new(
            1, 0,
        ))));

        for (op, expected) in [
            (Operator::Plus, vec![DAY, 2 * DAY]),
            (Operator::Minus, vec![-DAY, 0]),
        ] {
            let expr = BinaryExpr::new(col("a", &schema)?, op, one_day.clone());
            let result = expr.evaluate(&batch)?.into_array(batch.num_rows());
            let result = result
                .as_any()
                .downcast_ref::<TimestampNanosecondArray>()
                .expect("failed to downcast to TimestampNanosecondArray");
            assert_eq!(result.values(), expected.as_slice(), "{}", op);
        }

        Ok(())
    }

    #[test]
    fn timestamp_array_minus_timestamp_scalar() -> Result<()> {
        const DAY: i64 = 86_400_000_000_000;
        let ts_type = DataType::Timestamp(TimeUnit::Nanosecond, None);
        let schema = Schema::new(vec![
            Field::new("a", ts_type.clone(), false),
            Field::new("b", ts_type, false),
        ]);
        let a = TimestampNanosecondArray::from(vec![DAY, 2 * DAY]);
        let b = TimestampNanosecondArray::from(vec![DAY, DAY]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(a), Arc::new(b)],
        )?;

        let minus_scalar = BinaryExpr::new(
            col("a", &schema)?,
            Operator::Minus,
            lit(ScalarValue::TimestampNanosecond(Some(DAY), None)),
        )
        .evaluate(&batch)?
        .into_array(batch.num_rows());
        let minus_array =
            BinaryExpr::new(col("a", &schema)?, Operator::Minus, col("b", &schema)?)
                .evaluate(&batch)?
                .into_array(batch.num_rows());
        assert_eq!(&minus_scalar, &minus_array);

        Ok(())
    }

    #[test]
    fn number_array_times_interval_scalar() -> Result<()> {
        let schema = Schema::new(vec![Field::new("n", DataType::Int64, false)]);
        let n = Int64Array::from(vec![1, 2]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(n)])?;
        let one_day = lit(ScalarValue::IntervalDayTime(Some(IntervalDayTime::new(
            1, 0,
        ))));

        let n_times_interval =
            BinaryExpr::new(col("n", &schema)?, Operator::Multiply, one_day.clone())
                .evaluate(&batch)?
                .into_array(batch.num_rows());
        let interval_times_n =
            BinaryExpr::new(one_day, Operator::Multiply, col("n", &schema)?)
                .evaluate(&batch)?
                .into_array(batch.num_rows());
        assert_eq!(&n_times_interval, &interval_times_n);

        Ok(())
    }

    #[test]
    fn binary_nested() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]);
        let a = Int32Array::from(vec![2, 4, 6, 8, 10]);
        let b = Int32Array::from(vec![2, 5, 4, 8, 8]);

        // expression: "a < b OR a == b"
        let expr = binary_simple(
            binary_simple(
                col("a", &schema)?,
                Operator::Lt,
                col("b", &schema)?,
                &schema,
            ),
            Operator::Or,
            binary_simple(
                col("a", &schema)?,
                Operator::Eq,
                col("b", &schema)?,
                &schema,
            ),
            &schema,
        );
        let batch =
            RecordBatch::try_new(Arc::new(schema), vec![Arc::new(a), Arc::new(b)])?;

        assert_eq!("a@0 < b@1 OR a@0 = b@1", format!("{}", expr));

        let result = expr.evaluate(&batch)?.into_array(batch.num_rows());
        assert_eq!(result.len(), 5);

        let expected = [true, true, false, true, false];
        let result = result
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("failed to downcast to BooleanArray");
        for (i, &expected_item) in expected.iter().enumerate().take(5) {
            assert_eq!(result.value(i), expected_item);
        }

        Ok(())
    }

    // runs an end-to-end test of physical type coercion:
    // 1. construct a record batch with two columns of type A and B
    //  (*_ARRAY is the Rust Arrow array type, and *_TYPE is the DataType of the elements)
    // 2. construct a physical expression of A OP B
    // 3. evaluate the expression
    // 4. verify that the resulting expression is of type C
    // 5. verify that the results of evaluation are $VEC
    macro_rules! test_coercion {
        ($A_ARRAY:ident, $A_TYPE:expr, $A_VEC:expr, $B_ARRAY:ident, $B_TYPE:expr, $B_VEC:expr, $OP:expr, $C_ARRAY:ident, $C_TYPE:expr, $VEC:expr) => {{
            let schema = Schema::new(vec![
                Field::new("a", $A_TYPE, false),
                Field::new("b", $B_TYPE, false),
            ]);
            let a = $A_ARRAY::from($A_VEC);
            let b = $B_ARRAY::from($B_VEC);

            // verify that we can construct the expression
            let expression =
                binary(col("a", &schema)?, $OP, col("b", &schema)?, &schema)?;
            let batch = RecordBatch::try_new(
                Arc::new(schema.clone()),
                vec![Arc::new(a), Arc::new(b)],
            )?;

            // verify that the expression's type is correct
            assert_eq!(expression.data_type(&schema)?, $C_TYPE);

            // compute
            let result = expression.evaluate(&batch)?.into_array(batch.num_rows());

            // verify that the array's data_type is correct
            assert_eq!(*result.data_type(), $C_TYPE);

            // verify that the data itself is downcastable
            let result = result
                .as_any()
                .downcast_ref::<$C_ARRAY>()
                .expect("failed to downcast");
            // verify that the result itself is correct
            for (i, x) in $VEC.iter().enumerate() {
                assert_eq!(result.value(i), *x);
            }
        }};
    }

    #[test]
    fn test_type_coersion() -> Result<()> {
        test_coercion!(
            Int32Array,
            DataType::Int32,
            vec![1i32, 2i32],
            UInt32Array,
            DataType::UInt32,
            vec![1u32, 2u32],
            Operator::Plus,
            Int32Array,
            DataType::Int32,
            [2i32, 4i32]
        );
        test_coercion!(
            Int32Array,
            DataType::Int32,
            vec![1i32],
            UInt16Array,
            DataType::UInt16,
            vec![1u16],
            Operator::Plus,
            Int32Array,
            DataType::Int32,
            [2i32]
        );
        test_coercion!(
            Float32Array,
            DataType::Float32,
            vec![1f32],
            UInt16Array,
            DataType::UInt16,
            vec![1u16],
            Operator::Plus,
            Float32Array,
            DataType::Float32,
            [2f32]
        );
        test_coercion!(
            Float32Array,
            DataType::Float32,
            vec![2f32],
            UInt16Array,
            DataType::UInt16,
            vec![1u16],
            Operator::Multiply,
            Float32Array,
            DataType::Float32,
            [2f32]
        );
        test_coercion!(
            Int64Array,
            DataType::Int64,
            vec![2i64],
            Int64Array,
            DataType::Int64,
            vec![3i64],
            Operator::Exponentiate,
            Float64Array,
            DataType::Float64,
            [8f64]
        );
        test_coercion!(
            StringArray,
            DataType::Utf8,
            vec!["hello world", "world"],
            StringArray,
            DataType::Utf8,
            vec!["%hello%", "%hello%"],
            Operator::Like,
            BooleanArray,
            DataType::Boolean,
            [true, false]
        );
        test_coercion!(
            StringArray,
            DataType::Utf8,
            vec!["hello world", "world"],
            StringArray,
            DataType::Utf8,
            vec!["%hello%", "%hello%"],
            Operator::NotLike,
            BooleanArray,
            DataType::Boolean,
            [false, true]
        );
        test_coercion!(
            StringArray,
            DataType::Utf8,
            vec!["1994-12-13", "1995-01-26"],
            Date32Array,
            DataType::Date32,
            vec![9112, 9156],
            Operator::Eq,
            BooleanArray,
            DataType::Boolean,
            [true, true]
        );
        test_coercion!(
            StringArray,
            DataType::Utf8,
            vec!["1994-12-13", "1995-01-26"],
            Date32Array,
            DataType::Date32,
            vec![9113, 9154],
            Operator::Lt,
            BooleanArray,
            DataType::Boolean,
            [true, false]
        );
        test_coercion!(
            StringArray,
            DataType::Utf8,
            vec!["1994-12-13T12:34:56", "1995-01-26T01:23:45"],
            Date64Array,
            DataType::Date64,
            vec![787322096000, 791083425000],
            Operator::Eq,
            BooleanArray,
            DataType::Boolean,
            [true, true]
        );
        test_coercion!(
            StringArray,
            DataType::Utf8,
            vec!["1994-12-13T12:34:56", "1995-01-26T01:23:45"],
            Date64Array,
            DataType::Date64,
            vec![787322096001, 791083424999],
            Operator::Lt,
            BooleanArray,
            DataType::Boolean,
            [true, false]
        );
        test_coercion!(
            StringArray,
            DataType::Utf8,
            vec!["abc"; 5],
            StringArray,
            DataType::Utf8,
            vec!["^a", "^A", "(b|d)", "(B|D)", "^(b|c)"],
            Operator::RegexMatch,
            BooleanArray,
            DataType::Boolean,
            [true, false, true, false, false]
        );
        test_coercion!(
            StringArray,
            DataType::Utf8,
            vec!["abc"; 5],
            StringArray,
            DataType::Utf8,
            vec!["^a", "^A", "(b|d)", "(B|D)", "^(b|c)"],
            Operator::RegexIMatch,
            BooleanArray,
            DataType::Boolean,
            [true, true, true, true, false]
        );
        test_coercion!(
            StringArray,
            DataType::Utf8,
            vec!["abc"; 5],
            StringArray,
            DataType::Utf8,
            vec!["^a", "^A", "(b|d)", "(B|D)", "^(b|c)"],
            Operator::RegexNotMatch,
            BooleanArray,
            DataType::Boolean,
            [false, true, false, true, true]
        );
        test_coercion!(
            StringArray,
            DataType::Utf8,
            vec!["abc"; 5],
            StringArray,
            DataType::Utf8,
            vec!["^a", "^A", "(b|d)", "(B|D)", "^(b|c)"],
            Operator::RegexNotIMatch,
            BooleanArray,
            DataType::Boolean,
            [false, false, false, false, true]
        );
        test_coercion!(
            LargeStringArray,
            DataType::LargeUtf8,
            vec!["abc"; 5],
            LargeStringArray,
            DataType::LargeUtf8,
            vec!["^a", "^A", "(b|d)", "(B|D)", "^(b|c)"],
            Operator::RegexMatch,
            BooleanArray,
            DataType::Boolean,
            [true, false, true, false, false]
        );
        test_coercion!(
            LargeStringArray,
            DataType::LargeUtf8,
            vec!["abc"; 5],
            LargeStringArray,
            DataType::LargeUtf8,
            vec!["^a", "^A", "(b|d)", "(B|D)", "^(b|c)"],
            Operator::RegexIMatch,
            BooleanArray,
            DataType::Boolean,
            [true, true, true, true, false]
        );
        test_coercion!(
            LargeStringArray,
            DataType::LargeUtf8,
            vec!["abc"; 5],
            LargeStringArray,
            DataType::LargeUtf8,
            vec!["^a", "^A", "(b|d)", "(B|D)", "^(b|c)"],
            Operator::RegexNotMatch,
            BooleanArray,
            DataType::Boolean,
            [false, true, false, true, true]
        );
        test_coercion!(
            LargeStringArray,
            DataType::LargeUtf8,
            vec!["abc"; 5],
            LargeStringArray,
            DataType::LargeUtf8,
            vec!["^a", "^A", "(b|d)", "(B|D)", "^(b|c)"],
            Operator::RegexNotIMatch,
            BooleanArray,
            DataType::Boolean,
            [false, false, false, false, true]
        );
        test_coercion!(
            Int16Array,
            DataType::Int16,
            vec![1i16, 2i16, 3i16],
            Int64Array,
            DataType::Int64,
            vec![10i64, 4i64, 5i64],
            Operator::BitwiseAnd,
            Int64Array,
            DataType::Int64,
            [0i64, 0i64, 1i64]
        );
        test_coercion!(
            Int16Array,
            DataType::Int16,
            vec![1i16, 2i16, 3i16],
            Int64Array,
            DataType::Int64,
            vec![10i64, 4i64, 5i64],
            Operator::BitwiseOr,
            Int64Array,
            DataType::Int64,
            [11i64, 6i64, 7i64]
        );
        Ok(())
    }

    // Note it would be nice to use the same test_coercion macro as
    // above, but sadly the type of the values of the dictionary are
    // not encoded in the rust type of the DictionaryArray. Thus there
    // is no way at the time of this writing to create a dictionary
    // array using the `From` trait
    #[test]
    fn test_dictionary_type_to_array_coersion() -> Result<()> {
        // Test string  a string dictionary
        let dict_type =
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
        let string_type = DataType::Utf8;

        // build dictionary
        let mut dict_builder = StringDictionaryBuilder::<Int32Type>::new();

        dict_builder.append("one")?;
        dict_builder.append_null();
        dict_builder.append("three")?;
        dict_builder.append("four")?;
        let dict_array = dict_builder.finish();

        let str_array =
            StringArray::from(vec![Some("not one"), Some("two"), None, Some("four")]);

        let schema = Arc::new(Schema::new(vec![
            Field::new("dict", dict_type, true),
            Field::new("str", string_type, true),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(dict_array), Arc::new(str_array)],
        )?;

        let expected = "false\n\n\ntrue";

        // Test 1: dict = str

        // verify that we can construct the expression
        let expression = binary(
            col("dict", &schema)?,
            Operator::Eq,
            col("str", &schema)?,
            &schema,
        )?;
        assert_eq!(expression.data_type(&schema)?, DataType::Boolean);

        // evaluate and verify the result type matched
        let result = expression.evaluate(&batch)?.into_array(batch.num_rows());
        assert_eq!(result.data_type(), &DataType::Boolean);

        // verify that the result itself is correct
        assert_eq!(expected, array_to_string(&result)?);

        // Test 2: now test the other direction
        // str = dict

        // verify that we can construct the expression
        let expression = binary(
            col("str", &schema)?,
            Operator::Eq,
            col("dict", &schema)?,
            &schema,
        )?;
        assert_eq!(expression.data_type(&schema)?, DataType::Boolean);

        // evaluate and verify the result type matched
        let result = expression.evaluate(&batch)?.into_array(batch.num_rows());
        assert_eq!(result.data_type(), &DataType::Boolean);

        // verify that the result itself is correct
        assert_eq!(expected, array_to_string(&result)?);

        Ok(())
    }

    // Convert the array to a newline delimited string of pretty printed values
    fn array_to_string(array: &ArrayRef) -> Result<String> {
        let s = (0..array.len())
            .map(|i| array_value_to_string(array, i))
            .collect::<std::result::Result<Vec<_>, arrow::error::ArrowError>>()?
            .join("\n");
        Ok(s)
    }

    #[test]
    fn plus_op() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]);
        let a = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let b = Int32Array::from(vec![1, 2, 4, 8, 16]);

        apply_arithmetic::<Int32Type>(
            Arc::new(schema),
            vec![Arc::new(a), Arc::new(b)],
            Operator::Plus,
            Int32Array::from(vec![2, 4, 7, 12, 21]),
        )?;

        Ok(())
    }

    #[test]
    fn minus_op() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let a = Arc::new(Int32Array::from(vec![1, 2, 4, 8, 16]));
        let b = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]));

        apply_arithmetic::<Int32Type>(
            schema.clone(),
            vec![a.clone(), b.clone()],
            Operator::Minus,
            Int32Array::from(vec![0, 0, 1, 4, 11]),
        )?;

        // should handle have negative values in result (for signed)
        apply_arithmetic::<Int32Type>(
            schema,
            vec![b, a],
            Operator::Minus,
            Int32Array::from(vec![0, 0, -1, -4, -11]),
        )?;

        Ok(())
    }

    #[test]
    fn multiply_op() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let a = Arc::new(Int32Array::from(vec![4, 8, 16, 32, 64]));
        let b = Arc::new(Int32Array::from(vec![2, 4, 8, 16, 32]));

        apply_arithmetic::<Int32Type>(
            schema,
            vec![a, b],
            Operator::Multiply,
            Int32Array::from(vec![8, 32, 128, 512, 2048]),
        )?;

        Ok(())
    }

    #[test]
    fn divide_op() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let a = Arc::new(Int32Array::from(vec![8, 32, 128, 512, 2048]));
        let b = Arc::new(Int32Array::from(vec![2, 4, 8, 16, 32]));

        apply_arithmetic::<Int32Type>(
            schema,
            vec![a, b],
            Operator::Divide,
            Int32Array::from(vec![4, 8, 16, 32, 64]),
        )?;

        Ok(())
    }

    #[test]
    fn modulus_op() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let a = Arc::new(Int32Array::from(vec![8, 32, 128, 512, 2048]));
        let b = Arc::new(Int32Array::from(vec![2, 4, 7, 14, 32]));

        apply_arithmetic::<Int32Type>(
            schema,
            vec![a, b],
            Operator::Modulo,
            Int32Array::from(vec![0, 0, 2, 8, 0]),
        )?;

        Ok(())
    }

    #[test]
    fn exponentiate_op() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Float64, false),
        ]);
        let a = Int32Array::from(vec![2, 3, 4, -5, 16]);
        let b = Float64Array::from(vec![2.0, 5.0, -0.5, 2.0, 1.5]);

        apply_arithmetic::<Float64Type>(
            Arc::new(schema),
            vec![Arc::new(a), Arc::new(b)],
            Operator::Exponentiate,
            Float64Array::from(vec![4.0, 243.0, 0.5, 25.0, 64.0]),
        )?;

        Ok(())
    }

    fn apply_arithmetic<T: ArrowNumericType>(
        schema: SchemaRef,
        data: Vec<ArrayRef>,
        op: Operator,
        expected: PrimitiveArray<T>,
    ) -> Result<()> {
        let arithmetic_op =
            binary_simple(col("a", &schema)?, op, col("b", &schema)?, &schema);
        let batch = RecordBatch::try_new(schema, data)?;
        let result = arithmetic_op.evaluate(&batch)?.into_array(batch.num_rows());

        assert_eq!(result.as_ref(), &expected);
        Ok(())
    }

    fn apply_logic_op(
        schema: &SchemaRef,
        left: &ArrayRef,
        right: &ArrayRef,
        op: Operator,
        expected: BooleanArray,
    ) -> Result<()> {
        let arithmetic_op =
            binary_simple(col("a", schema)?, op, col("b", schema)?, schema);
        let data: Vec<ArrayRef> = vec![left.clone(), right.clone()];
        let batch = RecordBatch::try_new(schema.clone(), data)?;
        let result = arithmetic_op.evaluate(&batch)?.into_array(batch.num_rows());

        assert_eq!(result.as_ref(), &expected);
        Ok(())
    }

    // Test `scalar <op> arr` produces expected
    fn apply_logic_op_scalar_arr(
        schema: &SchemaRef,
        scalar: &ScalarValue,
        arr: &ArrayRef,
        op: Operator,
        expected: &BooleanArray,
    ) -> Result<()> {
        let scalar = lit(scalar.clone());

        let arithmetic_op = binary_simple(scalar, op, col("a", schema)?, schema);
        let batch = RecordBatch::try_new(Arc::clone(schema), vec![Arc::clone(arr)])?;
        let result = arithmetic_op.evaluate(&batch)?.into_array(batch.num_rows());
        assert_eq!(result.as_ref(), expected);

        Ok(())
    }

    // Test `arr <op> scalar` produces expected
    fn apply_logic_op_arr_scalar(
        schema: &SchemaRef,
        arr: &ArrayRef,
        scalar: &ScalarValue,
        op: Operator,
        expected: &BooleanArray,
    ) -> Result<()> {
        let scalar = lit(scalar.clone());

        let arithmetic_op = binary_simple(col("a", schema)?, op, scalar, schema);
        let batch = RecordBatch::try_new(Arc::clone(schema), vec![Arc::clone(arr)])?;
        let result = arithmetic_op.evaluate(&batch)?.into_array(batch.num_rows());
        assert_eq!(result.as_ref(), expected);

        Ok(())
    }

    #[test]
    fn and_with_nulls_op() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Boolean, true),
            Field::new("b", DataType::Boolean, true),
        ]);
        let a = Arc::new(BooleanArray::from(vec![
            Some(true),
            Some(false),
            None,
            Some(true),
            Some(false),
            None,
            Some(true),
            Some(false),
            None,
        ])) as ArrayRef;
        let b = Arc::new(BooleanArray::from(vec![
            Some(true),
            Some(true),
            Some(true),
            Some(false),
            Some(false),
            Some(false),
            None,
            None,
            None,
        ])) as ArrayRef;

        let expected = BooleanArray::from(vec![
            Some(true),
            Some(false),
            None,
            Some(false),
            Some(false),
            Some(false),
            None,
            Some(false),
            None,
        ]);
        apply_logic_op(&Arc::new(schema), &a, &b, Operator::And, expected)?;

        Ok(())
    }

    #[test]
    fn or_with_nulls_op() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Boolean, true),
            Field::new("b", DataType::Boolean, true),
        ]);
        let a = Arc::new(BooleanArray::from(vec![
            Some(true),
            Some(false),
            None,
            Some(true),
            Some(false),
            None,
            Some(true),
            Some(false),
            None,
        ])) as ArrayRef;
        let b = Arc::new(BooleanArray::from(vec![
            Some(true),
            Some(true),
            Some(true),
            Some(false),
            Some(false),
            Some(false),
            None,
            None,
            None,
        ])) as ArrayRef;

        let expected = BooleanArray::from(vec![
            Some(true),
            Some(true),
            Some(true),
            Some(true),
            Some(false),
            None,
            Some(true),
            None,
            None,
        ]);
        apply_logic_op(&Arc::new(schema), &a, &b, Operator::Or, expected)?;

        Ok(())
    }

    /// Returns (schema, a: BooleanArray, b: BooleanArray) with all possible inputs
    ///
    /// a: [true, true, true,  NULL, NULL, NULL,  false, false, false]
    /// b: [true, NULL, false, true, NULL, false, true,  NULL,  false]
    fn bool_test_arrays() -> (SchemaRef, ArrayRef, ArrayRef) {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Boolean, true),
            Field::new("b", DataType::Boolean, true),
        ]);
        let a: BooleanArray = [
            Some(true),
            Some(true),
            Some(true),
            None,
            None,
            None,
            Some(false),
            Some(false),
            Some(false),
        ]
        .iter()
        .collect();
        let b: BooleanArray = [
            Some(true),
            None,
            Some(false),
            Some(true),
            None,
            Some(false),
            Some(true),
            None,
            Some(false),
        ]
        .iter()
        .collect();
        (Arc::new(schema), Arc::new(a), Arc::new(b))
    }

    /// Returns (schema, BooleanArray) with [true, NULL, false]
    fn scalar_bool_test_array() -> (SchemaRef, ArrayRef) {
        let schema = Schema::new(vec![Field::new("a", DataType::Boolean, true)]);
        let a: BooleanArray = [Some(true), None, Some(false)].iter().collect();
        (Arc::new(schema), Arc::new(a))
    }

    #[test]
    fn eq_op_bool() {
        let (schema, a, b) = bool_test_arrays();
        let expected = [
            Some(true),
            None,
            Some(false),
            None,
            None,
            None,
            Some(false),
            None,
            Some(true),
        ]
        .iter()
        .collect();
        apply_logic_op(&schema, &a, &b, Operator::Eq, expected).unwrap();
    }

    #[test]
    fn eq_op_bool_scalar() {
        let (schema, a) = scalar_bool_test_array();
        let expected = [Some(true), None, Some(false)].iter().collect();
        apply_logic_op_scalar_arr(
            &schema,
            &ScalarValue::from(true),
            &a,
            Operator::Eq,
            &expected,
        )
        .unwrap();
        apply_logic_op_arr_scalar(
            &schema,
            &a,
            &ScalarValue::from(true),
            Operator::Eq,
            &expected,
        )
        .unwrap();

        let expected = [Some(false), None, Some(true)].iter().collect();
        apply_logic_op_scalar_arr(
            &schema,
            &ScalarValue::from(false),
            &a,
            Operator::Eq,
            &expected,
        )
        .unwrap();
        apply_logic_op_arr_scalar(
            &schema,
            &a,
            &ScalarValue::from(false),
            Operator::Eq,
            &expected,
        )
        .unwrap();
    }

    #[test]
    fn neq_op_bool() {
        let (schema, a, b) = bool_test_arrays();
        let expected = [
            Some(false),
            None,
            Some(true),
            None,
            None,
            None,
            Some(true),
            None,
            Some(false),
        ]
        .iter()
        .collect();
        apply_logic_op(&schema, &a, &b, Operator::NotEq, expected).unwrap();
    }

    #[test]
    fn neq_op_bool_scalar() {
        let (schema, a) = scalar_bool_test_array();
        let expected = [Some(false), None, Some(true)].iter().collect();
        apply_logic_op_scalar_arr(
            &schema,
            &ScalarValue::from(true),
            &a,
            Operator::NotEq,
            &expected,
        )
        .unwrap();
        apply_logic_op_arr_scalar(
            &schema,
            &a,
            &ScalarValue::from(true),
            Operator::NotEq,
            &expected,
        )
        .unwrap();

        let expected = [Some(true), None, Some(false)].iter().collect();
        apply_logic_op_scalar_arr(
            &schema,
            &ScalarValue::from(false),
            &a,
            Operator::NotEq,
            &expected,
        )
        .unwrap();
        apply_logic_op_arr_scalar(
            &schema,
            &a,
            &ScalarValue::from(false),
            Operator::NotEq,
            &expected,
        )
        .unwrap();
    }

    #[test]
    fn lt_op_bool() {
        let (schema, a, b) = bool_test_arrays();
        let expected = [
            Some(false),
            None,
            Some(false),
            None,
            None,
            None,
            Some(true),
            None,
            Some(false),
        ]
        .iter()
        .collect();
        apply_logic_op(&schema, &a, &b, Operator::Lt, expected).unwrap();
    }

    #[test]
    fn lt_op_bool_scalar() {
        let (schema, a) = scalar_bool_test_array();
        let expected = [Some(false), None, Some(false)].iter().collect();
        apply_logic_op_scalar_arr(
            &schema,
            &ScalarValue::from(true),
            &a,
            Operator::Lt,
            &expected,
        )
        .unwrap();

        let expected = [Some(false), None, Some(true)].iter().collect();
        apply_logic_op_arr_scalar(
            &schema,
            &a,
            &ScalarValue::from(true),
            Operator::Lt,
            &expected,
        )
        .unwrap();

        let expected = [Some(true), None, Some(false)].iter().collect();
        apply_logic_op_scalar_arr(
            &schema,
            &ScalarValue::from(false),
            &a,
            Operator::Lt,
            &expected,
        )
        .unwrap();

        let expected = [Some(false), None, Some(false)].iter().collect();
        apply_logic_op_arr_scalar(
            &schema,
            &a,
            &ScalarValue::from(false),
            Operator::Lt,
            &expected,
        )
        .unwrap();
    }

    #[test]
    fn lt_eq_op_bool() {
        let (schema, a, b) = bool_test_arrays();
        let expected = [
            Some(true),
            None,
            Some(false),
            None,
            None,
            None,
            Some(true),
            None,
            Some(true),
        ]
        .iter()
        .collect();
        apply_logic_op(&schema, &a, &b, Operator::LtEq, expected).unwrap();
    }

    #[test]
    fn lt_eq_op_bool_scalar() {
        let (schema, a) = scalar_bool_test_array();
        let expected = [Some(true), None, Some(false)].iter().collect();
        apply_logic_op_scalar_arr(
            &schema,
            &ScalarValue::from(true),
            &a,
            Operator::LtEq,
            &expected,
        )
        .unwrap();

        let expected = [Some(true), None, Some(true)].iter().collect();
        apply_logic_op_arr_scalar(
            &schema,
            &a,
            &ScalarValue::from(true),
            Operator::LtEq,
            &expected,
        )
        .unwrap();

        let expected = [Some(true), None, Some(true)].iter().collect();
        apply_logic_op_scalar_arr(
            &schema,
            &ScalarValue::from(false),
            &a,
            Operator::LtEq,
            &expected,
        )
        .unwrap();

        let expected = [Some(false), None, Some(true)].iter().collect();
        apply_logic_op_arr_scalar(
            &schema,
            &a,
            &ScalarValue::from(false),
            Operator::LtEq,
            &expected,
        )
        .unwrap();
    }

    #[test]
    fn gt_op_bool() {
        let (schema, a, b) = bool_test_arrays();
        let expected = [
            Some(false),
            None,
            Some(true),
            None,
            None,
            None,
            Some(false),
            None,
            Some(false),
        ]
        .iter()
        .collect();
        apply_logic_op(&schema, &a, &b, Operator::Gt, expected).unwrap();
    }

    #[test]
    fn gt_op_bool_scalar() {
        let (schema, a) = scalar_bool_test_array();
        let expected = [Some(false), None, Some(true)].iter().collect();
        apply_logic_op_scalar_arr(
            &schema,
            &ScalarValue::from(true),
            &a,
            Operator::Gt,
            &expected,
        )
        .unwrap();

        let expected = [Some(false), None, Some(false)].iter().collect();
        apply_logic_op_arr_scalar(
            &schema,
            &a,
            &ScalarValue::from(true),
            Operator::Gt,
            &expected,
        )
        .unwrap();

        let expected = [Some(false), None, Some(false)].iter().collect();
        apply_logic_op_scalar_arr(
            &schema,
            &ScalarValue::from(false),
            &a,
            Operator::Gt,
            &expected,
        )
        .unwrap();

        let expected = [Some(true), None, Some(false)].iter().collect();
        apply_logic_op_arr_scalar(
            &schema,
            &a,
            &ScalarValue::from(false),
            Operator::Gt,
            &expected,
        )
        .unwrap();
    }

    #[test]
    fn gt_eq_op_bool() {
        let (schema, a, b) = bool_test_arrays();
        let expected = [
            Some(true),
            None,
            Some(true),
            None,
            None,
            None,
            Some(false),
            None,
            Some(true),
        ]
        .iter()
        .collect();
        apply_logic_op(&schema, &a, &b, Operator::GtEq, expected).unwrap();
    }

    #[test]
    fn gt_eq_op_bool_scalar() {
        let (schema, a) = scalar_bool_test_array();
        let expected = [Some(true), None, Some(true)].iter().collect();
        apply_logic_op_scalar_arr(
            &schema,
            &ScalarValue::from(true),
            &a,
            Operator::GtEq,
            &expected,
        )
        .unwrap();

        let expected = [Some(true), None, Some(false)].iter().collect();
        apply_logic_op_arr_scalar(
            &schema,
            &a,
            &ScalarValue::from(true),
            Operator::GtEq,
            &expected,
        )
        .unwrap();

        let expected = [Some(false), None, Some(true)].iter().collect();
        apply_logic_op_scalar_arr(
            &schema,
            &ScalarValue::from(false),
            &a,
            Operator::GtEq,
            &expected,
        )
        .unwrap();

        let expected = [Some(true), None, Some(true)].iter().collect();
        apply_logic_op_arr_scalar(
            &schema,
            &a,
            &ScalarValue::from(false),
            Operator::GtEq,
            &expected,
        )
        .unwrap();
    }

    #[test]
    fn is_distinct_from_op_bool() {
        let (schema, a, b) = bool_test_arrays();
        let expected = [
            Some(false),
            Some(true),
            Some(true),
            Some(true),
            Some(false),
            Some(true),
            Some(true),
            Some(true),
            Some(false),
        ]
        .iter()
        .collect();
        apply_logic_op(&schema, &a, &b, Operator::IsDistinctFrom, expected).unwrap();
    }

    #[test]
    fn is_not_distinct_from_op_bool() {
        let (schema, a, b) = bool_test_arrays();
        let expected = [
            Some(true),
            Some(false),
            Some(false),
            Some(false),
            Some(true),
            Some(false),
            Some(false),
            Some(false),
            Some(true),
        ]
        .iter()
        .collect();
        apply_logic_op(&schema, &a, &b, Operator::IsNotDistinctFrom, expected).unwrap();
    }

    #[test]
    fn relatively_deeply_nested() {
        // Reproducer for https://github.com/apache/arrow-datafusion/issues/419

        // where even relatively shallow binary expressions overflowed
        // the stack in debug builds

        let input: Vec<_> = vec![1, 2, 3, 4, 5].into_iter().map(Some).collect();
        let a: Int32Array = input.iter().collect();

        let batch = RecordBatch::try_from_iter(vec![("a", Arc::new(a) as _)]).unwrap();
        let schema = batch.schema();

        // build a left deep tree ((((a + a) + a) + a ....
        let tree_depth: i32 = 100;
        let expr = (0..tree_depth)
            .map(|_| col("a", schema.as_ref()).unwrap())
            .reduce(|l, r| binary_simple(l, Operator::Plus, r, &schema))
            .unwrap();

        let result = expr
            .evaluate(&batch)
            .expect("evaluation")
            .into_array(batch.num_rows());

        let expected: Int32Array = input
            .into_iter()
            .map(|i| i.map(|i| i * tree_depth))
            .collect();
        assert_eq!(result.as_ref(), &expected);
    }

    fn create_decimal_array(
        array: &[Option<i128>],
        precision: u8,
        scale: i8,
    ) -> Result<Decimal128Array> {
        let array = array
            .iter()
            .copied()
            .collect::<Decimal128Array>()
            .with_precision_and_scale(precision, scale)?;
        array.validate_decimal_precision(precision)?;
        Ok(array)
    }

    #[test]
    fn comparison_decimal_op_test() -> Result<()> {
        let value_i128: i128 = 123;
        let decimal_array = create_decimal_array(
            &[
                Some(value_i128),
                None,
                Some(value_i128 - 1),
                Some(value_i128 + 1),
            ],
            25,
            3,
        )?;
        // eq: array = i128
        let result = eq_decimal_scalar(&decimal_array, value_i128)?;
        assert_eq!(
            BooleanArray::from(vec![Some(true), None, Some(false), Some(false)]),
            result
        );
        // neq: array != i128
        let result = neq_decimal_scalar(&decimal_array, value_i128)?;
        assert_eq!(
            BooleanArray::from(vec![Some(false), None, Some(true), Some(true)]),
            result
        );
        // lt: array < i128
        let result = lt_decimal_scalar(&decimal_array, value_i128)?;
        assert_eq!(
            BooleanArray::from(vec![Some(false), None, Some(true), Some(false)]),
            result
        );
        // lt_eq: array <= i128
        let result = lt_eq_decimal_scalar(&decimal_array, value_i128)?;
        assert_eq!(
            BooleanArray::from(vec![Some(true), None, Some(true), Some(false)]),
            result
        );
        // gt: array > i128
        let result = gt_decimal_scalar(&decimal_array, value_i128)?;
        assert_eq!(
            BooleanArray::from(vec![Some(false), None, Some(false), Some(true)]),
            result
        );
        // gt_eq: array >= i128
        let result = gt_eq_decimal_scalar(&decimal_array, value_i128)?;
        assert_eq!(
            BooleanArray::from(vec![Some(true), None, Some(false), Some(true)]),
            result
        );

        let left_decimal_array = decimal_array;
        let right_decimal_array = create_decimal_array(
            &[
                Some(value_i128 - 1),
                Some(value_i128),
                Some(value_i128 + 1),
                Some(value_i128 + 1),
            ],
            25,
            3,
        )?;
        // eq: left == right
        let result = eq_decimal(&left_decimal_array, &right_decimal_array)?;
        assert_eq!(
            BooleanArray::from(vec![Some(false), None, Some(false), Some(true)]),
            result
        );
        // neq: left != right
        let result = neq_decimal(&left_decimal_array, &right_decimal_array)?;
        assert_eq!(
            BooleanArray::from(vec![Some(true), None, Some(true), Some(false)]),
            result
        );
        // lt: left < right
        let result = lt_decimal(&left_decimal_array, &right_decimal_array)?;
        assert_eq!(
            BooleanArray::from(vec![Some(false), None, Some(true), Some(false)]),
            result
        );
        // lt_eq: left <= right
        let result = lt_eq_decimal(&left_decimal_array, &right_decimal_array)?;
        assert_eq!(
            BooleanArray::from(vec![Some(false), None, Some(true), Some(true)]),
            result
        );
        // gt: left > right
        let result = gt_decimal(&left_decimal_array, &right_decimal_array)?;
        assert_eq!(
            BooleanArray::from(vec![Some(true), None, Some(false), Some(false)]),
            result
        );
        // gt_eq: left >= right
        let result = gt_eq_decimal(&left_decimal_array, &right_decimal_array)?;
        assert_eq!(
            BooleanArray::from(vec![Some(true), None, Some(false), Some(true)]),
            result
        );
        // is_distinct: left distinct right
        let result = is_distinct_from_decimal(&left_decimal_array, &right_decimal_array)?;
        assert_eq!(
            BooleanArray::from(vec![Some(true), Some(true), Some(true), Some(false)]),
            result
        );
        // is_distinct: left distinct right
        let result =
            is_not_distinct_from_decimal(&left_decimal_array, &right_decimal_array)?;
        assert_eq!(
            BooleanArray::from(vec![Some(false), Some(false), Some(false), Some(true)]),
            result
        );
        Ok(())
    }

    #[test]
    fn comparison_decimal_expr_test() -> Result<()> {
        let decimal_scalar = ScalarValue::Decimal128(Some(123_456), 10, 3);
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true)]));
        // scalar == array
        apply_logic_op_scalar_arr(
            &schema,
            &decimal_scalar,
            &(Arc::new(Int64Array::from(vec![Some(124), None])) as ArrayRef),
            Operator::Eq,
            &BooleanArray::from(vec![Some(false), None]),
        )
        .unwrap();

        // array != scalar
        apply_logic_op_arr_scalar(
            &schema,
            &(Arc::new(Int64Array::from(vec![Some(123), None, Some(1)])) as ArrayRef),
            &decimal_scalar,
            Operator::NotEq,
            &BooleanArray::from(vec![Some(true), None, Some(true)]),
        )
        .unwrap();

        // array < scalar
        apply_logic_op_arr_scalar(
            &schema,
            &(Arc::new(Int64Array::from(vec![Some(123), None, Some(124)])) as ArrayRef),
            &decimal_scalar,
            Operator::Lt,
            &BooleanArray::from(vec![Some(true), None, Some(false)]),
        )
        .unwrap();

        // array > scalar
        apply_logic_op_arr_scalar(
            &schema,
            &(Arc::new(Int64Array::from(vec![Some(123), None, Some(124)])) as ArrayRef),
            &decimal_scalar,
            Operator::Gt,
            &BooleanArray::from(vec![Some(false), None, Some(true)]),
        )
        .unwrap();

        let schema =
            Arc::new(Schema::new(vec![Field::new("a", DataType::Float64, true)]));
        // array == scalar
        apply_logic_op_arr_scalar(
            &schema,
            &(Arc::new(Float64Array::from(vec![Some(123.456), None, Some(123.457)]))
                as ArrayRef),
            &decimal_scalar,
            Operator::Eq,
            &BooleanArray::from(vec![Some(true), None, Some(false)]),
        )
        .unwrap();

        // array <= scalar
        apply_logic_op_arr_scalar(
            &schema,
            &(Arc::new(Float64Array::from(vec![
                Some(123.456),
                None,
                Some(123.457),
                Some(123.45),
            ])) as ArrayRef),
            &decimal_scalar,
            Operator::LtEq,
            &BooleanArray::from(vec![Some(true), None, Some(false), Some(true)]),
        )
        .unwrap();
        // array >= scalar
        apply_logic_op_arr_scalar(
            &schema,
            &(Arc::new(Float64Array::from(vec![
                Some(123.456),
                None,
                Some(123.457),
                Some(123.45),
            ])) as ArrayRef),
            &decimal_scalar,
            Operator::GtEq,
            &BooleanArray::from(vec![Some(true), None, Some(true), Some(false)]),
        )
        .unwrap();

        // compare decimal array with other array type
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Decimal128(10, 0), true),
        ]));

        let value: i64 = 123;

        let decimal_array = Arc::new(create_decimal_array(
            &[
                Some(value as i128),
                None,
                Some((value - 1) as i128),
                Some((value + 1) as i128),
            ],
            10,
            0,
        )?) as ArrayRef;

        let int64_array = Arc::new(Int64Array::from(vec![
            Some(value),
            Some(value - 1),
            Some(value),
            Some(value + 1),
        ])) as ArrayRef;

        // eq: int64array == decimal array
        apply_logic_op(
            &schema,
            &int64_array,
            &decimal_array,
            Operator::Eq,
            BooleanArray::from(vec![Some(true), None, Some(false), Some(true)]),
        )
        .unwrap();
        // neq: int64array != decimal array
        apply_logic_op(
            &schema,
            &int64_array,
            &decimal_array,
            Operator::NotEq,
            BooleanArray::from(vec![Some(false), None, Some(true), Some(false)]),
        )
        .unwrap();

        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Float64, true),
            Field::new("b", DataType::Decimal128(10, 2), true),
        ]));

        let value: i128 = 123;
        let decimal_array = Arc::new(create_decimal_array(
            &[
                Some(value), // 1.23
                None,
                Some(value - 1), // 1.22
                Some(value + 1), // 1.24
            ],
            10,
            2,
        )?) as ArrayRef;
        let float64_array = Arc::new(Float64Array::from(vec![
            Some(1.23),
            Some(1.22),
            Some(1.23),
            Some(1.24),
        ])) as ArrayRef;
        // lt: float64array < decimal array
        apply_logic_op(
            &schema,
            &float64_array,
            &decimal_array,
            Operator::Lt,
            BooleanArray::from(vec![Some(false), None, Some(false), Some(false)]),
        )
        .unwrap();
        // lt_eq: float64array <= decimal array
        apply_logic_op(
            &schema,
            &float64_array,
            &decimal_array,
            Operator::LtEq,
            BooleanArray::from(vec![Some(true), None, Some(false), Some(true)]),
        )
        .unwrap();
        // gt: float64array > decimal array
        apply_logic_op(
            &schema,
            &float64_array,
            &decimal_array,
            Operator::Gt,
            BooleanArray::from(vec![Some(false), None, Some(true), Some(false)]),
        )
        .unwrap();
        apply_logic_op(
            &schema,
            &float64_array,
            &decimal_array,
            Operator::GtEq,
            BooleanArray::from(vec![Some(true), None, Some(true), Some(true)]),
        )
        .unwrap();
        // is distinct: float64array is distinct decimal array
        // TODO: now we do not refactor the `is distinct or is not distinct` rule of coercion.
        // traced by https://github.com/apache/arrow-datafusion/issues/1590
        // the decimal array will be casted to float64array
        apply_logic_op(
            &schema,
            &float64_array,
            &decimal_array,
            Operator::IsDistinctFrom,
            BooleanArray::from(vec![Some(false), Some(true), Some(true), Some(false)]),
        )
        .unwrap();
        // is not distinct
        apply_logic_op(
            &schema,
            &float64_array,
            &decimal_array,
            Operator::IsNotDistinctFrom,
            BooleanArray::from(vec![Some(true), Some(false), Some(false), Some(true)]),
        )
        .unwrap();

        Ok(())
    }

    #[test]
    fn arithmetic_decimal_op_test() -> Result<()> {
        let value_i128: i128 = 123;
        let left_decimal_array = create_decimal_array(
            &[
                Some(value_i128),
                None,
                Some(value_i128 - 1),
                Some(value_i128 + 1),
            ],
            25,
            3,
        )?;
        let right_decimal_array = create_decimal_array(
            &[
                Some(value_i128),
                Some(value_i128),
                Some(value_i128),
                Some(value_i128),
            ],
            25,
            3,
        )?;
        // add
        let result = add_decimal(&left_decimal_array, &right_decimal_array)?;
        let expect =
            create_decimal_array(&[Some(246), None, Some(245), Some(247)], 25, 3)?;
        assert_eq!(expect, result);
        // subtract
        let result = subtract_decimal(&left_decimal_array, &right_decimal_array)?;
        let expect = create_decimal_array(&[Some(0), None, Some(-1), Some(1)], 25, 3)?;
        assert_eq!(expect, result);
        // multiply
        let result = multiply_decimal(&left_decimal_array, &right_decimal_array)?;
        let expect = create_decimal_array(&[Some(15), None, Some(15), Some(15)], 25, 3)?;
        assert_eq!(expect, result);
        // divide
        let left_decimal_array = create_decimal_array(
            &[Some(1234567), None, Some(1234567), Some(1234567)],
            25,
            3,
        )?;
        let right_decimal_array =
            create_decimal_array(&[Some(10), Some(100), Some(55), Some(-123)], 25, 3)?;
        let result = divide_decimal(&left_decimal_array, &right_decimal_array)?;
        let expect = create_decimal_array(
            &[Some(123456700), None, Some(22446672), Some(-10037130)],
            25,
            3,
        )?;
        assert_eq!(expect, result);
        // modulus
        let result = modulus_decimal(&left_decimal_array, &right_decimal_array)?;
        let expect = create_decimal_array(&[Some(7), None, Some(37), Some(16)], 25, 3)?;
        assert_eq!(expect, result);

        Ok(())
    }

    fn apply_arithmetic_op(
        schema: &SchemaRef,
        left: &ArrayRef,
        right: &ArrayRef,
        op: Operator,
        expected: ArrayRef,
    ) -> Result<()> {
        let arithmetic_op =
            binary_simple(col("a", schema)?, op, col("b", schema)?, schema);
        let data: Vec<ArrayRef> = vec![left.clone(), right.clone()];
        let batch = RecordBatch::try_new(schema.clone(), data)?;
        let result = arithmetic_op.evaluate(&batch)?.into_array(batch.num_rows());

        assert_eq!(result.as_ref(), expected.as_ref());
        Ok(())
    }

    #[test]
    fn arithmetic_decimal_expr_test() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Decimal128(10, 2), true),
        ]));
        let value: i128 = 123;
        let decimal_array = Arc::new(create_decimal_array(
            &[
                Some(value), // 1.23
                None,
                Some(value - 1), // 1.22
                Some(value + 1), // 1.24
            ],
            10,
            2,
        )?) as ArrayRef;
        let int32_array = Arc::new(Int32Array::from(vec![
            Some(123),
            Some(122),
            Some(123),
            Some(124),
        ])) as ArrayRef;

        // add: Int32array add decimal array
        let expect = Arc::new(create_decimal_array(
            &[Some(12423), None, Some(12422), Some(12524)],
            13,
            2,
        )?) as ArrayRef;
        apply_arithmetic_op(
            &schema,
            &int32_array,
            &decimal_array,
            Operator::Plus,
            expect,
        )
        .unwrap();

        // subtract: decimal array subtract int32 array
        let schema = Arc::new(Schema::new(vec![
            Field::new("b", DataType::Int32, true),
            Field::new("a", DataType::Decimal128(10, 2), true),
        ]));
        let expect = Arc::new(create_decimal_array(
            &[Some(-12177), None, Some(-12178), Some(-12276)],
            13,
            2,
        )?) as ArrayRef;
        apply_arithmetic_op(
            &schema,
            &int32_array,
            &decimal_array,
            Operator::Minus,
            expect,
        )
        .unwrap();

        // multiply: decimal array multiply int32 array
        let expect = Arc::new(create_decimal_array(
            &[Some(15129), None, Some(15006), Some(15376)],
            21,
            2,
        )?) as ArrayRef;
        apply_arithmetic_op(
            &schema,
            &int32_array,
            &decimal_array,
            Operator::Multiply,
            expect,
        )
        .unwrap();
        // divide: int32 array divide decimal array
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Decimal128(10, 2), true),
        ]));
        let expect = Arc::new(create_decimal_array(
            &[
                Some(10000000000000),
                None,
                Some(10081967213114),
                Some(10000000000000),
            ],
            23,
            11,
        )?) as ArrayRef;
        apply_arithmetic_op(
            &schema,
            &int32_array,
            &decimal_array,
            Operator::Divide,
            expect,
        )
        .unwrap();
        // modulus: int32 array modulus decimal array
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Decimal128(10, 2), true),
        ]));
        let expect = Arc::new(create_decimal_array(
            &[Some(000), None, Some(100), Some(000)],
            10,
            2,
        )?) as ArrayRef;
        apply_arithmetic_op(
            &schema,
            &int32_array,
            &decimal_array,
            Operator::Modulo,
            expect,
        )
        .unwrap();

        Ok(())
    }

    #[test]
    fn bitwise_array_test() -> Result<()> {
        let left = Arc::new(Int32Array::from(vec![Some(12), None, Some(11)])) as ArrayRef;
        let right =
            Arc::new(Int32Array::from(vec![Some(1), Some(3), Some(7)])) as ArrayRef;
        let mut result = bitwise_and(left.clone(), right.clone())?;
        let expected = Int32Array::from(vec![Some(0), None, Some(3)]);
        assert_eq!(result.as_ref(), &expected);

        result = bitwise_or(left.clone(), right.clone())?;
        let expected = Int32Array::from(vec![Some(13), None, Some(15)]);
        assert_eq!(result.as_ref(), &expected);
        Ok(())
    }

    #[test]
    fn bitwise_shift_array_test() -> Result<()> {
        let input = Arc::new(Int32Array::from(vec![Some(2), None, Some(10)])) as ArrayRef;
        let modules =
            Arc::new(Int32Array::from(vec![Some(2), Some(4), Some(8)])) as ArrayRef;
        let mut result = bitwise_shift_left(input.clone(), modules.clone())?;

        let expected = Int32Array::from(vec![Some(8), None, Some(2560)]);
        assert_eq!(result.as_ref(), &expected);

        result = bitwise_shift_right(result.clone(), modules.clone())?;
        assert_eq!(result.as_ref(), &input);

        Ok(())
    }

    #[test]
    fn bitwise_scalar_test() -> Result<()> {
        let left = Arc::new(Int32Array::from(vec![Some(12), None, Some(11)])) as ArrayRef;
        let right = ScalarValue::from(3i32);
        let mut result = bitwise_and_scalar(&left, right.clone()).unwrap()?;
        let expected = Int32Array::from(vec![Some(0), None, Some(3)]);
        assert_eq!(result.as_ref(), &expected);

        result = bitwise_or_scalar(&left, right).unwrap()?;
        let expected = Int32Array::from(vec![Some(15), None, Some(11)]);
        assert_eq!(result.as_ref(), &expected);
        Ok(())
    }

    #[test]
    fn bitwise_shift_scalar_test() -> Result<()> {
        let input = Arc::new(Int32Array::from(vec![Some(2), None, Some(4)])) as ArrayRef;
        let module = ScalarValue::from(10i32);
        let mut result = bitwise_shift_left_scalar(&input, module.clone()).unwrap()?;

        let expected = Int32Array::from(vec![Some(2048), None, Some(4096)]);
        assert_eq!(result.as_ref(), &expected);

        result = bitwise_shift_right_scalar(&result, module).unwrap()?;
        assert_eq!(result.as_ref(), &input);

        Ok(())
    }

    #[test]
    fn bitwise_shift_scalar_test_i64() -> Result<()> {
        let input = Arc::new(Int64Array::from(vec![Some(2), None, Some(4)])) as ArrayRef;
        let module = ScalarValue::from(10i64);
        let mut result = bitwise_shift_left_scalar(&input, module.clone()).unwrap()?;

        let expected = Int64Array::from(vec![Some(2048), None, Some(4096)]);
        assert_eq!(result.as_ref(), &expected);

        result = bitwise_shift_right_scalar(&result, module).unwrap()?;
        assert_eq!(result.as_ref(), &input);

        Ok(())
    }
}
