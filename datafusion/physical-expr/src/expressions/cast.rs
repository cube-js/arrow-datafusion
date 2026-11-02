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

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use crate::datetime_expressions::{
    string_to_datetime_local, string_to_timestamp_nanos_local,
};
use crate::PhysicalExpr;
use arrow::array::{
    new_null_array, Array, ArrayRef, AsArray, Decimal128Array, GenericStringArray,
    OffsetSizeTrait, TimestampMicrosecondArray, TimestampMillisecondArray,
    TimestampNanosecondArray, TimestampSecondArray,
};
use arrow::compute;
use arrow::compute::kernels;
use arrow::compute::CastOptions;
use arrow::datatypes::{
    ArrowPrimitiveType, DataType, Date64Type, Decimal128Type, Float32Type, Float64Type,
    IntervalDayTimeType, IntervalMonthDayNanoType, IntervalUnit, IntervalYearMonthType,
    Schema, TimeUnit, TimestampMicrosecondType, TimestampMillisecondType,
    TimestampNanosecondType, TimestampSecondType, DECIMAL128_MAX_PRECISION,
    MAX_DECIMAL128_FOR_EACH_PRECISION, MIN_DECIMAL128_FOR_EACH_PRECISION,
};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use arrow::temporal_conversions::as_datetime;
use arrow::util::display::FormatOptions;
use datafusion_common::ScalarValue;
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::ColumnarValue;

/// provide DataFusion default cast options
pub const DEFAULT_DATAFUSION_CAST_OPTIONS: CastOptions<'static> = CastOptions {
    safe: false,
    format_options: FormatOptions::new(),
};

/// Returns true if a value of type `from_type` can be cast into a value of `to_type`
/// with [`cast_array_with_options`].
///
/// Unlike arrow's `can_cast_types`, every type can be cast to `DataType::Null`
/// (arrow 13 allowed it, the result is an array of nulls).
pub fn can_cast_types(from_type: &DataType, to_type: &DataType) -> bool {
    to_type == &DataType::Null || compute::can_cast_types(from_type, to_type)
}

/// Casts `array` to `to_type`, keeping the arrow 13 semantics DataFusion relies on:
///
/// * casting to `DataType::Null` yields an array of nulls;
/// * casting a timestamp without a timezone or a date to a timestamp with a timezone
///   only attaches the timezone (arrow's `cast` would interpret the values as local
///   wall-clock time of that timezone and shift them);
/// * casting a timestamp with a timezone to a date takes the date of the UTC value
///   (arrow's `cast` takes the local date in that timezone);
/// * timestamps and `Date64` are cast to strings with chrono's `NaiveDateTime` `Display`
///   of the UTC value (`2020-01-01 00:00:00.123`, arrow's `cast` uses the ISO 8601 `T`
///   separator and appends the offset for timestamps with a timezone), unless
///   `cast_options` sets the corresponding format;
/// * booleans are cast to the strings `1` and `0` (arrow's `cast` yields `true` and
///   `false`);
/// * intervals are cast to strings in PostgreSQL's default `postgres` interval style
///   (`1 year 2 mons 3 days 04:05:06.789000`, see [`interval_to_string`]), the format
///   the SQL API writes intervals to the wire with (arrow's `cast` yields
///   `1 years 2 mons 3 days 4 hours 5 mins 6.789 secs`);
/// * a cast between integer and floating point types turns a value that does not fit
///   the target type (overflow, NaN or infinity to an integer) into NULL, whatever
///   `cast_options.safe` says (arrow 13 ignored `safe` for numeric casts; arrow's `cast`
///   errors when `safe` is false);
/// * casts from floats, strings and decimals with a larger scale to decimals truncate
///   extra digits instead of rounding them, a float NaN becomes 0, strings are parsed as
///   `f64` (see [`string_to_decimal`]) and every value is checked against the target
///   precision with arrow 13's error;
/// * strings without an offset or time zone are cast to timestamps without a time zone
///   as wall-clock times of the process-local time zone (arrow's `cast` reads them as
///   UTC).
///
/// Errors are arrow errors, as returned by arrow's `cast`.
pub fn cast_array_with_options(
    array: &ArrayRef,
    to_type: &DataType,
    cast_options: &CastOptions,
) -> std::result::Result<ArrayRef, ArrowError> {
    let format_options = &cast_options.format_options;
    match (array.data_type(), to_type) {
        (from_type, DataType::Null) if from_type != &DataType::Null => {
            Ok(new_null_array(&DataType::Null, array.len()))
        }
        (
            DataType::Timestamp(_, None) | DataType::Date32 | DataType::Date64,
            DataType::Timestamp(unit, Some(tz)),
        ) => {
            let array = kernels::cast::cast_with_options(
                array,
                &DataType::Timestamp(*unit, None),
                cast_options,
            )?;
            timestamp_with_timezone(&array, Some(tz.clone()))
        }
        (DataType::Timestamp(_, Some(_)), DataType::Date32 | DataType::Date64) => {
            kernels::cast::cast_with_options(
                &timestamp_with_timezone(array, None)?,
                to_type,
                cast_options,
            )
        }
        (DataType::Timestamp(unit, tz), DataType::Utf8)
            if has_default_timestamp_format(tz, format_options) =>
        {
            Ok(timestamp_to_string::<i32>(array, unit))
        }
        (DataType::Timestamp(unit, tz), DataType::LargeUtf8)
            if has_default_timestamp_format(tz, format_options) =>
        {
            Ok(timestamp_to_string::<i64>(array, unit))
        }
        (DataType::Date64, DataType::Utf8)
            if format_options.datetime_format().is_none() =>
        {
            Ok(datetime_to_string::<Date64Type, i32>(array))
        }
        (DataType::Date64, DataType::LargeUtf8)
            if format_options.datetime_format().is_none() =>
        {
            Ok(datetime_to_string::<Date64Type, i64>(array))
        }
        (DataType::Boolean, DataType::Utf8) => Ok(boolean_to_string::<i32>(array)),
        (DataType::Boolean, DataType::LargeUtf8) => Ok(boolean_to_string::<i64>(array)),
        (DataType::Interval(_), DataType::Utf8) => interval_to_string::<i32>(array),
        (DataType::Interval(_), DataType::LargeUtf8) => interval_to_string::<i64>(array),
        (from_type, to_type)
            if is_integer_or_float(from_type) && is_integer_or_float(to_type) =>
        {
            kernels::cast::cast_with_options(
                array,
                to_type,
                &CastOptions {
                    safe: true,
                    ..cast_options.clone()
                },
            )
        }
        (DataType::Float32, DataType::Decimal128(precision, scale)) if *scale >= 0 => {
            float_to_decimal(
                array
                    .as_primitive::<Float32Type>()
                    .iter()
                    .map(|v| v.map(f64::from)),
                *precision,
                *scale,
            )
        }
        (DataType::Float64, DataType::Decimal128(precision, scale)) if *scale >= 0 => {
            float_to_decimal(
                array.as_primitive::<Float64Type>().iter(),
                *precision,
                *scale,
            )
        }
        (DataType::Decimal128(_, from_scale), DataType::Decimal128(precision, scale))
            if *scale >= 0 && from_scale > scale =>
        {
            let div = 10_i128.pow((*from_scale - *scale) as u32);
            decimal_array_checked(
                array
                    .as_primitive::<Decimal128Type>()
                    .iter()
                    .map(|v| v.map(|v| v / div)),
                *precision,
                *scale,
            )
        }
        (DataType::Utf8, DataType::Decimal128(precision, scale)) if *scale >= 0 => {
            string_to_decimal::<i32>(array, *precision, *scale)
        }
        (DataType::LargeUtf8, DataType::Decimal128(precision, scale)) if *scale >= 0 => {
            string_to_decimal::<i64>(array, *precision, *scale)
        }
        (DataType::Utf8, DataType::Timestamp(unit, None)) => {
            string_to_timestamp::<i32>(array, unit, cast_options)
        }
        (DataType::LargeUtf8, DataType::Timestamp(unit, None)) => {
            string_to_timestamp::<i64>(array, unit, cast_options)
        }
        _ => kernels::cast::cast_with_options(array, to_type, cast_options),
    }
}

/// Returns true if [`cast_array_with_options`] with `cast_options` can turn a non-null
/// value of `from_type` into NULL when casting it to `to_type`, i.e. whether the result
/// of the cast must be nullable even when its input is not.
///
/// This happens for casts that keep arrow 13's NULL-instead-of-error semantics:
/// integer/float casts that can overflow (or get a NaN or infinity), strings that are
/// not numbers cast to decimals, and casts to `DataType::Null`. With `cast_options.safe`
/// every failed cast is NULL.
pub fn cast_may_produce_null(
    from_type: &DataType,
    to_type: &DataType,
    cast_options: &CastOptions,
) -> bool {
    if from_type == to_type || from_type == &DataType::Null {
        return false;
    }
    if cast_options.safe {
        return true;
    }
    match (from_type, to_type) {
        (_, DataType::Null) => true,
        (DataType::Utf8 | DataType::LargeUtf8, DataType::Decimal128(_, scale)) => {
            *scale >= 0
        }
        (from_type, to_type)
            if is_integer_or_float(from_type) && is_integer_or_float(to_type) =>
        {
            !is_lossless_numeric_cast(from_type, to_type)
        }
        _ => false,
    }
}

/// Returns true if every value of the integer or float type `from_type` can be cast to
/// the integer or float type `to_type` without overflow (casts to floats saturate to
/// infinity instead of overflowing)
fn is_lossless_numeric_cast(from_type: &DataType, to_type: &DataType) -> bool {
    if to_type.is_floating() {
        return true;
    }
    if from_type.is_floating() {
        return false;
    }
    let (from_width, to_width) = (from_type.primitive_width(), to_type.primitive_width());
    match (from_type.is_signed_integer(), to_type.is_signed_integer()) {
        (true, true) | (false, false) => to_width >= from_width,
        (false, true) => to_width > from_width,
        (true, false) => false,
    }
}

/// Returns true for the types arrow 13 cast between with its `numeric_cast` (integers,
/// `Float32` and `Float64`)
fn is_integer_or_float(data_type: &DataType) -> bool {
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

/// Builds a `Decimal128(precision, scale)` array from `values`, checking every value
/// against `precision` like arrow 13's `DecimalArray::with_precision_and_scale` did for
/// the arrays its casts built (they had precision 38, so values were only checked for a
/// smaller target precision)
fn decimal_array_checked(
    values: impl Iterator<Item = Option<i128>>,
    precision: u8,
    scale: i8,
) -> std::result::Result<ArrayRef, ArrowError> {
    let array = values.collect::<Decimal128Array>();
    if precision < DECIMAL128_MAX_PRECISION {
        let max = MAX_DECIMAL128_FOR_EACH_PRECISION[precision as usize];
        let min = MIN_DECIMAL128_FOR_EACH_PRECISION[precision as usize];
        for value in array.iter().flatten() {
            if value > max {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "{} is too large to store in a Decimal of precision {}. Max is {}",
                    value, precision, max
                )));
            }
            if value < min {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "{} is too small to store in a Decimal of precision {}. Min is {}",
                    value, precision, min
                )));
            }
        }
    }
    Ok(Arc::new(array.with_precision_and_scale(precision, scale)?))
}

/// Casts floats to `Decimal128(precision, scale)` like arrow 13: `v * 10^scale`
/// truncated towards zero with a saturating `as i128` (NaN becomes 0)
fn float_to_decimal(
    values: impl Iterator<Item = Option<f64>>,
    precision: u8,
    scale: i8,
) -> std::result::Result<ArrayRef, ArrowError> {
    let mul = 10_f64.powi(scale as i32);
    decimal_array_checked(
        values.map(|v| v.map(|v| (v * mul) as i128)),
        precision,
        scale,
    )
}

/// Parses a string as `f64` for a cast to decimal, like arrow 13 did with
/// `lexical_core::parse::<f64>`: a string without any digit in its mantissa (such as
/// `abc` or `.`) is NULL, an empty string (or a lone sign) and any other malformed number
/// are an error.
fn parse_decimal_string(s: &str) -> std::result::Result<Option<f64>, ArrowError> {
    let unsigned = s.strip_prefix(['+', '-']).unwrap_or(s);
    if unsigned.is_empty() {
        return Err(ArrowError::CastError(
            "Cannot cast from string to decimal".to_string(),
        ));
    }
    match s.parse::<f64>() {
        Ok(value) => Ok(Some(value)),
        Err(_) => {
            let integer_digits =
                unsigned.bytes().take_while(|b| b.is_ascii_digit()).count();
            let fraction_digits = unsigned[integer_digits..]
                .strip_prefix('.')
                .map(|fraction| {
                    fraction.bytes().take_while(|b| b.is_ascii_digit()).count()
                })
                .unwrap_or(0);
            if integer_digits + fraction_digits == 0 {
                // lexical_core's `EmptyMantissa`, which arrow 13 cast to NULL
                Ok(None)
            } else {
                Err(ArrowError::CastError(
                    "Cannot cast from string to decimal".to_string(),
                ))
            }
        }
    }
}

/// Casts strings to `Decimal128(precision, scale)` like arrow 13: the string is parsed
/// as `f64` (see [`parse_decimal_string`]) and converted like [`float_to_decimal`].
/// Parse errors are returned whatever `CastOptions::safe` says, like in arrow 13.
fn string_to_decimal<O: OffsetSizeTrait>(
    array: &ArrayRef,
    precision: u8,
    scale: i8,
) -> std::result::Result<ArrayRef, ArrowError> {
    let values = array
        .as_string::<O>()
        .iter()
        .map(|v| v.map(parse_decimal_string).transpose().map(Option::flatten))
        .collect::<std::result::Result<Vec<_>, ArrowError>>()?;
    float_to_decimal(values.into_iter(), precision, scale)
}

/// Casts strings to timestamps without a time zone, reading strings without an offset
/// as process-local wall-clock times like arrow 13 (see
/// [`string_to_datetime_local`]). An unparsable string is NULL when `cast_options.safe`
/// is set and an error otherwise.
fn string_to_timestamp<O: OffsetSizeTrait>(
    array: &ArrayRef,
    unit: &TimeUnit,
    cast_options: &CastOptions,
) -> std::result::Result<ArrayRef, ArrowError> {
    let parse = |s: &str| -> std::result::Result<i64, ArrowError> {
        match unit {
            TimeUnit::Second => Ok(string_to_datetime_local(s)?.timestamp()),
            TimeUnit::Millisecond => Ok(string_to_datetime_local(s)?.timestamp_millis()),
            TimeUnit::Microsecond => Ok(string_to_datetime_local(s)?.timestamp_micros()),
            TimeUnit::Nanosecond => string_to_timestamp_nanos_local(s),
        }
    };
    let strings = array.as_string::<O>();
    let values = if cast_options.safe {
        strings
            .iter()
            .map(|v| v.and_then(|v| parse(v).ok()))
            .collect::<Vec<_>>()
    } else {
        strings
            .iter()
            .map(|v| v.map(parse).transpose())
            .collect::<std::result::Result<Vec<_>, ArrowError>>()?
    };
    Ok(match unit {
        TimeUnit::Second => Arc::new(TimestampSecondArray::from(values)),
        TimeUnit::Millisecond => Arc::new(TimestampMillisecondArray::from(values)),
        TimeUnit::Microsecond => Arc::new(TimestampMicrosecondArray::from(values)),
        TimeUnit::Nanosecond => Arc::new(TimestampNanosecondArray::from(values)),
    })
}

/// Replaces the timezone of a timestamp array without changing its values
fn timestamp_with_timezone(
    array: &ArrayRef,
    tz: Option<Arc<str>>,
) -> std::result::Result<ArrayRef, ArrowError> {
    Ok(match array.data_type() {
        DataType::Timestamp(TimeUnit::Second, _) => Arc::new(
            array
                .as_primitive::<TimestampSecondType>()
                .clone()
                .with_timezone_opt(tz),
        ),
        DataType::Timestamp(TimeUnit::Millisecond, _) => Arc::new(
            array
                .as_primitive::<TimestampMillisecondType>()
                .clone()
                .with_timezone_opt(tz),
        ),
        DataType::Timestamp(TimeUnit::Microsecond, _) => Arc::new(
            array
                .as_primitive::<TimestampMicrosecondType>()
                .clone()
                .with_timezone_opt(tz),
        ),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => Arc::new(
            array
                .as_primitive::<TimestampNanosecondType>()
                .clone()
                .with_timezone_opt(tz),
        ),
        other => {
            return Err(ArrowError::InvalidArgumentError(format!(
                "Expected a timestamp array, got {:?}",
                other
            )))
        }
    })
}

/// Returns true if `options` sets no format for timestamps with the timezone `tz`
fn has_default_timestamp_format(tz: &Option<Arc<str>>, options: &FormatOptions) -> bool {
    match tz {
        None => options.timestamp_format().is_none(),
        Some(_) => options.timestamp_tz_format().is_none(),
    }
}

/// Formats every value of a timestamp array (ignoring its timezone, i.e. as UTC) like
/// arrow 13's `cast` did
fn timestamp_to_string<O: OffsetSizeTrait>(
    array: &ArrayRef,
    unit: &TimeUnit,
) -> ArrayRef {
    match unit {
        TimeUnit::Second => datetime_to_string::<TimestampSecondType, O>(array),
        TimeUnit::Millisecond => datetime_to_string::<TimestampMillisecondType, O>(array),
        TimeUnit::Microsecond => datetime_to_string::<TimestampMicrosecondType, O>(array),
        TimeUnit::Nanosecond => datetime_to_string::<TimestampNanosecondType, O>(array),
    }
}

/// Formats every value of a timestamp or `Date64` array with chrono's `NaiveDateTime`
/// `Display`. A value chrono cannot represent becomes NULL, like in arrow 13.
fn datetime_to_string<T, O>(array: &ArrayRef) -> ArrayRef
where
    T: ArrowPrimitiveType<Native = i64>,
    O: OffsetSizeTrait,
{
    let array = array.as_primitive::<T>();
    Arc::new(
        array
            .iter()
            .map(|v| v.and_then(as_datetime::<T>).map(|v| v.to_string()))
            .collect::<GenericStringArray<O>>(),
    )
}

/// Formats every value of a boolean array as `1` or `0`, like arrow 13's `cast` did
fn boolean_to_string<O: OffsetSizeTrait>(array: &ArrayRef) -> ArrayRef {
    Arc::new(
        array
            .as_boolean()
            .iter()
            .map(|v| v.map(|v| if v { "1" } else { "0" }))
            .collect::<GenericStringArray<O>>(),
    )
}

/// Casts an interval array to strings in PostgreSQL's default `postgres` interval style
fn interval_to_string<O: OffsetSizeTrait>(
    array: &ArrayRef,
) -> std::result::Result<ArrayRef, ArrowError> {
    // (months, days, microseconds): PostgreSQL keeps intervals with microsecond precision
    let parts: Vec<Option<(i32, i32, i64)>> = match array.data_type() {
        DataType::Interval(IntervalUnit::YearMonth) => array
            .as_primitive::<IntervalYearMonthType>()
            .iter()
            .map(|v| v.map(|months| (months, 0, 0)))
            .collect(),
        DataType::Interval(IntervalUnit::DayTime) => array
            .as_primitive::<IntervalDayTimeType>()
            .iter()
            .map(|v| v.map(|v| (0, v.days, v.milliseconds as i64 * 1_000)))
            .collect(),
        DataType::Interval(IntervalUnit::MonthDayNano) => array
            .as_primitive::<IntervalMonthDayNanoType>()
            .iter()
            .map(|v| v.map(|v| (v.months, v.days, v.nanoseconds / 1_000)))
            .collect(),
        other => {
            return Err(ArrowError::CastError(format!(
                "Unsupported CAST from {} to Utf8",
                other
            )))
        }
    };

    Ok(Arc::new(
        parts
            .into_iter()
            .map(|v| {
                v.map(|(months, days, micros)| format_interval(months, days, micros))
            })
            .collect::<GenericStringArray<O>>(),
    ))
}

/// Formats an interval like PostgreSQL's `postgres` interval style, e.g.
/// `1 year 2 mons 3 days 04:05:06.789000`, `-1 days -02:00:00` or `00:00:00`
fn format_interval(months: i32, days: i32, micros: i64) -> String {
    fn unit(res: &mut String, value: i32, singular: &str, plural: &str) {
        if value != 0 {
            let name = if value == 1 { singular } else { plural };
            res.push_str(&format!("{} {} ", value, name));
        }
    }

    if months == 0 && days == 0 && micros == 0 {
        return "00:00:00".to_string();
    }

    let mut res = String::new();
    unit(&mut res, months / 12, "year", "years");
    unit(&mut res, months % 12, "mon", "mons");
    unit(&mut res, days, "day", "days");

    if micros != 0 {
        if micros < 0 {
            res.push('-');
        }
        let micros = micros.unsigned_abs();
        let secs = micros / 1_000_000;
        res.push_str(&format!(
            "{:02}:{:02}:{:02}",
            secs / 3600,
            secs / 60 % 60,
            secs % 60
        ));
        let fraction = micros % 1_000_000;
        if fraction != 0 {
            res.push_str(&format!(".{:06}", fraction));
        }
    }

    res.trim_end().to_string()
}

/// CAST expression casts an expression to a specific data type and returns a runtime error on invalid cast
#[derive(Debug)]
pub struct CastExpr {
    /// The expression to cast
    expr: Arc<dyn PhysicalExpr>,
    /// The data type to cast to
    cast_type: DataType,
    /// Cast options
    cast_options: CastOptions<'static>,
}

impl CastExpr {
    /// Create a new CastExpr
    pub fn new(
        expr: Arc<dyn PhysicalExpr>,
        cast_type: DataType,
        cast_options: CastOptions<'static>,
    ) -> Self {
        Self {
            expr,
            cast_type,
            cast_options,
        }
    }

    /// The expression to cast
    pub fn expr(&self) -> &Arc<dyn PhysicalExpr> {
        &self.expr
    }

    /// The data type to cast to
    pub fn cast_type(&self) -> &DataType {
        &self.cast_type
    }
}

impl fmt::Display for CastExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "CAST({} AS {:?})", self.expr, self.cast_type)
    }
}

impl PhysicalExpr for CastExpr {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(self.cast_type.clone())
    }

    fn nullable(&self, input_schema: &Schema) -> Result<bool> {
        // a cast can turn values into NULL (e.g. an overflowing numeric cast): the result
        // is then nullable even if the input is not, otherwise arrow rejects the batch
        Ok(self.expr.nullable(input_schema)?
            || cast_may_produce_null(
                &self.expr.data_type(input_schema)?,
                &self.cast_type,
                &self.cast_options,
            ))
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let value = self.expr.evaluate(batch)?;
        cast_column(&value, &self.cast_type, &self.cast_options)
    }
}

/// Internal cast function for casting ColumnarValue -> ColumnarValue for cast_type
pub fn cast_column(
    value: &ColumnarValue,
    cast_type: &DataType,
    cast_options: &CastOptions,
) -> Result<ColumnarValue> {
    match value {
        ColumnarValue::Array(array) => Ok(ColumnarValue::Array(cast_array_with_options(
            array,
            cast_type,
            cast_options,
        )?)),
        ColumnarValue::Scalar(scalar) => {
            let scalar_array = scalar.to_array();
            let cast_array =
                cast_array_with_options(&scalar_array, cast_type, cast_options)?;
            let cast_scalar = ScalarValue::try_from_array(&cast_array, 0)?;
            Ok(ColumnarValue::Scalar(cast_scalar))
        }
    }
}

/// Return a PhysicalExpression representing `expr` casted to
/// `cast_type`, if any casting is needed.
///
/// Note that such casts may lose type information
pub fn cast_with_options(
    expr: Arc<dyn PhysicalExpr>,
    input_schema: &Schema,
    cast_type: DataType,
    cast_options: CastOptions<'static>,
) -> Result<Arc<dyn PhysicalExpr>> {
    let expr_type = expr.data_type(input_schema)?;
    if expr_type == cast_type {
        Ok(expr.clone())
    } else if can_cast_types(&expr_type, &cast_type) {
        Ok(Arc::new(CastExpr::new(expr, cast_type, cast_options)))
    } else {
        Err(DataFusionError::Internal(format!(
            "Unsupported CAST from {:?} to {:?}",
            expr_type, cast_type
        )))
    }
}

/// Return a PhysicalExpression representing `expr` casted to
/// `cast_type`, if any casting is needed.
///
/// Note that such casts may lose type information
pub fn cast(
    expr: Arc<dyn PhysicalExpr>,
    input_schema: &Schema,
    cast_type: DataType,
) -> Result<Arc<dyn PhysicalExpr>> {
    cast_with_options(
        expr,
        input_schema,
        cast_type,
        DEFAULT_DATAFUSION_CAST_OPTIONS,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expressions::col;
    use arrow::compute::CUBESQL_CAST_OPTIONS;
    use arrow::{
        array::{
            Array, BooleanArray, Date32Array, Date64Array, Decimal128Array, Float32Array,
            Float64Array, Int16Array, Int32Array, Int64Array, Int8Array,
            IntervalDayTimeArray, IntervalMonthDayNanoArray, IntervalYearMonthArray,
            StringArray, Time64NanosecondArray, TimestampNanosecondArray,
            TimestampSecondArray, UInt32Array,
        },
        datatypes::*,
    };
    use datafusion_common::Result;

    // runs an end-to-end test of physical type cast
    // 1. construct a record batch with a column "a" of type A
    // 2. construct a physical expression of CAST(a AS B)
    // 3. evaluate the expression
    // 4. verify that the resulting expression is of type B
    // 5. verify that the resulting values are downcastable and correct
    macro_rules! generic_decimal_to_other_test_cast {
        ($DECIMAL_ARRAY:ident, $A_TYPE:expr, $TYPEARRAY:ident, $TYPE:expr, $VEC:expr,$CAST_OPTIONS:expr) => {{
            let schema = Schema::new(vec![Field::new("a", $A_TYPE, true)]);
            let batch = RecordBatch::try_new(
                Arc::new(schema.clone()),
                vec![Arc::new($DECIMAL_ARRAY)],
            )?;
            // verify that we can construct the expression
            let expression =
                cast_with_options(col("a", &schema)?, &schema, $TYPE, $CAST_OPTIONS)?;

            // verify that its display is correct
            assert_eq!(
                format!("CAST(a@0 AS {:?})", $TYPE),
                format!("{}", expression)
            );

            // verify that the expression's type is correct
            assert_eq!(expression.data_type(&schema)?, $TYPE);

            // compute
            let result = expression.evaluate(&batch)?.into_array(batch.num_rows());

            // verify that the array's data_type is correct
            assert_eq!(*result.data_type(), $TYPE);

            // verify that the data itself is downcastable
            let result = result
                .as_any()
                .downcast_ref::<$TYPEARRAY>()
                .expect("failed to downcast");

            // verify that the result itself is correct
            for (i, x) in $VEC.iter().enumerate() {
                match x {
                    Some(x) => assert_eq!(result.value(i), *x),
                    None => assert!(!result.is_valid(i)),
                }
            }
        }};
    }

    // runs an end-to-end test of physical type cast
    // 1. construct a record batch with a column "a" of type A
    // 2. construct a physical expression of CAST(a AS B)
    // 3. evaluate the expression
    // 4. verify that the resulting expression is of type B
    // 5. verify that the resulting values are downcastable and correct
    macro_rules! generic_test_cast {
        ($A_ARRAY:ident, $A_TYPE:expr, $A_VEC:expr, $TYPEARRAY:ident, $TYPE:expr, $VEC:expr, $CAST_OPTIONS:expr) => {{
            let schema = Schema::new(vec![Field::new("a", $A_TYPE, true)]);
            let a = $A_ARRAY::from($A_VEC);
            let batch =
                RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

            // verify that we can construct the expression
            let expression =
                cast_with_options(col("a", &schema)?, &schema, $TYPE, $CAST_OPTIONS)?;

            // verify that its display is correct
            assert_eq!(
                format!("CAST(a@0 AS {:?})", $TYPE),
                format!("{}", expression)
            );

            // verify that the expression's type is correct
            assert_eq!(expression.data_type(&schema)?, $TYPE);

            // compute
            let result = expression.evaluate(&batch)?.into_array(batch.num_rows());

            // verify that the array's data_type is correct
            assert_eq!(*result.data_type(), $TYPE);

            // verify that the len is correct
            assert_eq!(result.len(), $A_VEC.len());

            // verify that the data itself is downcastable
            let result = result
                .as_any()
                .downcast_ref::<$TYPEARRAY>()
                .expect("failed to downcast");

            // verify that the result itself is correct
            for (i, x) in $VEC.iter().enumerate() {
                match x {
                    Some(x) => assert_eq!(result.value(i), *x),
                    None => assert!(!result.is_valid(i)),
                }
            }
        }};
    }

    #[test]
    fn test_cast_decimal_to_decimal() -> Result<()> {
        let array = [
            Some(1234),
            Some(2222),
            Some(3),
            Some(4000),
            Some(5000),
            None,
        ];

        let decimal_array = array
            .iter()
            .collect::<Decimal128Array>()
            .with_precision_and_scale(10, 3)?;

        generic_decimal_to_other_test_cast!(
            decimal_array,
            DataType::Decimal128(10, 3),
            Decimal128Array,
            DataType::Decimal128(20, 6),
            [
                Some(1_234_000_i128),
                Some(2_222_000_i128),
                Some(3_000_i128),
                Some(4_000_000_i128),
                Some(5_000_000_i128),
                None
            ],
            DEFAULT_DATAFUSION_CAST_OPTIONS
        );

        let decimal_array = array
            .iter()
            .collect::<Decimal128Array>()
            .with_precision_and_scale(10, 3)?;

        generic_decimal_to_other_test_cast!(
            decimal_array,
            DataType::Decimal128(10, 3),
            Decimal128Array,
            DataType::Decimal128(10, 2),
            [
                Some(123_i128),
                Some(222_i128),
                Some(0_i128),
                Some(400_i128),
                Some(500_i128),
                None
            ],
            DEFAULT_DATAFUSION_CAST_OPTIONS
        );

        Ok(())
    }

    #[test]
    fn test_cast_decimal_to_numeric() -> Result<()> {
        let array = [Some(1), Some(2), Some(3), Some(4), Some(5), None];
        // decimal to i8
        let decimal_array = array
            .iter()
            .collect::<Decimal128Array>()
            .with_precision_and_scale(10, 0)?;
        generic_decimal_to_other_test_cast!(
            decimal_array,
            DataType::Decimal128(10, 0),
            Int8Array,
            DataType::Int8,
            [
                Some(1_i8),
                Some(2_i8),
                Some(3_i8),
                Some(4_i8),
                Some(5_i8),
                None
            ],
            DEFAULT_DATAFUSION_CAST_OPTIONS
        );

        // decimal to i16
        let decimal_array = array
            .iter()
            .collect::<Decimal128Array>()
            .with_precision_and_scale(10, 0)?;
        generic_decimal_to_other_test_cast!(
            decimal_array,
            DataType::Decimal128(10, 0),
            Int16Array,
            DataType::Int16,
            [
                Some(1_i16),
                Some(2_i16),
                Some(3_i16),
                Some(4_i16),
                Some(5_i16),
                None
            ],
            DEFAULT_DATAFUSION_CAST_OPTIONS
        );

        // decimal to i32
        let decimal_array = array
            .iter()
            .collect::<Decimal128Array>()
            .with_precision_and_scale(10, 0)?;
        generic_decimal_to_other_test_cast!(
            decimal_array,
            DataType::Decimal128(10, 0),
            Int32Array,
            DataType::Int32,
            [
                Some(1_i32),
                Some(2_i32),
                Some(3_i32),
                Some(4_i32),
                Some(5_i32),
                None
            ],
            DEFAULT_DATAFUSION_CAST_OPTIONS
        );

        // decimal to i64
        let decimal_array = array
            .iter()
            .collect::<Decimal128Array>()
            .with_precision_and_scale(10, 0)?;
        generic_decimal_to_other_test_cast!(
            decimal_array,
            DataType::Decimal128(10, 0),
            Int64Array,
            DataType::Int64,
            [
                Some(1_i64),
                Some(2_i64),
                Some(3_i64),
                Some(4_i64),
                Some(5_i64),
                None
            ],
            DEFAULT_DATAFUSION_CAST_OPTIONS
        );

        // decimal to float32
        let array = vec![
            Some(1234),
            Some(2222),
            Some(3),
            Some(4000),
            Some(5000),
            None,
        ];
        let decimal_array = array
            .iter()
            .collect::<Decimal128Array>()
            .with_precision_and_scale(10, 3)?;
        generic_decimal_to_other_test_cast!(
            decimal_array,
            DataType::Decimal128(10, 3),
            Float32Array,
            DataType::Float32,
            [
                Some(1.234_f32),
                Some(2.222_f32),
                Some(0.003_f32),
                Some(4.0_f32),
                Some(5.0_f32),
                None
            ],
            DEFAULT_DATAFUSION_CAST_OPTIONS
        );

        // decimal to float64
        let decimal_array = array
            .into_iter()
            .collect::<Decimal128Array>()
            .with_precision_and_scale(20, 6)?;
        generic_decimal_to_other_test_cast!(
            decimal_array,
            DataType::Decimal128(20, 6),
            Float64Array,
            DataType::Float64,
            [
                Some(0.001234_f64),
                Some(0.002222_f64),
                Some(0.000003_f64),
                Some(0.004_f64),
                Some(0.005_f64),
                None
            ],
            DEFAULT_DATAFUSION_CAST_OPTIONS
        );
        Ok(())
    }

    #[test]
    fn test_cast_numeric_to_decimal() -> Result<()> {
        // int8
        generic_test_cast!(
            Int8Array,
            DataType::Int8,
            vec![1, 2, 3, 4, 5],
            Decimal128Array,
            DataType::Decimal128(3, 0),
            [
                Some(1_i128),
                Some(2_i128),
                Some(3_i128),
                Some(4_i128),
                Some(5_i128)
            ],
            DEFAULT_DATAFUSION_CAST_OPTIONS
        );

        // int16
        generic_test_cast!(
            Int16Array,
            DataType::Int16,
            vec![1, 2, 3, 4, 5],
            Decimal128Array,
            DataType::Decimal128(5, 0),
            [
                Some(1_i128),
                Some(2_i128),
                Some(3_i128),
                Some(4_i128),
                Some(5_i128)
            ],
            DEFAULT_DATAFUSION_CAST_OPTIONS
        );

        // int32
        generic_test_cast!(
            Int32Array,
            DataType::Int32,
            vec![1, 2, 3, 4, 5],
            Decimal128Array,
            DataType::Decimal128(10, 0),
            [
                Some(1_i128),
                Some(2_i128),
                Some(3_i128),
                Some(4_i128),
                Some(5_i128)
            ],
            DEFAULT_DATAFUSION_CAST_OPTIONS
        );

        // int64
        generic_test_cast!(
            Int64Array,
            DataType::Int64,
            vec![1, 2, 3, 4, 5],
            Decimal128Array,
            DataType::Decimal128(20, 0),
            [
                Some(1_i128),
                Some(2_i128),
                Some(3_i128),
                Some(4_i128),
                Some(5_i128)
            ],
            DEFAULT_DATAFUSION_CAST_OPTIONS
        );

        // int64 to different scale
        generic_test_cast!(
            Int64Array,
            DataType::Int64,
            vec![1, 2, 3, 4, 5],
            Decimal128Array,
            DataType::Decimal128(20, 2),
            [
                Some(100_i128),
                Some(200_i128),
                Some(300_i128),
                Some(400_i128),
                Some(500_i128)
            ],
            DEFAULT_DATAFUSION_CAST_OPTIONS
        );

        // float32
        generic_test_cast!(
            Float32Array,
            DataType::Float32,
            vec![1.5, 2.5, 3.0, 1.123_456_8, 5.50],
            Decimal128Array,
            DataType::Decimal128(10, 2),
            [
                Some(150_i128),
                Some(250_i128),
                Some(300_i128),
                Some(112_i128),
                Some(550_i128)
            ],
            DEFAULT_DATAFUSION_CAST_OPTIONS
        );

        // float64
        generic_test_cast!(
            Float64Array,
            DataType::Float64,
            vec![1.5, 2.5, 3.0, 1.123_456_8, 5.50],
            Decimal128Array,
            DataType::Decimal128(20, 4),
            [
                Some(15000_i128),
                Some(25000_i128),
                Some(30000_i128),
                Some(11234_i128),
                Some(55000_i128)
            ],
            DEFAULT_DATAFUSION_CAST_OPTIONS
        );
        Ok(())
    }

    #[test]
    fn test_cast_i32_u32() -> Result<()> {
        generic_test_cast!(
            Int32Array,
            DataType::Int32,
            vec![1, 2, 3, 4, 5],
            UInt32Array,
            DataType::UInt32,
            [
                Some(1_u32),
                Some(2_u32),
                Some(3_u32),
                Some(4_u32),
                Some(5_u32)
            ],
            DEFAULT_DATAFUSION_CAST_OPTIONS
        );
        Ok(())
    }

    #[test]
    fn test_cast_i32_utf8() -> Result<()> {
        generic_test_cast!(
            Int32Array,
            DataType::Int32,
            vec![1, 2, 3, 4, 5],
            StringArray,
            DataType::Utf8,
            [Some("1"), Some("2"), Some("3"), Some("4"), Some("5")],
            DEFAULT_DATAFUSION_CAST_OPTIONS
        );
        Ok(())
    }

    #[allow(clippy::redundant_clone)]
    #[test]
    fn test_cast_i64_t64() -> Result<()> {
        let original = vec![1, 2, 3, 4, 5];
        let expected: Vec<Option<i64>> = original
            .iter()
            .map(|i| Some(Time64NanosecondArray::from(vec![*i]).value(0)))
            .collect();
        generic_test_cast!(
            Int64Array,
            DataType::Int64,
            original.clone(),
            TimestampNanosecondArray,
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            expected,
            DEFAULT_DATAFUSION_CAST_OPTIONS
        );
        Ok(())
    }

    #[test]
    fn invalid_cast() {
        // Ensure a useful error happens at plan time if invalid casts are used
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);

        let result = cast(
            col("a", &schema).unwrap(),
            &schema,
            DataType::Interval(IntervalUnit::MonthDayNano),
        );
        result.expect_err("expected Invalid CAST");
    }

    #[test]
    fn invalid_cast_with_options_error() -> Result<()> {
        // Ensure a useful error happens at plan time if invalid casts are used
        let schema = Schema::new(vec![Field::new("a", DataType::Utf8, false)]);
        let a = StringArray::from(vec!["9.1"]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;
        let expression = cast_with_options(
            col("a", &schema)?,
            &schema,
            DataType::Int32,
            DEFAULT_DATAFUSION_CAST_OPTIONS,
        )?;
        let result = expression.evaluate(&batch);

        match result {
            Ok(_) => panic!("expected error"),
            Err(e) => {
                assert!(e.to_string().contains(
                    "Cast error: Cannot cast string '9.1' to value of Int32 type"
                ))
            }
        }
        Ok(())
    }

    fn cast_array(array: ArrayRef, to_type: &DataType) -> Result<ArrayRef> {
        Ok(cast_array_with_options(
            &array,
            to_type,
            &DEFAULT_DATAFUSION_CAST_OPTIONS,
        )?)
    }

    fn string_values(array: &ArrayRef) -> Vec<Option<&str>> {
        array.as_string::<i32>().iter().collect()
    }

    // 2020-01-01 00:00:00.123 UTC
    const TS_NANOS: i64 = 1_577_836_800_123_000_000;

    #[test]
    fn test_cast_to_null() -> Result<()> {
        let array: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), None]));
        let result = cast_array(array, &DataType::Null)?;
        assert_eq!(result.data_type(), &DataType::Null);
        assert_eq!(result.len(), 2);
        assert!(can_cast_types(&DataType::Utf8, &DataType::Null));
        Ok(())
    }

    #[test]
    fn test_cast_timestamp_to_timestamp_with_timezone() -> Result<()> {
        // only attaches the timezone, the values are kept
        let array: ArrayRef =
            Arc::new(TimestampNanosecondArray::from(vec![Some(TS_NANOS), None]));
        let to_type = DataType::Timestamp(TimeUnit::Millisecond, Some("+01:00".into()));
        let result = cast_array(array, &to_type)?;
        assert_eq!(result.data_type(), &to_type);
        let result = result.as_primitive::<TimestampMillisecondType>();
        assert_eq!(result.value(0), 1_577_836_800_123);
        assert!(result.is_null(1));
        Ok(())
    }

    #[test]
    fn test_cast_date_to_timestamp_with_timezone() -> Result<()> {
        // only attaches the timezone, the values are kept
        let to_type = DataType::Timestamp(TimeUnit::Nanosecond, Some("+01:00".into()));
        let date32: ArrayRef = Arc::new(Date32Array::from(vec![Some(18262), None]));
        let date64: ArrayRef =
            Arc::new(Date64Array::from(vec![Some(1_577_836_800_000), None]));
        for array in [date32, date64] {
            let result = cast_array(array, &to_type)?;
            assert_eq!(result.data_type(), &to_type);
            let result = result.as_primitive::<TimestampNanosecondType>();
            assert_eq!(result.value(0), 1_577_836_800_000_000_000);
            assert!(result.is_null(1));
        }
        Ok(())
    }

    #[test]
    fn test_cast_timestamp_with_timezone_to_date() -> Result<()> {
        // 2019-12-31 23:00:00 UTC is 2020-01-01 in +01:00, the date of the UTC value is used
        let array: ArrayRef = Arc::new(
            TimestampNanosecondArray::from(vec![Some(1_577_833_200_000_000_000), None])
                .with_timezone("+01:00"),
        );
        let result = cast_array(array.clone(), &DataType::Date32)?;
        let result = result.as_primitive::<Date32Type>();
        assert_eq!(result.value(0), 18261);
        assert!(result.is_null(1));

        let result = cast_array(array, &DataType::Date64)?;
        let result = result.as_primitive::<Date64Type>();
        assert_eq!(result.value(0), 1_577_750_400_000);
        assert!(result.is_null(1));
        Ok(())
    }

    #[test]
    fn test_cast_timestamp_to_string() -> Result<()> {
        let array: ArrayRef = Arc::new(TimestampNanosecondArray::from(vec![
            Some(TS_NANOS),
            Some(1_577_836_800_000_000_000),
            Some(1_577_836_800_000_001_000),
            None,
        ]));
        let expected = vec![
            Some("2020-01-01 00:00:00.123"),
            Some("2020-01-01 00:00:00"),
            Some("2020-01-01 00:00:00.000001"),
            None,
        ];
        let result = cast_array(array.clone(), &DataType::Utf8)?;
        assert_eq!(string_values(&result), expected);

        let result = cast_array(array, &DataType::LargeUtf8)?;
        let result = result.as_string::<i64>().iter().collect::<Vec<_>>();
        assert_eq!(result, expected);

        // the UTC value is formatted, without the offset
        let array: ArrayRef = Arc::new(
            TimestampSecondArray::from(vec![Some(1_577_836_800), None])
                .with_timezone("+01:00"),
        );
        let result = cast_array(array, &DataType::Utf8)?;
        assert_eq!(
            string_values(&result),
            vec![Some("2020-01-01 00:00:00"), None]
        );
        Ok(())
    }

    #[test]
    fn test_cast_timestamp_to_string_with_format() -> Result<()> {
        // an explicit format is passed through to arrow
        let array: ArrayRef = Arc::new(TimestampNanosecondArray::from(vec![TS_NANOS]));
        let options = CastOptions {
            safe: false,
            format_options: FormatOptions::new().with_timestamp_format(Some("%Y")),
        };
        let result = cast_array_with_options(&array, &DataType::Utf8, &options)?;
        assert_eq!(string_values(&result), vec![Some("2020")]);
        Ok(())
    }

    #[test]
    fn test_cast_date_to_string() -> Result<()> {
        let array: ArrayRef =
            Arc::new(Date64Array::from(vec![Some(1_577_836_800_000), None]));
        let result = cast_array(array, &DataType::Utf8)?;
        assert_eq!(
            string_values(&result),
            vec![Some("2020-01-01 00:00:00"), None]
        );

        let array: ArrayRef = Arc::new(Date32Array::from(vec![Some(18262), None]));
        let result = cast_array(array, &DataType::Utf8)?;
        assert_eq!(string_values(&result), vec![Some("2020-01-01"), None]);
        Ok(())
    }

    #[test]
    fn test_cast_boolean_to_string() -> Result<()> {
        let array: ArrayRef =
            Arc::new(BooleanArray::from(vec![Some(true), Some(false), None]));
        let result = cast_array(array.clone(), &DataType::Utf8)?;
        assert_eq!(string_values(&result), vec![Some("1"), Some("0"), None]);

        let result = cast_array(array, &DataType::LargeUtf8)?;
        let result = result.as_string::<i64>().iter().collect::<Vec<_>>();
        assert_eq!(result, vec![Some("1"), Some("0"), None]);
        Ok(())
    }

    #[test]
    fn test_cast_interval_to_string() -> Result<()> {
        let year_month: ArrayRef = Arc::new(IntervalYearMonthArray::from(vec![
            Some(14),
            Some(-1),
            Some(0),
            None,
        ]));
        let result = cast_array(year_month, &DataType::Utf8)?;
        assert_eq!(
            string_values(&result),
            vec![
                Some("1 year 2 mons"),
                Some("-1 mons"),
                Some("00:00:00"),
                None
            ]
        );

        let day_time: ArrayRef = Arc::new(IntervalDayTimeArray::from(vec![
            Some(IntervalDayTime::new(1, 7_200_000)),
            Some(IntervalDayTime::new(2, 1_500)),
            Some(IntervalDayTime::new(-1, -7_200_000)),
            Some(IntervalDayTime::new(0, 90 * 60_000)),
            None,
        ]));
        let result = cast_array(day_time, &DataType::Utf8)?;
        assert_eq!(
            string_values(&result),
            vec![
                Some("1 day 02:00:00"),
                Some("2 days 00:00:01.500000"),
                Some("-1 days -02:00:00"),
                Some("01:30:00"),
                None
            ]
        );

        let month_day_nano: ArrayRef = Arc::new(IntervalMonthDayNanoArray::from(vec![
            Some(IntervalMonthDayNano::new(
                14,
                3,
                ((4 * 60 + 5) * 60 + 6) * 1_000_000_000 + 789_000_000,
            )),
            Some(IntervalMonthDayNano::new(0, 0, 1_999)),
            Some(IntervalMonthDayNano::new(0, 0, 100 * 3_600 * 1_000_000_000)),
            None,
        ]));
        let result = cast_array(month_day_nano.clone(), &DataType::Utf8)?;
        assert_eq!(
            string_values(&result),
            vec![
                Some("1 year 2 mons 3 days 04:05:06.789000"),
                Some("00:00:00.000001"),
                Some("100:00:00"),
                None
            ]
        );

        let result = cast_array(month_day_nano, &DataType::LargeUtf8)?;
        let result = result.as_string::<i64>().iter().collect::<Vec<_>>();
        assert_eq!(result[0], Some("1 year 2 mons 3 days 04:05:06.789000"));
        Ok(())
    }

    #[test]
    fn test_cast_column_scalar() -> Result<()> {
        let value = ColumnarValue::Scalar(ScalarValue::TimestampNanosecond(
            Some(TS_NANOS),
            Some("UTC".into()),
        ));
        let result =
            cast_column(&value, &DataType::Utf8, &DEFAULT_DATAFUSION_CAST_OPTIONS)?;
        match result {
            ColumnarValue::Scalar(ScalarValue::Utf8(v)) => {
                assert_eq!(v.as_deref(), Some("2020-01-01 00:00:00.123"))
            }
            other => panic!("unexpected result {:?}", other),
        }
        Ok(())
    }
    #[test]
    fn test_cast_numeric_overflow_is_null() -> Result<()> {
        // arrow 13 ignored `safe` for numeric casts: values that don't fit become NULL
        for options in [DEFAULT_DATAFUSION_CAST_OPTIONS, CUBESQL_CAST_OPTIONS] {
            let array: ArrayRef =
                Arc::new(Int64Array::from(vec![Some(3_000_000_000), Some(-7), None]));
            let result = cast_array_with_options(&array, &DataType::Int32, &options)?;
            assert_eq!(
                result.as_primitive::<Int32Type>(),
                &Int32Array::from(vec![None, Some(-7), None])
            );
            let result = cast_array_with_options(&array, &DataType::UInt32, &options)?;
            assert_eq!(
                result.as_primitive::<UInt32Type>(),
                &UInt32Array::from(vec![Some(3_000_000_000), None, None])
            );

            let array: ArrayRef = Arc::new(Float64Array::from(vec![
                f64::NAN,
                1e20,
                f64::INFINITY,
                2.9,
                -1.5,
            ]));
            let result = cast_array_with_options(&array, &DataType::Int64, &options)?;
            assert_eq!(
                result.as_primitive::<Int64Type>(),
                &Int64Array::from(vec![None, None, None, Some(2), Some(-1)])
            );

            let array: ArrayRef = Arc::new(Int32Array::from(vec![300, 100]));
            let result = cast_array_with_options(&array, &DataType::Int8, &options)?;
            assert_eq!(
                result.as_primitive::<Int8Type>(),
                &Int8Array::from(vec![None, Some(100)])
            );
        }
        Ok(())
    }

    #[test]
    fn test_cast_to_decimal_truncates() -> Result<()> {
        // floats: truncated towards zero, NaN is 0
        let array: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(1.123_456_8),
            Some(-1.234_56),
            Some(f64::NAN),
            None,
        ]));
        let result = cast_array(array, &DataType::Decimal128(20, 4))?;
        assert_eq!(result.data_type(), &DataType::Decimal128(20, 4));
        assert_eq!(
            result
                .as_primitive::<Decimal128Type>()
                .iter()
                .collect::<Vec<_>>(),
            vec![Some(11234), Some(-12345), Some(0), None]
        );
        let array: ArrayRef = Arc::new(Float32Array::from(vec![2.56_f32]));
        let result = cast_array(array, &DataType::Decimal128(10, 1))?;
        assert_eq!(result.as_primitive::<Decimal128Type>().value(0), 25);

        // decimals with a larger scale: truncated
        let array: ArrayRef = Arc::new(
            Decimal128Array::from(vec![Some(1235), Some(-1239), None])
                .with_precision_and_scale(10, 3)?,
        );
        let result = cast_array(array, &DataType::Decimal128(10, 2))?;
        assert_eq!(
            result
                .as_primitive::<Decimal128Type>()
                .iter()
                .collect::<Vec<_>>(),
            vec![Some(123), Some(-123), None]
        );
        let array: ArrayRef =
            Arc::new(Decimal128Array::from(vec![99999]).with_precision_and_scale(5, 3)?);
        let result = cast_array(array, &DataType::Decimal128(3, 1))?;
        assert_eq!(result.as_primitive::<Decimal128Type>().value(0), 999);

        // values are checked against the target precision
        let array: ArrayRef = Arc::new(Float64Array::from(vec![1000.0]));
        let err = cast_array(array, &DataType::Decimal128(3, 0)).unwrap_err();
        assert!(
            err.to_string()
                .contains("1000 is too large to store in a Decimal of precision 3"),
            "{}",
            err
        );

        // strings: parsed as f64 and truncated, no digit in the mantissa is NULL
        let array: ArrayRef = Arc::new(StringArray::from(vec![
            Some("1.239"),
            Some("-2.5"),
            Some("1e2"),
            Some("abc"),
            Some("."),
            None,
        ]));
        let result = cast_array(array, &DataType::Decimal128(10, 2))?;
        assert_eq!(
            result
                .as_primitive::<Decimal128Type>()
                .iter()
                .collect::<Vec<_>>(),
            vec![Some(123), Some(-250), Some(10000), None, None, None]
        );
        for invalid in ["", "-", "1x", "1.2.3"] {
            let array: ArrayRef = Arc::new(StringArray::from(vec![invalid]));
            let err = cast_array(array, &DataType::Decimal128(10, 2)).unwrap_err();
            assert!(
                err.to_string()
                    .contains("Cannot cast from string to decimal"),
                "{}: {}",
                invalid,
                err
            );
        }
        Ok(())
    }

    #[test]
    fn test_cast_string_to_timestamp_is_local_time() -> Result<()> {
        use chrono::{Local, NaiveDate, TimeZone};

        let naive = NaiveDate::from_ymd_opt(2020, 9, 8)
            .unwrap()
            .and_hms_opt(13, 42, 29)
            .unwrap();
        let local = Local.from_local_datetime(&naive).earliest().unwrap();

        let array: ArrayRef = Arc::new(StringArray::from(vec![
            Some("2020-09-08 13:42:29"),
            Some("2020-09-08T13:42:29Z"),
            Some("2020-09-08T13:42:29+02:00"),
            None,
        ]));
        let expected = vec![
            Some(local.timestamp()),
            Some(1_599_572_549),
            Some(1_599_565_349),
            None,
        ];
        let to_type = DataType::Timestamp(TimeUnit::Nanosecond, None);
        let result = cast_array(array.clone(), &to_type)?;
        assert_eq!(
            result
                .as_primitive::<TimestampNanosecondType>()
                .iter()
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|v| v.map(|v| v * 1_000_000_000))
                .collect::<Vec<_>>()
        );
        let to_type = DataType::Timestamp(TimeUnit::Second, None);
        let result = cast_array(array, &to_type)?;
        assert_eq!(
            result
                .as_primitive::<TimestampSecondType>()
                .iter()
                .collect::<Vec<_>>(),
            expected
        );

        // invalid strings: an error, or NULL for a safe cast
        let array: ArrayRef = Arc::new(StringArray::from(vec!["not a timestamp"]));
        let to_type = DataType::Timestamp(TimeUnit::Nanosecond, None);
        let err = cast_array(array.clone(), &to_type).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Arrow error: Cast error: Error parsing 'not a timestamp' as timestamp"
        );
        let options = CastOptions {
            safe: true,
            format_options: FormatOptions::new(),
        };
        let result = cast_array_with_options(&array, &to_type, &options)?;
        assert!(result.is_null(0));
        Ok(())
    }

    #[test]
    fn cast_nullability() -> Result<()> {
        let options = DEFAULT_DATAFUSION_CAST_OPTIONS;
        let may_null =
            |from: DataType, to: DataType| cast_may_produce_null(&from, &to, &options);
        // lossless casts
        assert!(!may_null(DataType::Int32, DataType::Int32));
        assert!(!may_null(DataType::Int8, DataType::Int64));
        assert!(!may_null(DataType::UInt32, DataType::Int64));
        assert!(!may_null(DataType::UInt8, DataType::UInt16));
        assert!(!may_null(DataType::Int64, DataType::Float32));
        assert!(!may_null(DataType::Float64, DataType::Float32));
        assert!(!may_null(DataType::Int32, DataType::Utf8));
        assert!(!may_null(DataType::Utf8, DataType::Int32));
        // casts that turn values into NULL
        assert!(may_null(DataType::Int64, DataType::Int32));
        assert!(may_null(DataType::Int32, DataType::UInt64));
        assert!(may_null(DataType::UInt32, DataType::Int32));
        assert!(may_null(DataType::Float64, DataType::Int64));
        assert!(may_null(DataType::Utf8, DataType::Decimal128(10, 2)));
        assert!(may_null(DataType::Int32, DataType::Null));
        assert!(cast_may_produce_null(
            &DataType::Utf8,
            &DataType::Int32,
            &CastOptions {
                safe: true,
                format_options: FormatOptions::new(),
            }
        ));

        // the nullability of a cast expression follows
        let schema = Schema::new(vec![Field::new("a", DataType::Int64, false)]);
        let expr = cast_with_options(
            col("a", &schema)?,
            &schema,
            DataType::Int32,
            options.clone(),
        )?;
        assert!(expr.nullable(&schema)?);
        let expr =
            cast_with_options(col("a", &schema)?, &schema, DataType::Float64, options)?;
        assert!(!expr.nullable(&schema)?);
        Ok(())
    }
}
