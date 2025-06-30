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

use arrow::array::{
    Array, Date32Array, Date64Array, Float64Array, Int32Array, Int32Builder,
    IntervalDayTimeArray, IntervalMonthDayNanoArray, IntervalYearMonthArray,
    PrimitiveArray, TimestampMicrosecondArray, TimestampMillisecondArray,
    TimestampNanosecondArray, TimestampSecondArray,
};
use arrow::compute::kernels::{arity::unary, temporal as arrow_temporal};
use arrow::datatypes::{
    ArrowNumericType, ArrowPrimitiveType, ArrowTemporalType, DataType, Date32Type,
    Date64Type, Float64Type, IntervalDayTimeType, IntervalMonthDayNanoType,
    IntervalYearMonthType, TimestampMicrosecondType, TimestampMillisecondType,
    TimestampNanosecondType, TimestampSecondType,
};
use arrow::error::{ArrowError, Result};

use chrono::format::strftime::StrftimeItems;
use chrono::format::{parse, Parsed};
use chrono::FixedOffset;
use chrono::{Datelike, NaiveDateTime};

pub fn using_chrono_tz_and_utc_naive_date_time(
    _tz: &str,
    _utc: chrono::NaiveDateTime,
) -> Option<FixedOffset> {
    None
}

macro_rules! extract_component_from_array {
    ($array:ident, $builder:ident, $extract_fn:ident, $using:ident) => {
        for i in 0..$array.len() {
            if $array.is_null(i) {
                $builder.append_null()?;
            } else {
                match $array.$using(i) {
                    Some(dt) => $builder.append_value(dt.$extract_fn() as i32)?,
                    None => $builder.append_null()?,
                }
            }
        }
    };
    ($array:ident, $builder:ident, $extract_fn1:ident, $extract_fn2:ident, $using:ident) => {
        for i in 0..$array.len() {
            if $array.is_null(i) {
                $builder.append_null()?;
            } else {
                match $array.$using(i) {
                    Some(dt) => {
                        $builder.append_value(dt.$extract_fn1().$extract_fn2() as i32)?
                    }
                    None => $builder.append_null()?,
                }
            }
        }
    };
    ($array:ident, $builder:ident, $extract_fn:ident, $using:ident, $tz:ident, $parsed:ident) => {
        if ($tz.starts_with('+') || $tz.starts_with('-')) && !$tz.contains(':') {
            return_compute_error_with!(
                "Invalid timezone",
                "Expected format [+-]XX:XX".to_string()
            )
        } else {
            let tz_parse_result = parse(&mut $parsed, $tz, StrftimeItems::new("%z"));
            let fixed_offset_from_parsed = match tz_parse_result {
                Ok(_) => match $parsed.to_fixed_offset() {
                    Ok(fo) => Some(fo),
                    err => return_compute_error_with!("Invalid timezone", err),
                },
                _ => None,
            };

            for i in 0..$array.len() {
                if $array.is_null(i) {
                    $builder.append_null()?;
                } else {
                    match $array.value_as_datetime(i) {
                        Some(utc) => {
                            let fixed_offset = match fixed_offset_from_parsed {
                                Some(fo) => fo,
                                None => match using_chrono_tz_and_utc_naive_date_time(
                                    $tz, utc,
                                ) {
                                    Some(fo) => fo,
                                    err => return_compute_error_with!(
                                        "Unable to parse timezone",
                                        err
                                    ),
                                },
                            };
                            match $array.$using(i, fixed_offset) {
                                Some(dt) => {
                                    $builder.append_value(dt.$extract_fn() as i32)?
                                }
                                None => $builder.append_null()?,
                            }
                        }
                        err => return_compute_error_with!(
                            "Unable to read value as datetime",
                            err
                        ),
                    }
                }
            }
        }
    };
}

macro_rules! return_compute_error_with {
    ($msg:expr, $param:expr) => {
        return { Err(ArrowError::ComputeError(format!("{}: {:?}", $msg, $param))) }
    };
}

/// Extracts the day of year of a given temporal array as an array of integers
pub fn doy<T>(array: &PrimitiveArray<T>) -> Result<Int32Array>
where
    T: ArrowTemporalType + ArrowNumericType,
    i64: std::convert::From<T::Native>,
{
    let mut b = Int32Builder::new(array.len());
    match array.data_type() {
        &DataType::Date32 | &DataType::Date64 | &DataType::Timestamp(_, None) => {
            extract_component_from_array!(array, b, ordinal, value_as_datetime)
        }
        &DataType::Timestamp(_, Some(ref tz)) => {
            let mut scratch = Parsed::new();
            extract_component_from_array!(
                array,
                b,
                ordinal,
                value_as_datetime_with_tz,
                tz,
                scratch
            )
        }
        dt => return_compute_error_with!("doy does not support", dt),
    }

    Ok(b.finish())
}

fn postgres_months_epoch(n: i32) -> f64 {
    let years = n / 12;
    let remainder = n % 12;
    // Note that this arithmetic produces exact integer calculations with no floating point error.
    let seconds_in_a_day = 86400_f64;
    (years as f64) * (seconds_in_a_day * 365.25)
        + (remainder as f64) * (seconds_in_a_day * 30.0)
}

pub trait Epochable: ArrowPrimitiveType + Sized {
    fn get_epoch(array: &PrimitiveArray<Self>) -> PrimitiveArray<Float64Type>;
}

impl Epochable for TimestampSecondType {
    fn get_epoch(
        array: &PrimitiveArray<TimestampSecondType>,
    ) -> PrimitiveArray<Float64Type> {
        unary(array, |n| n as f64)
    }
}

impl Epochable for TimestampMillisecondType {
    fn get_epoch(
        array: &PrimitiveArray<TimestampMillisecondType>,
    ) -> PrimitiveArray<Float64Type> {
        unary(array, |n| n as f64 / 1_000.0)
    }
}

impl Epochable for TimestampMicrosecondType {
    fn get_epoch(
        array: &PrimitiveArray<TimestampMicrosecondType>,
    ) -> PrimitiveArray<Float64Type> {
        unary(array, |n| n as f64 / 1_000_000.0)
    }
}

impl Epochable for TimestampNanosecondType {
    fn get_epoch(
        array: &PrimitiveArray<TimestampNanosecondType>,
    ) -> PrimitiveArray<Float64Type> {
        unary(array, |n| n as f64 / 1_000_000_000.0)
    }
}

impl Epochable for Date32Type {
    fn get_epoch(array: &PrimitiveArray<Date32Type>) -> PrimitiveArray<Float64Type> {
        unary(array, |n| {
            let seconds_in_a_day = 86400.0;
            n as f64 * seconds_in_a_day
        })
    }
}

impl Epochable for Date64Type {
    fn get_epoch(array: &PrimitiveArray<Date64Type>) -> PrimitiveArray<Float64Type> {
        unary(array, |n| n as f64 / 1_000.0)
    }
}

impl Epochable for IntervalYearMonthType {
    fn get_epoch(
        array: &PrimitiveArray<IntervalYearMonthType>,
    ) -> PrimitiveArray<Float64Type> {
        unary(array, postgres_months_epoch)
    }
}

impl Epochable for IntervalDayTimeType {
    fn get_epoch(
        array: &PrimitiveArray<IntervalDayTimeType>,
    ) -> PrimitiveArray<Float64Type> {
        unary(array, |n| {
            let (days, millis) = IntervalDayTimeType::to_parts(n);
            let seconds_in_a_day = 86400.0;
            (days as f64) * seconds_in_a_day + (millis as f64) / 1_000.0
        })
    }
}

impl Epochable for IntervalMonthDayNanoType {
    fn get_epoch(
        array: &PrimitiveArray<IntervalMonthDayNanoType>,
    ) -> PrimitiveArray<Float64Type> {
        unary(array, |n| {
            let seconds_in_a_day = 86400_f64;
            let (month, day, nano) = IntervalMonthDayNanoType::to_parts(n);
            let month_epoch: f64 = postgres_months_epoch(month);
            month_epoch
                + (day as f64) * seconds_in_a_day
                + (nano as f64) / 1_000_000_000.0
        })
    }
}

pub fn epoch<T>(array: &PrimitiveArray<T>) -> Result<Float64Array>
where
    T: Epochable,
{
    let b = Epochable::get_epoch(array);
    Ok(b)
}

trait ChronoDateLikeExt {
    fn weekday_from_sunday(&self) -> i32;
}

impl ChronoDateLikeExt for NaiveDateTime {
    fn weekday_from_sunday(&self) -> i32 {
        self.weekday().num_days_from_sunday() as i32
    }
}

/// Extracts the day of week of a given temporal array as an array of integers
pub fn dow<T>(array: &PrimitiveArray<T>) -> Result<Int32Array>
where
    T: ArrowTemporalType + ArrowNumericType,
    i64: std::convert::From<T::Native>,
{
    let mut b = Int32Builder::new(array.len());
    match array.data_type() {
        &DataType::Date32 | &DataType::Date64 | &DataType::Timestamp(_, None) => {
            extract_component_from_array!(
                array,
                b,
                weekday_from_sunday,
                value_as_datetime
            )
        }
        &DataType::Timestamp(_, Some(ref tz)) => {
            let mut scratch = Parsed::new();
            extract_component_from_array!(
                array,
                b,
                weekday_from_sunday,
                value_as_datetime_with_tz,
                tz,
                scratch
            )
        }
        dt => return_compute_error_with!("dow does not support", dt),
    }

    Ok(b.finish())
}

/// This macro will generate a trait `DatePartable` that is automatically implemented
/// for all Arrow temporal types.
macro_rules! date_partable {
    ($($gran:ident),*) => {
        pub trait DatePartable: ArrowPrimitiveType + Sized {
            $(
                fn $gran(array: &PrimitiveArray<Self>) -> Result<Int32Array>;
            )*
        }

        impl DatePartable for TimestampSecondType {
            $(
                fn $gran(array: &TimestampSecondArray) -> Result<Int32Array> {
                    arrow_temporal::$gran(array)
                }
            )*
        }

        impl DatePartable for TimestampMillisecondType {
            $(
                fn $gran(array: &TimestampMillisecondArray) -> Result<Int32Array> {
                    arrow_temporal::$gran(array)
                }
            )*
        }

        impl DatePartable for TimestampMicrosecondType {
            $(
                fn $gran(array: &TimestampMicrosecondArray) -> Result<Int32Array> {
                    arrow_temporal::$gran(array)
                }
            )*
        }

        impl DatePartable for TimestampNanosecondType {
            $(
                fn $gran(array: &TimestampNanosecondArray) -> Result<Int32Array> {
                    arrow_temporal::$gran(array)
                }
            )*
        }

        impl DatePartable for Date32Type {
            $(
                fn $gran(array: &Date32Array) -> Result<Int32Array> {
                    arrow_temporal::$gran(array)
                }
            )*
        }

        impl DatePartable for Date64Type {
            $(
                fn $gran(array: &Date64Array) -> Result<Int32Array> {
                    arrow_temporal::$gran(array)
                }
            )*
        }

        $(
            pub fn $gran<T>(array: &PrimitiveArray<T>) -> Result<Int32Array>
            where
                T: DatePartable,
            {
                DatePartable::$gran(array)
            }
        )*
    };
}

date_partable!(year, quarter, month, day, hour, minute, second);

fn interval_year_month_op(
    array: &IntervalYearMonthArray,
    op: fn(i32) -> Result<i32>,
) -> Result<Int32Array> {
    let mut builder = Int32Builder::new(array.len());
    for i in 0..array.len() {
        if array.is_null(i) {
            builder.append_null()?;
            continue;
        }
        let value = array.value(i);
        let result = op(value)?;
        builder.append_value(result)?;
    }
    Ok(builder.finish())
}

fn interval_day_time_op(
    array: &IntervalDayTimeArray,
    op: fn(i64) -> Result<i32>,
) -> Result<Int32Array> {
    let mut builder = Int32Builder::new(array.len());
    for i in 0..array.len() {
        if array.is_null(i) {
            builder.append_null()?;
            continue;
        }
        let value = array.value(i);
        let result = op(value)?;
        builder.append_value(result)?;
    }
    Ok(builder.finish())
}

fn interval_month_day_nano_op(
    array: &IntervalMonthDayNanoArray,
    op: fn(i128) -> Result<i32>,
) -> Result<Int32Array> {
    let mut builder = Int32Builder::new(array.len());
    for i in 0..array.len() {
        if array.is_null(i) {
            builder.append_null()?;
            continue;
        }
        let value = array.value(i);
        let result = op(value)?;
        builder.append_value(result)?;
    }
    Ok(builder.finish())
}

impl DatePartable for IntervalYearMonthType {
    fn year(array: &IntervalYearMonthArray) -> Result<Int32Array> {
        interval_year_month_op(array, |v| Ok(v / 12))
    }

    fn quarter(array: &IntervalYearMonthArray) -> Result<Int32Array> {
        interval_year_month_op(array, |v| Ok(v % 12 / 3 + 1))
    }

    fn month(array: &IntervalYearMonthArray) -> Result<Int32Array> {
        interval_year_month_op(array, |v| Ok(v % 12))
    }

    fn day(array: &IntervalYearMonthArray) -> Result<Int32Array> {
        interval_year_month_op(array, |_| Ok(0))
    }

    fn hour(array: &IntervalYearMonthArray) -> Result<Int32Array> {
        interval_year_month_op(array, |_| Ok(0))
    }

    fn minute(array: &IntervalYearMonthArray) -> Result<Int32Array> {
        interval_year_month_op(array, |_| Ok(0))
    }

    fn second(array: &IntervalYearMonthArray) -> Result<Int32Array> {
        interval_year_month_op(array, |_| Ok(0))
    }
}

impl DatePartable for IntervalDayTimeType {
    fn year(array: &IntervalDayTimeArray) -> Result<Int32Array> {
        interval_day_time_op(array, |_| Ok(0))
    }

    fn quarter(array: &IntervalDayTimeArray) -> Result<Int32Array> {
        interval_day_time_op(array, |_| Ok(1))
    }

    fn month(array: &IntervalDayTimeArray) -> Result<Int32Array> {
        interval_day_time_op(array, |_| Ok(0))
    }

    fn day(array: &IntervalDayTimeArray) -> Result<Int32Array> {
        interval_day_time_op(array, |v| {
            let (days, _) = IntervalDayTimeType::to_parts(v);
            Ok(days)
        })
    }

    fn hour(array: &IntervalDayTimeArray) -> Result<Int32Array> {
        interval_day_time_op(array, |v| {
            let (_, millis) = IntervalDayTimeType::to_parts(v);
            Ok(millis / 3_600_000)
        })
    }

    fn minute(array: &IntervalDayTimeArray) -> Result<Int32Array> {
        interval_day_time_op(array, |v| {
            let (_, millis) = IntervalDayTimeType::to_parts(v);
            Ok(millis % 3_600_000 / 60_000)
        })
    }

    fn second(array: &IntervalDayTimeArray) -> Result<Int32Array> {
        // NOTE: this technically should return Float64 with millis in the decimal part,
        // but we are returning Int32 for compatibility with the original implementation.
        interval_day_time_op(array, |v| {
            let (_, millis) = IntervalDayTimeType::to_parts(v);
            Ok(millis % 60_000 / 1_000)
        })
    }
}

impl DatePartable for IntervalMonthDayNanoType {
    fn year(array: &IntervalMonthDayNanoArray) -> Result<Int32Array> {
        interval_month_day_nano_op(array, |v| {
            let (months, _, _) = IntervalMonthDayNanoType::to_parts(v);
            Ok(months / 12)
        })
    }

    fn quarter(array: &IntervalMonthDayNanoArray) -> Result<Int32Array> {
        interval_month_day_nano_op(array, |v| {
            let (months, _, _) = IntervalMonthDayNanoType::to_parts(v);
            Ok(months % 12 / 3 + 1)
        })
    }

    fn month(array: &IntervalMonthDayNanoArray) -> Result<Int32Array> {
        interval_month_day_nano_op(array, |v| {
            let (months, _, _) = IntervalMonthDayNanoType::to_parts(v);
            Ok(months % 12)
        })
    }

    fn day(array: &IntervalMonthDayNanoArray) -> Result<Int32Array> {
        interval_month_day_nano_op(array, |v| {
            let (_, days, _) = IntervalMonthDayNanoType::to_parts(v);
            Ok(days)
        })
    }

    fn hour(array: &IntervalMonthDayNanoArray) -> Result<Int32Array> {
        interval_month_day_nano_op(array, |v| {
            let (_, _, nanos) = IntervalMonthDayNanoType::to_parts(v);
            (nanos / 3_600_000_000_000).try_into().map_err(|_| {
                ArrowError::ComputeError("Unable to convert i64 nanos to i32".to_string())
            })
        })
    }

    fn minute(array: &IntervalMonthDayNanoArray) -> Result<Int32Array> {
        interval_month_day_nano_op(array, |v| {
            let (_, _, nanos) = IntervalMonthDayNanoType::to_parts(v);
            (nanos % 3_600_000_000_000 / 60_000_000_000)
                .try_into()
                .map_err(|_| {
                    ArrowError::ComputeError(
                        "Unable to convert i64 nanos to i32".to_string(),
                    )
                })
        })
    }

    fn second(array: &IntervalMonthDayNanoArray) -> Result<Int32Array> {
        // NOTE: this technically should return Float64 with millis in the decimal part,
        // but we are returning Int32 for compatibility with the original implementation.
        interval_month_day_nano_op(array, |v| {
            let (_, _, nanos) = IntervalMonthDayNanoType::to_parts(v);
            (nanos % 60_000_000_000 / 1_000_000_000)
                .try_into()
                .map_err(|_| {
                    ArrowError::ComputeError(
                        "Unable to convert i64 nanos to i32".to_string(),
                    )
                })
        })
    }
}
