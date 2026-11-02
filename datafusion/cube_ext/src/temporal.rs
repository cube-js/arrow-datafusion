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

use arrow::array::timezone::Tz;
use arrow::array::{
    Array, Date32Array, Date64Array, Float64Array, Int32Array, Int32Builder,
    IntervalDayTimeArray, IntervalMonthDayNanoArray, IntervalYearMonthArray,
    PrimitiveArray, TimestampMicrosecondArray, TimestampMillisecondArray,
    TimestampNanosecondArray, TimestampSecondArray,
};
use arrow::compute::kernels::arity::unary;
use arrow::datatypes::{
    ArrowNumericType, ArrowPrimitiveType, ArrowTemporalType, DataType, Date32Type,
    Date64Type, Float64Type, IntervalDayTime, IntervalDayTimeType, IntervalMonthDayNano,
    IntervalMonthDayNanoType, IntervalYearMonthType, TimestampMicrosecondType,
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType,
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

/// Resolves an IANA time zone name to the offset in effect at the `utc` instant.
///
/// This is arrow 13's `using_chrono_tz_and_utc_naive_date_time` as compiled with
/// the `chrono-tz` feature, which the arrow 13 kernels behind `year` ... `second`
/// used (`doy` and `dow` keep using the stub above, as before). arrow 60's [`Tz`]
/// also parses fixed offsets (e.g. "+08:60"), which arrow 13's chrono-tz lookup
/// never accepted, so signed strings are rejected here: no IANA name starts with
/// a sign.
fn chrono_tz_offset_at_utc(tz: &str, utc: NaiveDateTime) -> Option<FixedOffset> {
    use chrono::{Offset, TimeZone};

    if tz.starts_with('+') || tz.starts_with('-') {
        return None;
    }
    tz.parse::<Tz>()
        .map(|tz| tz.offset_from_utc_datetime(&utc).fix())
        .ok()
}

// The macros below come from arrow 13's `compute::kernels::temporal`. arrow 60
// replaced those kernels with `date_part`, whose semantics differ (e.g. `year`
// of a timestamp with a time zone, `minute`/`second` of a Date32), so the
// original loops are kept here to preserve the results bit for bit.
//
// arrow 13's `value_as_datetime_with_tz(i, offset)` returned the naive UTC
// datetime shifted by the offset; arrow 60's method of that name takes a `Tz`
// and returns a `DateTime<Tz>`, so the shift is done here with
// `checked_add_offset`.
macro_rules! extract_component_from_array {
    ($array:ident, $builder:ident, $extract_fn:ident, $using:ident) => {
        for i in 0..$array.len() {
            if $array.is_null(i) {
                $builder.append_null();
            } else {
                match $array.$using(i) {
                    Some(dt) => $builder.append_value(dt.$extract_fn() as i32),
                    None => $builder.append_null(),
                }
            }
        }
    };
    ($array:ident, $builder:ident, $extract_fn1:ident, $extract_fn2:ident, $using:ident) => {
        for i in 0..$array.len() {
            if $array.is_null(i) {
                $builder.append_null();
            } else {
                match $array.$using(i) {
                    Some(dt) => {
                        $builder.append_value(dt.$extract_fn1().$extract_fn2() as i32)
                    }
                    None => $builder.append_null(),
                }
            }
        }
    };
    ($array:ident, $builder:ident, $extract_fn:ident, $tz:ident, $parsed:ident, $resolve_tz:path) => {
        extract_mapped_component_from_array!(
            $array,
            $builder,
            $extract_fn,
            $tz,
            $parsed,
            $resolve_tz,
            |v: i32| v
        )
    };
}

macro_rules! extract_mapped_component_from_array {
    ($array:ident, $builder:ident, $extract_fn:ident, $using:ident, $mapper:expr) => {
        for i in 0..$array.len() {
            if $array.is_null(i) {
                $builder.append_null();
            } else {
                match $array.$using(i) {
                    Some(dt) => $builder.append_value($mapper(dt.$extract_fn() as i32)),
                    None => $builder.append_null(),
                }
            }
        }
    };
    ($array:ident, $builder:ident, $extract_fn:ident, $tz:ident, $parsed:ident, $resolve_tz:path, $mapper:expr) => {
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
                    $builder.append_null();
                } else {
                    match $array.value_as_datetime(i) {
                        Some(utc) => {
                            let fixed_offset = match fixed_offset_from_parsed {
                                Some(fo) => fo,
                                None => match $resolve_tz($tz, utc) {
                                    Some(fo) => fo,
                                    err => return_compute_error_with!(
                                        "Unable to parse timezone",
                                        err
                                    ),
                                },
                            };
                            match utc.checked_add_offset(fixed_offset) {
                                Some(dt) => $builder
                                    .append_value($mapper(dt.$extract_fn() as i32)),
                                None => $builder.append_null(),
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
    let mut b = Int32Builder::with_capacity(array.len());
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
                tz,
                scratch,
                using_chrono_tz_and_utc_naive_date_time
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
    let mut b = Int32Builder::with_capacity(array.len());
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
                tz,
                scratch,
                using_chrono_tz_and_utc_naive_date_time
            )
        }
        dt => return_compute_error_with!("dow does not support", dt),
    }

    Ok(b.finish())
}

/// arrow 13's `compute::kernels::temporal::{year, quarter, month, day, hour,
/// minute, second}`, kept with their exact type support, null handling, time zone
/// handling and error messages. arrow 60 only offers `date_part`, which differs
/// for some inputs (see the module-level macros comment).
mod kernels {
    use super::*;
    use chrono::Timelike;

    /// Extracts the hours of a given temporal array as an array of integers
    pub fn hour<T>(array: &PrimitiveArray<T>) -> Result<Int32Array>
    where
        T: ArrowTemporalType + ArrowNumericType,
        i64: std::convert::From<T::Native>,
    {
        let mut b = Int32Builder::with_capacity(array.len());
        match array.data_type() {
            &DataType::Time32(_) | &DataType::Time64(_) => {
                extract_component_from_array!(array, b, hour, value_as_time)
            }
            &DataType::Date32 | &DataType::Date64 | &DataType::Timestamp(_, None) => {
                extract_component_from_array!(array, b, hour, value_as_datetime)
            }
            &DataType::Timestamp(_, Some(ref tz)) => {
                let mut scratch = Parsed::new();
                extract_component_from_array!(
                    array,
                    b,
                    hour,
                    tz,
                    scratch,
                    chrono_tz_offset_at_utc
                )
            }
            dt => return_compute_error_with!("hour does not support", dt),
        }

        Ok(b.finish())
    }

    /// Extracts the years of a given temporal array as an array of integers
    pub fn year<T>(array: &PrimitiveArray<T>) -> Result<Int32Array>
    where
        T: ArrowTemporalType + ArrowNumericType,
        i64: std::convert::From<T::Native>,
    {
        let mut b = Int32Builder::with_capacity(array.len());
        match array.data_type() {
            // The time zone is ignored: this is the year of the UTC value.
            &DataType::Date32 | &DataType::Date64 | &DataType::Timestamp(_, _) => {
                extract_component_from_array!(array, b, year, value_as_datetime)
            }
            dt => return_compute_error_with!("year does not support", dt),
        }

        Ok(b.finish())
    }

    /// Extracts the quarter of a given temporal array as an array of integers
    pub fn quarter<T>(array: &PrimitiveArray<T>) -> Result<Int32Array>
    where
        T: ArrowTemporalType + ArrowNumericType,
        i64: std::convert::From<T::Native>,
    {
        let mapper = |month| (month - 1) / 3 + 1;
        let mut b = Int32Builder::with_capacity(array.len());
        match array.data_type() {
            &DataType::Date32 | &DataType::Date64 | &DataType::Timestamp(_, None) => {
                extract_mapped_component_from_array!(
                    array,
                    b,
                    month,
                    value_as_datetime,
                    mapper
                )
            }
            &DataType::Timestamp(_, Some(ref tz)) => {
                let mut scratch = Parsed::new();
                extract_mapped_component_from_array!(
                    array,
                    b,
                    month,
                    tz,
                    scratch,
                    chrono_tz_offset_at_utc,
                    mapper
                )
            }
            dt => return_compute_error_with!("quarter does not support", dt),
        }

        Ok(b.finish())
    }

    /// Extracts the month of a given temporal array as an array of integers
    pub fn month<T>(array: &PrimitiveArray<T>) -> Result<Int32Array>
    where
        T: ArrowTemporalType + ArrowNumericType,
        i64: std::convert::From<T::Native>,
    {
        let mut b = Int32Builder::with_capacity(array.len());
        match array.data_type() {
            &DataType::Date32 | &DataType::Date64 | &DataType::Timestamp(_, None) => {
                extract_component_from_array!(array, b, month, value_as_datetime)
            }
            &DataType::Timestamp(_, Some(ref tz)) => {
                let mut scratch = Parsed::new();
                extract_component_from_array!(
                    array,
                    b,
                    month,
                    tz,
                    scratch,
                    chrono_tz_offset_at_utc
                )
            }
            dt => return_compute_error_with!("month does not support", dt),
        }

        Ok(b.finish())
    }

    /// Extracts the day of a given temporal array as an array of integers
    pub fn day<T>(array: &PrimitiveArray<T>) -> Result<Int32Array>
    where
        T: ArrowTemporalType + ArrowNumericType,
        i64: std::convert::From<T::Native>,
    {
        let mut b = Int32Builder::with_capacity(array.len());
        match array.data_type() {
            &DataType::Date32 | &DataType::Date64 | &DataType::Timestamp(_, None) => {
                extract_component_from_array!(array, b, day, value_as_datetime)
            }
            &DataType::Timestamp(_, Some(ref tz)) => {
                let mut scratch = Parsed::new();
                extract_component_from_array!(
                    array,
                    b,
                    day,
                    tz,
                    scratch,
                    chrono_tz_offset_at_utc
                )
            }
            dt => return_compute_error_with!("day does not support", dt),
        }

        Ok(b.finish())
    }

    /// Extracts the minutes of a given temporal array as an array of integers
    pub fn minute<T>(array: &PrimitiveArray<T>) -> Result<Int32Array>
    where
        T: ArrowTemporalType + ArrowNumericType,
        i64: std::convert::From<T::Native>,
    {
        let mut b = Int32Builder::with_capacity(array.len());
        match array.data_type() {
            // Date32 is not supported (an error, not zeros).
            &DataType::Date64 | &DataType::Timestamp(_, None) => {
                extract_component_from_array!(array, b, minute, value_as_datetime)
            }
            &DataType::Timestamp(_, Some(ref tz)) => {
                let mut scratch = Parsed::new();
                extract_component_from_array!(
                    array,
                    b,
                    minute,
                    tz,
                    scratch,
                    chrono_tz_offset_at_utc
                )
            }
            dt => return_compute_error_with!("minute does not support", dt),
        }

        Ok(b.finish())
    }

    /// Extracts the seconds of a given temporal array as an array of integers
    pub fn second<T>(array: &PrimitiveArray<T>) -> Result<Int32Array>
    where
        T: ArrowTemporalType + ArrowNumericType,
        i64: std::convert::From<T::Native>,
    {
        let mut b = Int32Builder::with_capacity(array.len());
        match array.data_type() {
            // Date32 is not supported (an error, not zeros).
            &DataType::Date64 | &DataType::Timestamp(_, None) => {
                extract_component_from_array!(array, b, second, value_as_datetime)
            }
            &DataType::Timestamp(_, Some(ref tz)) => {
                let mut scratch = Parsed::new();
                extract_component_from_array!(
                    array,
                    b,
                    second,
                    tz,
                    scratch,
                    chrono_tz_offset_at_utc
                )
            }
            dt => return_compute_error_with!("second does not support", dt),
        }

        Ok(b.finish())
    }
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
                    kernels::$gran(array)
                }
            )*
        }

        impl DatePartable for TimestampMillisecondType {
            $(
                fn $gran(array: &TimestampMillisecondArray) -> Result<Int32Array> {
                    kernels::$gran(array)
                }
            )*
        }

        impl DatePartable for TimestampMicrosecondType {
            $(
                fn $gran(array: &TimestampMicrosecondArray) -> Result<Int32Array> {
                    kernels::$gran(array)
                }
            )*
        }

        impl DatePartable for TimestampNanosecondType {
            $(
                fn $gran(array: &TimestampNanosecondArray) -> Result<Int32Array> {
                    kernels::$gran(array)
                }
            )*
        }

        impl DatePartable for Date32Type {
            $(
                fn $gran(array: &Date32Array) -> Result<Int32Array> {
                    kernels::$gran(array)
                }
            )*
        }

        impl DatePartable for Date64Type {
            $(
                fn $gran(array: &Date64Array) -> Result<Int32Array> {
                    kernels::$gran(array)
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
    let mut builder = Int32Builder::with_capacity(array.len());
    for i in 0..array.len() {
        if array.is_null(i) {
            builder.append_null();
            continue;
        }
        let value = array.value(i);
        let result = op(value)?;
        builder.append_value(result);
    }
    Ok(builder.finish())
}

fn interval_day_time_op(
    array: &IntervalDayTimeArray,
    op: fn(IntervalDayTime) -> Result<i32>,
) -> Result<Int32Array> {
    let mut builder = Int32Builder::with_capacity(array.len());
    for i in 0..array.len() {
        if array.is_null(i) {
            builder.append_null();
            continue;
        }
        let value = array.value(i);
        let result = op(value)?;
        builder.append_value(result);
    }
    Ok(builder.finish())
}

fn interval_month_day_nano_op(
    array: &IntervalMonthDayNanoArray,
    op: fn(IntervalMonthDayNano) -> Result<i32>,
) -> Result<Int32Array> {
    let mut builder = Int32Builder::with_capacity(array.len());
    for i in 0..array.len() {
        if array.is_null(i) {
            builder.append_null();
            continue;
        }
        let value = array.value(i);
        let result = op(value)?;
        builder.append_value(result);
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

#[cfg(test)]
mod tests {
    //! Unless a test says otherwise, the expected values were produced by the
    //! arrow 13 build of this module (before the arrow 60 migration), so they pin
    //! the results bit for bit across arrow upgrades.

    use super::*;

    fn show_ints(r: Result<Int32Array>) -> String {
        match r {
            Ok(a) => {
                let v: Vec<String> = a
                    .iter()
                    .map(|v| v.map_or("null".to_string(), |v| v.to_string()))
                    .collect();
                format!("[{}]", v.join(", "))
            }
            Err(ArrowError::ComputeError(m)) => format!("ERR ComputeError({})", m),
            Err(e) => format!("ERR {:?}", e),
        }
    }

    /// `{:?}` of an f64 round-trips, so equal strings mean bit-identical values.
    fn show_floats(r: Result<Float64Array>) -> String {
        let v: Vec<String> = r
            .unwrap()
            .iter()
            .map(|v| v.map_or("null".to_string(), |v| format!("{:?}", v)))
            .collect();
        format!("[{}]", v.join(", "))
    }

    fn parts<T: DatePartable>(a: &PrimitiveArray<T>) -> Vec<String> {
        vec![
            format!("year: {}", show_ints(year(a))),
            format!("quarter: {}", show_ints(quarter(a))),
            format!("month: {}", show_ints(month(a))),
            format!("day: {}", show_ints(day(a))),
            format!("hour: {}", show_ints(hour(a))),
            format!("minute: {}", show_ints(minute(a))),
            format!("second: {}", show_ints(second(a))),
        ]
    }

    fn interval_parts<T: DatePartable + Epochable>(a: &PrimitiveArray<T>) -> Vec<String> {
        let mut v = parts(a);
        v.push(format!("epoch: {}", show_floats(epoch(a))));
        v
    }

    fn temporal_parts<T>(a: &PrimitiveArray<T>) -> Vec<String>
    where
        T: DatePartable + Epochable + ArrowTemporalType + ArrowNumericType,
        i64: From<T::Native>,
    {
        let mut v = parts(a);
        v.push(format!("doy: {}", show_ints(doy(a))));
        v.push(format!("dow: {}", show_ints(dow(a))));
        v.push(format!("epoch: {}", show_floats(epoch(a))));
        v
    }

    const SECONDS: [Option<i64>; 13] = [
        Some(0),
        Some(-1),
        Some(1),
        None,
        Some(1_599_572_549),   // 2020-09-08T13:42:29Z
        Some(1_615_705_199),   // 2021-03-14T06:59:59Z, America/New_York 01:59:59 EST
        Some(1_615_705_200),   // 2021-03-14T07:00:00Z, America/New_York 03:00:00 EDT
        Some(1_609_457_400),   // 2020-12-31T23:30:00Z
        Some(1_582_934_400),   // 2020-02-29T00:00:00Z
        Some(-86_401),         // 1969-12-30T23:59:59Z
        Some(-62_135_596_800), // 0001-01-01T00:00:00Z
        Some(253_402_300_799), // 9999-12-31T23:59:59Z
        Some(1_640_995_199),   // 2021-12-31T23:59:59Z
    ];

    fn seconds(tz: Option<&str>) -> TimestampSecondArray {
        TimestampSecondArray::from(SECONDS.to_vec()).with_timezone_opt(tz)
    }

    const SECONDS_YEAR: &str =
        "year: [1970, 1969, 1970, null, 2020, 2021, 2021, 2020, 2020, 1969, 1, 9999, 2021]";
    const SECONDS_EPOCH: &str = "epoch: [0.0, -1.0, 1.0, null, 1599572549.0, 1615705199.0, 1615705200.0, 1609457400.0, 1582934400.0, -86401.0, -62135596800.0, 253402300799.0, 1640995199.0]";

    #[test]
    fn timestamp_without_tz() {
        assert_eq!(
            temporal_parts(&seconds(None)),
            [
                SECONDS_YEAR,
                "quarter: [1, 4, 1, null, 3, 1, 1, 4, 1, 4, 1, 4, 4]",
                "month: [1, 12, 1, null, 9, 3, 3, 12, 2, 12, 1, 12, 12]",
                "day: [1, 31, 1, null, 8, 14, 14, 31, 29, 30, 1, 31, 31]",
                "hour: [0, 23, 0, null, 13, 6, 7, 23, 0, 23, 0, 23, 23]",
                "minute: [0, 59, 0, null, 42, 59, 0, 30, 0, 59, 0, 59, 59]",
                "second: [0, 59, 1, null, 29, 59, 0, 0, 0, 59, 0, 59, 59]",
                "doy: [1, 365, 1, null, 252, 73, 73, 366, 60, 364, 1, 365, 365]",
                "dow: [4, 3, 4, null, 2, 0, 0, 4, 6, 2, 1, 5, 5]",
                SECONDS_EPOCH,
            ]
        );
    }

    #[test]
    fn timestamp_with_fixed_offset_tz() {
        // `year` ignores the time zone; the other parts use the local time.
        assert_eq!(
            temporal_parts(&seconds(Some("+05:30"))),
            [
                SECONDS_YEAR,
                "quarter: [1, 1, 1, null, 3, 1, 1, 1, 1, 4, 1, 1, 1]",
                "month: [1, 1, 1, null, 9, 3, 3, 1, 2, 12, 1, 1, 1]",
                "day: [1, 1, 1, null, 8, 14, 14, 1, 29, 31, 1, 1, 1]",
                "hour: [5, 5, 5, null, 19, 12, 12, 5, 5, 5, 5, 5, 5]",
                "minute: [30, 29, 30, null, 12, 29, 30, 0, 30, 29, 30, 29, 29]",
                "second: [0, 59, 1, null, 29, 59, 0, 0, 0, 59, 0, 59, 59]",
                "doy: [1, 1, 1, null, 252, 73, 73, 1, 60, 365, 1, 1, 1]",
                "dow: [4, 4, 4, null, 2, 0, 0, 5, 6, 3, 1, 6, 6]",
                SECONDS_EPOCH,
            ]
        );
        assert_eq!(
            temporal_parts(&seconds(Some("-00:30"))),
            [
                SECONDS_YEAR,
                "quarter: [4, 4, 4, null, 3, 1, 1, 4, 1, 4, 4, 4, 4]",
                "month: [12, 12, 12, null, 9, 3, 3, 12, 2, 12, 12, 12, 12]",
                "day: [31, 31, 31, null, 8, 14, 14, 31, 28, 30, 31, 31, 31]",
                "hour: [23, 23, 23, null, 13, 6, 6, 23, 23, 23, 23, 23, 23]",
                "minute: [30, 29, 30, null, 12, 29, 30, 0, 30, 29, 30, 29, 29]",
                "second: [0, 59, 1, null, 29, 59, 0, 0, 0, 59, 0, 59, 59]",
                "doy: [365, 365, 365, null, 252, 73, 73, 366, 59, 364, 366, 365, 365]",
                "dow: [3, 3, 3, null, 2, 0, 0, 4, 5, 2, 0, 5, 5]",
                SECONDS_EPOCH,
            ]
        );
    }

    #[test]
    fn timestamp_with_named_tz() {
        // DST switch between rows 5 and 6; 0001-01-01 uses the LMT offset
        // (-04:56:02). `doy` and `dow` only accept fixed offsets.
        assert_eq!(
            temporal_parts(&seconds(Some("America/New_York"))),
            [
                SECONDS_YEAR,
                "quarter: [4, 4, 4, null, 3, 1, 1, 4, 1, 4, 4, 4, 4]",
                "month: [12, 12, 12, null, 9, 3, 3, 12, 2, 12, 12, 12, 12]",
                "day: [31, 31, 31, null, 8, 14, 14, 31, 28, 30, 31, 31, 31]",
                "hour: [19, 18, 19, null, 9, 1, 3, 18, 19, 18, 19, 18, 18]",
                "minute: [0, 59, 0, null, 42, 59, 0, 30, 0, 59, 3, 59, 59]",
                "second: [0, 59, 1, null, 29, 59, 0, 0, 0, 59, 58, 59, 59]",
                "doy: ERR ComputeError(Unable to parse timezone: None)",
                "dow: ERR ComputeError(Unable to parse timezone: None)",
                SECONDS_EPOCH,
            ]
        );
    }

    #[test]
    fn timestamp_with_invalid_tz() {
        for (tz, error) in [
            ("+0800", "Invalid timezone: \"Expected format [+-]XX:XX\""),
            ("+24:00", "Invalid timezone: Err(ParseError(OutOfRange))"),
            ("+08:60", "Unable to parse timezone: None"),
            ("Invalid/Zone", "Unable to parse timezone: None"),
            ("utc", "Unable to parse timezone: None"),
        ] {
            let error = format!("ERR ComputeError({})", error);
            let mut expected = vec![SECONDS_YEAR.to_string()];
            for part in ["quarter", "month", "day", "hour", "minute", "second", "doy"] {
                expected.push(format!("{}: {}", part, error));
            }
            expected.push(format!("dow: {}", error));
            expected.push(SECONDS_EPOCH.to_string());
            assert_eq!(temporal_parts(&seconds(Some(tz))), expected, "tz {}", tz);
        }
    }

    #[test]
    fn sliced_timestamp_with_tz() {
        let a = TimestampNanosecondArray::from(vec![
            Some(0),
            Some(1_000_000_000),
            Some(-1_000_000_000),
            None,
            Some(1_599_572_549_190_855_123),
            Some(1_615_705_199_999_999_999),
            Some(1_615_705_200_000_000_000),
            Some(9_223_372_036_000_000_000),
        ])
        .with_timezone("+05:30")
        .slice(4, 3);
        assert_eq!(
            temporal_parts(&a),
            [
                "year: [2020, 2021, 2021]",
                "quarter: [3, 1, 1]",
                "month: [9, 3, 3]",
                "day: [8, 14, 14]",
                "hour: [19, 12, 12]",
                "minute: [12, 29, 30]",
                "second: [29, 59, 0]",
                "doy: [252, 73, 73]",
                "dow: [2, 0, 0]",
                "epoch: [1599572549.1908553, 1615705200.0, 1615705200.0]",
            ]
        );
    }

    #[test]
    fn date32() {
        let a = Date32Array::from(vec![
            Some(0),
            Some(-1),
            Some(1),
            None,
            Some(18513),   // 2020-09-08
            Some(18321),   // 2020-02-29
            Some(-719162), // 0001-01-01
            Some(2932896), // 9999-12-31
            Some(18627),   // 2020-12-31
        ]);
        // `minute` and `second` do not support Date32 (an error, not zeros).
        assert_eq!(
            temporal_parts(&a),
            [
                "year: [1970, 1969, 1970, null, 2020, 2020, 1, 9999, 2020]",
                "quarter: [1, 4, 1, null, 3, 1, 1, 4, 4]",
                "month: [1, 12, 1, null, 9, 2, 1, 12, 12]",
                "day: [1, 31, 2, null, 8, 29, 1, 31, 31]",
                "hour: [0, 0, 0, null, 0, 0, 0, 0, 0]",
                "minute: ERR ComputeError(minute does not support: Date32)",
                "second: ERR ComputeError(second does not support: Date32)",
                "doy: [1, 365, 2, null, 252, 60, 1, 365, 366]",
                "dow: [4, 3, 5, null, 2, 6, 1, 5, 4]",
                "epoch: [0.0, -86400.0, 86400.0, null, 1599523200.0, 1582934400.0, -62135596800.0, 253402214400.0, 1609372800.0]",
            ]
        );
    }

    #[test]
    fn date64() {
        let a = Date64Array::from(vec![
            Some(0),
            Some(1000),
            Some(-86_400_000),
            None,
            Some(1_514_764_800_000),
            Some(1_550_636_625_000),
            Some(1_599_572_549_190),
        ]);
        assert_eq!(
            temporal_parts(&a),
            [
                "year: [1970, 1970, 1969, null, 2018, 2019, 2020]",
                "quarter: [1, 1, 4, null, 1, 1, 3]",
                "month: [1, 1, 12, null, 1, 2, 9]",
                "day: [1, 1, 31, null, 1, 20, 8]",
                "hour: [0, 0, 0, null, 0, 4, 13]",
                "minute: [0, 0, 0, null, 0, 23, 42]",
                "second: [0, 1, 0, null, 0, 45, 29]",
                "doy: [1, 1, 365, null, 1, 51, 252]",
                "dow: [4, 4, 3, null, 1, 3, 2]",
                "epoch: [0.0, 1.0, -86400.0, null, 1514764800.0, 1550636625.0, 1599572549.19]",
            ]
        );
    }

    /// arrow 13 panicked on pre-epoch values that are not whole seconds; arrow 60
    /// converts them correctly (floor semantics, like whole-second values).
    #[test]
    fn pre_epoch_sub_second_values() {
        let expected = [
            "year: [1969]",
            "quarter: [4]",
            "month: [12]",
            "day: [31]",
            "hour: [23]",
            "minute: [59]",
            "second: [59]",
            "doy: [365]",
            "dow: [3]",
            "epoch: [-0.001]",
        ];
        assert_eq!(
            temporal_parts(&TimestampMillisecondArray::from(vec![-1])),
            expected
        );
        assert_eq!(temporal_parts(&Date64Array::from(vec![-1])), expected);
        assert_eq!(
            temporal_parts(
                &TimestampMicrosecondArray::from(vec![-1_500_000])
                    .with_timezone("America/New_York")
            ),
            [
                "year: [1969]",
                "quarter: [4]",
                "month: [12]",
                "day: [31]",
                "hour: [18]",
                "minute: [59]",
                "second: [58]",
                "doy: ERR ComputeError(Unable to parse timezone: None)",
                "dow: ERR ComputeError(Unable to parse timezone: None)",
                "epoch: [-1.5]",
            ]
        );
    }

    /// arrow 13 panicked on values chrono can't represent; arrow 60 conversions
    /// return None, which these kernels turn into NULL (or into their
    /// "Unable to read value as datetime" error on the time zone path).
    #[test]
    fn out_of_range_values() {
        let nulls = |error: Option<&str>| {
            let mut v = vec!["year: [null]".to_string()];
            for part in [
                "quarter", "month", "day", "hour", "minute", "second", "doy", "dow",
            ] {
                v.push(match error {
                    Some(e) => format!("{}: ERR ComputeError({})", part, e),
                    None => format!("{}: [null]", part),
                });
            }
            v.push("epoch: [9.223372036854776e18]".to_string());
            v
        };
        assert_eq!(
            temporal_parts(&TimestampSecondArray::from(vec![i64::MAX])),
            nulls(None)
        );
        assert_eq!(
            temporal_parts(
                &TimestampSecondArray::from(vec![i64::MAX]).with_timezone("+05:30")
            ),
            nulls(Some("Unable to read value as datetime: None"))
        );

        // In range as UTC, out of range once shifted to the local time.
        let mut expected = vec!["year: [262142]".to_string()];
        for part in [
            "quarter", "month", "day", "hour", "minute", "second", "doy", "dow",
        ] {
            expected.push(format!("{}: [null]", part));
        }
        expected.push("epoch: [8210266876799.0]".to_string());
        assert_eq!(
            temporal_parts(
                &TimestampSecondArray::from(vec![8_210_266_876_799])
                    .with_timezone("+05:30")
            ),
            expected
        );

        assert_eq!(
            temporal_parts(&Date32Array::from(vec![i32::MAX])),
            [
                "year: [null]",
                "quarter: [null]",
                "month: [null]",
                "day: [null]",
                "hour: [null]",
                "minute: ERR ComputeError(minute does not support: Date32)",
                "second: ERR ComputeError(second does not support: Date32)",
                "doy: [null]",
                "dow: [null]",
                "epoch: [185542587100800.0]",
            ]
        );
    }

    #[test]
    fn interval_year_month() {
        let a = IntervalYearMonthArray::from(vec![
            Some(0),
            Some(1),
            Some(2),
            Some(3),
            Some(11),
            Some(12),
            Some(13),
            Some(25),
            None,
            Some(-1),
            Some(-2),
            Some(-3),
            Some(-11),
            Some(-12),
            Some(-13),
            Some(-25),
            Some(i32::MAX),
            Some(i32::MIN),
        ]);
        assert_eq!(
            interval_parts(&a),
            [
                "year: [0, 0, 0, 0, 0, 1, 1, 2, null, 0, 0, 0, 0, -1, -1, -2, 178956970, -178956970]",
                "quarter: [1, 1, 1, 2, 4, 1, 1, 1, null, 1, 1, 0, -2, 1, 1, 1, 3, -1]",
                "month: [0, 1, 2, 3, 11, 0, 1, 1, null, -1, -2, -3, -11, 0, -1, -1, 7, -8]",
                "day: [0, 0, 0, 0, 0, 0, 0, 0, null, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
                "hour: [0, 0, 0, 0, 0, 0, 0, 0, null, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
                "minute: [0, 0, 0, 0, 0, 0, 0, 0, null, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
                "second: [0, 0, 0, 0, 0, 0, 0, 0, null, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
                "epoch: [0.0, 2592000.0, 5184000.0, 7776000.0, 28512000.0, 31557600.0, 34149600.0, 65707200.0, null, -2592000.0, -5184000.0, -7776000.0, -28512000.0, -31557600.0, -34149600.0, -65707200.0, 5647452494616000.0, -5647452497208000.0]",
            ]
        );
    }

    #[test]
    fn interval_day_time() {
        let a = IntervalDayTimeArray::from(
            [
                Some((0, 0)),
                Some((1, 0)),
                Some((-1, 0)),
                Some((0, 1)),
                Some((0, -1)),
                None,
                Some((0, 999)),
                Some((0, 1000)),
                Some((0, 59_999)),
                Some((0, 60_000)),
                Some((0, 3_599_999)),
                Some((0, 3_600_000)),
                Some((1, 86_399_999)),
                Some((-3, -7_200_001)),
                Some((5, -1)),
                Some((-5, 1)),
                Some((40, 90_061_001)),
                Some((i32::MAX, i32::MAX)),
                Some((i32::MIN, i32::MIN)),
                Some((i32::MAX, i32::MIN)),
                Some((i32::MIN, i32::MAX)),
            ]
            .into_iter()
            .map(|v| v.map(|(d, ms)| IntervalDayTimeType::make_value(d, ms)))
            .collect::<Vec<_>>(),
        );
        assert_eq!(
            interval_parts(&a),
            [
                "year: [0, 0, 0, 0, 0, null, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
                "quarter: [1, 1, 1, 1, 1, null, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]",
                "month: [0, 0, 0, 0, 0, null, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
                "day: [0, 1, -1, 0, 0, null, 0, 0, 0, 0, 0, 0, 1, -3, 5, -5, 40, 2147483647, -2147483648, 2147483647, -2147483648]",
                "hour: [0, 0, 0, 0, 0, null, 0, 0, 0, 0, 0, 1, 23, -2, 0, 0, 25, 596, -596, -596, 596]",
                "minute: [0, 0, 0, 0, 0, null, 0, 0, 0, 1, 59, 0, 59, 0, 0, 0, 1, 31, -31, -31, 31]",
                "second: [0, 0, 0, 0, 0, null, 0, 1, 59, 0, 59, 0, 59, 0, 0, 0, 1, 23, -23, -23, 23]",
                "epoch: [0.0, 86400.0, -86400.0, 0.001, -0.001, null, 0.999, 1.0, 59.999, 60.0, 3599.999, 3600.0, 172799.999, -266400.001, 431999.999, -431999.999, 3546061.001, 185542589248283.66, -185542589334683.66, 185542584953316.34, -185542585039716.34]",
            ]
        );
    }

    fn month_day_nano_array() -> IntervalMonthDayNanoArray {
        IntervalMonthDayNanoArray::from(
            [
                Some((0, 0, 0)),
                Some((1, 0, 0)),
                Some((12, 0, 0)),
                Some((13, 2, 3_600_000_000_001)),
                Some((-13, -2, -3_661_000_000_000)),
                None,
                Some((0, 0, 1)),
                Some((0, 0, -1)),
                Some((0, 0, 999_999_999)),
                Some((0, 0, 59_999_999_999)),
                Some((0, 0, 60_000_000_000)),
                Some((0, 0, 86_400_000_000_000)),
                Some((-1, 1, -1)),
                Some((1, -1, 1)),
                Some((25, 31, 90_061_000_000_001)),
                Some((i32::MAX, i32::MAX, i64::MAX)),
                Some((i32::MIN, i32::MIN, i64::MIN)),
                Some((i32::MAX, i32::MIN, i64::MIN)),
                Some((i32::MIN, i32::MAX, i64::MAX)),
                Some((-1, -1, -1)),
            ]
            .into_iter()
            .map(|v| v.map(|(m, d, ns)| IntervalMonthDayNanoType::make_value(m, d, ns)))
            .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn interval_month_day_nano() {
        assert_eq!(
            interval_parts(&month_day_nano_array()),
            [
                "year: [0, 0, 1, 1, -1, null, 0, 0, 0, 0, 0, 0, 0, 0, 2, 178956970, -178956970, 178956970, -178956970, 0]",
                "quarter: [1, 1, 1, 1, 1, null, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, -1, 3, -1, 1]",
                "month: [0, 1, 0, 1, -1, null, 0, 0, 0, 0, 0, 0, -1, 1, 1, 7, -8, 7, -8, -1]",
                "day: [0, 0, 0, 2, -2, null, 0, 0, 0, 0, 0, 0, 1, -1, 31, 2147483647, -2147483648, -2147483648, 2147483647, -1]",
                "hour: [0, 0, 0, 1, -1, null, 0, 0, 0, 0, 0, 24, 0, 0, 25, 2562047, -2562047, -2562047, 2562047, 0]",
                "minute: [0, 0, 0, 0, -1, null, 0, 0, 0, 0, 1, 0, 0, 0, 1, 47, -47, -47, 47, 0]",
                "second: [0, 0, 0, 0, -1, null, 0, 0, 0, 59, 0, 0, 0, 0, 1, 16, -16, -16, 16, 0]",
                "epoch: [0.0, 2592000.0, 31557600.0, 34326000.0, -34326061.0, null, 1e-9, -1e-9, 0.999999999, 59.999999999, 60.0, 86400.0, -2505600.000000001, 2505600.000000001, 68475661.0, 5833004305088837.0, -5833004307767237.0, 5461900684056763.0, -5461900686735163.0, -2678400.000000001]",
            ]
        );
    }

    #[test]
    fn sliced_interval_month_day_nano() {
        assert_eq!(
            interval_parts(&month_day_nano_array().slice(3, 10)),
            [
                "year: [1, -1, null, 0, 0, 0, 0, 0, 0, 0]",
                "quarter: [1, 1, null, 1, 1, 1, 1, 1, 1, 1]",
                "month: [1, -1, null, 0, 0, 0, 0, 0, 0, -1]",
                "day: [2, -2, null, 0, 0, 0, 0, 0, 0, 1]",
                "hour: [1, -1, null, 0, 0, 0, 0, 0, 24, 0]",
                "minute: [0, -1, null, 0, 0, 0, 0, 1, 0, 0]",
                "second: [0, -1, null, 0, 0, 0, 59, 0, 0, 0]",
                "epoch: [34326000.0, -34326061.0, null, 1e-9, -1e-9, 0.999999999, 59.999999999, 60.0, 86400.0, -2505600.000000001]",
            ]
        );
    }
}
