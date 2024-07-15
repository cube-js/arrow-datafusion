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

use std::sync::Arc;

use arrow::{
    array::{
        Array, ArrayRef, Date32Array, Int64Array, IntervalDayTimeArray,
        IntervalMonthDayNanoArray, IntervalYearMonthArray, TimestampNanosecondArray,
    },
    datatypes::{
        ArrowPrimitiveType, DataType, Date32Type, IntervalDayTimeType,
        IntervalMonthDayNanoType, IntervalUnit, IntervalYearMonthType,
        TimestampNanosecondType,
    },
    temporal_conversions::{date32_to_datetime, timestamp_ns_to_datetime},
};
use chrono::{Datelike, Days, Duration, Months, NaiveDate, NaiveDateTime};
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::Operator;

type TimestampNanosecond = <TimestampNanosecondType as ArrowPrimitiveType>::Native;
type Date32 = <Date32Type as ArrowPrimitiveType>::Native;
type IntervalYearMonth = <IntervalYearMonthType as ArrowPrimitiveType>::Native;
type IntervalDayTime = <IntervalDayTimeType as ArrowPrimitiveType>::Native;
type IntervalMonthDayNano = <IntervalMonthDayNanoType as ArrowPrimitiveType>::Native;

/// return whether the binary expression with distinct types can be evaluated.
pub fn distinct_types_allowed(
    left_type: &DataType,
    op: &Operator,
    right_type: &DataType,
) -> bool {
    use arrow::datatypes::TimeUnit::*;
    use DataType::*;

    match op {
        Operator::Plus => matches!(
            (left_type, right_type),
            (Interval(_), Timestamp(Nanosecond, _))
                | (Timestamp(Nanosecond, _), Interval(_))
        ),
        Operator::Minus => matches!(
            (left_type, right_type),
            (Timestamp(Nanosecond, _), Interval(_))
                | (Timestamp(Nanosecond, _), Timestamp(Nanosecond, _))
                | (Timestamp(Nanosecond, _), Date32)
        ),
        Operator::Multiply => matches!(
            (left_type, right_type),
            (Int64, Interval(_)) | (Interval(_), Int64)
        ),
        _ => false,
    }
}

/// return two types for physical expressions to be coerced to when two types
/// must be distinct.
pub fn coerce_types_distinct(
    lhs_type: &DataType,
    op: &Operator,
    rhs_type: &DataType,
) -> Option<(DataType, DataType)> {
    use arrow::datatypes::TimeUnit::*;
    use DataType::*;
    use IntervalUnit::*;

    match op {
        Operator::Plus => match (lhs_type, rhs_type) {
            (Interval(iunit), Timestamp(_, tz)) => {
                Some((Interval(iunit.clone()), Timestamp(Nanosecond, tz.clone())))
            }
            (Null, Timestamp(_, tz)) => {
                Some((Interval(MonthDayNano), Timestamp(Nanosecond, tz.clone())))
            }
            (Timestamp(_, tz), Interval(iunit)) => {
                Some((Timestamp(Nanosecond, tz.clone()), Interval(iunit.clone())))
            }
            (Timestamp(_, tz), Null) => {
                Some((Timestamp(Nanosecond, tz.clone()), Interval(MonthDayNano)))
            }
            (Interval(iunit), Date64) | (Interval(iunit), Date32) => {
                Some((Interval(iunit.clone()), Timestamp(Nanosecond, None)))
            }
            (Date64, Interval(iunit)) | (Date32, Interval(iunit)) => {
                Some((Timestamp(Nanosecond, None), Interval(iunit.clone())))
            }
            _ => None,
        },
        Operator::Minus => match (lhs_type, rhs_type) {
            (Timestamp(_, tz), Interval(iunit)) => {
                Some((Timestamp(Nanosecond, tz.clone()), Interval(iunit.clone())))
            }
            (Timestamp(_, tz), Null) => {
                Some((Timestamp(Nanosecond, tz.clone()), Interval(MonthDayNano)))
            }
            (Date64, Interval(iunit)) | (Date32, Interval(iunit)) => {
                Some((Timestamp(Nanosecond, None), Interval(iunit.clone())))
            }
            (Timestamp(_, tz), Timestamp(_, tz2)) => Some((
                Timestamp(Nanosecond, tz.clone()),
                Timestamp(Nanosecond, tz2.clone()),
            )),
            (Timestamp(_, tz), Date32) => {
                Some((Timestamp(Nanosecond, tz.clone()), Date32))
            }
            _ => None,
        },
        Operator::Multiply => match (lhs_type, rhs_type) {
            // TODO: this isn't exactly correct and Utf8 should go the Float64 route,
            // but will crash rather than yield incorrect results, so is acceptable
            (Utf8, Interval(unit)) => Some((Int64, Interval(unit.clone()))),
            (Interval(unit), Utf8) => Some((Interval(unit.clone()), Int64)),
            (Int64, Interval(unit))
            | (Int32, Interval(unit))
            | (Int16, Interval(unit))
            | (Int8, Interval(unit))
            | (UInt64, Interval(unit))
            | (UInt32, Interval(unit))
            | (UInt16, Interval(unit))
            | (UInt8, Interval(unit))
            | (Null, Interval(unit)) => Some((Int64, Interval(unit.clone()))),
            (Interval(unit), Int64)
            | (Interval(unit), Int32)
            | (Interval(unit), Int16)
            | (Interval(unit), Int8)
            | (Interval(unit), UInt64)
            | (Interval(unit), UInt32)
            | (Interval(unit), UInt16)
            | (Interval(unit), UInt8)
            | (Interval(unit), Null) => Some((Interval(unit.clone()), Int64)),
            _ => None,
        },
        _ => None,
    }
}

/// try to evaluate the expression having distinct args,
/// returns None if no operator matches left and right data types
pub fn evaluate_distinct_with_resolved_args(
    left: Arc<dyn Array>,
    left_data_type: &DataType,
    op: &Operator,
    right: Arc<dyn Array>,
    right_data_type: &DataType,
) -> Option<Result<ArrayRef>> {
    use arrow::datatypes::TimeUnit::*;
    use DataType::*;

    match op {
        Operator::Plus => match (left_data_type, right_data_type) {
            (Timestamp(Nanosecond, None), Interval(_)) => {
                Some(timestamp_add_interval(left, right, false))
            }
            (Timestamp(Nanosecond, Some(tz)), Interval(_)) if tz == "UTC" => {
                Some(timestamp_add_interval(left, right, false))
            }
            (Interval(_), Timestamp(Nanosecond, None)) => {
                Some(timestamp_add_interval(right, left, false))
            }
            (Interval(_), Timestamp(Nanosecond, Some(tz))) if tz == "UTC" => {
                Some(timestamp_add_interval(right, left, false))
            }
            _ => None,
        },
        Operator::Minus => match (left_data_type, right_data_type) {
            (Timestamp(Nanosecond, None), Interval(_)) => {
                Some(timestamp_add_interval(left, right, true))
            }
            (Timestamp(Nanosecond, Some(tz)), Interval(_)) if tz == "UTC" => {
                Some(timestamp_add_interval(left, right, true))
            }
            (Timestamp(Nanosecond, None), Timestamp(Nanosecond, None)) => {
                // TODO: Implement postgres behavior with time zones
                Some(timestamp_subtract_timestamp(left, right))
            }
            (Timestamp(Nanosecond, None), Date32) => {
                Some(timestamp_subtract_date(left, right))
            }
            _ => None,
        },
        Operator::Multiply => match (left_data_type, right_data_type) {
            (Int64, Interval(_)) => Some(interval_multiply_int(right, left)),
            (Interval(_), Int64) => Some(interval_multiply_int(left, right)),
            _ => None,
        },
        _ => None,
    }
}

fn interval_multiply_int(
    intervals: Arc<dyn Array>,
    multipliers: Arc<dyn Array>,
) -> Result<ArrayRef> {
    let multipliers = match multipliers.data_type() {
        DataType::Int64 => multipliers,
        t => {
            return Err(DataFusionError::Execution(format!(
                "unsupported multiplicand type {}",
                t
            )))
        }
    };
    let multipliers = multipliers.as_any().downcast_ref::<Int64Array>().unwrap();

    match intervals.data_type() {
        DataType::Interval(IntervalUnit::YearMonth) => {
            let intervals = intervals
                .as_any()
                .downcast_ref::<IntervalYearMonthArray>()
                .unwrap();
            let result = intervals
                .iter()
                .zip(multipliers.iter())
                .map(|(i, m)| scalar_interval_year_month_mul_int(i, m))
                .collect::<Result<IntervalYearMonthArray>>()?;
            Ok(Arc::new(result))
        }
        DataType::Interval(IntervalUnit::DayTime) => {
            let intervals = intervals
                .as_any()
                .downcast_ref::<IntervalDayTimeArray>()
                .unwrap();
            let result = intervals
                .iter()
                .zip(multipliers.iter())
                .map(|(i, m)| scalar_interval_day_time_mul_int(i, m))
                .collect::<Result<IntervalDayTimeArray>>()?;
            Ok(Arc::new(result))
        }
        DataType::Interval(IntervalUnit::MonthDayNano) => {
            let intervals = intervals
                .as_any()
                .downcast_ref::<IntervalMonthDayNanoArray>()
                .unwrap();
            let result = intervals
                .iter()
                .zip(multipliers.iter())
                .map(|(i, m)| scalar_interval_month_day_nano_time_mul_int(i, m))
                .collect::<Result<IntervalMonthDayNanoArray>>()?;
            Ok(Arc::new(result))
        }
        t => Err(DataFusionError::Execution(format!(
            "multiplication expected Interval, got {}",
            t
        ))),
    }
}

fn scalar_interval_year_month_mul_int(
    interval: Option<IntervalYearMonth>,
    multiplier: Option<i64>,
) -> Result<Option<IntervalYearMonth>> {
    if interval.is_none() || multiplier.is_none() {
        return Ok(None);
    }
    let interval = interval.unwrap();
    let multiplier = multiplier.unwrap();

    let multiplier = i32::try_from(multiplier).map_err(|err| {
        DataFusionError::Execution(format!(
            "unable to convert interval multiplier to Int32: {}",
            err
        ))
    })?;
    let result = interval
        .checked_mul(multiplier)
        .ok_or_else(|| DataFusionError::Execution("interval out of range".to_string()))?;
    Ok(Some(result))
}

fn scalar_interval_day_time_mul_int(
    interval: Option<IntervalDayTime>,
    multiplier: Option<i64>,
) -> Result<Option<IntervalDayTime>> {
    if interval.is_none() || multiplier.is_none() {
        return Ok(None);
    }
    let interval = interval.unwrap();
    let multiplier = multiplier.unwrap();

    let (days, milliseconds) = IntervalDayTimeType::to_parts(interval);
    let multiplier = i32::try_from(multiplier).map_err(|err| {
        DataFusionError::Execution(format!(
            "unable to convert interval multiplier to Int32: {}",
            err
        ))
    })?;
    let days = days
        .checked_mul(multiplier)
        .ok_or_else(|| DataFusionError::Execution("interval out of range".to_string()))?;
    let milliseconds = milliseconds
        .checked_mul(multiplier)
        .ok_or_else(|| DataFusionError::Execution("interval out of range".to_string()))?;
    let interval_product = IntervalDayTimeType::make_value(days, milliseconds);
    Ok(Some(interval_product))
}

fn scalar_interval_month_day_nano_time_mul_int(
    interval: Option<IntervalMonthDayNano>,
    multiplier: Option<i64>,
) -> Result<Option<IntervalMonthDayNano>> {
    if interval.is_none() || multiplier.is_none() {
        return Ok(None);
    }
    let interval = interval.unwrap();
    let multiplier = multiplier.unwrap();

    let (months, days, nanos) = IntervalMonthDayNanoType::to_parts(interval);

    let multiplier = i32::try_from(multiplier).map_err(|err| {
        DataFusionError::Execution(format!(
            "unable to convert interval multiplier to Int32: {}",
            err
        ))
    })?;

    let months = months.checked_mul(multiplier).ok_or_else(|| {
        DataFusionError::Execution("interval out of range (months)".to_string())
    })?;
    let days = days.checked_mul(multiplier).ok_or_else(|| {
        DataFusionError::Execution("interval out of range (days)".to_string())
    })?;
    let nanos = nanos.checked_mul(multiplier as i64).ok_or_else(|| {
        DataFusionError::Execution("interval out of range (nanos)".to_string())
    })?;

    let interval = IntervalMonthDayNanoType::make_value(months, days, nanos);
    Ok(Some(interval))
}

fn timestamp_add_interval(
    timestamps: Arc<dyn Array>,
    intervals: Arc<dyn Array>,
    negated: bool,
) -> Result<ArrayRef> {
    let timestamps = timestamps
        .as_any()
        .downcast_ref::<TimestampNanosecondArray>()
        .unwrap();

    match intervals.data_type() {
        DataType::Interval(IntervalUnit::YearMonth) => {
            let intervals = intervals
                .as_any()
                .downcast_ref::<IntervalYearMonthArray>()
                .unwrap();
            let result = timestamps
                .iter()
                .zip(intervals.iter())
                .map(|(t, i)| scalar_timestamp_add_interval_year_month(t, i, negated))
                .collect::<Result<TimestampNanosecondArray>>()?;
            Ok(Arc::new(result))
        }
        DataType::Interval(IntervalUnit::DayTime) => {
            let intervals = intervals
                .as_any()
                .downcast_ref::<IntervalDayTimeArray>()
                .unwrap();
            let result = timestamps
                .iter()
                .zip(intervals.iter())
                .map(|(t, i)| scalar_timestamp_add_interval_day_time(t, i, negated))
                .collect::<Result<TimestampNanosecondArray>>()?;
            Ok(Arc::new(result))
        }
        DataType::Interval(IntervalUnit::MonthDayNano) => {
            let intervals = intervals
                .as_any()
                .downcast_ref::<IntervalMonthDayNanoArray>()
                .unwrap();
            let result = timestamps
                .iter()
                .zip(intervals.iter())
                .map(|(t, i)| scalar_timestamp_add_interval_month_day_nano(t, i, negated))
                .collect::<Result<TimestampNanosecondArray>>()?;
            Ok(Arc::new(result))
        }
        t => Err(DataFusionError::Execution(format!(
            "addition expected Interval, got {}",
            t
        ))),
    }
}

fn timestamp_subtract_timestamp(
    left: Arc<dyn Array>,
    right: Arc<dyn Array>,
) -> Result<ArrayRef> {
    let left = left
        .as_any()
        .downcast_ref::<TimestampNanosecondArray>()
        .unwrap();
    let right = right
        .as_any()
        .downcast_ref::<TimestampNanosecondArray>()
        .unwrap();

    let result = left
        .iter()
        .zip(right.iter())
        .map(|(t_l, t_r)| scalar_timestamp_subtract_timestamp(t_l, t_r))
        .collect::<Result<IntervalMonthDayNanoArray>>()?;
    Ok(Arc::new(result))
}

fn timestamp_subtract_date(
    left: Arc<dyn Array>,
    right: Arc<dyn Array>,
) -> Result<ArrayRef> {
    let left = left
        .as_any()
        .downcast_ref::<TimestampNanosecondArray>()
        .unwrap();
    let right = right.as_any().downcast_ref::<Date32Array>().unwrap();

    let result = left
        .iter()
        .zip(right.iter())
        .map(|(t_l, t_r)| scalar_timestamp_subtract_date(t_l, t_r))
        .collect::<Result<IntervalMonthDayNanoArray>>()?;
    Ok(Arc::new(result))
}

fn scalar_timestamp_add_interval_year_month(
    timestamp: Option<TimestampNanosecond>,
    interval: Option<IntervalYearMonth>,
    negated: bool,
) -> Result<Option<TimestampNanosecond>> {
    if timestamp.is_none() || interval.is_none() {
        return Ok(None);
    }
    let timestamp = timestamp.unwrap();
    let interval = interval.unwrap();

    let timestamp = timestamp_ns_to_datetime(timestamp);
    let interval = match negated {
        true => interval.checked_neg().ok_or_else(|| {
            DataFusionError::Execution("interval out of range".to_string())
        })?,
        false => interval,
    };

    // TODO: legacy code, check validity

    let mut year = timestamp.year();
    // Note month is numbered 0..11 in this function.
    let mut month = timestamp.month() as i32 - 1;

    year += interval / 12;
    month += interval % 12;

    if month < 0 {
        year -= 1;
        month += 12;
    }
    debug_assert!(0 <= month);
    year += month / 12;
    month %= 12;

    let result = change_ym(timestamp, year, 1 + month as u32)?;
    Ok(Some(result.timestamp_nanos()))
}

fn scalar_timestamp_add_interval_day_time(
    timestamp: Option<TimestampNanosecond>,
    interval: Option<IntervalDayTime>,
    negated: bool,
) -> Result<Option<TimestampNanosecond>> {
    if timestamp.is_none() || interval.is_none() {
        return Ok(None);
    }
    let timestamp = timestamp.unwrap();
    let interval = interval.unwrap();

    let timestamp = timestamp_ns_to_datetime(timestamp);
    let (days, millis) = IntervalDayTimeType::to_parts(interval);
    let (days, millis) = match negated {
        true => {
            let days = days.checked_neg().ok_or_else(|| {
                DataFusionError::Execution("interval out of range".to_string())
            })?;
            let millis = millis.checked_neg().ok_or_else(|| {
                DataFusionError::Execution("interval out of range".to_string())
            })?;
            (days, millis)
        }
        false => (days, millis),
    };

    let result =
        timestamp + Duration::days(days as i64) + Duration::milliseconds(millis as i64);
    Ok(Some(result.timestamp_nanos()))
}

fn scalar_timestamp_add_interval_month_day_nano(
    timestamp: Option<TimestampNanosecond>,
    interval: Option<IntervalMonthDayNano>,
    negated: bool,
) -> Result<Option<TimestampNanosecond>> {
    if timestamp.is_none() || interval.is_none() {
        return Ok(None);
    }
    let timestamp = timestamp.unwrap();
    let interval = interval.unwrap();

    let timestamp = timestamp_ns_to_datetime(timestamp);

    let (months, days, nanos) = IntervalMonthDayNanoType::to_parts(interval);

    let (months, days, nanos) = match negated {
        true => {
            let months = months.checked_neg().ok_or_else(|| {
                DataFusionError::Execution("interval out of range".to_string())
            })?;
            let days = days.checked_neg().ok_or_else(|| {
                DataFusionError::Execution("interval out of range".to_string())
            })?;
            let nanos = nanos.checked_neg().ok_or_else(|| {
                DataFusionError::Execution("interval out of range".to_string())
            })?;
            (months, days, nanos)
        }
        false => (months, days, nanos),
    };

    let result = if months >= 0 {
        timestamp.checked_add_months(Months::new(months as u32))
    } else {
        timestamp.checked_sub_months(Months::new(months.abs() as u32))
    };

    let result = if days >= 0 {
        result.and_then(|t| t.checked_add_days(Days::new(days as u64)))
    } else {
        result.and_then(|t| t.checked_sub_days(Days::new(days.abs() as u64)))
    };

    let result = result.and_then(|t| t.checked_add_signed(Duration::nanoseconds(nanos)));

    let result = result.ok_or_else(|| {
        DataFusionError::Execution(format!(
            "Failed to add interval: {} month {} day {} nano",
            months, days, nanos
        ))
    })?;
    Ok(Some(result.timestamp_nanos()))
}

fn scalar_timestamp_subtract_timestamp(
    timestamp_left: Option<TimestampNanosecond>,
    timestamp_right: Option<TimestampNanosecond>,
) -> Result<Option<IntervalMonthDayNano>> {
    if timestamp_left.is_none() || timestamp_right.is_none() {
        return Ok(None);
    }

    let datetime_left: NaiveDateTime = timestamp_ns_to_datetime(timestamp_left.unwrap());
    let datetime_right: NaiveDateTime =
        timestamp_ns_to_datetime(timestamp_right.unwrap());
    let duration = datetime_left.signed_duration_since(datetime_right);

    duration_to_interval_day_nano(duration)

    // TODO: How can day, above, in scalar_timestamp_add_interval_month_day_nano, be negative?
}

fn scalar_timestamp_subtract_date(
    timestamp_left: Option<TimestampNanosecond>,
    timestamp_right: Option<Date32>,
) -> Result<Option<IntervalMonthDayNano>> {
    if timestamp_left.is_none() || timestamp_right.is_none() {
        return Ok(None);
    }

    let datetime_left: NaiveDateTime = timestamp_ns_to_datetime(timestamp_left.unwrap());
    let datetime_right: NaiveDateTime = date32_to_datetime(timestamp_right.unwrap());
    let duration = datetime_left.signed_duration_since(datetime_right);

    duration_to_interval_day_nano(duration)
}

fn duration_to_interval_day_nano(
    duration: Duration,
) -> Result<Option<IntervalMonthDayNano>> {
    // TODO: What is Postgres behavior?  E.g. if these timestamp values are i64::MIN and i32/i64::MAX,
    // we needlessly have a range error.
    let nanos: i64 = duration.num_nanoseconds().ok_or_else(|| {
        DataFusionError::Execution("Interval value is out of range".to_string())
    })?;

    let days = nanos / 86_400_000_000_000;
    let nanos_rem = nanos % 86_400_000_000_000;
    Ok(Some(
        (((days as i128) & 0xFFFF_FFFF) << 64)
            | ((nanos_rem as i128) & 0xFFFF_FFFF_FFFF_FFFF),
    ))
}

fn change_ym(t: NaiveDateTime, y: i32, m: u32) -> Result<NaiveDateTime> {
    // TODO: legacy code, check validity
    debug_assert!((1..=12).contains(&m));
    let mut d = t.day();
    d = d.min(last_day_of_month(y, m)?);
    t.with_day(1)
        .ok_or_else(|| DataFusionError::Execution("Invalid year month".to_string()))?
        .with_year(y)
        .ok_or_else(|| DataFusionError::Execution("Invalid year month".to_string()))?
        .with_month(m)
        .ok_or_else(|| DataFusionError::Execution("Invalid year month".to_string()))?
        .with_day(d)
        .ok_or_else(|| DataFusionError::Execution("Invalid year month".to_string()))
}

fn last_day_of_month(y: i32, m: u32) -> Result<u32> {
    // TODO: legacy code, check validity
    debug_assert!((1..=12).contains(&m));
    if m == 12 {
        return Ok(31);
    }
    Ok(NaiveDate::from_ymd_opt(y, m + 1, 1)
        .ok_or_else(|| {
            DataFusionError::Execution(format!("Invalid year month: {}-{}", y, m))
        })?
        .pred_opt()
        .ok_or_else(|| {
            DataFusionError::Execution(format!("Invalid year month: {}-{}", y, m))
        })?
        .day())
}
