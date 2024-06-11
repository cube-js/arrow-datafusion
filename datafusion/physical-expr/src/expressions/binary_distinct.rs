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
        Array, ArrayRef, Int64Array, IntervalDayTimeArray, IntervalMonthDayNanoArray,
        IntervalYearMonthArray, TimestampNanosecondArray,
    },
    datatypes::{DataType, IntervalUnit},
    temporal_conversions::timestamp_ns_to_datetime,
};
use chrono::{Datelike, Days, Duration, Months, NaiveDate, NaiveDateTime};

use datafusion_common::{DataFusionError, Result};
use datafusion_expr::Operator;

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
        DataType::Interval(unit) => Err(DataFusionError::Execution(format!(
            "multiplication of interval type {:?} is not supported",
            unit
        ))),
        t => Err(DataFusionError::Execution(format!(
            "multiplication expected Interval, got {}",
            t
        ))),
    }
}

fn scalar_interval_year_month_mul_int(
    interval: Option<i32>,
    multiplier: Option<i64>,
) -> Result<Option<i32>> {
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
    interval: Option<i64>,
    multiplier: Option<i64>,
) -> Result<Option<i64>> {
    if interval.is_none() || multiplier.is_none() {
        return Ok(None);
    }
    let interval = interval.unwrap();
    let multiplier = multiplier.unwrap();

    let interval = interval as u64;
    let days: i32 = ((interval & 0xFFFFFFFF00000000) >> 32) as i32;
    let milliseconds: i32 = (interval & 0xFFFFFFFF) as i32;
    let multiplier = i32::try_from(multiplier).map_err(|err| {
        DataFusionError::Execution(format!(
            "unable to convert interval multiplier to Int32: {}",
            err
        ))
    })?;
    let days_product = days
        .checked_mul(multiplier)
        .ok_or_else(|| DataFusionError::Execution("interval out of range".to_string()))?;
    let milliseconds_product = milliseconds
        .checked_mul(multiplier)
        .ok_or_else(|| DataFusionError::Execution("interval out of range".to_string()))?;
    let interval_product =
        (((days_product as u64) << 32) | (milliseconds_product as u64)) as i64;
    Ok(Some(interval_product))
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

fn scalar_timestamp_add_interval_year_month(
    timestamp: Option<i64>,
    interval: Option<i32>,
    negated: bool,
) -> Result<Option<i64>> {
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
    timestamp: Option<i64>,
    interval: Option<i64>,
    negated: bool,
) -> Result<Option<i64>> {
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
    let days: i64 = interval.signum() * (interval.abs() >> 32);
    let millis: i64 = interval.signum() * ((interval.abs() << 32) >> 32);
    let result = timestamp + Duration::days(days) + Duration::milliseconds(millis);
    Ok(Some(result.timestamp_nanos()))
}

fn scalar_timestamp_add_interval_month_day_nano(
    timestamp: Option<i64>,
    interval: Option<i128>,
    negated: bool,
) -> Result<Option<i64>> {
    if timestamp.is_none() || interval.is_none() {
        return Ok(None);
    }
    let timestamp = timestamp.unwrap();
    let interval = interval.unwrap();

    let timestamp = timestamp_ns_to_datetime(timestamp);

    // TODO: legacy code, check validity
    let month = (interval >> (64 + 32)) & 0xFFFFFFFF;
    let day = (interval >> 64) & 0xFFFFFFFF;
    let nano = interval & 0xFFFFFFFFFFFFFFFF;

    let result = if month > 0 && !negated || month < 0 && negated {
        timestamp.checked_add_months(Months::new(month as u32))
    } else {
        timestamp.checked_sub_months(Months::new(month.abs() as u32))
    };

    let result = if day > 0 && !negated || day < 0 && negated {
        result.and_then(|t| t.checked_add_days(Days::new(day as u64)))
    } else {
        result.and_then(|t| t.checked_sub_days(Days::new(day.abs() as u64)))
    };

    let result = result.and_then(|t| {
        t.checked_add_signed(Duration::nanoseconds(
            (nano as i64) * (if negated { -1 } else { 1 }),
        ))
    });

    let result = result.ok_or_else(|| {
        DataFusionError::Execution(format!(
            "Failed to add interval: {} month {} day {} nano",
            month, day, nano
        ))
    })?;
    Ok(Some(result.timestamp_nanos()))
}

fn scalar_timestamp_subtract_timestamp(
    timestamp_left: Option<i64>,
    timestamp_right: Option<i64>,
) -> Result<Option<i128>> {
    if timestamp_left.is_none() || timestamp_right.is_none() {
        return Ok(None);
    }

    let datetime_left: NaiveDateTime = timestamp_ns_to_datetime(timestamp_left.unwrap());
    let datetime_right: NaiveDateTime =
        timestamp_ns_to_datetime(timestamp_right.unwrap());
    let duration = datetime_left.signed_duration_since(datetime_right);
    // TODO: What is Postgres behavior?  E.g. if these timestamp values are i64::MAX and i64::MIN,
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

    // TODO: How can day, above, in scalar_timestamp_add_interval_month_day_nano, be negative?
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
