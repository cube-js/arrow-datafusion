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

use crate::scalar::ScalarValue;
use arrow::array::ArrayRef;
use arrow::compute::{total_cmp_32, total_cmp_64};
use std::cmp::Ordering;

/// Generic code to help implement generic operations on arrays.
/// See usages for examples.
#[macro_export]
macro_rules! cube_match_array {
    ($array: expr, $matcher: ident) => {{
        use $crate::arrow::array::*;
        use $crate::arrow::datatypes::*;
        let a = $array;
        match a.data_type() {
            DataType::Null => panic!("null type is not supported"),
            DataType::Boolean => ($matcher!(a, BooleanArray, BooleanBuilder, Boolean)),
            DataType::Int8 => ($matcher!(a, Int8Array, PrimitiveBuilder<Int8Type>, Int8)),
            DataType::Int16 => {
                ($matcher!(a, Int16Array, PrimitiveBuilder<Int16Type>, Int16))
            }
            DataType::Int32 => {
                ($matcher!(a, Int32Array, PrimitiveBuilder<Int32Type>, Int32))
            }
            DataType::Int64 => {
                ($matcher!(a, Int64Array, PrimitiveBuilder<Int64Type>, Int64))
            }
            DataType::Int96 => {
                ($matcher!(a, Int96Array, PrimitiveBuilder<Int96Type>, Int96))
            }
            DataType::UInt8 => {
                ($matcher!(a, UInt8Array, PrimitiveBuilder<UInt8Type>, UInt8))
            }
            DataType::UInt16 => {
                ($matcher!(a, UInt16Array, PrimitiveBuilder<UInt16Type>, UInt16))
            }
            DataType::UInt32 => {
                ($matcher!(a, UInt32Array, PrimitiveBuilder<UInt32Type>, UInt32))
            }
            DataType::UInt64 => {
                ($matcher!(a, UInt64Array, PrimitiveBuilder<UInt64Type>, UInt64))
            }
            DataType::Float16 => panic!("float 16 is not supported"),
            DataType::Float32 => {
                ($matcher!(a, Float32Array, PrimitiveBuilder<Float32Type>, Float32))
            }
            DataType::Float64 => {
                ($matcher!(a, Float64Array, PrimitiveBuilder<Float64Type>, Float64))
            }
            DataType::Timestamp(TimeUnit::Second, _) => {
                ($matcher!(
                    a,
                    TimestampSecondArray,
                    TimestampSecondBuilder,
                    TimestampSecond
                ))
            }
            DataType::Timestamp(TimeUnit::Millisecond, _) => {
                ($matcher!(
                    a,
                    TimestampMillisecondArray,
                    TimestampMillisecondBuilder,
                    TimestampMillisecond
                ))
            }
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                ($matcher!(
                    a,
                    TimestampMicrosecondArray,
                    TimestampMicrosecondBuilder,
                    TimestampMicrosecond
                ))
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                ($matcher!(
                    a,
                    TimestampNanosecondArray,
                    TimestampNanosecondBuilder,
                    TimestampNanosecond
                ))
            }
            DataType::Date32 => {
                ($matcher!(a, Date32Array, PrimitiveBuilder<Date32Type>, Date32))
            }
            DataType::Date64 => {
                ($matcher!(a, Date64Array, PrimitiveBuilder<Date64Type>, Date64))
            }
            DataType::Time32(_) => panic!("time32 not supported"),
            DataType::Time64(_) => panic!("time64 not supported"),
            DataType::Duration(_) => panic!("duration not supported"),
            DataType::FixedSizeBinary(_) => panic!("fixed size binary not supported"),
            DataType::Interval(IntervalUnit::YearMonth) => {
                ($matcher!(
                    a,
                    IntervalYearMonthArray,
                    PrimitiveBuilder<IntervalYearMonthType>,
                    IntervalYearMonth
                ))
            }
            DataType::Interval(IntervalUnit::DayTime) => {
                ($matcher!(
                    a,
                    IntervalDayTimeArray,
                    PrimitiveBuilder<IntervalDayTimeType>,
                    IntervalDayTime
                ))
            }
            DataType::Binary => ($matcher!(a, BinaryArray, BinaryBuilder, Binary)),
            DataType::LargeBinary => {
                ($matcher!(a, LargeBinaryArray, LargeBinaryBuilder, LargeBinary))
            }
            DataType::Utf8 => ($matcher!(a, StringArray, StringBuilder, Utf8)),
            DataType::LargeUtf8 => {
                ($matcher!(a, LargeStringArray, LargeStringBuilder, Utf8))
            }
            DataType::List(_)
            | DataType::FixedSizeList(_, _)
            | DataType::LargeList(_) => {
                panic!("list not supported")
            }
            DataType::Struct(_) | DataType::Union(_) => {
                panic!("struct and union not supported")
            }
            DataType::Dictionary(_, _) => panic!("dictionary not supported"),
            DataType::Decimal(_, _) => panic!("decimal not supported"),
            DataType::Int64Decimal(0) => {
                ($matcher!(a, Int64Decimal0Array, Int64Decimal0Builder, Int64Decimal, 0))
            }
            DataType::Int64Decimal(1) => {
                ($matcher!(a, Int64Decimal1Array, Int64Decimal1Builder, Int64Decimal, 1))
            }
            DataType::Int64Decimal(2) => {
                ($matcher!(a, Int64Decimal2Array, Int64Decimal2Builder, Int64Decimal, 2))
            }
            DataType::Int64Decimal(3) => {
                ($matcher!(a, Int64Decimal3Array, Int64Decimal3Builder, Int64Decimal, 3))
            }
            DataType::Int64Decimal(4) => {
                ($matcher!(a, Int64Decimal4Array, Int64Decimal4Builder, Int64Decimal, 4))
            }
            DataType::Int64Decimal(5) => {
                ($matcher!(a, Int64Decimal5Array, Int64Decimal5Builder, Int64Decimal, 5))
            }
            DataType::Int64Decimal(10) => {
                ($matcher!(
                    a,
                    Int64Decimal10Array,
                    Int64Decimal10Builder,
                    Int64Decimal,
                    10
                ))
            }
            DataType::Int64Decimal(_) => panic!("unsupported scale for decimal"),
            DataType::Int96Decimal(0) => {
                ($matcher!(a, Int96Decimal0Array, Int96Decimal0Builder, Int96Decimal, 0))
            }
            DataType::Int96Decimal(1) => {
                ($matcher!(a, Int96Decimal1Array, Int96Decimal1Builder, Int96Decimal, 1))
            }
            DataType::Int96Decimal(2) => {
                ($matcher!(a, Int96Decimal2Array, Int96Decimal2Builder, Int96Decimal, 2))
            }
            DataType::Int96Decimal(3) => {
                ($matcher!(a, Int96Decimal3Array, Int96Decimal3Builder, Int96Decimal, 3))
            }
            DataType::Int96Decimal(4) => {
                ($matcher!(a, Int96Decimal4Array, Int96Decimal4Builder, Int96Decimal, 4))
            }
            DataType::Int96Decimal(5) => {
                ($matcher!(a, Int96Decimal5Array, Int96Decimal5Builder, Int96Decimal, 5))
            }
            DataType::Int96Decimal(10) => {
                ($matcher!(
                    a,
                    Int96Decimal10Array,
                    Int96Decimal10Builder,
                    Int96Decimal,
                    10
                ))
            }
            DataType::Int96Decimal(_) => panic!("unsupported scale for decimal"),
        }
    }};
}

/// Generic code to help implement generic operations on scalars.
/// Callers must [ScalarValue] to use this.
/// See usages for examples.
#[macro_export]
macro_rules! cube_match_scalar {
    ($scalar: expr, $matcher: ident $(, $arg: tt)*) => {{
        use $crate::arrow::array::*;
        match $scalar {
            ScalarValue::Boolean(v) => ($matcher!($($arg ,)* v, BooleanBuilder)),
            ScalarValue::Float32(v) => ($matcher!($($arg ,)* v, Float32Builder)),
            ScalarValue::Float64(v) => ($matcher!($($arg ,)* v, Float64Builder)),
            ScalarValue::Int8(v) => ($matcher!($($arg ,)* v, Int8Builder)),
            ScalarValue::Int16(v) => ($matcher!($($arg ,)* v, Int16Builder)),
            ScalarValue::Int32(v) => ($matcher!($($arg ,)* v, Int32Builder)),
            ScalarValue::Int64(v) => ($matcher!($($arg ,)* v, Int64Builder)),
            ScalarValue::Int96(v) => ($matcher!($($arg ,)* v, Int96Builder)),
            ScalarValue::Int64Decimal(v, 0) => ($matcher!($($arg ,)* v, Int64Decimal0Builder)),
            ScalarValue::Int64Decimal(v, 1) => ($matcher!($($arg ,)* v, Int64Decimal1Builder)),
            ScalarValue::Int64Decimal(v, 2) => ($matcher!($($arg ,)* v, Int64Decimal2Builder)),
            ScalarValue::Int64Decimal(v, 3) => ($matcher!($($arg ,)* v, Int64Decimal3Builder)),
            ScalarValue::Int64Decimal(v, 4) => ($matcher!($($arg ,)* v, Int64Decimal4Builder)),
            ScalarValue::Int64Decimal(v, 5) => ($matcher!($($arg ,)* v, Int64Decimal5Builder)),
            ScalarValue::Int64Decimal(v, 10) => ($matcher!($($arg ,)* v, Int64Decimal10Builder)),
            ScalarValue::Int64Decimal(v, scale) => {
                panic!("unhandled scale for decimal: {}", scale)
            }
            ScalarValue::Int96Decimal(v, 0) => ($matcher!($($arg ,)* v, Int96Decimal0Builder)),
            ScalarValue::Int96Decimal(v, 1) => ($matcher!($($arg ,)* v, Int96Decimal1Builder)),
            ScalarValue::Int96Decimal(v, 2) => ($matcher!($($arg ,)* v, Int96Decimal2Builder)),
            ScalarValue::Int96Decimal(v, 3) => ($matcher!($($arg ,)* v, Int96Decimal3Builder)),
            ScalarValue::Int96Decimal(v, 4) => ($matcher!($($arg ,)* v, Int96Decimal4Builder)),
            ScalarValue::Int96Decimal(v, 5) => ($matcher!($($arg ,)* v, Int96Decimal5Builder)),
            ScalarValue::Int96Decimal(v, 10) => ($matcher!($($arg ,)* v, Int96Decimal10Builder)),
            ScalarValue::Int96Decimal(v, scale) => {
                panic!("unhandled scale for decimal: {}", scale)
            }
            ScalarValue::UInt8(v) => ($matcher!($($arg ,)* v, UInt8Builder)),
            ScalarValue::UInt16(v) => ($matcher!($($arg ,)* v, UInt16Builder)),
            ScalarValue::UInt32(v) => ($matcher!($($arg ,)* v, UInt32Builder)),
            ScalarValue::UInt64(v) => ($matcher!($($arg ,)* v, UInt64Builder)),
            ScalarValue::Utf8(v) => ($matcher!($($arg ,)* v, StringBuilder)),
            ScalarValue::LargeUtf8(v) => ($matcher!($($arg ,)* v, LargeStringBuilder)),
            ScalarValue::Date32(v) => ($matcher!($($arg ,)* v, Date32Builder)),
            ScalarValue::Date64(v) => ($matcher!($($arg ,)* v, Date64Builder)),
            ScalarValue::TimestampMicrosecond(v) => {
                ($matcher!($($arg ,)* v, TimestampMicrosecondBuilder))
            }
            ScalarValue::TimestampNanosecond(v) => {
                ($matcher!($($arg ,)* v, TimestampNanosecondBuilder))
            }
            ScalarValue::TimestampMillisecond(v) => {
                ($matcher!($($arg ,)* v, TimestampMillisecondBuilder))
            }
            ScalarValue::TimestampSecond(v) => ($matcher!($($arg ,)* v, TimestampSecondBuilder)),
            ScalarValue::IntervalYearMonth(v) => ($matcher!($($arg ,)* v, IntervalYearMonthBuilder)),
            ScalarValue::IntervalDayTime(v) => ($matcher!($($arg ,)* v, IntervalDayTimeBuilder)),
            ScalarValue::List(v, box dt) => ($matcher!($($arg ,)* v, dt, ListBuilder)),
            ScalarValue::Binary(v) => ($matcher!($($arg ,)* v, BinaryBuilder)),
            ScalarValue::LargeBinary(v) => ($matcher!($($arg ,)* v, LargeBinaryBuilder)),
        }
    }};
}

/// Panics if scalars are of different types.
pub fn cmp_same_types(
    l: &ScalarValue,
    r: &ScalarValue,
    nulls_first: bool,
    asc: bool,
) -> Ordering {
    match (l.is_null(), r.is_null()) {
        (true, true) => return Ordering::Equal,
        (true, false) => {
            return if nulls_first {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }
        (false, true) => {
            return if nulls_first {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        }
        (false, false) => {} // fallthrough.
    }

    let o = match (l, r) {
        (ScalarValue::Boolean(Some(l)), ScalarValue::Boolean(Some(r))) => l.cmp(r),
        (ScalarValue::Float32(Some(l)), ScalarValue::Float32(Some(r))) => {
            total_cmp_32(*l, *r)
        }
        (ScalarValue::Float64(Some(l)), ScalarValue::Float64(Some(r))) => {
            total_cmp_64(*l, *r)
        }
        (ScalarValue::Int8(Some(l)), ScalarValue::Int8(Some(r))) => l.cmp(r),
        (ScalarValue::Int16(Some(l)), ScalarValue::Int16(Some(r))) => l.cmp(r),
        (ScalarValue::Int32(Some(l)), ScalarValue::Int32(Some(r))) => l.cmp(r),
        (ScalarValue::Int64(Some(l)), ScalarValue::Int64(Some(r))) => l.cmp(r),
        (ScalarValue::Int96(Some(l)), ScalarValue::Int96(Some(r))) => l.cmp(r),
        (
            ScalarValue::Int64Decimal(Some(l), lscale),
            ScalarValue::Int64Decimal(Some(r), rscale),
        ) => {
            assert_eq!(lscale, rscale);
            l.cmp(r)
        }
        (
            ScalarValue::Int96Decimal(Some(l), lscale),
            ScalarValue::Int96Decimal(Some(r), rscale),
        ) => {
            assert_eq!(lscale, rscale);
            l.cmp(r)
        }
        (ScalarValue::UInt8(Some(l)), ScalarValue::UInt8(Some(r))) => l.cmp(r),
        (ScalarValue::UInt16(Some(l)), ScalarValue::UInt16(Some(r))) => l.cmp(r),
        (ScalarValue::UInt32(Some(l)), ScalarValue::UInt32(Some(r))) => l.cmp(r),
        (ScalarValue::UInt64(Some(l)), ScalarValue::UInt64(Some(r))) => l.cmp(r),
        (ScalarValue::Utf8(Some(l)), ScalarValue::Utf8(Some(r))) => l.cmp(r),
        (ScalarValue::LargeUtf8(Some(l)), ScalarValue::LargeUtf8(Some(r))) => l.cmp(r),
        (ScalarValue::Binary(Some(l)), ScalarValue::Binary(Some(r))) => l.cmp(r),
        (ScalarValue::LargeBinary(Some(l)), ScalarValue::LargeBinary(Some(r))) => {
            l.cmp(r)
        }
        (ScalarValue::Date32(Some(l)), ScalarValue::Date32(Some(r))) => l.cmp(r),
        (ScalarValue::Date64(Some(l)), ScalarValue::Date64(Some(r))) => l.cmp(r),
        (
            ScalarValue::TimestampSecond(Some(l)),
            ScalarValue::TimestampSecond(Some(r)),
        ) => l.cmp(r),
        (
            ScalarValue::TimestampMillisecond(Some(l)),
            ScalarValue::TimestampMillisecond(Some(r)),
        ) => l.cmp(r),
        (
            ScalarValue::TimestampMicrosecond(Some(l)),
            ScalarValue::TimestampMicrosecond(Some(r)),
        ) => l.cmp(r),
        (
            ScalarValue::TimestampNanosecond(Some(l)),
            ScalarValue::TimestampNanosecond(Some(r)),
        ) => l.cmp(r),
        (
            ScalarValue::IntervalYearMonth(Some(l)),
            ScalarValue::IntervalYearMonth(Some(r)),
        ) => l.cmp(r),
        (
            ScalarValue::IntervalDayTime(Some(l)),
            ScalarValue::IntervalDayTime(Some(r)),
        ) => l.cmp(r),
        (ScalarValue::List(_, _), ScalarValue::List(_, _)) => {
            panic!("list as accumulator result is not supported")
        }
        (l, r) => panic!(
            "unhandled types in comparison: {} and {}",
            l.get_datatype(),
            r.get_datatype()
        ),
    };
    if asc {
        o
    } else {
        o.reverse()
    }
}

/// Panics if arrays are of different types. Comparison is ascending, null first.
pub fn cmp_array_row_same_types(
    l: &ArrayRef,
    l_row: usize,
    r: &ArrayRef,
    r_row: usize,
) -> Ordering {
    let l_null = l.is_null(l_row);
    let r_null = r.is_null(r_row);
    if l_null && r_null {
        return Ordering::Equal;
    }
    if l_null && !r_null {
        return Ordering::Less;
    }
    if !l_null && r_null {
        return Ordering::Greater;
    }

    macro_rules! cmp_row {
        ($l: expr, Float32Array, $($rest: tt)*) => {{
            let l = $l.as_any().downcast_ref::<Float32Array>().unwrap();
            let r = r.as_any().downcast_ref::<Float32Array>().unwrap();
            return arrow::compute::total_cmp_32(l.value(l_row), r.value(r_row));
        }};
        ($l: expr, Float64Array, $($rest: tt)*) => {{
            let l = $l.as_any().downcast_ref::<Float64Array>().unwrap();
            let r = r.as_any().downcast_ref::<Float64Array>().unwrap();
            return arrow::compute::total_cmp_64(l.value(l_row), r.value(r_row));
        }};
        ($l: expr, $arr: ty, $($rest: tt)*) => {{
            let l = $l.as_any().downcast_ref::<$arr>().unwrap();
            let r = r.as_any().downcast_ref::<$arr>().unwrap();
            return l.value(l_row).cmp(&r.value(r_row));
        }};
    }

    cube_match_array!(l, cmp_row);
}

pub fn lexcmp_array_rows<'a>(
    cols: impl Iterator<Item = &'a ArrayRef>,
    l_row: usize,
    r_row: usize,
) -> Ordering {
    for c in cols {
        let o = cmp_array_row_same_types(c, l_row, c, r_row);
        if o != Ordering::Equal {
            return o;
        }
    }
    Ordering::Equal
}
