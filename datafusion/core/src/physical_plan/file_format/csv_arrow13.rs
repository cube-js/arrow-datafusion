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

//! CSV schema inference and value parsing with the semantics of the arrow 13 CSV
//! reader, on top of the arrow 60 CSV tokenizer.
//!
//! The arrow 60 CSV reader differs from arrow 13 in ways that are visible to queries:
//! empty `Utf8` values are read as NULL (arrow 13: empty strings), naive timestamps are
//! read as UTC (arrow 13: process-local time), decimals are rounded (arrow 13:
//! truncated), more boolean / date / float spellings are accepted, and schema inference
//! returns other types (`Timestamp` instead of `Date64`, `Null` for empty columns, ...)
//! and always nullable fields. To keep the old behavior, the arrow 60 reader is only
//! used to split the input into string records (every column is read as `Utf8`) and
//! the typed parsing / inference below is a port of the arrow 13 code.

use std::collections::HashSet;
use std::io::{Cursor, Read};
use std::ops::Neg;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Decimal128Array, DictionaryArray, PrimitiveArray,
    StringArray,
};
use arrow::csv::reader::Format;
use arrow::csv::ReaderBuilder;
use arrow::datatypes::*;
use arrow::error::{ArrowError, Result as ArrowResult};
use arrow::record_batch::{RecordBatch, RecordBatchOptions};
use chrono::Datelike;
use datafusion_physical_expr::datetime_expressions::string_to_timestamp_nanos_local;
use lazy_static::lazy_static;
use regex::{Regex, RegexBuilder};

lazy_static! {
    static ref PARSE_DECIMAL_RE: Regex =
        Regex::new(r"^-?(\d+\.?\d*|\d*\.?\d+)$").unwrap();
    static ref DECIMAL_RE: Regex =
        Regex::new(r"^-?((\d*\.\d+|\d+\.\d*)([eE]-?\d+)?|\d+([eE]-?\d+))$").unwrap();
    static ref INTEGER_RE: Regex = Regex::new(r"^-?(\d+)$").unwrap();
    static ref BOOLEAN_RE: Regex = RegexBuilder::new(r"^(true)$|^(false)$")
        .case_insensitive(true)
        .build()
        .unwrap();
    static ref DATE_RE: Regex = Regex::new(r"^\d{4}-\d\d-\d\d$").unwrap();
    static ref DATETIME_RE: Regex =
        Regex::new(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d$").unwrap();
}

/// Value of row `i` of a column read as `Utf8` by the arrow 60 reader: it reads empty
/// values as NULL, arrow 13 kept them as empty strings.
#[inline]
fn value(strings: &StringArray, i: usize) -> &str {
    if strings.is_null(i) {
        ""
    } else {
        strings.value(i)
    }
}

/// An arrow 60 CSV reader builder that reads every column as `Utf8`
fn string_reader_builder(
    column_names: impl IntoIterator<Item = String>,
    has_header: bool,
    delimiter: u8,
    batch_size: usize,
    max_records: Option<usize>,
    projection: Option<Vec<usize>>,
) -> ReaderBuilder {
    let string_schema = Schema::new(
        column_names
            .into_iter()
            .map(|name| Field::new(name, DataType::Utf8, true))
            .collect::<Vec<_>>(),
    );
    let mut builder = ReaderBuilder::new(Arc::new(string_schema))
        .with_header(has_header)
        .with_delimiter(delimiter)
        .with_batch_size(batch_size.max(1));
    if let Some(max_records) = max_records {
        // the header line is added to the bounds by the arrow 60 reader
        if max_records < usize::MAX {
            builder = builder.with_bounds(0, max_records);
        }
    }
    if let Some(projection) = projection {
        builder = builder.with_projection(projection);
    }
    builder
}

// ---------------------------------------------------------------------------------
// Schema inference (arrow 13 `csv::reader::infer_reader_schema`)
// ---------------------------------------------------------------------------------

/// Infer the data type of a record (arrow 13 `infer_field_schema`)
fn infer_field_schema(string: &str) -> DataType {
    // when quoting is enabled in the reader, these quotes aren't escaped, we default to
    // Utf8 for them
    if string.starts_with('"') {
        return DataType::Utf8;
    }
    // match regex in a particular order
    if BOOLEAN_RE.is_match(string) {
        DataType::Boolean
    } else if DECIMAL_RE.is_match(string) {
        DataType::Float64
    } else if INTEGER_RE.is_match(string) {
        DataType::Int64
    } else if DATETIME_RE.is_match(string) {
        DataType::Date64
    } else if DATE_RE.is_match(string) {
        DataType::Date32
    } else {
        DataType::Utf8
    }
}

/// A `Read` that keeps a copy of every byte it reads, so that the input can be read
/// again from the start without seeking
struct RecordingReader<R> {
    inner: R,
    recorded: Vec<u8>,
}

impl<R: Read> Read for RecordingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.recorded.extend_from_slice(&buf[..n]);
        Ok(n)
    }
}

/// Infer the schema of CSV records provided by `reader` like arrow 13's
/// `arrow::csv::reader::infer_reader_schema`: `Boolean`, `Float64`, `Int64`, `Date64`
/// (`YYYY-MM-DDTHH:MM:SS`), `Date32` (`YYYY-MM-DD`) or `Utf8` (also for conflicting
/// types, except `Int64` + `Float64` -> `Float64`, and for columns without values); a
/// field is nullable only if an empty value was seen.
///
/// `max_read_records` controls the maximum number of records to read; all records are
/// read if it is `None`. Returns the inferred schema and the number of records read.
pub(crate) fn infer_reader_schema<R: Read>(
    reader: R,
    delimiter: u8,
    max_read_records: Option<usize>,
    has_header: bool,
) -> ArrowResult<(Schema, usize)> {
    let mut recording = RecordingReader {
        inner: reader,
        recorded: vec![],
    };
    // get or create header names (`column_{i}` when there is no header), without
    // reading any record
    let (header_schema, _) = Format::default()
        .with_delimiter(delimiter)
        .with_header(has_header)
        .infer_schema(&mut recording, Some(0))?;
    let headers: Vec<String> = header_schema
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    let header_length = headers.len();
    if header_length == 0 {
        // no record at all
        return Ok((Schema::new(Vec::<Field>::new()), 0));
    }

    let RecordingReader { inner, recorded } = recording;
    let max_records = max_read_records.unwrap_or(usize::MAX);
    let records = string_reader_builder(
        headers.iter().cloned(),
        has_header,
        delimiter,
        max_records.min(1024),
        Some(max_records),
        None,
    )
    .build(Cursor::new(recorded).chain(inner))?;

    // keep track of inferred field types
    let mut column_types: Vec<HashSet<DataType>> = vec![HashSet::new(); header_length];
    // keep track of columns with nulls
    let mut nulls: Vec<bool> = vec![false; header_length];
    let mut records_count = 0;

    for batch in records {
        let batch = batch?;
        records_count += batch.num_rows();
        for i in 0..header_length {
            let strings = batch
                .column(i)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("CSV columns are read as strings");
            for row in 0..strings.len() {
                let string = value(strings, row);
                if string.is_empty() {
                    nulls[i] = true;
                } else {
                    column_types[i].insert(infer_field_schema(string));
                }
            }
        }
    }

    // build schema from inference results
    let fields = (0..header_length)
        .map(|i| {
            let possibilities = &column_types[i];
            let has_nulls = nulls[i];
            let field_name = &headers[i];

            // determine data type based on possible types
            // if there are incompatible types, use DataType::Utf8
            let data_type = match possibilities.len() {
                1 => possibilities.iter().next().unwrap().clone(),
                2 if possibilities.contains(&DataType::Int64)
                    && possibilities.contains(&DataType::Float64) =>
                {
                    // we have an integer and double, fall down to double
                    DataType::Float64
                }
                // default to Utf8 for conflicting datatypes (e.g bool and int)
                _ => DataType::Utf8,
            };
            Field::new(field_name, data_type, has_nulls)
        })
        .collect::<Vec<_>>();

    Ok((Schema::new(fields), records_count))
}

// ---------------------------------------------------------------------------------
// Reading (arrow 13 `csv::Reader`)
// ---------------------------------------------------------------------------------

/// CSV reader producing the same record batches as arrow 13's `csv::Reader`
pub(crate) struct Arrow13CsvReader<R: Read> {
    /// arrow 60 reader returning the projected columns as `Utf8`
    records: arrow::csv::Reader<R>,
    /// Explicit schema for the CSV file
    schema: SchemaRef,
    /// File column indices to load
    projection: Vec<usize>,
    /// Schema of the returned batches
    projected_schema: SchemaRef,
    /// Current line number (for error messages)
    line_number: usize,
}

impl<R: Read> Arrow13CsvReader<R> {
    /// Create a new reader, like arrow 13 `csv::Reader::new(reader, schema, has_header,
    /// Some(delimiter), batch_size, limit.map(|l| (0, l + has_header)), projection,
    /// None)`: at most `limit` records are read (when set).
    pub(crate) fn try_new(
        reader: R,
        schema: SchemaRef,
        has_header: bool,
        delimiter: u8,
        batch_size: usize,
        limit: Option<usize>,
        projection: Option<Vec<usize>>,
    ) -> ArrowResult<Self> {
        let projection =
            projection.unwrap_or_else(|| (0..schema.fields().len()).collect());
        let records = string_reader_builder(
            schema.fields().iter().map(|f| f.name().clone()),
            has_header,
            delimiter,
            batch_size,
            limit,
            Some(projection.clone()),
        )
        // arrow 13 read the columns missing at the end of a record (e.g. a schema with
        // more columns than the file) as NULL
        .with_truncated_rows(true)
        .build(reader)?;
        let projected_schema = Arc::new(Schema::new_with_metadata(
            projection
                .iter()
                .map(|i| schema.field(*i).clone())
                .collect::<Vec<_>>(),
            schema.metadata().clone(),
        ));
        Ok(Self {
            records,
            schema,
            projection,
            projected_schema,
            line_number: usize::from(has_header),
        })
    }

    fn parse(&self, batch: &RecordBatch) -> ArrowResult<RecordBatch> {
        let arrays = self
            .projection
            .iter()
            .enumerate()
            .map(|(column, col_idx)| {
                let strings = batch
                    .column(column)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("CSV columns are read as strings");
                parse_column(
                    strings,
                    self.schema.field(*col_idx).data_type(),
                    *col_idx,
                    self.line_number,
                )
            })
            .collect::<ArrowResult<Vec<_>>>()?;
        RecordBatch::try_new_with_options(
            Arc::clone(&self.projected_schema),
            arrays,
            &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
        )
    }
}

impl<R: Read> Iterator for Arrow13CsvReader<R> {
    type Item = ArrowResult<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = match self.records.next()? {
            Ok(batch) => batch,
            Err(e) => return Some(Err(e)),
        };
        let result = self.parse(&batch);
        self.line_number += batch.num_rows();
        Some(result)
    }
}

fn parse_error(s: &str, col_idx: usize, line_number: usize) -> ArrowError {
    ArrowError::ParseError(format!(
        // TODO: we should surface the underlying error here.
        "Error while parsing value {} for column {} at line {}",
        s, col_idx, line_number
    ))
}

/// Parse a column read as strings into `data_type` (arrow 13 `csv::reader::parse`)
fn parse_column(
    strings: &StringArray,
    data_type: &DataType,
    col_idx: usize,
    line_number: usize,
) -> ArrowResult<ArrayRef> {
    match data_type {
        DataType::Boolean => build_boolean_array(strings, col_idx, line_number),
        DataType::Decimal128(precision, scale) => {
            build_decimal_array(strings, *precision, *scale)
        }
        DataType::Int8 => {
            build_primitive_array::<Int8Type>(strings, col_idx, line_number, |s| {
                s.parse().ok()
            })
        }
        DataType::Int16 => {
            build_primitive_array::<Int16Type>(strings, col_idx, line_number, |s| {
                s.parse().ok()
            })
        }
        DataType::Int32 => {
            build_primitive_array::<Int32Type>(strings, col_idx, line_number, |s| {
                s.parse().ok()
            })
        }
        DataType::Int64 => {
            build_primitive_array::<Int64Type>(strings, col_idx, line_number, |s| {
                s.parse().ok()
            })
        }
        DataType::UInt8 => {
            build_primitive_array::<UInt8Type>(strings, col_idx, line_number, |s| {
                s.parse().ok()
            })
        }
        DataType::UInt16 => {
            build_primitive_array::<UInt16Type>(strings, col_idx, line_number, |s| {
                s.parse().ok()
            })
        }
        DataType::UInt32 => {
            build_primitive_array::<UInt32Type>(strings, col_idx, line_number, |s| {
                s.parse().ok()
            })
        }
        DataType::UInt64 => {
            build_primitive_array::<UInt64Type>(strings, col_idx, line_number, |s| {
                s.parse().ok()
            })
        }
        DataType::Float32 => build_primitive_array::<Float32Type>(
            strings,
            col_idx,
            line_number,
            parse_float::<Float32Type>,
        ),
        DataType::Float64 => build_primitive_array::<Float64Type>(
            strings,
            col_idx,
            line_number,
            parse_float::<Float64Type>,
        ),
        DataType::Date32 => {
            build_primitive_array::<Date32Type>(strings, col_idx, line_number, |s| {
                /// Number of days between 0001-01-01 and 1970-01-01
                const EPOCH_DAYS_FROM_CE: i32 = 719_163;
                let date = s.parse::<chrono::NaiveDate>().ok()?;
                Some(date.num_days_from_ce() - EPOCH_DAYS_FROM_CE)
            })
        }
        DataType::Date64 => {
            build_primitive_array::<Date64Type>(strings, col_idx, line_number, |s| {
                let date_time = s.parse::<chrono::NaiveDateTime>().ok()?;
                Some(date_time.and_utc().timestamp_millis())
            })
        }
        DataType::Timestamp(TimeUnit::Microsecond, tz) => {
            let array = build_primitive::<TimestampMicrosecondType>(
                strings,
                col_idx,
                line_number,
                |s| {
                    // truncating division, like arrow 13
                    string_to_timestamp_nanos_local(s).ok().map(|x| x / 1000)
                },
            )?;
            Ok(Arc::new(array.with_timezone_opt(tz.clone())))
        }
        DataType::Timestamp(TimeUnit::Nanosecond, tz) => {
            let array = build_primitive::<TimestampNanosecondType>(
                strings,
                col_idx,
                line_number,
                |s| string_to_timestamp_nanos_local(s).ok(),
            )?;
            Ok(Arc::new(array.with_timezone_opt(tz.clone())))
        }
        DataType::Utf8 => {
            if strings.null_count() == 0 {
                Ok(Arc::new(strings.clone()))
            } else {
                Ok(Arc::new(StringArray::from_iter_values(
                    (0..strings.len()).map(|i| value(strings, i)),
                )))
            }
        }
        DataType::Dictionary(key_type, value_type)
            if value_type.as_ref() == &DataType::Utf8 =>
        {
            match key_type.as_ref() {
                DataType::Int8 => Ok(build_dictionary_array::<Int8Type>(strings)),
                DataType::Int16 => Ok(build_dictionary_array::<Int16Type>(strings)),
                DataType::Int32 => Ok(build_dictionary_array::<Int32Type>(strings)),
                DataType::Int64 => Ok(build_dictionary_array::<Int64Type>(strings)),
                DataType::UInt8 => Ok(build_dictionary_array::<UInt8Type>(strings)),
                DataType::UInt16 => Ok(build_dictionary_array::<UInt16Type>(strings)),
                DataType::UInt32 => Ok(build_dictionary_array::<UInt32Type>(strings)),
                DataType::UInt64 => Ok(build_dictionary_array::<UInt64Type>(strings)),
                _ => Err(ArrowError::ParseError(format!(
                    "Unsupported dictionary key type {:?}",
                    key_type
                ))),
            }
        }
        other => Err(ArrowError::ParseError(format!(
            "Unsupported data type {:?}",
            other
        ))),
    }
}

/// arrow 13 parsed floats with `lexical_core::parse` (no surrounding whitespace allowed);
/// arrow 60's parser also accepts surrounding whitespace.
fn parse_float<T>(s: &str) -> Option<T::Native>
where
    T: ArrowPrimitiveType + arrow::compute::kernels::cast_utils::Parser,
{
    if s.trim() != s {
        return None;
    }
    T::parse(s)
}

fn build_primitive<T: ArrowPrimitiveType>(
    strings: &StringArray,
    col_idx: usize,
    line_number: usize,
    parse: impl Fn(&str) -> Option<T::Native>,
) -> ArrowResult<PrimitiveArray<T>> {
    (0..strings.len())
        .map(|row_index| {
            let s = value(strings, row_index);
            if s.is_empty() {
                return Ok(None);
            }
            match parse(s) {
                Some(v) => Ok(Some(v)),
                None => Err(parse_error(s, col_idx, line_number + row_index)),
            }
        })
        .collect::<ArrowResult<PrimitiveArray<T>>>()
}

fn build_primitive_array<T: ArrowPrimitiveType>(
    strings: &StringArray,
    col_idx: usize,
    line_number: usize,
    parse: impl Fn(&str) -> Option<T::Native>,
) -> ArrowResult<ArrayRef> {
    Ok(Arc::new(build_primitive::<T>(
        strings,
        col_idx,
        line_number,
        parse,
    )?))
}

fn parse_bool(string: &str) -> Option<bool> {
    if string.eq_ignore_ascii_case("false") {
        Some(false)
    } else if string.eq_ignore_ascii_case("true") {
        Some(true)
    } else {
        None
    }
}

fn build_boolean_array(
    strings: &StringArray,
    col_idx: usize,
    line_number: usize,
) -> ArrowResult<ArrayRef> {
    (0..strings.len())
        .map(|row_index| {
            let s = value(strings, row_index);
            if s.is_empty() {
                return Ok(None);
            }
            match parse_bool(s) {
                Some(v) => Ok(Some(v)),
                None => Err(parse_error(s, col_idx, line_number + row_index)),
            }
        })
        .collect::<ArrowResult<BooleanArray>>()
        .map(|a| Arc::new(a) as ArrayRef)
}

fn build_dictionary_array<K: ArrowDictionaryKeyType>(strings: &StringArray) -> ArrayRef {
    Arc::new(
        (0..strings.len())
            .map(|i| Some(value(strings, i)))
            .collect::<DictionaryArray<K>>(),
    )
}

fn build_decimal_array(
    strings: &StringArray,
    precision: u8,
    scale: i8,
) -> ArrowResult<ArrayRef> {
    let scale = usize::try_from(scale).map_err(|_| {
        ArrowError::ParseError(format!(
            "Unsupported data type {:?}",
            DataType::Decimal128(precision, scale)
        ))
    })?;
    let array = (0..strings.len())
        .map(|i| {
            let s = value(strings, i);
            if s.is_empty() {
                Ok(None)
            } else {
                parse_decimal_with_parameter(s, precision as usize, scale).map(Some)
            }
        })
        .collect::<ArrowResult<Decimal128Array>>()?
        .with_precision_and_scale(precision, scale as i8)?;
    Ok(Arc::new(array))
}

/// arrow 13 `validate_decimal_precision`
fn validate_decimal_precision(value: i128, precision: usize) -> ArrowResult<i128> {
    let max = 10_i128
        .checked_pow(precision as u32)
        .map(|p| p - 1)
        .unwrap_or(i128::MAX);
    let min = -max;

    if value > max {
        Err(ArrowError::InvalidArgumentError(format!(
            "{} is too large to store in a Decimal of precision {}. Max is {}",
            value, precision, max
        )))
    } else if value < min {
        Err(ArrowError::InvalidArgumentError(format!(
            "{} is too small to store in a Decimal of precision {}. Min is {}",
            value, precision, min
        )))
    } else {
        Ok(value)
    }
}

// Parse the string format decimal value to i128 format and checking the precision and
// scale. Digits beyond the scale are truncated (arrow 60 rounds).
fn parse_decimal_with_parameter(
    s: &str,
    precision: usize,
    scale: usize,
) -> ArrowResult<i128> {
    if PARSE_DECIMAL_RE.is_match(s) {
        let mut offset = s.len();
        let len = s.len();
        let mut base: i128 = 1;

        // handle the value after the '.' and meet the scale
        let delimiter_position = s.find('.');
        match delimiter_position {
            None => {
                // there is no '.'
                base = 10_i128.pow(scale as u32);
            }
            Some(mid) => {
                // there is the '.'
                if len - mid >= scale + 1 {
                    // If the string value is "123.12345" and the scale is 2, we should
                    // just remain '.12' and drop the '345' value.
                    offset -= len - mid - 1 - scale;
                } else {
                    // If the string value is "123.12" and the scale is 4, we should
                    // append '00' to the tail.
                    base = 10_i128.pow((scale + 1 + mid - len) as u32);
                }
            }
        };

        // each byte is digit、'-' or '.'
        let bytes = s.as_bytes();
        let mut negative = false;
        let mut result: i128 = 0;

        bytes[0..offset].iter().rev().for_each(|&byte| match byte {
            b'-' => {
                negative = true;
            }
            b'0'..=b'9' => {
                result += i128::from(byte - b'0') * base;
                base *= 10;
            }
            // because of the PARSE_DECIMAL_RE, bytes just contains digit、'-' and '.'.
            _ => {}
        });

        if negative {
            result = result.neg();
        }
        validate_decimal_precision(result, precision)
            .map_err(|e| ArrowError::ParseError(format!("parse decimal overflow: {}", e)))
    } else {
        Err(ArrowError::ParseError(format!(
            "can't parse the string value {} to decimal",
            s
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Date32Array, Date64Array, Float64Array, Int64Array};

    fn infer(csv: &str, has_header: bool, max: Option<usize>) -> (Schema, usize) {
        infer_reader_schema(Cursor::new(csv.to_string()), b',', max, has_header).unwrap()
    }

    #[test]
    fn infer_schema_like_arrow13() {
        let (schema, n) = infer(
            "a,b,c,d,e,f,g,h,i\n\
             1,1.5,true,2020-01-01,2020-01-01T00:00:00,x,,1,+1\n\
             2,,FALSE,2020-01-02,2020-01-01T00:00:01,,,2.5,2\n",
            true,
            None,
        );
        assert_eq!(n, 2);
        let fields = schema
            .fields()
            .iter()
            .map(|f| (f.name().as_str(), f.data_type().clone(), f.is_nullable()))
            .collect::<Vec<_>>();
        assert_eq!(
            fields,
            vec![
                ("a", DataType::Int64, false),
                ("b", DataType::Float64, true),
                ("c", DataType::Boolean, false),
                ("d", DataType::Date32, false),
                ("e", DataType::Date64, false),
                ("f", DataType::Utf8, true),
                ("g", DataType::Utf8, true),
                ("h", DataType::Float64, false),
                ("i", DataType::Utf8, false),
            ]
        );
    }

    #[test]
    fn infer_schema_limits_and_headers() {
        let csv = "1,a\n2,b\nx,c\n";
        let (schema, n) = infer(csv, false, Some(2));
        assert_eq!(n, 2);
        assert_eq!(schema.field(0).name(), "column_1");
        assert_eq!(schema.field(0).data_type(), &DataType::Int64);
        assert_eq!(schema.field(1).name(), "column_2");

        let (schema, n) = infer(csv, false, None);
        assert_eq!(n, 3);
        assert_eq!(schema.field(0).data_type(), &DataType::Utf8);

        let (schema, n) = infer("", true, None);
        assert_eq!(n, 0);
        assert_eq!(schema.fields().len(), 0);

        let (schema, n) = infer("a,b\n", true, Some(10));
        assert_eq!(n, 0);
        assert_eq!(schema.fields().len(), 2);
    }

    #[test]
    fn read_like_arrow13() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("s", DataType::Utf8, false),
            Field::new("i", DataType::Int64, true),
            Field::new("f", DataType::Float64, true),
            Field::new("d", DataType::Date32, true),
            Field::new("dt", DataType::Date64, true),
            Field::new("dec", DataType::Decimal128(5, 2), true),
        ]));
        let csv = "s,i,f,d,dt,dec\n\
                   ,1,1.5,2020-01-02,2020-01-01T00:00:01,1.239\n\
                   x,,,,,-12\n";
        let mut reader = Arrow13CsvReader::try_new(
            Cursor::new(csv.to_string()),
            schema,
            true,
            b',',
            1024,
            None,
            None,
        )
        .unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert!(reader.next().is_none());
        assert_eq!(batch.num_rows(), 2);

        let s = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(s.null_count(), 0);
        assert_eq!(s.value(0), "");
        assert_eq!(s.value(1), "x");
        let i = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(i.iter().collect::<Vec<_>>(), vec![Some(1), None]);
        let f = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(f.iter().collect::<Vec<_>>(), vec![Some(1.5), None]);
        let d = batch
            .column(3)
            .as_any()
            .downcast_ref::<Date32Array>()
            .unwrap();
        assert_eq!(d.iter().collect::<Vec<_>>(), vec![Some(18263), None]);
        let dt = batch
            .column(4)
            .as_any()
            .downcast_ref::<Date64Array>()
            .unwrap();
        assert_eq!(
            dt.iter().collect::<Vec<_>>(),
            vec![Some(1577836801000), None]
        );
        let dec = batch
            .column(5)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .unwrap();
        // truncated, not rounded
        assert_eq!(dec.iter().collect::<Vec<_>>(), vec![Some(123), Some(-1200)]);
    }

    #[test]
    fn read_errors_and_limit() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Boolean, true),
        ]));
        let csv = "a,b\n1,true\n2,yes\n3,false\n";
        let mut reader = Arrow13CsvReader::try_new(
            Cursor::new(csv.to_string()),
            Arc::clone(&schema),
            true,
            b',',
            1024,
            None,
            Some(vec![1]),
        )
        .unwrap();
        let err = reader.next().unwrap().unwrap_err();
        assert_eq!(
            err.to_string(),
            "Parser error: Error while parsing value yes for column 1 at line 2"
        );

        let reader = Arrow13CsvReader::try_new(
            Cursor::new(csv.to_string()),
            schema,
            true,
            b',',
            1,
            Some(1),
            Some(vec![0]),
        )
        .unwrap();
        let batches = reader.collect::<ArrowResult<Vec<_>>>().unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 1);
        assert_eq!(batches[0].schema().fields().len(), 1);
        assert_eq!(batches[0].schema().field(0).name(), "a");
    }
}
