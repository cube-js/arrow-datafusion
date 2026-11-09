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

//! Join related functionality used both on logical and physical plans

use crate::error::{DataFusionError, Result};
use crate::logical_plan::JoinType;
use crate::physical_plan::expressions::Column;
use arrow::datatypes::{FieldRef, Schema};
use std::collections::HashSet;

/// The on clause of the join, as vector of (left, right) columns.
pub type JoinOn = Vec<(Column, Column)>;
/// Reference for JoinOn.
pub type JoinOnRef<'a> = &'a [(Column, Column)];

/// Checks whether the schemas "left" and "right" and columns "on" represent a valid join.
/// They are valid whenever their columns' intersection equals the set `on`
pub fn check_join_is_valid(left: &Schema, right: &Schema, on: JoinOnRef) -> Result<()> {
    let left: HashSet<Column> = left
        .fields()
        .iter()
        .enumerate()
        .map(|(idx, f)| Column::new(f.name(), idx))
        .collect();
    let right: HashSet<Column> = right
        .fields()
        .iter()
        .enumerate()
        .map(|(idx, f)| Column::new(f.name(), idx))
        .collect();

    check_join_set_is_valid(&left, &right, on)
}

/// Checks whether the sets left, right and on compose a valid join.
/// They are valid whenever their intersection equals the set `on`
fn check_join_set_is_valid(
    left: &HashSet<Column>,
    right: &HashSet<Column>,
    on: &[(Column, Column)],
) -> Result<()> {
    let left = &left
        .iter()
        .map(|column| Column::new(&column.name().to_ascii_lowercase(), column.index()))
        .collect();
    let right = &right
        .iter()
        .map(|column| Column::new(&column.name().to_ascii_lowercase(), column.index()))
        .collect();

    let on_left = &on
        .iter()
        .map(|on| Column::new(&on.0.name().to_ascii_lowercase(), on.0.index()))
        .collect::<HashSet<_>>();
    let left_missing = on_left.difference(left).collect::<HashSet<_>>();

    let on_right = &on
        .iter()
        .map(|on| Column::new(&on.1.name().to_ascii_lowercase(), on.1.index()))
        .collect::<HashSet<_>>();
    let right_missing = on_right.difference(right).collect::<HashSet<_>>();

    if !left_missing.is_empty() | !right_missing.is_empty() {
        return Err(DataFusionError::Plan(format!(
                "The left or right side of the join does not have all columns on \"on\": \nMissing on the left: {:?}\nMissing on the right: {:?}",
                left_missing,
                right_missing,
            )));
    };

    Ok(())
}

/// Used in ColumnIndex to distinguish which side the index is for
#[derive(Debug, Clone)]
pub enum JoinSide {
    /// Left side of the join
    Left,
    /// Right side of the join
    Right,
}

/// Information about the index and placement (left or right) of the columns
#[derive(Debug, Clone)]
pub struct ColumnIndex {
    /// Index of the column
    pub index: usize,
    /// Whether the column is at the left or right side
    pub side: JoinSide,
}

/// Returns the output field given the input field. Outer joins may
/// insert nulls even if the input was not null
/// (port of upstream DataFusion b73a39a1b1: arrow 60 `RecordBatch::try_new`
/// rejects nulls in a non-nullable field, arrow 13 did not check)
fn output_join_field(
    old_field: &FieldRef,
    join_type: &JoinType,
    is_left: bool,
) -> FieldRef {
    let force_nullable = match join_type {
        JoinType::Inner => false,
        JoinType::Left => !is_left, // right input is padded with nulls
        JoinType::Right => is_left, // left input is padded with nulls
        JoinType::Full => true,     // both inputs can be padded with nulls
        JoinType::Semi => false,    // doesn't introduce nulls
        JoinType::Anti => false,    // doesn't introduce nulls
    };

    if force_nullable && !old_field.is_nullable() {
        std::sync::Arc::new(old_field.as_ref().clone().with_nullable(true))
    } else {
        old_field.clone()
    }
}

/// Creates a schema for a join operation.
/// The fields from the left side are first
pub fn build_join_schema(
    left: &Schema,
    right: &Schema,
    join_type: &JoinType,
) -> (Schema, Vec<ColumnIndex>) {
    let (fields, column_indices): (Vec<FieldRef>, Vec<ColumnIndex>) = match join_type {
        JoinType::Inner | JoinType::Left | JoinType::Full | JoinType::Right => {
            let left_fields = left
                .fields()
                .iter()
                .map(|f| output_join_field(f, join_type, true))
                .enumerate()
                .map(|(index, f)| {
                    (
                        f,
                        ColumnIndex {
                            index,
                            side: JoinSide::Left,
                        },
                    )
                });
            let right_fields = right
                .fields()
                .iter()
                .map(|f| output_join_field(f, join_type, false))
                .enumerate()
                .map(|(index, f)| {
                    (
                        f,
                        ColumnIndex {
                            index,
                            side: JoinSide::Right,
                        },
                    )
                });

            // left then right
            left_fields.chain(right_fields).unzip()
        }
        JoinType::Semi | JoinType::Anti => left
            .fields()
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, f)| {
                (
                    f,
                    ColumnIndex {
                        index,
                        side: JoinSide::Left,
                    },
                )
            })
            .unzip(),
    };

    (Schema::new(fields), column_indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(left: &[Column], right: &[Column], on: &[(Column, Column)]) -> Result<()> {
        let left = left
            .iter()
            .map(|x| x.to_owned())
            .collect::<HashSet<Column>>();
        let right = right
            .iter()
            .map(|x| x.to_owned())
            .collect::<HashSet<Column>>();
        check_join_set_is_valid(&left, &right, on)
    }

    #[test]
    fn check_valid() -> Result<()> {
        let left = vec![Column::new("a", 0), Column::new("b1", 1)];
        let right = vec![Column::new("a", 0), Column::new("b2", 1)];
        let on = &[(Column::new("a", 0), Column::new("a", 0))];

        check(&left, &right, on)?;
        Ok(())
    }

    #[test]
    fn check_not_in_right() {
        let left = vec![Column::new("a", 0), Column::new("b", 1)];
        let right = vec![Column::new("b", 0)];
        let on = &[(Column::new("a", 0), Column::new("a", 0))];

        assert!(check(&left, &right, on).is_err());
    }

    #[test]
    fn check_not_in_left() {
        let left = vec![Column::new("b", 0)];
        let right = vec![Column::new("a", 0)];
        let on = &[(Column::new("a", 0), Column::new("a", 0))];

        assert!(check(&left, &right, on).is_err());
    }

    #[test]
    fn check_collision() {
        // column "a" would appear both in left and right
        let left = vec![Column::new("a", 0), Column::new("c", 1)];
        let right = vec![Column::new("a", 0), Column::new("b", 1)];
        let on = &[(Column::new("a", 0), Column::new("b", 1))];

        assert!(check(&left, &right, on).is_ok());
    }

    #[test]
    fn check_in_right() {
        let left = vec![Column::new("a", 0), Column::new("c", 1)];
        let right = vec![Column::new("b", 0)];
        let on = &[(Column::new("a", 0), Column::new("b", 0))];

        assert!(check(&left, &right, on).is_ok());
    }

    #[test]
    fn test_join_schema() -> Result<()> {
        use arrow::datatypes::{DataType, Field};

        let a = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
        let a_nulls = Schema::new(vec![Field::new("a", DataType::Int32, true)]);
        let b = Schema::new(vec![Field::new("b", DataType::Int32, false)]);
        let b_nulls = Schema::new(vec![Field::new("b", DataType::Int32, true)]);

        let cases = vec![
            (&a, &b, JoinType::Inner, &a, &b),
            (&a, &b_nulls, JoinType::Inner, &a, &b_nulls),
            (&a_nulls, &b, JoinType::Inner, &a_nulls, &b),
            (&a_nulls, &b_nulls, JoinType::Inner, &a_nulls, &b_nulls),
            // right input of a `LEFT` join can be null, regardless of input nullness
            (&a, &b, JoinType::Left, &a, &b_nulls),
            (&a, &b_nulls, JoinType::Left, &a, &b_nulls),
            (&a_nulls, &b, JoinType::Left, &a_nulls, &b_nulls),
            (&a_nulls, &b_nulls, JoinType::Left, &a_nulls, &b_nulls),
            // left input of a `RIGHT` join can be null, regardless of input nullness
            (&a, &b, JoinType::Right, &a_nulls, &b),
            (&a, &b_nulls, JoinType::Right, &a_nulls, &b_nulls),
            (&a_nulls, &b, JoinType::Right, &a_nulls, &b),
            (&a_nulls, &b_nulls, JoinType::Right, &a_nulls, &b_nulls),
            // Either input of a `FULL` join can be null
            (&a, &b, JoinType::Full, &a_nulls, &b_nulls),
            (&a, &b_nulls, JoinType::Full, &a_nulls, &b_nulls),
            (&a_nulls, &b, JoinType::Full, &a_nulls, &b_nulls),
            (&a_nulls, &b_nulls, JoinType::Full, &a_nulls, &b_nulls),
        ];

        for (left_in, right_in, join_type, left_out, right_out) in cases {
            let (schema, _) = build_join_schema(left_in, right_in, &join_type);

            let expected_fields = left_out
                .fields()
                .iter()
                .cloned()
                .chain(right_out.fields().iter().cloned())
                .collect::<Vec<_>>();

            let expected_schema = Schema::new(expected_fields);
            assert_eq!(
                schema,
                expected_schema,
                "Mismatch with left_in={}:{}, right_in={}:{}, join_type={:?}",
                left_in.fields()[0].name(),
                left_in.fields()[0].is_nullable(),
                right_in.fields()[0].name(),
                right_in.fields()[0].is_nullable(),
                join_type
            );
        }

        Ok(())
    }
}
