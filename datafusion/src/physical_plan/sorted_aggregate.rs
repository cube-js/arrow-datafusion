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

//! Inplace aggregation for pre-sorted inputs

use crate::error::{DataFusionError, Result};
use crate::physical_plan::group_scalar::GroupByScalar;
use crate::physical_plan::hash_aggregate::{
    create_accumulators, create_group_by_value, create_group_by_values,
    write_group_result_row, AccumulatorSet, AggregateMode,
};
use crate::physical_plan::AggregateExpr;
use crate::scalar::ScalarValue;
use arrow::array::{ArrayBuilder, ArrayRef, LargeStringArray, StringArray};
use arrow::datatypes::{Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use itertools::Itertools;
use smallvec::smallvec;
use smallvec::SmallVec;
use std::sync::Arc;

pub(crate) struct Agg {
    key: SmallVec<[GroupByScalar; 2]>,
    accumulators: AccumulatorSet,
}

pub(crate) struct SortedAggState {
    processed_keys: Vec<Box<dyn ArrayBuilder>>,
    processed_values: Vec<Box<dyn ArrayBuilder>>,
    current_agg: Option<Agg>,
}

// TODO: find a safe alternative.
// This assumes builders are well-behaved.
unsafe impl Send for SortedAggState {}

impl SortedAggState {
    pub fn new() -> SortedAggState {
        SortedAggState {
            processed_keys: Vec::new(),
            processed_values: Vec::new(),
            current_agg: None,
        }
    }

    pub fn finish(
        mut self,
        mode: AggregateMode,
        schema: SchemaRef,
    ) -> arrow::error::Result<RecordBatch> {
        if let Some(agg) = self.current_agg {
            write_group_result_row(
                mode,
                &agg.key,
                &agg.accumulators,
                &schema.fields()[0..agg.key.len()],
                &mut self.processed_keys,
                &mut self.processed_values,
            )
            .map_err(DataFusionError::into_arrow_external_error)?;
        }
        let columns = self
            .processed_keys
            .into_iter()
            .chain(self.processed_values.into_iter())
            .map(|mut c| c.finish())
            .collect_vec();
        if columns.is_empty() {
            Ok(RecordBatch::new_empty(schema))
        } else {
            RecordBatch::try_new(schema, columns)
        }
    }

    pub fn add_batch(
        &mut self,
        mode: AggregateMode,
        agg_exprs: &Vec<Arc<dyn AggregateExpr>>,
        key_columns: &[ArrayRef],
        aggr_input_values: &[Vec<ArrayRef>],
        out_schema: &Schema,
    ) -> Result<()> {
        assert_ne!(key_columns.len(), 0);
        assert_eq!(aggr_input_values.len(), agg_exprs.len());
        let mut values_buffer = Vec::with_capacity(aggr_input_values.len());
        let mut value_scalars_buffer = Vec::with_capacity(2);

        let num_rows = key_columns[0].len();
        if num_rows == 0 {
            return Ok(());
        }
        if self.current_agg.is_none() {
            let mut key = smallvec![GroupByScalar::Int64(0); key_columns.len()];
            create_group_by_values(key_columns, 0, &mut key)?;
            self.current_agg = Some(Agg {
                key,
                accumulators: create_accumulators(agg_exprs)?,
            });

            // If this does not hold, the while below loops forever. Ensure we panic instead.
            assert!(
                agg_key_equals(&self.current_agg.as_ref().unwrap().key, key_columns, 0)?,
                "grouping key not equal to its input"
            );
        }

        let mut row_i = 0;
        while row_i < num_rows {
            let current_agg = self.current_agg.as_mut().unwrap();

            let start = row_i;
            let mut end = start;
            while end < key_columns[0].len()
                && agg_key_equals(&current_agg.key, key_columns, end)?
            {
                end += 1
            }

            if start == end {
                // Start a new group, next iteration will do the actual aggregation.
                write_group_result_row(
                    mode,
                    &current_agg.key,
                    &current_agg.accumulators,
                    &out_schema.fields()[0..current_agg.key.len()],
                    &mut self.processed_keys,
                    &mut self.processed_values,
                )?;
                create_group_by_values(key_columns, start, &mut current_agg.key)?;
                for a in &mut current_agg.accumulators {
                    a.reset();
                }
            } else {
                if end - start < 8 {
                    // Update individual values, inputs are small.
                    for i in 0..aggr_input_values.len() {
                        for agg_row in start..end {
                            value_scalars_buffer.clear();
                            for a in &aggr_input_values[i] {
                                value_scalars_buffer
                                    .push(ScalarValue::try_from_array(a, agg_row)?)
                            }

                            match mode {
                                AggregateMode::Partial | AggregateMode::Full => {
                                    current_agg.accumulators[i]
                                        .update(&value_scalars_buffer)?
                                }
                                AggregateMode::Final
                                | AggregateMode::FinalPartitioned => current_agg
                                    .accumulators[i]
                                    .merge(&value_scalars_buffer)?,
                            }
                        }
                    }
                } else {
                    // Update in batches, the inputs are large.
                    for i in 0..aggr_input_values.len() {
                        values_buffer.clear();
                        for inp in &aggr_input_values[i] {
                            values_buffer.push(inp.slice(start, end - start));
                        }
                        match mode {
                            AggregateMode::Partial | AggregateMode::Full => current_agg
                                .accumulators[i]
                                .update_batch(&values_buffer)?,
                            AggregateMode::Final | AggregateMode::FinalPartitioned => {
                                current_agg.accumulators[i].merge_batch(&values_buffer)?
                            }
                        }
                    }
                }
            }

            row_i = end;
        }
        Ok(())
    }
}

pub fn agg_key_equals(
    key: &[GroupByScalar],
    key_columns: &[ArrayRef],
    row: usize,
) -> Result<bool> {
    assert_eq!(key.len(), key_columns.len());
    for i in 0..key.len() {
        match &key[i] {
            // Optimize string comparisons to avoid allocations.
            GroupByScalar::Utf8(l) => {
                let r;
                if let Some(a) = key_columns[i].as_any().downcast_ref::<StringArray>() {
                    r = a.value(row);
                } else if let Some(a) =
                    key_columns[i].as_any().downcast_ref::<LargeStringArray>()
                {
                    r = a.value(row);
                } else {
                    return Err(DataFusionError::Internal(
                        "Failed to downcast to StringArray".to_string(),
                    ));
                }
                if l != r {
                    return Ok(false);
                }
            }
            l => {
                if l != &create_group_by_value(&key_columns[i], row)? {
                    return Ok(false);
                }
            }
        }
    }
    return Ok(true);
}
