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

//! Defines physical expressions that can evaluated at runtime during query execution

use std::any::Any;
use std::mem::size_of;
use std::sync::Arc;

use crate::error::{DataFusionError, Result};
use crate::physical_plan::groups_accumulator::{EmitTo, GroupsAccumulator};
use crate::physical_plan::null_state::accumulate_indices;
use crate::physical_plan::{Accumulator, AggregateExpr, PhysicalExpr};
use crate::scalar::ScalarValue;
use arrow::array::{Array, ArrayData, BooleanArray, PrimitiveArray};
use arrow::buffer::Buffer;
use arrow::compute;
use arrow::datatypes::{DataType, UInt64Type};
use arrow::{
    array::{ArrayRef, UInt64Array},
    datatypes::Field,
};

use super::format_state_name;
use smallvec::smallvec;
use smallvec::SmallVec;

/// COUNT aggregate expression
/// Returns the amount of non-null values of the given expression.
#[derive(Debug)]
pub struct Count {
    name: String,
    data_type: DataType,
    nullable: bool,
    expr: Arc<dyn PhysicalExpr>,
}

impl Count {
    /// Create a new COUNT aggregate function.
    pub fn new(
        expr: Arc<dyn PhysicalExpr>,
        name: impl Into<String>,
        data_type: DataType,
    ) -> Self {
        Self {
            name: name.into(),
            expr,
            data_type,
            nullable: true,
        }
    }
}

impl AggregateExpr for Count {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn field(&self) -> Result<Field> {
        Ok(Field::new(
            &self.name,
            self.data_type.clone(),
            self.nullable,
        ))
    }

    fn state_fields(&self) -> Result<Vec<Field>> {
        Ok(vec![Field::new(
            &format_state_name(&self.name, "count"),
            self.data_type.clone(),
            true,
        )])
    }

    fn expressions(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        vec![self.expr.clone()]
    }

    fn create_accumulator(&self) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(CountAccumulator::new()))
    }

    fn uses_groups_accumulator(&self) -> bool {
        return true;
    }

    fn create_groups_accumulator(
        &self,
    ) -> arrow::error::Result<Option<Box<dyn GroupsAccumulator>>> {
        Ok(Some(Box::new(CountGroupsAccumulator::new())))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Debug)]
struct CountAccumulator {
    count: u64,
}

impl CountAccumulator {
    /// new count accumulator
    pub fn new() -> Self {
        Self { count: 0 }
    }
}

impl Accumulator for CountAccumulator {
    fn reset(&mut self) {
        self.count = 0;
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let array = &values[0];
        self.count += (array.len() - array.data().null_count()) as u64;
        Ok(())
    }

    fn update(&mut self, values: &[ScalarValue]) -> Result<()> {
        let value = &values[0];
        if !value.is_null() {
            self.count += 1;
        }
        Ok(())
    }

    fn merge(&mut self, states: &[ScalarValue]) -> Result<()> {
        let count = &states[0];
        if let ScalarValue::UInt64(Some(delta)) = count {
            self.count += *delta;
        } else {
            unreachable!()
        }
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let counts = states[0].as_any().downcast_ref::<UInt64Array>().unwrap();
        let delta = &compute::sum(counts);
        if let Some(d) = delta {
            self.count += *d;
        }
        Ok(())
    }

    fn state(&self) -> Result<SmallVec<[ScalarValue; 2]>> {
        Ok(smallvec![ScalarValue::UInt64(Some(self.count))])
    }

    fn evaluate(&self) -> Result<ScalarValue> {
        Ok(ScalarValue::UInt64(Some(self.count)))
    }
}

/// An accumulator to compute the counts of [`PrimitiveArray<T>`].
/// Stores values as native types, and does overflow checking
///
/// Unlike most other accumulators, COUNT never produces NULLs. If no
/// non-null values are seen in any group the output is 0. Thus, this
/// accumulator has no additional null or seen filter tracking.
#[derive(Debug)]
struct CountGroupsAccumulator {
    /// Count per group.
    ///
    /// Note that in upstream this is a Vec<i64>, and the count output and intermediate data type is
    /// `DataType::Int64`.  But here we are still using UInt64.
    counts: Vec<u64>,
}

impl CountGroupsAccumulator {
    pub fn new() -> Self {
        Self { counts: vec![] }
    }
}

impl GroupsAccumulator for CountGroupsAccumulator {
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        assert_eq!(values.len(), 1, "single argument to update_batch");
        let values = &values[0];

        // Add one to each group's counter for each non null, non
        // filtered value
        self.counts.resize(total_num_groups, 0);
        accumulate_indices(
            group_indices,
            values
                .data_ref()
                .null_bitmap()
                .as_ref()
                .map(|bitmap| (bitmap, values.offset(), values.len())),
            opt_filter,
            |group_index| {
                self.counts[group_index] += 1;
            },
        );

        Ok(())
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        assert_eq!(values.len(), 1, "one argument to merge_batch");
        // first batch is counts, second is partial sums
        let partial_counts = match values[0]
            .as_any()
            .downcast_ref::<PrimitiveArray<UInt64Type>>()
        {
            Some(x) => x,
            None => {
                panic!("values[0] is of unexpected type {:?}, expecting UInt64Type for intermediate count batch", values[0].data_type());
            }
        };

        // intermediate counts are always created as non null
        assert_eq!(partial_counts.null_count(), 0);
        let partial_counts = partial_counts.values();

        // Adds the counts with the partial counts
        self.counts.resize(total_num_groups, 0);
        match opt_filter {
            Some(filter) => filter
                .iter()
                .zip(group_indices.iter())
                .zip(partial_counts.iter())
                .for_each(|((filter_value, &group_index), partial_count)| {
                    if let Some(true) = filter_value {
                        self.counts[group_index] += partial_count;
                    }
                }),
            None => group_indices.iter().zip(partial_counts.iter()).for_each(
                |(&group_index, partial_count)| {
                    self.counts[group_index] += partial_count;
                },
            ),
        }

        Ok(())
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        let counts = emit_to.take_needed(&mut self.counts);

        // Count is always non null (null inputs just don't contribute to the overall values)

        // TODO: This copies.  Ideally, don't.  Note: Avoiding this memcpy had minimal effect in PrimitiveGroupsAccumulator
        let buffers = vec![Buffer::from_slice_ref(&counts)];

        let data = ArrayData::new(
            DataType::UInt64,
            counts.len(),
            None,
            None,
            0, /* offset */
            buffers,
            vec![],
        );
        Ok(Arc::new(PrimitiveArray::<UInt64Type>::from(data)))
    }

    // return arrays for counts
    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        let counts = emit_to.take_needed(&mut self.counts);
        // Backporting note:  UInt64Array::from actually does copy here in old DF.
        let counts: PrimitiveArray<UInt64Type> = UInt64Array::from(counts); // zero copy, no nulls
        Ok(vec![Arc::new(counts) as ArrayRef])
    }

    /// Converts an input batch directly to a state batch
    ///
    /// The state of `COUNT` is always a single Int64Array (backporting note: it is a UInt64Array):
    /// * `1` (for non-null, non filtered values)
    /// * `0` (for null values)
    fn convert_to_state(
        &self,
        _values: &[ArrayRef],
        _opt_filter: Option<&BooleanArray>,
    ) -> Result<Vec<ArrayRef>> {
        // convert_to_state only gets used in upstream datafusion, and we set
        // supports_convert_to_state to false. Because values.data_ref().offset() and the null
        // bitmap have differences that require care to backport, we comment this out instead.
        return Err(DataFusionError::NotImplemented(
            "Input batch conversion to state not implemented".to_owned(),
        ));
        /*
        let values = &values[0];

        let state_array = match (values.logical_nulls(), opt_filter) {
            (None, None) => {
                // In case there is no nulls in input and no filter, returning array of 1
                Arc::new(Int64Array::from_value(1, values.len()))
            }
            (Some(nulls), None) => {
                // If there are any nulls in input values -- casting `nulls` (true for values, false for nulls)
                // of input array to Int64
                let nulls = BooleanArray::new(nulls.into_inner(), None);
                compute::cast(&nulls, &DataType::Int64)?
            }
            (None, Some(filter)) => {
                // If there is only filter
                // - applying filter null mask to filter values by bitand filter values and nulls buffers
                //   (using buffers guarantees absence of nulls in result)
                // - casting result of bitand to Int64 array
                let (filter_values, filter_nulls) = filter.clone().into_parts();

                let state_buf = match filter_nulls {
                    Some(filter_nulls) => &filter_values & filter_nulls.inner(),
                    None => filter_values,
                };

                let boolean_state = BooleanArray::new(state_buf, None);
                compute::cast(&boolean_state, &DataType::Int64)?
            }
            (Some(nulls), Some(filter)) => {
                // For both input nulls and filter
                // - applying filter null mask to filter values by bitand filter values and nulls buffers
                //   (using buffers guarantees absence of nulls in result)
                // - applying values null mask to filter buffer by another bitand on filter result and
                //   nulls from input values
                // - casting result to Int64 array
                let (filter_values, filter_nulls) = filter.clone().into_parts();

                let filter_buf = match filter_nulls {
                    Some(filter_nulls) => &filter_values & filter_nulls.inner(),
                    None => filter_values,
                };
                let state_buf = &filter_buf & nulls.inner();

                let boolean_state = BooleanArray::new(state_buf, None);
                compute::cast(&boolean_state, &DataType::Int64)?
            }
        };

        Ok(vec![state_array])
        */
    }

    fn supports_convert_to_state(&self) -> bool {
        // Is set to true in upstream (as it's implemented above in upstream).  But convert_to_state
        // is not used in this branch anyway.
        false
    }

    fn size(&self) -> usize {
        self.counts.capacity() * size_of::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physical_plan::expressions::col;
    use crate::physical_plan::expressions::tests::aggregate;
    use crate::{error::Result, generic_test_op};
    use arrow::record_batch::RecordBatch;
    use arrow::{array::*, datatypes::*};

    #[test]
    fn count_elements() -> Result<()> {
        let a: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]));
        generic_test_op!(
            a,
            DataType::Int32,
            Count,
            ScalarValue::from(5u64),
            DataType::UInt64
        )
    }

    #[test]
    fn count_with_nulls() -> Result<()> {
        let a: ArrayRef = Arc::new(Int32Array::from(vec![
            Some(1),
            Some(2),
            None,
            None,
            Some(3),
            None,
        ]));
        generic_test_op!(
            a,
            DataType::Int32,
            Count,
            ScalarValue::from(3u64),
            DataType::UInt64
        )
    }

    #[test]
    fn count_all_nulls() -> Result<()> {
        let a: ArrayRef = Arc::new(BooleanArray::from(vec![
            None, None, None, None, None, None, None, None,
        ]));
        generic_test_op!(
            a,
            DataType::Boolean,
            Count,
            ScalarValue::from(0u64),
            DataType::UInt64
        )
    }

    #[test]
    fn count_empty() -> Result<()> {
        let a: Vec<bool> = vec![];
        let a: ArrayRef = Arc::new(BooleanArray::from(a));
        generic_test_op!(
            a,
            DataType::Boolean,
            Count,
            ScalarValue::from(0u64),
            DataType::UInt64
        )
    }

    #[test]
    fn count_utf8() -> Result<()> {
        let a: ArrayRef =
            Arc::new(StringArray::from(vec!["a", "bb", "ccc", "dddd", "ad"]));
        generic_test_op!(
            a,
            DataType::Utf8,
            Count,
            ScalarValue::from(5u64),
            DataType::UInt64
        )
    }

    #[test]
    fn count_large_utf8() -> Result<()> {
        let a: ArrayRef =
            Arc::new(LargeStringArray::from(vec!["a", "bb", "ccc", "dddd", "ad"]));
        generic_test_op!(
            a,
            DataType::LargeUtf8,
            Count,
            ScalarValue::from(5u64),
            DataType::UInt64
        )
    }
}
