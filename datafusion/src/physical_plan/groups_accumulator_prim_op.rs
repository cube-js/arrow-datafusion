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

//! PrimitiveGroupsAccumulator

use std::any::type_name;
use std::marker::PhantomData;
use std::mem::size_of;
use std::sync::Arc;

use crate::error::{DataFusionError, Result};
use crate::physical_plan::groups_accumulator::{EmitTo, GroupsAccumulator};
use arrow::array::{
    ArrayData, ArrayRef, BooleanArray, BooleanBufferBuilder, PrimitiveArray,
};
use arrow::bitmap::Bitmap;
use arrow::buffer::Buffer;
use arrow::datatypes::ArrowPrimitiveType;
use arrow::datatypes::DataType;

use crate::physical_plan::null_state::NullState;

/// An accumulator that implements a single operation over
/// [`ArrowPrimitiveType`] where the accumulated state is the same as
/// the input type (such as `Sum`)
///
/// F: The function to apply to two elements. The first argument is
/// the existing value and should be updated with the second value
/// (e.g. [`BitAndAssign`] style).
///
/// [`BitAndAssign`]: std::ops::BitAndAssign
#[derive(Debug)]
pub struct PrimitiveGroupsAccumulator<T, U, F, G>
where
    T: ArrowPrimitiveType + Send,
    U: ArrowPrimitiveType + Send,
    F: Fn(&mut T::Native, U::Native) + Send + Sync + Copy,
    G: Fn(&mut T::Native, T::Native) + Send + Sync + Copy,
{
    /// values per group, stored as the native type
    values: Vec<T::Native>,

    /// The output type (needed for Decimal precision and scale)
    data_type: DataType,

    /// The starting value for new groups
    starting_value: T::Native,

    /// Track nulls in the input / filters
    null_state: NullState,

    /// Function that computes the update result from input values
    update_fn: F,

    /// Function that computes the merge result from state values
    merge_fn: G,

    _marker: std::marker::PhantomData<U>,
}

impl<T, U, F, G> PrimitiveGroupsAccumulator<T, U, F, G>
where
    T: ArrowPrimitiveType + Send,
    U: ArrowPrimitiveType + Send,
    F: Fn(&mut T::Native, U::Native) + Send + Sync + Copy,
    G: Fn(&mut T::Native, T::Native) + Send + Sync + Copy,
{
    #[allow(missing_docs)]
    pub fn new(data_type: &DataType, update_fn: F, merge_fn: G) -> Self {
        Self {
            values: vec![],
            data_type: data_type.clone(),
            null_state: NullState::new(),
            starting_value: T::default_value(),
            update_fn,
            merge_fn,
            _marker: PhantomData,
        }
    }

    /// Set the starting values for new groups
    pub fn with_starting_value(mut self, starting_value: T::Native) -> Self {
        self.starting_value = starting_value;
        self
    }

    /// Helper for update_batch and merge_batch -- (V, H) is either (T, G) or (U, F) respectively.
    fn update_or_merge_batch<
        V: ArrowPrimitiveType + Send,
        H: Fn(&mut T::Native, V::Native) + Send + Sync,
    >(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
        binary_fn: H,
    ) -> Result<()> {
        assert_eq!(values.len(), 1, "single argument to update_batch");
        let values = match values[0].as_any().downcast_ref::<PrimitiveArray<V>>() {
            Some(x) => x,
            None => {
                panic!(
                    "values[0] is of unexpected type {:?} while we are of type {:?} (T = {}, U = {}",
                    values[0].data_type(),
                    self.data_type,
                    type_name::<T>(),
                    type_name::<U>(),
                );
            }
        };

        // update values
        self.values.resize(total_num_groups, self.starting_value);

        // NullState dispatches / handles tracking nulls and groups that saw no values
        self.null_state.accumulate(
            group_indices,
            values,
            opt_filter,
            total_num_groups,
            |group_index, new_value| {
                let value = &mut self.values[group_index];
                (binary_fn)(value, new_value);
            },
        );

        Ok(())
    }
}

impl<T, U, F, G> GroupsAccumulator for PrimitiveGroupsAccumulator<T, U, F, G>
where
    T: ArrowPrimitiveType + Send,
    U: ArrowPrimitiveType + Send,
    F: Fn(&mut T::Native, U::Native) + Send + Sync + Copy,
    G: Fn(&mut T::Native, T::Native) + Send + Sync + Copy,
{
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        // update / merge are almost the same except we're adding T's vs. adding U's.
        self.update_or_merge_batch::<U, F>(
            values,
            group_indices,
            opt_filter,
            total_num_groups,
            self.update_fn,
        )
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        let values = emit_to.take_needed(&mut self.values);
        let nulls = self.null_state.build(emit_to);

        let buffers = vec![Buffer::from_slice_ref(&values)]; // TODO: This copies.  Ideally, don't.  Note: Avoiding this memcpy has minimal performance impact.

        let data = ArrayData::new(
            self.data_type.clone(),
            values.len(),
            None,
            Some(nulls.into_buffer()),
            0, /* offset */
            buffers,
            vec![],
        );
        Ok(Arc::new(PrimitiveArray::<T>::from(data)))
    }

    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        self.evaluate(emit_to).map(|arr| vec![arr])
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        // update / merge are almost the same except we're adding T's vs. adding U's.
        self.update_or_merge_batch::<T, G>(
            values,
            group_indices,
            opt_filter,
            total_num_groups,
            self.merge_fn,
        )
    }

    /// Converts an input batch directly to a state batch
    ///
    /// The state is:
    /// - self.prim_fn for all non null, non filtered values
    /// - null otherwise
    ///
    fn convert_to_state(
        &self,
        values: &[ArrayRef],
        opt_filter: Option<&BooleanArray>,
    ) -> Result<Vec<ArrayRef>> {
        let values: PrimitiveArray<U> = values[0]
            .as_any()
            .downcast_ref::<PrimitiveArray<U>>()
            .unwrap()
            .clone();

        // Initializing state with starting values
        let initial_state =
            PrimitiveArray::<T>::from_value(self.starting_value, values.len());

        // Recalculating values in case there is filter
        let values = match opt_filter {
            None => values,
            Some(filter) => {
                let (filter_values, filter_nulls, filter_offset, filter_length) =
                    filter.clone().into_parts();
                // Calculating filter mask as a result of bitand of filter, and converting it to null buffer
                let filter_bool: Buffer = match filter_nulls {
                    Some(filter_nulls) => (&filter_nulls.into_buffer() & &filter_values)?,
                    None => filter_values,
                };
                let filter_nulls: Bitmap = Bitmap::from(filter_bool);

                // Rebuilding input values with a new nulls mask, which is equal to
                // the union of original nulls and filter mask
                let values_data: ArrayData = values.into_data();
                let dt = values_data.data_type().clone();
                let (values_buf, original_nulls, values_offset, values_length) =
                    values_data.into_1_dimensional_parts();
                let nulls_buf = null_buffer_union(
                    original_nulls,
                    values_offset,
                    values_length,
                    filter_nulls,
                    filter_offset,
                    filter_length,
                )?;

                let data = ArrayData::new(
                    dt,
                    values_length,
                    None,
                    Some(nulls_buf.into_buffer()),
                    values_offset,
                    vec![values_buf],
                    vec![],
                );
                PrimitiveArray::<U>::from(data)
            }
        };

        // TODO: Use a function math_op_mut, like upstream, with initial_state passed by value.
        let state_values = arrow::compute::math_op_with_data_type(
            self.data_type.clone(),
            &initial_state,
            &values,
            |mut x, y| {
                (self.update_fn)(&mut x, y);
                x
            },
        );
        let state_values: PrimitiveArray<T> =
            state_values.map_err(DataFusionError::from)?;

        Ok(vec![Arc::new(state_values)])
    }

    fn supports_convert_to_state(&self) -> bool {
        true
    }

    fn size(&self) -> usize {
        self.values.capacity() * size_of::<T::Native>() + self.null_state.size()
    }
}

/// Returns a bitmap whose offset is lhs_offset.
fn null_buffer_union(
    lhs: Option<Bitmap>,
    lhs_offset: usize,
    lhs_len: usize,
    rhs: Bitmap,
    rhs_offset: usize,
    rhs_len: usize,
) -> Result<Bitmap> {
    assert_eq!(lhs_len, rhs_len);
    match lhs {
        Some(lhs) => {
            if lhs_offset == rhs_offset && lhs.len() == rhs.len() {
                // TODO: Do &= instead.
                // TODO: We shouldn't need lhs.len() == rhs.len(), but it makes it more convenient to use the Bitmap & operator, but... they probably are in the happy path, anyway.

                // The bitmaps are true for non-null "valid" entries -- hence the union operation bitwise anding.
                let new_bitmap: Bitmap = (&lhs & &rhs)?;
                Ok(new_bitmap)
            } else {
                // TODO: Dog-awful performance.
                let mut ret = BooleanBufferBuilder::new(lhs_offset + lhs_len);
                for _ in 0..lhs_offset {
                    ret.append(false);
                }
                for i in 0..lhs_offset {
                    ret.append(lhs.is_set(i + lhs_offset) & rhs.is_set(i + rhs_offset));
                }
                Ok(Bitmap::from(ret.finish()))
            }
        }
        None => {
            if lhs_offset == rhs_offset {
                Ok(rhs)
            } else {
                // TODO: Dog-awful performance.
                let mut ret = BooleanBufferBuilder::new(lhs_offset + lhs_len);
                for _ in 0..lhs_offset {
                    ret.append(false);
                }
                for i in 0..lhs_len {
                    ret.append(rhs.is_set(i + rhs_offset));
                }
                Ok(Bitmap::from(ret.finish()))
            }
        }
    }
}
