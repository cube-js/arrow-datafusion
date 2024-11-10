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

//! [`GroupsAccumulator`] helpers: [`NullState`] and [`accumulate_indices`]
//!
//! [`GroupsAccumulator`]: datafusion_expr_common::groups_accumulator::GroupsAccumulator

use arrow::array::{Array, BooleanArray, BooleanBufferBuilder, PrimitiveArray};
use arrow::bitmap::Bitmap;
use arrow::buffer::{Buffer, MutableBuffer};
// use arrow::buffer::{BooleanBuffer, NullBuffer};
use arrow::datatypes::ArrowPrimitiveType;

use crate::physical_plan::groups_accumulator::EmitTo;
/// Track the accumulator null state per row: if any values for that
/// group were null and if any values have been seen at all for that group.
///
/// This is part of the inner loop for many [`GroupsAccumulator`]s,
/// and thus the performance is critical and so there are multiple
/// specialized implementations, invoked depending on the specific
/// combinations of the input.
///
/// Typically there are 4 potential combinations of inputs must be
/// special cased for performance:
///
/// * With / Without filter
/// * With / Without nulls in the input
///
/// If the input has nulls, then the accumulator must potentially
/// handle each input null value specially (e.g. for `SUM` to mark the
/// corresponding sum as null)
///
/// If there are filters present, `NullState` tracks if it has seen
/// *any* value for that group (as some values may be filtered
/// out). Without a filter, the accumulator is only passed groups that
/// had at least one value to accumulate so they do not need to track
/// if they have seen values for a particular group.
///
/// [`GroupsAccumulator`]: datafusion_expr_common::groups_accumulator::GroupsAccumulator
#[derive(Debug)]
pub struct NullState {
    /// Have we seen any non-filtered input values for `group_index`?
    ///
    /// If `seen_values[i]` is true, have seen at least one non null
    /// value for group `i`
    ///
    /// If `seen_values[i]` is false, have not seen any values that
    /// pass the filter yet for group `i`
    seen_values: BooleanBufferBuilder,
}

impl Default for NullState {
    fn default() -> Self {
        Self::new()
    }
}

impl NullState {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Self {
            seen_values: BooleanBufferBuilder::new(0),
        }
    }

    /// return the size of all buffers allocated by this null state, not including self
    pub fn size(&self) -> usize {
        // capacity is in bits, so convert to bytes
        self.seen_values.capacity() / 8
    }

    /// Gets `seen_values[index]`, i.e. looks up the bitmap.  Unused but exists for
    /// debugging or documentation.
    pub fn get_is_valid(&self, index: usize) -> bool {
        self.seen_values.get_bit(index)
    }

    /// Invokes `value_fn(group_index, value)` for each non null, non
    /// filtered value of `value`, while tracking which groups have
    /// seen null inputs and which groups have seen any inputs if necessary
    //
    /// # Arguments:
    ///
    /// * `values`: the input arguments to the accumulator
    /// * `group_indices`:  To which groups do the rows in `values` belong, (aka group_index)
    /// * `opt_filter`: if present, only rows for which is Some(true) are included
    /// * `value_fn`: function invoked for  (group_index, value) where value is non null
    ///
    /// See [`accumulate`], for more details on how value_fn is called
    ///
    /// When value_fn is called it also sets
    ///
    /// 1. `self.seen_values[group_index]` to true for all rows that had a non null value
    pub fn accumulate<T, F>(
        &mut self,
        group_indices: &[usize],
        values: &PrimitiveArray<T>,
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
        mut value_fn: F,
    ) where
        T: ArrowPrimitiveType + Send,
        F: FnMut(usize, T::Native) + Send,
    {
        // ensure the seen_values is big enough (start everything at
        // "not seen" valid)
        let seen_values =
            initialize_builder(&mut self.seen_values, total_num_groups, false);
        accumulate(group_indices, values, opt_filter, |group_index, value| {
            seen_values.set_bit(group_index, true);
            value_fn(group_index, value);
        });
    }

    /// Invokes `value_fn(group_index, value)` for each non null, non
    /// filtered value in `values`, while tracking which groups have
    /// seen null inputs and which groups have seen any inputs, for
    /// [`BooleanArray`]s.
    ///
    /// Since `BooleanArray` is not a [`PrimitiveArray`] it must be
    /// handled specially.
    ///
    /// See [`Self::accumulate`], which handles `PrimitiveArray`s, for
    /// more details on other arguments.
    pub fn accumulate_boolean<F>(
        &mut self,
        group_indices: &[usize],
        values: &BooleanArray,
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
        mut value_fn: F,
    ) where
        F: FnMut(usize, bool) + Send,
    {
        // Upstream this is a BooleanBuffer.  To minimize code changes, we work with a buffer (of packed bits) directly here.
        let data: &arrow::buffer::Buffer = values.values();
        assert_eq!(data.len(), group_indices.len().div_ceil(8));

        // ensure the seen_values is big enough (start everything at
        // "not seen" valid)
        let seen_values =
            initialize_builder(&mut self.seen_values, total_num_groups, false);

        // TODO: Using values.value(i), values.is_null(i), etc., is probably bad performance, but we
        // were already using fairly similar iterators.

        // These could be made more performant by iterating in chunks of 64 bits at a time
        match (values.null_count() > 0, opt_filter) {
            // no nulls, no filter,
            (false, None) => {
                // if we have previously seen nulls, ensure the null
                // buffer is big enough (start everything at valid)
                for (i, &group_index) in group_indices.iter().enumerate() {
                    let new_value = values.value(i);
                    seen_values.set_bit(group_index, true);
                    value_fn(group_index, new_value);
                }
            }
            // nulls, no filter
            (true, None) => {
                // let nulls = values.nulls().unwrap();  // TODO: Try faster usage with direct access to null bitmaps.
                for (i, &group_index) in group_indices.iter().enumerate() {
                    let new_value = values.value(i);
                    let is_valid = !values.is_null(i);
                    if is_valid {
                        seen_values.set_bit(group_index, true);
                        value_fn(group_index, new_value);
                    }
                }
            }
            // no nulls, but a filter
            (false, Some(filter)) => {
                assert_eq!(filter.len(), group_indices.len());
                for (i, &group_index) in group_indices.iter().enumerate() {
                    let new_value = values.value(i);
                    let filter_value = !filter.is_null(i) && filter.value(i);
                    if filter_value {
                        seen_values.set_bit(group_index, true);
                        value_fn(group_index, new_value);
                    }
                }
            }
            // both null values and filters
            (true, Some(filter)) => {
                assert_eq!(filter.len(), group_indices.len());
                for (i, &group_index) in group_indices.iter().enumerate() {
                    let new_value = values.value(i);
                    let new_value_is_valid = !values.is_null(i);
                    let filter_value = !filter.is_null(i) & filter.value(i);
                    if filter_value {
                        if new_value_is_valid {
                            seen_values.set_bit(group_index, true);
                            value_fn(group_index, new_value)
                        }
                    }
                }
            }
        }
    }

    /// Creates the a [`NullBuffer`] representing which group_indices
    /// should have null values (because they never saw any values)
    /// for the `emit_to` rows.
    ///
    /// resets the internal state appropriately
    pub fn build(&mut self, emit_to: EmitTo) -> Bitmap {
        let original_len = self.seen_values.len();
        let nulls: Buffer = self.seen_values.finish();

        let nulls: Buffer = match emit_to {
            EmitTo::All => nulls,
            EmitTo::First(n) => {
                // split off the first N values in seen_values

                let new_len = original_len - n;

                //
                // TODO make this more efficient rather than two
                // copies and bitwise manipulation (upstream df comment)
                // TODO: Of course use 64 bits, at least.  Or simd or such.  Sigh.
                let num_bytes = n.div_ceil(8);
                let nulls_slice = nulls.as_slice();
                let first_n_null: Buffer = Buffer::from(&nulls_slice[0..num_bytes]);
                if n % 8 == 0 {
                    let mut mutable_buffer =
                        MutableBuffer::with_capacity(nulls_slice.len());
                    mutable_buffer.extend_from_slice(&nulls_slice[num_bytes..]);
                    self.seen_values =
                        BooleanBufferBuilder::new_from_buffer(mutable_buffer, new_len);
                } else {
                    let mut mutable_buffer =
                        MutableBuffer::with_capacity(nulls_slice.len());

                    // TODO: We could find a fast library function for this... or add an offset field to self.seen_values.  At least do 64 bit.
                    let misalignment = n % 8;
                    // Because n % 8 != 0, we know num_bytes > 0.
                    let mut leftover = nulls_slice[num_bytes - 1] >> misalignment;
                    for seen in &nulls_slice[num_bytes..] {
                        let seen: u8 = *seen;
                        let rejoined = leftover | (seen << (8 - misalignment));
                        mutable_buffer.push(rejoined);
                        leftover = seen >> misalignment;
                    }
                    mutable_buffer.push(leftover);
                    mutable_buffer.resize(new_len.div_ceil(8), 0);
                    self.seen_values =
                        BooleanBufferBuilder::new_from_buffer(mutable_buffer, new_len);
                }

                first_n_null
            }
        };
        Bitmap::from(nulls)
    }
}

/// Invokes `value_fn(group_index, value)` for each non null, non
/// filtered value of `value`,
///
/// # Arguments:
///
/// * `group_indices`:  To which groups do the rows in `values` belong, (aka group_index)
/// * `values`: the input arguments to the accumulator
/// * `opt_filter`: if present, only rows for which is Some(true) are included
/// * `value_fn`: function invoked for  (group_index, value) where value is non null
///
/// # Example
///
/// ```text
///  ┌─────────┐   ┌─────────┐   ┌ ─ ─ ─ ─ ┐
///  │ ┌─────┐ │   │ ┌─────┐ │     ┌─────┐
///  │ │  2  │ │   │ │ 200 │ │   │ │  t  │ │
///  │ ├─────┤ │   │ ├─────┤ │     ├─────┤
///  │ │  2  │ │   │ │ 100 │ │   │ │  f  │ │
///  │ ├─────┤ │   │ ├─────┤ │     ├─────┤
///  │ │  0  │ │   │ │ 200 │ │   │ │  t  │ │
///  │ ├─────┤ │   │ ├─────┤ │     ├─────┤
///  │ │  1  │ │   │ │ 200 │ │   │ │NULL │ │
///  │ ├─────┤ │   │ ├─────┤ │     ├─────┤
///  │ │  0  │ │   │ │ 300 │ │   │ │  t  │ │
///  │ └─────┘ │   │ └─────┘ │     └─────┘
///  └─────────┘   └─────────┘   └ ─ ─ ─ ─ ┘
///
/// group_indices   values        opt_filter
/// ```
///
/// In the example above, `value_fn` is invoked for each (group_index,
/// value) pair where `opt_filter[i]` is true and values is non null
///
/// ```text
/// value_fn(2, 200)
/// value_fn(0, 200)
/// value_fn(0, 300)
/// ```
pub fn accumulate<T, F>(
    group_indices: &[usize],
    values: &PrimitiveArray<T>,
    opt_filter: Option<&BooleanArray>,
    mut value_fn: F,
) where
    T: ArrowPrimitiveType + Send,
    F: FnMut(usize, T::Native) + Send,
{
    let data: &[T::Native] = values.values();
    assert_eq!(data.len(), group_indices.len());

    match (values.null_count() > 0, opt_filter) {
        // no nulls, no filter,
        (false, None) => {
            let iter = group_indices.iter().zip(data.iter());
            for (&group_index, &new_value) in iter {
                value_fn(group_index, new_value);
            }
        }
        // nulls, no filter
        (true, None) => {
            let valids: &Bitmap = values.data().null_bitmap().as_ref().unwrap();
            // This is based on (ahem, COPY/PASTE) arrow::compute::aggregate::sum
            // iterate over in chunks of 64 bits for more efficient null checking
            let group_indices_chunks = group_indices.chunks_exact(64);
            let data_chunks = data.chunks_exact(64);
            let bit_chunks = valids
                .buffer_ref()
                .bit_chunks(values.offset(), values.len());

            let group_indices_remainder = group_indices_chunks.remainder();
            let data_remainder = data_chunks.remainder();

            group_indices_chunks
                .zip(data_chunks)
                .zip(bit_chunks.iter())
                .for_each(|((group_index_chunk, data_chunk), mask)| {
                    // index_mask has value 1 << i in the loop
                    let mut index_mask = 1;
                    group_index_chunk.iter().zip(data_chunk.iter()).for_each(
                        |(&group_index, &new_value)| {
                            // valid bit was set, real value
                            let is_valid = (mask & index_mask) != 0;
                            if is_valid {
                                value_fn(group_index, new_value);
                            }
                            index_mask <<= 1;
                        },
                    )
                });

            // handle any remaining bits (after the initial 64)
            let remainder_bits = bit_chunks.remainder_bits();
            group_indices_remainder
                .iter()
                .zip(data_remainder.iter())
                .enumerate()
                .for_each(|(i, (&group_index, &new_value))| {
                    let is_valid = remainder_bits & (1 << i) != 0;
                    if is_valid {
                        value_fn(group_index, new_value);
                    }
                });
        }
        // no nulls, but a filter
        (false, Some(filter)) => {
            assert_eq!(filter.len(), group_indices.len());
            // The performance with a filter could be improved by
            // iterating over the filter in chunks, rather than a single
            // iterator. TODO file a ticket (upstream df comment)
            group_indices
                .iter()
                .zip(data.iter())
                .zip(filter.iter())
                .for_each(|((&group_index, &new_value), filter_value)| {
                    if let Some(true) = filter_value {
                        value_fn(group_index, new_value);
                    }
                })
        }
        // both null values and filters
        (true, Some(filter)) => {
            assert_eq!(filter.len(), group_indices.len());
            // The performance with a filter could be improved by
            // iterating over the filter in chunks, rather than using
            // iterators. TODO file a ticket (upstream df comment)
            filter
                .iter()
                .zip(group_indices.iter())
                .zip(values.iter())
                .for_each(|((filter_value, &group_index), new_value)| {
                    if let Some(true) = filter_value {
                        if let Some(new_value) = new_value {
                            value_fn(group_index, new_value)
                        }
                    }
                })
        }
    }
}

/// This function is called to update the accumulator state per row
/// when the value is not needed (e.g. COUNT)
///
/// `F`: Invoked like `value_fn(group_index) for all non null values
/// passing the filter. Note that no tracking is done for null inputs
/// or which groups have seen any values
///
/// See [`NullState::accumulate`], for more details on other
/// arguments.
pub fn accumulate_indices<F>(
    group_indices: &[usize],
    nulls: Option<(&Bitmap, usize, usize)>,
    opt_filter: Option<&BooleanArray>,
    mut index_fn: F,
) where
    F: FnMut(usize) + Send,
{
    match (nulls, opt_filter) {
        (None, None) => {
            for &group_index in group_indices.iter() {
                index_fn(group_index)
            }
        }
        (None, Some(filter)) => {
            assert_eq!(filter.len(), group_indices.len());
            // The performance with a filter could be improved by
            // iterating over the filter in chunks, rather than a single
            // iterator. TODO file a ticket (upstream df comment)
            let iter = group_indices.iter().zip(filter.iter());
            for (&group_index, filter_value) in iter {
                if let Some(true) = filter_value {
                    index_fn(group_index)
                }
            }
        }
        (Some((valids, valids_offset, valids_len)), None) => {
            assert_eq!(valids_len, group_indices.len());
            // This is based on (ahem, COPY/PASTA) arrow::compute::aggregate::sum
            // iterate over in chunks of 64 bits for more efficient null checking
            let group_indices_chunks = group_indices.chunks_exact(64);
            let bit_chunks = valids.buffer_ref().bit_chunks(valids_offset, valids_len);

            let group_indices_remainder = group_indices_chunks.remainder();

            group_indices_chunks.zip(bit_chunks.iter()).for_each(
                |(group_index_chunk, mask)| {
                    // index_mask has value 1 << i in the loop
                    let mut index_mask = 1;
                    group_index_chunk.iter().for_each(|&group_index| {
                        // valid bit was set, real vale
                        let is_valid = (mask & index_mask) != 0;
                        if is_valid {
                            index_fn(group_index);
                        }
                        index_mask <<= 1;
                    })
                },
            );

            // handle any remaining bits (after the initial 64)
            let remainder_bits = bit_chunks.remainder_bits();
            group_indices_remainder
                .iter()
                .enumerate()
                .for_each(|(i, &group_index)| {
                    let is_valid = remainder_bits & (1 << i) != 0;
                    if is_valid {
                        index_fn(group_index)
                    }
                });
        }

        (Some((valids, valids_offset, valids_len)), Some(filter)) => {
            assert_eq!(filter.len(), group_indices.len());
            assert_eq!(valids_len, group_indices.len());

            // The performance with a filter could likely be improved by
            // iterating over the filter in chunks, rather than using
            // iterators. TODO file a ticket (upstream df comment)
            filter
                .iter()
                .zip(group_indices.iter())
                .zip(valids.make_iter(valids_offset, valids_len))
                .for_each(|((filter_value, &group_index), is_valid)| {
                    if let (Some(true), true) = (filter_value, is_valid) {
                        index_fn(group_index)
                    }
                })
        }
    }
}

/// Ensures that `builder` contains a `BooleanBufferBuilder with at
/// least `total_num_groups`.
///
/// All new entries are initialized to `default_value`
fn initialize_builder(
    builder: &mut BooleanBufferBuilder,
    total_num_groups: usize,
    default_value: bool,
) -> &mut BooleanBufferBuilder {
    if builder.len() < total_num_groups {
        let new_groups = total_num_groups - builder.len();
        builder.append_n(new_groups, default_value);
    }
    builder
}

#[cfg(test)]
mod test {
    use super::*;

    use arrow::array::UInt32Array;
    use rand::{rngs::ThreadRng, Rng};
    use std::collections::HashSet;

    #[test]
    fn accumulate() {
        let group_indices = (0..100).collect();
        let values = (0..100).map(|i| (i + 1) * 10).collect();
        let values_with_nulls = (0..100)
            .map(|i| if i % 3 == 0 { None } else { Some((i + 1) * 10) })
            .collect();

        // default to every fifth value being false, every even
        // being null
        let filter: BooleanArray = (0..100)
            .map(|i| {
                let is_even = i % 2 == 0;
                let is_fifth = i % 5 == 0;
                if is_even {
                    None
                } else if is_fifth {
                    Some(false)
                } else {
                    Some(true)
                }
            })
            .collect();

        Fixture {
            group_indices,
            values,
            values_with_nulls,
            filter,
        }
        .run()
    }

    #[test]
    fn accumulate_fuzz() {
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            Fixture::new_random(&mut rng).run();
        }
    }

    /// Values for testing (there are enough values to exercise the 64 bit chunks
    struct Fixture {
        /// 100..0
        group_indices: Vec<usize>,

        /// 10, 20, ... 1010
        values: Vec<u32>,

        /// same as values, but every third is null:
        /// None, Some(20), Some(30), None ...
        values_with_nulls: Vec<Option<u32>>,

        /// filter (defaults to None)
        filter: BooleanArray,
    }

    impl Fixture {
        fn new_random(rng: &mut ThreadRng) -> Self {
            // Number of input values in a batch
            let num_values: usize = rng.gen_range(1..200);
            // number of distinct groups
            let num_groups: usize = rng.gen_range(2..1000);
            let max_group = num_groups - 1;

            let group_indices: Vec<usize> = (0..num_values)
                .map(|_| rng.gen_range(0..max_group))
                .collect();

            let values: Vec<u32> = (0..num_values).map(|_| rng.gen()).collect();

            // 10% chance of false
            // 10% change of null
            // 80% chance of true
            let filter: BooleanArray = (0..num_values)
                .map(|_| {
                    let filter_value = rng.gen_range(0.0..1.0);
                    if filter_value < 0.1 {
                        Some(false)
                    } else if filter_value < 0.2 {
                        None
                    } else {
                        Some(true)
                    }
                })
                .collect();

            // random values with random number and location of nulls
            // random null percentage
            let null_pct: f32 = rng.gen_range(0.0..1.0);
            let values_with_nulls: Vec<Option<u32>> = (0..num_values)
                .map(|_| {
                    let is_null = null_pct < rng.gen_range(0.0..1.0);
                    if is_null {
                        None
                    } else {
                        Some(rng.gen())
                    }
                })
                .collect();

            Self {
                group_indices,
                values,
                values_with_nulls,
                filter,
            }
        }

        /// returns `Self::values` an Array
        fn values_array(&self) -> UInt32Array {
            UInt32Array::from(self.values.clone())
        }

        /// returns `Self::values_with_nulls` as an Array
        fn values_with_nulls_array(&self) -> UInt32Array {
            UInt32Array::from(self.values_with_nulls.clone())
        }

        /// Calls `NullState::accumulate` and `accumulate_indices`
        /// with all combinations of nulls and filter values
        fn run(&self) {
            let total_num_groups = *self.group_indices.iter().max().unwrap() + 1;

            let group_indices = &self.group_indices;
            let values_array = self.values_array();
            let values_with_nulls_array = self.values_with_nulls_array();
            let filter = &self.filter;

            // no null, no filters
            Self::accumulate_test(group_indices, &values_array, None, total_num_groups);

            // nulls, no filters
            Self::accumulate_test(
                group_indices,
                &values_with_nulls_array,
                None,
                total_num_groups,
            );

            // no nulls, filters
            Self::accumulate_test(
                group_indices,
                &values_array,
                Some(filter),
                total_num_groups,
            );

            // nulls, filters
            Self::accumulate_test(
                group_indices,
                &values_with_nulls_array,
                Some(filter),
                total_num_groups,
            );
        }

        /// Calls `NullState::accumulate` and `accumulate_indices` to
        /// ensure it generates the correct values.
        ///
        fn accumulate_test(
            group_indices: &[usize],
            values: &UInt32Array,
            opt_filter: Option<&BooleanArray>,
            total_num_groups: usize,
        ) {
            Self::accumulate_values_test(
                group_indices,
                values,
                opt_filter,
                total_num_groups,
            );
            Self::accumulate_indices_test(
                group_indices,
                values
                    .data()
                    .null_bitmap()
                    .as_ref()
                    .map(|bitmap| (bitmap, values.data().offset(), values.data().len())),
                opt_filter,
            );

            // Convert values into a boolean array (anything above the
            // average is true, otherwise false)
            let avg: usize = values.iter().filter_map(|v| v.map(|v| v as usize)).sum();
            let boolean_values: BooleanArray =
                values.iter().map(|v| v.map(|v| v as usize > avg)).collect();
            Self::accumulate_boolean_test(
                group_indices,
                &boolean_values,
                opt_filter,
                total_num_groups,
            );
        }

        /// This is effectively a different implementation of
        /// accumulate that we compare with the above implementation
        fn accumulate_values_test(
            group_indices: &[usize],
            values: &UInt32Array,
            opt_filter: Option<&BooleanArray>,
            total_num_groups: usize,
        ) {
            let mut accumulated_values = vec![];
            let mut null_state = NullState::new();

            null_state.accumulate(
                group_indices,
                values,
                opt_filter,
                total_num_groups,
                |group_index, value| {
                    accumulated_values.push((group_index, value));
                },
            );

            // Figure out the expected values
            let mut expected_values = vec![];
            let mut mock = MockNullState::new();

            match opt_filter {
                None => group_indices.iter().zip(values.iter()).for_each(
                    |(&group_index, value)| {
                        if let Some(value) = value {
                            mock.saw_value(group_index);
                            expected_values.push((group_index, value));
                        }
                    },
                ),
                Some(filter) => {
                    group_indices
                        .iter()
                        .zip(values.iter())
                        .zip(filter.iter())
                        .for_each(|((&group_index, value), is_included)| {
                            // if value passed filter
                            if let Some(true) = is_included {
                                if let Some(value) = value {
                                    mock.saw_value(group_index);
                                    expected_values.push((group_index, value));
                                }
                            }
                        });
                }
            }

            assert_eq!(accumulated_values, expected_values,
                       "\n\naccumulated_values:{accumulated_values:#?}\n\nexpected_values:{expected_values:#?}");
            let seen_values = Bitmap::from(null_state.seen_values.finish_cloned());
            mock.validate_seen_values((&seen_values, 0, null_state.seen_values.len()));

            // Validate the final buffer (one value per group)
            let expected_null_buffer = mock.expected_null_buffer(total_num_groups);

            let null_buffer = null_state.build(EmitTo::All);

            assert_eq!(null_buffer, expected_null_buffer);
        }

        // Calls `accumulate_indices`
        // and opt_filter and ensures it calls the right values
        fn accumulate_indices_test(
            group_indices: &[usize],
            nulls: Option<(&Bitmap, usize, usize)>,
            opt_filter: Option<&BooleanArray>,
        ) {
            let mut accumulated_values = vec![];

            accumulate_indices(group_indices, nulls, opt_filter, |group_index| {
                accumulated_values.push(group_index);
            });

            // Figure out the expected values
            let mut expected_values = vec![];

            match (nulls, opt_filter) {
                (None, None) => group_indices.iter().for_each(|&group_index| {
                    expected_values.push(group_index);
                }),
                (Some(nulls), None) => group_indices
                    .iter()
                    .zip(nulls.0.make_iter(nulls.1, nulls.2))
                    .for_each(|(&group_index, is_valid)| {
                        if is_valid {
                            expected_values.push(group_index);
                        }
                    }),
                (None, Some(filter)) => group_indices.iter().zip(filter.iter()).for_each(
                    |(&group_index, is_included)| {
                        if let Some(true) = is_included {
                            expected_values.push(group_index);
                        }
                    },
                ),
                (Some(nulls), Some(filter)) => {
                    group_indices
                        .iter()
                        .zip(nulls.0.make_iter(nulls.1, nulls.2))
                        .zip(filter.iter())
                        .for_each(|((&group_index, is_valid), is_included)| {
                            // if value passed filter
                            if let (true, Some(true)) = (is_valid, is_included) {
                                expected_values.push(group_index);
                            }
                        });
                }
            }

            assert_eq!(accumulated_values, expected_values,
                       "\n\naccumulated_values:{accumulated_values:#?}\n\nexpected_values:{expected_values:#?}");
        }

        /// This is effectively a different implementation of
        /// accumulate_boolean that we compare with the above implementation
        fn accumulate_boolean_test(
            group_indices: &[usize],
            values: &BooleanArray,
            opt_filter: Option<&BooleanArray>,
            total_num_groups: usize,
        ) {
            let mut accumulated_values = vec![];
            let mut null_state = NullState::new();

            null_state.accumulate_boolean(
                group_indices,
                values,
                opt_filter,
                total_num_groups,
                |group_index, value| {
                    accumulated_values.push((group_index, value));
                },
            );

            // Figure out the expected values
            let mut expected_values = vec![];
            let mut mock = MockNullState::new();

            match opt_filter {
                None => group_indices.iter().zip(values.iter()).for_each(
                    |(&group_index, value)| {
                        if let Some(value) = value {
                            mock.saw_value(group_index);
                            expected_values.push((group_index, value));
                        }
                    },
                ),
                Some(filter) => {
                    group_indices
                        .iter()
                        .zip(values.iter())
                        .zip(filter.iter())
                        .for_each(|((&group_index, value), is_included)| {
                            // if value passed filter
                            if let Some(true) = is_included {
                                if let Some(value) = value {
                                    mock.saw_value(group_index);
                                    expected_values.push((group_index, value));
                                }
                            }
                        });
                }
            }

            assert_eq!(accumulated_values, expected_values,
                       "\n\naccumulated_values:{accumulated_values:#?}\n\nexpected_values:{expected_values:#?}");

            let seen_values = Bitmap::from(null_state.seen_values.finish_cloned());
            mock.validate_seen_values((&seen_values, 0, null_state.seen_values.len()));

            // Validate the final buffer (one value per group)
            let expected_null_buffer = mock.expected_null_buffer(total_num_groups);

            let null_buffer = null_state.build(EmitTo::All);

            assert_eq!(null_buffer, expected_null_buffer);
        }
    }

    /// Parallel implementation of NullState to check expected values
    #[derive(Debug, Default)]
    struct MockNullState {
        /// group indices that had values that passed the filter
        seen_values: HashSet<usize>,
    }

    impl MockNullState {
        fn new() -> Self {
            Default::default()
        }

        fn saw_value(&mut self, group_index: usize) {
            self.seen_values.insert(group_index);
        }

        /// did this group index see any input?
        fn expected_seen(&self, group_index: usize) -> bool {
            self.seen_values.contains(&group_index)
        }

        /// Validate that the seen_values matches self.seen_values
        fn validate_seen_values(&self, seen_values: (&Bitmap, usize, usize)) {
            for (group_index, is_seen) in seen_values
                .0
                .make_iter(seen_values.1, seen_values.2)
                .enumerate()
            {
                let expected_seen = self.expected_seen(group_index);
                assert_eq!(
                    expected_seen, is_seen,
                    "mismatch at for group {group_index}"
                );
            }
        }

        /// Create the expected null buffer based on if the input had nulls and a filter
        fn expected_null_buffer(&self, total_num_groups: usize) -> Bitmap {
            let buf: Buffer = (0..total_num_groups)
                .map(|group_index| self.expected_seen(group_index))
                .collect();
            Bitmap::from(buf)
        }
    }
}
