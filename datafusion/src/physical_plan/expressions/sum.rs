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
use std::convert::TryFrom;
use std::sync::Arc;

use crate::error::{DataFusionError, Result};
use crate::physical_plan::groups_accumulator::GroupsAccumulator;
use crate::physical_plan::groups_accumulator_flat_adapter::GroupsAccumulatorFlatAdapter;
use crate::physical_plan::groups_accumulator_prim_op::PrimitiveGroupsAccumulator;
use crate::physical_plan::{Accumulator, AggregateExpr, PhysicalExpr};
use crate::scalar::ScalarValue;
use arrow::compute;
use arrow::datatypes::DataType;
use arrow::{
    array::{
        ArrayRef, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array,
        Int64Decimal0Array, Int64Decimal10Array, Int64Decimal1Array, Int64Decimal2Array,
        Int64Decimal3Array, Int64Decimal4Array, Int64Decimal5Array, Int8Array,
        Int96Array, Int96Decimal0Array, Int96Decimal10Array, Int96Decimal1Array,
        Int96Decimal2Array, Int96Decimal3Array, Int96Decimal4Array, Int96Decimal5Array,
        UInt16Array, UInt32Array, UInt64Array, UInt8Array,
    },
    datatypes::Field,
};

use super::format_state_name;
use smallvec::smallvec;
use smallvec::SmallVec;

/// SUM aggregate expression
#[derive(Debug)]
pub struct Sum {
    name: String,
    data_type: DataType,
    input_data_type: DataType,
    expr: Arc<dyn PhysicalExpr>,
    nullable: bool,
}

/// function return type of a sum
pub fn sum_return_type(arg_type: &DataType) -> Result<DataType> {
    match arg_type {
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
            Ok(DataType::Int64)
        }
        DataType::Int96 => Ok(DataType::Int96),
        DataType::Int64Decimal(scale) => Ok(DataType::Int64Decimal(*scale)),
        DataType::Int96Decimal(scale) => Ok(DataType::Int96Decimal(*scale)),
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            Ok(DataType::UInt64)
        }
        DataType::Float32 => Ok(DataType::Float32),
        DataType::Float64 => Ok(DataType::Float64),
        other => Err(DataFusionError::Plan(format!(
            "SUM does not support type \"{:?}\"",
            other
        ))),
    }
}

impl Sum {
    /// Create a new SUM aggregate function
    pub fn new(
        expr: Arc<dyn PhysicalExpr>,
        name: impl Into<String>,
        data_type: DataType,
        input_data_type: &DataType,
    ) -> Self {
        // Note: data_type = sum_return_type(input_data_type) in the actual caller, so we don't
        // really need two params.  But, we keep the four params to break symmetry with other
        // accumulators and any code that might use 3 params, such as the generic_test_op macro.
        Self {
            name: name.into(),
            expr,
            data_type,
            input_data_type: input_data_type.clone(),
            nullable: true,
        }
    }
}

impl AggregateExpr for Sum {
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
            &format_state_name(&self.name, "sum"),
            self.data_type.clone(),
            self.nullable,
        )])
    }

    fn expressions(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        vec![self.expr.clone()]
    }

    fn create_accumulator(&self) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(SumAccumulator::try_new(&self.data_type)?))
    }

    fn uses_groups_accumulator(&self) -> bool {
        return true;
    }

    fn create_groups_accumulator(
        &self,
    ) -> arrow::error::Result<Option<Box<dyn GroupsAccumulator>>> {
        use arrow::datatypes::ArrowPrimitiveType;

        macro_rules! make_accumulator {
            ($T:ty, $U:ty) => {
                Box::new(PrimitiveGroupsAccumulator::<$T, $U, _, _>::new(
                    &<$T as ArrowPrimitiveType>::DATA_TYPE,
                    |x: &mut <$T as ArrowPrimitiveType>::Native,
                     y: <$U as ArrowPrimitiveType>::Native| {
                        *x = *x + (y as <$T as ArrowPrimitiveType>::Native);
                    },
                    |x: &mut <$T as ArrowPrimitiveType>::Native,
                     y: <$T as ArrowPrimitiveType>::Native| {
                        *x = *x + y;
                    },
                ))
            };
        }

        // Note that upstream uses x.add_wrapping(y) for the sum functions -- but here we just mimic
        // the current datafusion Sum accumulator implementation using native +.  (That native +
        // specifically is the one in the expressions *x = *x + ... above.)
        Ok(Some(match (&self.data_type, &self.input_data_type) {
            (DataType::Int64, DataType::Int64) => make_accumulator!(
                arrow::datatypes::Int64Type,
                arrow::datatypes::Int64Type
            ),
            (DataType::Int64, DataType::Int32) => make_accumulator!(
                arrow::datatypes::Int64Type,
                arrow::datatypes::Int32Type
            ),
            (DataType::Int64, DataType::Int16) => make_accumulator!(
                arrow::datatypes::Int64Type,
                arrow::datatypes::Int16Type
            ),
            (DataType::Int64, DataType::Int8) => {
                make_accumulator!(arrow::datatypes::Int64Type, arrow::datatypes::Int8Type)
            }

            (DataType::Int96, DataType::Int96) => make_accumulator!(
                arrow::datatypes::Int96Type,
                arrow::datatypes::Int96Type
            ),

            (DataType::Int64Decimal(0), DataType::Int64Decimal(0)) => make_accumulator!(
                arrow::datatypes::Int64Decimal0Type,
                arrow::datatypes::Int64Decimal0Type
            ),
            (DataType::Int64Decimal(1), DataType::Int64Decimal(1)) => make_accumulator!(
                arrow::datatypes::Int64Decimal1Type,
                arrow::datatypes::Int64Decimal1Type
            ),
            (DataType::Int64Decimal(2), DataType::Int64Decimal(2)) => make_accumulator!(
                arrow::datatypes::Int64Decimal2Type,
                arrow::datatypes::Int64Decimal2Type
            ),
            (DataType::Int64Decimal(3), DataType::Int64Decimal(3)) => make_accumulator!(
                arrow::datatypes::Int64Decimal3Type,
                arrow::datatypes::Int64Decimal3Type
            ),
            (DataType::Int64Decimal(4), DataType::Int64Decimal(4)) => make_accumulator!(
                arrow::datatypes::Int64Decimal4Type,
                arrow::datatypes::Int64Decimal4Type
            ),
            (DataType::Int64Decimal(5), DataType::Int64Decimal(5)) => make_accumulator!(
                arrow::datatypes::Int64Decimal5Type,
                arrow::datatypes::Int64Decimal5Type
            ),
            (DataType::Int64Decimal(10), DataType::Int64Decimal(10)) => {
                make_accumulator!(
                    arrow::datatypes::Int64Decimal10Type,
                    arrow::datatypes::Int64Decimal10Type
                )
            }

            (DataType::Int96Decimal(0), DataType::Int96Decimal(0)) => make_accumulator!(
                arrow::datatypes::Int96Decimal0Type,
                arrow::datatypes::Int96Decimal0Type
            ),
            (DataType::Int96Decimal(1), DataType::Int96Decimal(1)) => make_accumulator!(
                arrow::datatypes::Int96Decimal1Type,
                arrow::datatypes::Int96Decimal1Type
            ),
            (DataType::Int96Decimal(2), DataType::Int96Decimal(2)) => make_accumulator!(
                arrow::datatypes::Int96Decimal2Type,
                arrow::datatypes::Int96Decimal2Type
            ),
            (DataType::Int96Decimal(3), DataType::Int96Decimal(3)) => make_accumulator!(
                arrow::datatypes::Int96Decimal3Type,
                arrow::datatypes::Int96Decimal3Type
            ),
            (DataType::Int96Decimal(4), DataType::Int96Decimal(4)) => make_accumulator!(
                arrow::datatypes::Int96Decimal4Type,
                arrow::datatypes::Int96Decimal4Type
            ),
            (DataType::Int96Decimal(5), DataType::Int96Decimal(5)) => make_accumulator!(
                arrow::datatypes::Int96Decimal5Type,
                arrow::datatypes::Int96Decimal5Type
            ),
            (DataType::Int96Decimal(10), DataType::Int96Decimal(10)) => {
                make_accumulator!(
                    arrow::datatypes::Int96Decimal10Type,
                    arrow::datatypes::Int96Decimal10Type
                )
            }

            (DataType::UInt64, DataType::UInt64) => make_accumulator!(
                arrow::datatypes::UInt64Type,
                arrow::datatypes::UInt64Type
            ),
            (DataType::UInt64, DataType::UInt32) => make_accumulator!(
                arrow::datatypes::UInt64Type,
                arrow::datatypes::UInt32Type
            ),
            (DataType::UInt64, DataType::UInt16) => make_accumulator!(
                arrow::datatypes::UInt64Type,
                arrow::datatypes::UInt16Type
            ),
            (DataType::UInt64, DataType::UInt8) => make_accumulator!(
                arrow::datatypes::UInt64Type,
                arrow::datatypes::UInt8Type
            ),

            (DataType::Float32, DataType::Float32) => make_accumulator!(
                arrow::datatypes::Float32Type,
                arrow::datatypes::Float32Type
            ),
            (DataType::Float64, DataType::Float64) => make_accumulator!(
                arrow::datatypes::Float64Type,
                arrow::datatypes::Float64Type
            ),

            _ => {
                // This case should never be reached because we've handled all sum_return_type
                // arg_type values.  Nonetheless:
                let data_type = self.data_type.clone();

                Box::new(GroupsAccumulatorFlatAdapter::<SumAccumulator>::new(
                    move || SumAccumulator::try_new(&data_type),
                ))
            }
        }))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Debug)]
struct SumAccumulator {
    sum: ScalarValue,
}

impl SumAccumulator {
    /// new sum accumulator
    pub fn try_new(data_type: &DataType) -> Result<Self> {
        Ok(Self {
            sum: ScalarValue::try_from(data_type)?,
        })
    }
}

// returns the new value after sum with the new values, taking nullability into account
macro_rules! typed_sum_delta_batch {
    ($VALUES:expr, $ARRAYTYPE:ident, Int64Decimal, $SCALE:expr) => {{
        let array = $VALUES.as_any().downcast_ref::<$ARRAYTYPE>().unwrap();
        let delta = compute::sum(array);
        ScalarValue::Int64Decimal(delta, $SCALE)
    }};
    ($VALUES:expr, $ARRAYTYPE:ident, Int96Decimal, $SCALE:expr) => {{
        let array = $VALUES.as_any().downcast_ref::<$ARRAYTYPE>().unwrap();
        let delta = compute::sum(array);
        ScalarValue::Int96Decimal(delta, $SCALE)
    }};
    ($VALUES:expr, $ARRAYTYPE:ident, $SCALAR:ident) => {{
        let array = $VALUES.as_any().downcast_ref::<$ARRAYTYPE>().unwrap();
        let delta = compute::sum(array);
        ScalarValue::$SCALAR(delta)
    }};
}

// sums the array and returns a ScalarValue of its corresponding type.
pub(super) fn sum_batch(values: &ArrayRef) -> Result<ScalarValue> {
    Ok(match values.data_type() {
        DataType::Float64 => typed_sum_delta_batch!(values, Float64Array, Float64),
        DataType::Float32 => typed_sum_delta_batch!(values, Float32Array, Float32),
        DataType::Int64 => typed_sum_delta_batch!(values, Int64Array, Int64),
        DataType::Int96 => typed_sum_delta_batch!(values, Int96Array, Int96),
        DataType::Int64Decimal(0) => {
            typed_sum_delta_batch!(values, Int64Decimal0Array, Int64Decimal, 0)
        }
        DataType::Int64Decimal(1) => {
            typed_sum_delta_batch!(values, Int64Decimal1Array, Int64Decimal, 1)
        }
        DataType::Int64Decimal(2) => {
            typed_sum_delta_batch!(values, Int64Decimal2Array, Int64Decimal, 2)
        }
        DataType::Int64Decimal(3) => {
            typed_sum_delta_batch!(values, Int64Decimal3Array, Int64Decimal, 3)
        }
        DataType::Int64Decimal(4) => {
            typed_sum_delta_batch!(values, Int64Decimal4Array, Int64Decimal, 4)
        }
        DataType::Int64Decimal(5) => {
            typed_sum_delta_batch!(values, Int64Decimal5Array, Int64Decimal, 5)
        }
        DataType::Int64Decimal(10) => {
            typed_sum_delta_batch!(values, Int64Decimal10Array, Int64Decimal, 10)
        }
        DataType::Int96Decimal(0) => {
            typed_sum_delta_batch!(values, Int96Decimal0Array, Int96Decimal, 0)
        }
        DataType::Int96Decimal(1) => {
            typed_sum_delta_batch!(values, Int96Decimal1Array, Int96Decimal, 1)
        }
        DataType::Int96Decimal(2) => {
            typed_sum_delta_batch!(values, Int96Decimal2Array, Int96Decimal, 2)
        }
        DataType::Int96Decimal(3) => {
            typed_sum_delta_batch!(values, Int96Decimal3Array, Int96Decimal, 3)
        }
        DataType::Int96Decimal(4) => {
            typed_sum_delta_batch!(values, Int96Decimal4Array, Int96Decimal, 4)
        }
        DataType::Int96Decimal(5) => {
            typed_sum_delta_batch!(values, Int96Decimal5Array, Int96Decimal, 5)
        }
        DataType::Int96Decimal(10) => {
            typed_sum_delta_batch!(values, Int96Decimal10Array, Int96Decimal, 10)
        }
        DataType::Int32 => typed_sum_delta_batch!(values, Int32Array, Int32),
        DataType::Int16 => typed_sum_delta_batch!(values, Int16Array, Int16),
        DataType::Int8 => typed_sum_delta_batch!(values, Int8Array, Int8),
        DataType::UInt64 => typed_sum_delta_batch!(values, UInt64Array, UInt64),
        DataType::UInt32 => typed_sum_delta_batch!(values, UInt32Array, UInt32),
        DataType::UInt16 => typed_sum_delta_batch!(values, UInt16Array, UInt16),
        DataType::UInt8 => typed_sum_delta_batch!(values, UInt8Array, UInt8),
        e => {
            return Err(DataFusionError::Internal(format!(
                "Sum is not expected to receive the type {:?}",
                e
            )))
        }
    })
}

// returns the sum of two scalar values, including coercion into $TYPE.
macro_rules! typed_sum {
    ($OLD_VALUE:expr, $DELTA:expr, Int64Decimal, $TYPE:ident, $SCALE:expr) => {{
        ScalarValue::Int64Decimal(
            match ($OLD_VALUE, $DELTA) {
                (None, None) => None,
                (Some(a), None) => Some(a.clone()),
                (None, Some(b)) => Some(b.clone() as $TYPE),
                (Some(a), Some(b)) => Some(a + (*b as $TYPE)),
            },
            $SCALE,
        )
    }};
    ($OLD_VALUE:expr, $DELTA:expr, Int96Decimal, $TYPE:ident, $SCALE:expr) => {{
        ScalarValue::Int96Decimal(
            match ($OLD_VALUE, $DELTA) {
                (None, None) => None,
                (Some(a), None) => Some(a.clone()),
                (None, Some(b)) => Some(b.clone() as $TYPE),
                (Some(a), Some(b)) => Some(a + (*b as $TYPE)),
            },
            $SCALE,
        )
    }};
    ($OLD_VALUE:expr, $DELTA:expr, $SCALAR:ident, $TYPE:ident) => {{
        ScalarValue::$SCALAR(match ($OLD_VALUE, $DELTA) {
            (None, None) => None,
            (Some(a), None) => Some(a.clone()),
            (None, Some(b)) => Some(b.clone() as $TYPE),
            (Some(a), Some(b)) => Some(a + (*b as $TYPE)),
        })
    }};
}

pub(super) fn sum(lhs: &ScalarValue, rhs: &ScalarValue) -> Result<ScalarValue> {
    Ok(match (lhs, rhs) {
        // float64 coerces everything to f64
        (ScalarValue::Float64(lhs), ScalarValue::Float64(rhs)) => {
            typed_sum!(lhs, rhs, Float64, f64)
        }
        (ScalarValue::Float64(lhs), ScalarValue::Float32(rhs)) => {
            typed_sum!(lhs, rhs, Float64, f64)
        }
        (ScalarValue::Float64(lhs), ScalarValue::Int64(rhs)) => {
            typed_sum!(lhs, rhs, Float64, f64)
        }
        (ScalarValue::Float64(lhs), ScalarValue::Int32(rhs)) => {
            typed_sum!(lhs, rhs, Float64, f64)
        }
        (ScalarValue::Float64(lhs), ScalarValue::Int16(rhs)) => {
            typed_sum!(lhs, rhs, Float64, f64)
        }
        (ScalarValue::Float64(lhs), ScalarValue::Int8(rhs)) => {
            typed_sum!(lhs, rhs, Float64, f64)
        }
        (ScalarValue::Float64(lhs), ScalarValue::UInt64(rhs)) => {
            typed_sum!(lhs, rhs, Float64, f64)
        }
        (ScalarValue::Float64(lhs), ScalarValue::UInt32(rhs)) => {
            typed_sum!(lhs, rhs, Float64, f64)
        }
        (ScalarValue::Float64(lhs), ScalarValue::UInt16(rhs)) => {
            typed_sum!(lhs, rhs, Float64, f64)
        }
        (ScalarValue::Float64(lhs), ScalarValue::UInt8(rhs)) => {
            typed_sum!(lhs, rhs, Float64, f64)
        }
        // float32 has no cast
        (ScalarValue::Float32(lhs), ScalarValue::Float32(rhs)) => {
            typed_sum!(lhs, rhs, Float32, f32)
        }
        // u64 coerces u* to u64
        (ScalarValue::UInt64(lhs), ScalarValue::UInt64(rhs)) => {
            typed_sum!(lhs, rhs, UInt64, u64)
        }
        (ScalarValue::UInt64(lhs), ScalarValue::UInt32(rhs)) => {
            typed_sum!(lhs, rhs, UInt64, u64)
        }
        (ScalarValue::UInt64(lhs), ScalarValue::UInt16(rhs)) => {
            typed_sum!(lhs, rhs, UInt64, u64)
        }
        (ScalarValue::UInt64(lhs), ScalarValue::UInt8(rhs)) => {
            typed_sum!(lhs, rhs, UInt64, u64)
        }
        // i64 coerces i* to u64
        (ScalarValue::Int64(lhs), ScalarValue::Int64(rhs)) => {
            typed_sum!(lhs, rhs, Int64, i64)
        }
        (ScalarValue::Int64(lhs), ScalarValue::Int32(rhs)) => {
            typed_sum!(lhs, rhs, Int64, i64)
        }
        (ScalarValue::Int64(lhs), ScalarValue::Int16(rhs)) => {
            typed_sum!(lhs, rhs, Int64, i64)
        }
        (ScalarValue::Int64(lhs), ScalarValue::Int8(rhs)) => {
            typed_sum!(lhs, rhs, Int64, i64)
        }
        (ScalarValue::Int96(lhs), ScalarValue::Int96(rhs)) => {
            typed_sum!(lhs, rhs, Int96, i128)
        }
        (
            ScalarValue::Int64Decimal(lhs, l_scale),
            ScalarValue::Int64Decimal(rhs, r_scale),
        ) => {
            if l_scale != r_scale {
                return Err(DataFusionError::Internal(format!(
                    "Scale doesn't match: {} and {}",
                    l_scale, r_scale
                )));
            }
            typed_sum!(lhs, rhs, Int64Decimal, i64, *l_scale)
        }
        (
            ScalarValue::Int96Decimal(lhs, l_scale),
            ScalarValue::Int96Decimal(rhs, r_scale),
        ) => {
            if l_scale != r_scale {
                return Err(DataFusionError::Internal(format!(
                    "Scale doesn't match: {} and {}",
                    l_scale, r_scale
                )));
            }
            typed_sum!(lhs, rhs, Int96Decimal, i128, *l_scale)
        }
        e => {
            return Err(DataFusionError::Internal(format!(
                "Sum is not expected to receive a scalar {:?}",
                e
            )))
        }
    })
}

impl Accumulator for SumAccumulator {
    fn reset(&mut self) {
        self.sum = ScalarValue::try_from(&self.sum.get_datatype())
            .expect("scalar changed type?");
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let values = &values[0];
        self.sum = sum(&self.sum, &sum_batch(values)?)?;
        Ok(())
    }

    fn update(&mut self, values: &[ScalarValue]) -> Result<()> {
        // sum(v1, v2, v3) = v1 + v2 + v3
        self.sum = sum(&self.sum, &values[0])?;
        Ok(())
    }

    fn merge(&mut self, states: &[ScalarValue]) -> Result<()> {
        // sum(sum1, sum2) = sum1 + sum2
        self.update(states)
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        // sum(sum1, sum2, sum3, ...) = sum1 + sum2 + sum3 + ...
        self.update_batch(states)
    }

    fn state(&self) -> Result<SmallVec<[ScalarValue; 2]>> {
        Ok(smallvec![self.sum.clone()])
    }

    fn evaluate(&self) -> Result<ScalarValue> {
        Ok(self.sum.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physical_plan::expressions::col;
    use crate::{error::Result, generic_test_op};
    use arrow::datatypes::*;
    use arrow::record_batch::RecordBatch;

    // A wrapper to make Sum::new, which now has an input_type argument, work with
    // generic_test_op!.
    struct SumTestStandin;
    impl SumTestStandin {
        fn new(
            expr: Arc<dyn PhysicalExpr>,
            name: impl Into<String>,
            data_type: DataType,
        ) -> Sum {
            Sum::new(expr, name, data_type.clone(), &data_type)
        }
    }

    #[test]
    fn sum_i32() -> Result<()> {
        let a: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]));

        generic_test_op!(
            a,
            DataType::Int32,
            SumTestStandin,
            ScalarValue::from(15i64),
            DataType::Int64
        )
    }

    #[test]
    fn sum_i32_with_nulls() -> Result<()> {
        let a: ArrayRef = Arc::new(Int32Array::from(vec![
            Some(1),
            None,
            Some(3),
            Some(4),
            Some(5),
        ]));
        generic_test_op!(
            a,
            DataType::Int32,
            SumTestStandin,
            ScalarValue::from(13i64),
            DataType::Int64
        )
    }

    #[test]
    fn sum_i32_all_nulls() -> Result<()> {
        let a: ArrayRef = Arc::new(Int32Array::from(vec![None, None]));
        generic_test_op!(
            a,
            DataType::Int32,
            SumTestStandin,
            ScalarValue::Int64(None),
            DataType::Int64
        )
    }

    #[test]
    fn sum_u32() -> Result<()> {
        let a: ArrayRef =
            Arc::new(UInt32Array::from(vec![1_u32, 2_u32, 3_u32, 4_u32, 5_u32]));
        generic_test_op!(
            a,
            DataType::UInt32,
            SumTestStandin,
            ScalarValue::from(15u64),
            DataType::UInt64
        )
    }

    #[test]
    fn sum_f32() -> Result<()> {
        let a: ArrayRef =
            Arc::new(Float32Array::from(vec![1_f32, 2_f32, 3_f32, 4_f32, 5_f32]));
        generic_test_op!(
            a,
            DataType::Float32,
            SumTestStandin,
            ScalarValue::from(15_f32),
            DataType::Float32
        )
    }

    #[test]
    fn sum_f64() -> Result<()> {
        let a: ArrayRef =
            Arc::new(Float64Array::from(vec![1_f64, 2_f64, 3_f64, 4_f64, 5_f64]));
        generic_test_op!(
            a,
            DataType::Float64,
            SumTestStandin,
            ScalarValue::from(15_f64),
            DataType::Float64
        )
    }

    fn aggregate(
        batch: &RecordBatch,
        agg: Arc<dyn AggregateExpr>,
    ) -> Result<ScalarValue> {
        let mut accum = agg.create_accumulator()?;
        let expr = agg.expressions();
        let values = expr
            .iter()
            .map(|e| e.evaluate(batch))
            .map(|r| r.map(|v| v.into_array(batch.num_rows())))
            .collect::<Result<Vec<_>>>()?;
        accum.update_batch(&values)?;
        accum.evaluate()
    }
}
