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

//! Negation (-) expression

use std::any::Any;
use std::sync::Arc;

use arrow::array::ArrayRef;
use arrow::compute::kernels::{arity::unary, numeric::neg_wrapping};
use arrow::{
    array::{IntervalDayTimeArray, IntervalYearMonthArray},
    datatypes::{
        DataType, IntervalDayTimeType, IntervalUnit, IntervalYearMonthType, Schema,
    },
    record_batch::RecordBatch,
};

use crate::PhysicalExpr;
use datafusion_common::{scalar_negate_interval_day_time, DataFusionError, Result};
use datafusion_expr::{
    binary_rule::{is_interval, is_signed_numeric},
    ColumnarValue,
};

/// Negative expression
#[derive(Debug)]
pub struct NegativeExpr {
    /// Input expression
    arg: Arc<dyn PhysicalExpr>,
}

impl NegativeExpr {
    /// Create new not expression
    pub fn new(arg: Arc<dyn PhysicalExpr>) -> Self {
        Self { arg }
    }

    /// Get the input expression
    pub fn arg(&self) -> &Arc<dyn PhysicalExpr> {
        &self.arg
    }
}

impl std::fmt::Display for NegativeExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "(- {})", self.arg)
    }
}

impl PhysicalExpr for NegativeExpr {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, input_schema: &Schema) -> Result<DataType> {
        self.arg.data_type(input_schema)
    }

    fn nullable(&self, input_schema: &Schema) -> Result<bool> {
        self.arg.nullable(input_schema)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let arg = self.arg.evaluate(batch)?;
        match arg {
            ColumnarValue::Array(array) => {
                let result: Result<ArrayRef> = match array.data_type() {
                    // integers wrap on overflow, like arrow 13's `negate` did
                    DataType::Int8
                    | DataType::Int16
                    | DataType::Int32
                    | DataType::Int64
                    | DataType::Float32
                    | DataType::Float64 => Ok(neg_wrapping(array.as_ref())?),
                    DataType::Interval(IntervalUnit::YearMonth) => {
                        let array = array.as_any().downcast_ref::<IntervalYearMonthArray>().unwrap();
                        Ok(Arc::new(unary::<_, _, IntervalYearMonthType>(array, |x: i32| x.wrapping_neg())))
                    }
                    DataType::Interval(IntervalUnit::DayTime) => {
                        let array = array.as_any().downcast_ref::<IntervalDayTimeArray>().unwrap();
                        Ok(Arc::new(unary::<_, _, IntervalDayTimeType>(array, scalar_negate_interval_day_time)))
                    }
                    _ => Err(DataFusionError::Internal(format!(
                        "(- '{:?}') can't be evaluated because the expression's type is {:?}, which is not supported",
                        self,
                        array.data_type(),
                    ))),
                };
                result.map(ColumnarValue::Array)
            }
            ColumnarValue::Scalar(scalar) => {
                Ok(ColumnarValue::Scalar(scalar.arithmetic_negate()))
            }
        }
    }
}

/// Creates a unary expression NEGATIVE
///
/// # Errors
///
/// This function errors when the argument's type is not signed numeric
pub fn negative(
    arg: Arc<dyn PhysicalExpr>,
    input_schema: &Schema,
) -> Result<Arc<dyn PhysicalExpr>> {
    let data_type = arg.data_type(input_schema)?;
    if !is_signed_numeric(&data_type) && !is_interval(&data_type) {
        Err(DataFusionError::Internal(
            format!(
                "(- '{:?}') can't be evaluated because the expression's type is {:?}, which is not supported",
                arg, data_type,
            ),
        ))
    } else {
        Ok(Arc::new(NegativeExpr::new(arg)))
    }
}
