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

use super::{format_state_name, Literal};

use crate::{AggregateExpr, PhysicalExpr};
use arrow::datatypes::{DataType, Field};
use datafusion_common::DataFusionError;
use datafusion_common::Result;
use datafusion_common::ScalarValue;
use datafusion_expr::Accumulator;

use std::{any::Any, sync::Arc};

/// PERCENTILE_CONT aggregate expression
#[derive(Debug)]
pub struct PercentileCont {
    name: String,
    input_data_type: DataType,
    _percentile: f64,
    expr: Arc<dyn PhysicalExpr>,
    _asc: bool,
    _nulls_first: bool,
}

impl PercentileCont {
    /// Create a new [`PercentileCont`] aggregate function.
    pub fn new(
        name: impl Into<String>,
        expr: Vec<Arc<dyn PhysicalExpr>>,
        input_data_type: DataType,
        within_group: Vec<(Arc<dyn PhysicalExpr>, bool, bool)>,
    ) -> Result<Self> {
        // Arguments should be [DesiredPercentileLiteral]
        debug_assert_eq!(expr.len(), 1);

        // Extract the desired percentile literal
        let lit = expr[0]
            .as_any()
            .downcast_ref::<Literal>()
            .ok_or_else(|| {
                DataFusionError::Internal(
                    "desired percentile argument must be float literal".to_string(),
                )
            })?
            .value();
        let percentile = match lit {
            ScalarValue::Float32(Some(q)) => *q as f64,
            ScalarValue::Float64(Some(q)) => *q as f64,
            got => return Err(DataFusionError::NotImplemented(format!(
                "Percentile value for 'PERCENTILE_CONT' must be Float32 or Float64 literal (got data type {})",
                got
            )))
        };

        // Ensure the percentile is between 0 and 1.
        if !(0.0..=1.0).contains(&percentile) {
            return Err(DataFusionError::Plan(format!(
                "Percentile value must be between 0.0 and 1.0 inclusive, {} is invalid",
                percentile
            )));
        }

        // Ensure that WITHIN GROUP contains exactly one value
        if within_group.len() != 1 {
            return Err(DataFusionError::Plan(
                "PERCENTILE_CONT ... WITHIN GROUP must have exactly one expression in ORDER BY".to_string(),
            ));
        }
        let (order_by, asc, nulls_first) = &within_group[0];

        // ORDER BY type must be Float64 or one of the Interval types
        match input_data_type {
            DataType::Float64 | DataType::Interval(_) => (),
            typ => {
                return Err(DataFusionError::Plan(format!(
                    "WITHIN GROUP (ORDER BY ...) must be Float64 or Interval, got {}",
                    typ
                )))
            }
        }

        Ok(Self {
            name: name.into(),
            input_data_type,
            _percentile: percentile,
            // The physical expr to evaluate during accumulation
            expr: order_by.clone(),
            _asc: *asc,
            _nulls_first: *nulls_first,
        })
    }
}

impl AggregateExpr for PercentileCont {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn field(&self) -> Result<Field> {
        Ok(Field::new(&self.name, self.input_data_type.clone(), false))
    }

    #[allow(rustdoc::private_intra_doc_links)]
    /// See [`TDigest::to_scalar_state()`] for a description of the serialised
    /// state.
    fn state_fields(&self) -> Result<Vec<Field>> {
        Ok(vec![
            Field::new(
                &format_state_name(&self.name, "max_size"),
                DataType::UInt64,
                false,
            ),
            Field::new(
                &format_state_name(&self.name, "sum"),
                DataType::Float64,
                false,
            ),
            Field::new(
                &format_state_name(&self.name, "count"),
                DataType::Float64,
                false,
            ),
            Field::new(
                &format_state_name(&self.name, "max"),
                DataType::Float64,
                false,
            ),
            Field::new(
                &format_state_name(&self.name, "min"),
                DataType::Float64,
                false,
            ),
            Field::new(
                &format_state_name(&self.name, "centroids"),
                DataType::List(Box::new(Field::new("item", DataType::Float64, true))),
                false,
            ),
        ])
    }

    fn expressions(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        vec![self.expr.clone()]
    }

    fn create_accumulator(&self) -> Result<Box<dyn Accumulator>> {
        Err(DataFusionError::NotImplemented(
            "percentile_cont(...) execution is not implemented".to_string(),
        ))
    }

    fn name(&self) -> &str {
        &self.name
    }
}
