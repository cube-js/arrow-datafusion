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

use arrow::error::ArrowError;
use futures::future::FutureExt;
use std::fmt::{Display, Formatter};
use std::future::Future;
use std::panic::{catch_unwind, AssertUnwindSafe};
use datafusion_common::DataFusionError;

#[derive(PartialEq, Debug)]
pub struct PanicError {
    pub msg: String,
}

impl PanicError {
    pub fn new(msg: String) -> PanicError {
        PanicError { msg }
    }
}

impl Display for PanicError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Panic: {}", self.msg)
    }
}

impl From<PanicError> for ArrowError {
    fn from(error: PanicError) -> Self {
        ArrowError::ComputeError(format!("Panic: {}", error.msg))
    }
}

impl From<PanicError> for DataFusionError {
    fn from(error: PanicError) -> Self {
        DataFusionError::Internal(error.msg)
    }
}

pub fn try_with_catch_unwind<F, R>(f: F) -> Result<R, PanicError>
where
    F: FnOnce() -> R,
{
    let result = catch_unwind(AssertUnwindSafe(f));
    match result {
        Ok(x) => Ok(x),
        Err(e) => match e.downcast::<String>() {
            Ok(s) => Err(PanicError::new(*s)),
            Err(e) => match e.downcast::<&str>() {
                Ok(m1) => Err(PanicError::new(m1.to_string())),
                Err(_) => Err(PanicError::new("unknown cause".to_string())),
            },
        },
    }
}

pub async fn async_try_with_catch_unwind<F, R>(future: F) -> Result<R, PanicError>
where
    F: Future<Output = R>,
{
    let result = AssertUnwindSafe(future).catch_unwind().await;
    match result {
        Ok(x) => Ok(x),
        Err(e) => match e.downcast::<String>() {
            Ok(s) => Err(PanicError::new(*s)),
            Err(e) => match e.downcast::<&str>() {
                Ok(m1) => Err(PanicError::new(m1.to_string())),
                Err(_) => Err(PanicError::new("unknown cause".to_string())),
            },
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic;

    #[test]
    fn test_try_with_catch_unwind() {
        assert_eq!(
            try_with_catch_unwind(|| "ok".to_string()),
            Ok("ok".to_string())
        );
        assert_eq!(
            try_with_catch_unwind(|| panic!("oops")),
            Err(PanicError::new("oops".to_string()))
        );
        assert_eq!(
            try_with_catch_unwind(|| panic!("oops{}", "ie")),
            Err(PanicError::new("oopsie".to_string()))
        );
    }

    #[tokio::test]
    async fn test_async_try_with_catch_unwind() {
        assert_eq!(
            async_try_with_catch_unwind(async { "ok".to_string() }).await,
            Ok("ok".to_string())
        );
        assert_eq!(
            async_try_with_catch_unwind(async { panic!("oops") }).await,
            Err(PanicError::new("oops".to_string()))
        );
        assert_eq!(
            async_try_with_catch_unwind(async { panic!("oops{}", "ie") }).await,
            Err(PanicError::new("oopsie".to_string()))
        );
    }
}
