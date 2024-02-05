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

use super::*;

#[tokio::test]
async fn subquery_select_no_from() -> Result<()> {
    let ctx = SessionContext::new();
    register_aggregate_simple_csv(&ctx).await?;

    let sql = "SELECT c1, (SELECT c1 + 1) FROM aggregate_simple ORDER BY c1 LIMIT 2";
    let actual = execute_to_batches(&ctx, sql).await;

    let expected = vec![
        "+---------+------------------+",
        "| c1      | c1 Plus Int64(1) |",
        "+---------+------------------+",
        "| 0.00001 | 1.00001          |",
        "| 0.00002 | 1.00002          |",
        "+---------+------------------+",
    ];
    assert_batches_eq!(expected, &actual);

    Ok(())
}

#[tokio::test]
async fn subquery_select_with_from() -> Result<()> {
    let ctx = SessionContext::new();
    register_aggregate_simple_csv(&ctx).await?;

    let sql = "SELECT c1, (SELECT o.c1 + i.c1 FROM aggregate_simple AS i WHERE o.c1 = i.c1 LIMIT 1) FROM aggregate_simple o ORDER BY c1 LIMIT 2";
    let actual = execute_to_batches(&ctx, sql).await;

    let expected = vec![
        "+---------+----------------+",
        "| c1      | o.c1 Plus i.c1 |",
        "+---------+----------------+",
        "| 0.00001 | 0.00002        |",
        "| 0.00002 | 0.00004        |",
        "+---------+----------------+",
    ];
    assert_batches_eq!(expected, &actual);

    Ok(())
}

#[tokio::test]
async fn subquery_where_no_from() -> Result<()> {
    let ctx = SessionContext::new();
    register_aggregate_simple_csv(&ctx).await?;

    let sql =
        "SELECT DISTINCT c1 FROM aggregate_simple o WHERE (SELECT NOT c3) ORDER BY c1";
    let actual = execute_to_batches(&ctx, sql).await;

    let expected = vec![
        "+---------+",
        "| c1      |",
        "+---------+",
        "| 0.00002 |",
        "| 0.00004 |",
        "+---------+",
    ];
    assert_batches_eq!(expected, &actual);

    Ok(())
}

#[tokio::test]
async fn subquery_where_with_from() -> Result<()> {
    let ctx = SessionContext::new();
    register_aggregate_simple_csv(&ctx).await?;

    let sql = "SELECT DISTINCT c1 FROM aggregate_simple o WHERE (SELECT c3 FROM aggregate_simple p WHERE o.c1 = p.c1 LIMIT 1) ORDER BY c1";
    let actual = execute_to_batches(&ctx, sql).await;

    let expected = vec![
        "+---------+",
        "| c1      |",
        "+---------+",
        "| 0.00001 |",
        "| 0.00003 |",
        "| 0.00005 |",
        "+---------+",
    ];
    assert_batches_eq!(expected, &actual);

    Ok(())
}

// TODO: plans but does not execute
#[tokio::test]
async fn subquery_select_and_where_no_from() -> Result<()> {
    let ctx = SessionContext::new();
    register_aggregate_simple_csv(&ctx).await?;

    let sql = "SELECT c1, (SELECT c1 + 1) FROM aggregate_simple o WHERE (SELECT NOT c3) ORDER BY c1 LIMIT 3";
    let actual = execute_to_batches(&ctx, sql).await;

    let expected = vec![
        "+---------+------------------+",
        "| c1      | c1 Plus Int64(1) |",
        "+---------+------------------+",
        "| 0.00002 | 1.00002          |",
        "| 0.00002 | 1.00002          |",
        "| 0.00004 | 1.00004          |",
        "+---------+------------------+",
    ];
    assert_batches_eq!(expected, &actual);

    Ok(())
}

// TODO: plans but does not execute
#[tokio::test]
async fn subquery_select_and_where_with_from() -> Result<()> {
    let ctx = SessionContext::new();
    register_aggregate_simple_csv(&ctx).await?;

    let sql = "SELECT c1, (SELECT c1 + 1) FROM aggregate_simple o WHERE (SELECT c3 FROM aggregate_simple p WHERE o.c1 = p.c1 LIMIT 1) ORDER BY c1 LIMIT 2";
    let actual = execute_to_batches(&ctx, sql).await;

    let expected = vec![
        "+---------+------------------+",
        "| c1      | c1 Plus Int64(1) |",
        "+---------+------------------+",
        "| 0.00001 | 1.00001          |",
        "| 0.00003 | 1.00003          |",
        "+---------+------------------+",
    ];
    assert_batches_eq!(expected, &actual);

    Ok(())
}

#[tokio::test]
async fn subquery_projection_pushdown() -> Result<()> {
    let ctx = SessionContext::new();
    register_aggregate_simple_csv(&ctx).await?;

    let sql = "SELECT c1, (SELECT o.c2 FROM aggregate_simple AS i WHERE o.c1 = i.c1 LIMIT 1) FROM aggregate_simple o ORDER BY c1 LIMIT 2";
    let actual = execute_to_batches(&ctx, sql).await;

    let expected = vec![
        "+---------+----------------+",
        "| c1      | o.c2           |",
        "+---------+----------------+",
        "| 0.00001 | 0.000000000001 |",
        "| 0.00002 | 0.000000000002 |",
        "+---------+----------------+",
    ];
    assert_batches_eq!(expected, &actual);

    Ok(())
}
