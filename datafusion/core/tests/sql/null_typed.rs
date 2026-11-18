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

//! Queries over `Null`-typed columns, e.g. `NULL AS n` from a subquery

use super::*;

/// t(a BIGINT, b BIGINT) = (1, 1), (1, 5), (2, NULL), (NULL, 3)
fn null_typed_ctx() -> Result<SessionContext> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, true),
        Field::new("b", DataType::Int64, true),
    ]));
    let data = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![Some(1), Some(1), Some(2), None])),
            Arc::new(Int64Array::from(vec![Some(1), Some(5), None, Some(3)])),
        ],
    )?;
    let table = MemTable::try_new(schema, vec![vec![data]])?;
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(table))?;
    Ok(ctx)
}

#[tokio::test]
async fn in_list_with_null_typed_column() -> Result<()> {
    let ctx = null_typed_ctx()?;
    let sql = "SELECT a, a IN (n, 1) AS i, a NOT IN (n, 1) AS ni, a IN (n) AS i1 \
               FROM (SELECT a, NULL AS n FROM t) s ORDER BY a";
    let actual = execute_to_batches(&ctx, sql).await;
    let expected = [
        "+---+------+-------+----+",
        "| a | i    | ni    | i1 |",
        "+---+------+-------+----+",
        "| 1 | true | false |    |",
        "| 1 | true | false |    |",
        "| 2 |      |       |    |",
        "|   |      |       |    |",
        "+---+------+-------+----+",
    ];
    assert_batches_eq!(expected, &actual);
    Ok(())
}

#[tokio::test]
async fn in_list_with_columns() -> Result<()> {
    let ctx = null_typed_ctx()?;
    let sql = "SELECT a, b, a IN (b, 2) AS i, a NOT IN (b, 2) AS ni \
               FROM t ORDER BY a, b";
    let actual = execute_to_batches(&ctx, sql).await;
    let expected = [
        "+---+---+-------+-------+",
        "| a | b | i     | ni    |",
        "+---+---+-------+-------+",
        "| 1 | 1 | true  | false |",
        "| 1 | 5 | false | true  |",
        "| 2 |   | true  | false |",
        "|   | 3 |       |       |",
        "+---+---+-------+-------+",
    ];
    assert_batches_eq!(expected, &actual);

    let sql = "SELECT a, b, b IN (a, NULL) AS i, b NOT IN (a, NULL) AS ni \
               FROM t ORDER BY a, b";
    let actual = execute_to_batches(&ctx, sql).await;
    let expected = [
        "+---+---+------+-------+",
        "| a | b | i    | ni    |",
        "+---+---+------+-------+",
        "| 1 | 1 | true | false |",
        "| 1 | 5 |      |       |",
        "| 2 |   |      |       |",
        "|   | 3 |      |       |",
        "+---+---+------+-------+",
    ];
    assert_batches_eq!(expected, &actual);

    let sql = "SELECT a, b, 1 IN (b, 2) AS i, 1 NOT IN (a) AS ni FROM t ORDER BY a, b";
    let actual = execute_to_batches(&ctx, sql).await;
    let expected = [
        "+---+---+-------+-------+",
        "| a | b | i     | ni    |",
        "+---+---+-------+-------+",
        "| 1 | 1 | true  | false |",
        "| 1 | 5 | false | false |",
        "| 2 |   |       | true  |",
        "|   | 3 | false |       |",
        "+---+---+-------+-------+",
    ];
    assert_batches_eq!(expected, &actual);
    Ok(())
}

#[tokio::test]
async fn count_distinct_null_typed() -> Result<()> {
    let ctx = null_typed_ctx()?;
    let sql = "SELECT count(DISTINCT n) AS cn, count(DISTINCT a) AS ca \
               FROM (SELECT a, NULL AS n FROM t) s";
    let actual = execute_to_batches(&ctx, sql).await;
    let expected = [
        "+----+----+",
        "| cn | ca |",
        "+----+----+",
        "| 0  | 2  |",
        "+----+----+",
    ];
    assert_batches_eq!(expected, &actual);

    let sql = "SELECT a, count(DISTINCT n) AS cn, count(DISTINCT b) AS cb \
               FROM (SELECT a, b, NULL AS n FROM t) s GROUP BY a ORDER BY a";
    let actual = execute_to_batches(&ctx, sql).await;
    let expected = [
        "+---+----+----+",
        "| a | cn | cb |",
        "+---+----+----+",
        "| 1 | 0  | 2  |",
        "| 2 | 0  | 0  |",
        "|   | 0  | 1  |",
        "+---+----+----+",
    ];
    assert_batches_eq!(expected, &actual);
    Ok(())
}

#[tokio::test]
async fn min_max_null_typed() -> Result<()> {
    let ctx = null_typed_ctx()?;
    let sql = "SELECT min(n) AS mi, max(n) AS ma, min(a) AS mia, max(a) AS maa \
               FROM (SELECT a, NULL AS n FROM t) s";
    let actual = execute_to_batches(&ctx, sql).await;
    let expected = [
        "+----+----+-----+-----+",
        "| mi | ma | mia | maa |",
        "+----+----+-----+-----+",
        "|    |    | 1   | 2   |",
        "+----+----+-----+-----+",
    ];
    assert_batches_eq!(expected, &actual);

    let sql = "SELECT a, min(n) AS mi, max(n) AS ma \
               FROM (SELECT a, NULL AS n FROM t) s GROUP BY a ORDER BY a";
    let actual = execute_to_batches(&ctx, sql).await;
    let expected = [
        "+---+----+----+",
        "| a | mi | ma |",
        "+---+----+----+",
        "| 1 |    |    |",
        "| 2 |    |    |",
        "|   |    |    |",
        "+---+----+----+",
    ];
    assert_batches_eq!(expected, &actual);

    let sql = "SELECT a, min(n) OVER (PARTITION BY a) AS mi, \
               max(n) OVER (ORDER BY a) AS ma \
               FROM (SELECT a, NULL AS n FROM t) s ORDER BY a";
    let actual = execute_to_batches(&ctx, sql).await;
    let expected = [
        "+---+----+----+",
        "| a | mi | ma |",
        "+---+----+----+",
        "| 1 |    |    |",
        "| 1 |    |    |",
        "| 2 |    |    |",
        "|   |    |    |",
        "+---+----+----+",
    ];
    assert_batches_eq!(expected, &actual);
    Ok(())
}

#[tokio::test]
async fn all_null_case_in_aggregates_and_in_list() -> Result<()> {
    let ctx = null_typed_ctx()?;
    let sql = "SELECT count(DISTINCT CASE WHEN a > 100 THEN NULL END) AS cn, \
               min(CASE WHEN a > 100 THEN NULL END) AS mi, \
               max(CASE WHEN a > 100 THEN NULL END) AS ma, \
               count(DISTINCT a) AS ca FROM t";
    let actual = execute_to_batches(&ctx, sql).await;
    let expected = [
        "+----+----+----+----+",
        "| cn | mi | ma | ca |",
        "+----+----+----+----+",
        "| 0  |    |    | 2  |",
        "+----+----+----+----+",
    ];
    assert_batches_eq!(expected, &actual);

    let sql =
        "SELECT a, a IN (CASE WHEN a > 100 THEN NULL END, 1) AS i FROM t ORDER BY a";
    let actual = execute_to_batches(&ctx, sql).await;
    let expected = [
        "+---+------+",
        "| a | i    |",
        "+---+------+",
        "| 1 | true |",
        "| 1 | true |",
        "| 2 |      |",
        "|   |      |",
        "+---+------+",
    ];
    assert_batches_eq!(expected, &actual);
    Ok(())
}
