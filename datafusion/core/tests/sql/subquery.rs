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

    let expected = [
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

    let expected = [
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

    let expected = [
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

    let expected = [
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

    let expected = [
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

    let expected = [
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
async fn subquery_exists() -> Result<()> {
    let ctx = SessionContext::new();
    register_aggregate_simple_csv(&ctx).await?;

    let sql = "SELECT DISTINCT c1 FROM aggregate_simple o WHERE EXISTS(SELECT 1 FROM aggregate_simple p WHERE o.c1 * 2 = p.c1) ORDER BY c1";
    let actual = execute_to_batches(&ctx, sql).await;

    let expected = [
        "+---------+",
        "| c1      |",
        "+---------+",
        "| 0.00001 |",
        "| 0.00002 |",
        "+---------+",
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

    let expected = [
        "+---------+-------+",
        "| c1      | o.c2  |",
        "+---------+-------+",
        "| 0.00001 | 1e-12 |",
        "| 0.00002 | 2e-12 |",
        "+---------+-------+",
    ];
    assert_batches_eq!(expected, &actual);

    Ok(())
}
#[tokio::test]
async fn subquery_any() -> Result<()> {
    let ctx = SessionContext::new();
    register_aggregate_simple_csv(&ctx).await?;

    let sql = "SELECT DISTINCT c1 FROM aggregate_simple o WHERE c1 = ANY(SELECT c1 FROM aggregate_simple p WHERE c3) ORDER BY c1";
    let actual = execute_to_batches(&ctx, sql).await;

    let expected = [
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

#[tokio::test]
async fn subquery_all() -> Result<()> {
    let ctx = SessionContext::new();
    register_aggregate_simple_csv(&ctx).await?;

    let sql = "SELECT DISTINCT c1 FROM aggregate_simple o WHERE c1 > ALL(SELECT DISTINCT c1 FROM aggregate_simple p ORDER BY c1 LIMIT 3) ORDER BY c1";
    let actual = execute_to_batches(&ctx, sql).await;

    let expected = [
        "+---------+",
        "| c1      |",
        "+---------+",
        "| 0.00004 |",
        "| 0.00005 |",
        "+---------+",
    ];
    assert_batches_eq!(expected, &actual);

    Ok(())
}

#[tokio::test]
async fn subquery_in() -> Result<()> {
    let ctx = SessionContext::new();
    register_aggregate_simple_csv(&ctx).await?;

    let sql = "SELECT DISTINCT c1 FROM aggregate_simple o WHERE c1 IN (SELECT c1 FROM aggregate_simple p WHERE c3) ORDER BY c1";
    let actual = execute_to_batches(&ctx, sql).await;

    let expected = [
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

#[tokio::test]
async fn subquery_in_cte() -> Result<()> {
    let ctx = SessionContext::new();
    register_aggregate_simple_csv(&ctx).await?;

    let sql = "WITH cte as (SELECT c1 v FROM aggregate_simple where c3) SELECT DISTINCT c1 FROM aggregate_simple o WHERE c1 in (SELECT v FROM cte) ORDER BY c1";
    let actual = execute_to_batches(&ctx, sql).await;

    let expected = [
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
#[tokio::test]
async fn subquery_not_in_cte() -> Result<()> {
    let ctx = SessionContext::new();
    register_aggregate_simple_csv(&ctx).await?;

    let sql = "WITH cte as (SELECT c1 v FROM aggregate_simple where c3 = false) SELECT DISTINCT c1 FROM aggregate_simple o WHERE c1 not in (SELECT v FROM cte) ORDER BY c1";
    let actual = execute_to_batches(&ctx, sql).await;

    let expected = [
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

#[tokio::test]
async fn subquery_topn() -> Result<()> {
    let ctx = SessionContext::new();
    register_aggregate_simple_csv(&ctx).await?;

    let sql = "SELECT c3, ROUND(SUM(c1), 5) AS sum FROM aggregate_simple WHERE c3 IN (SELECT c3 FROM aggregate_simple GROUP BY c3 ORDER BY SUM(c1) DESC LIMIT 1) GROUP BY c3";
    let actual = execute_to_batches(&ctx, sql).await;

    let expected = [
        "+------+---------+",
        "| c3   | sum     |",
        "+------+---------+",
        "| true | 0.00035 |",
        "+------+---------+",
    ];
    assert_batches_eq!(expected, &actual);

    Ok(())
}

/// Table `t(a BIGINT, s TEXT)` with rows (1,'x'), (1,'y'), (2,'z'), (NULL,NULL)
/// spread over 2 partitions / 3 batches, so the outer query of a correlated
/// subquery runs several partitions concurrently.
fn register_multi_partition_table(ctx: &SessionContext) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, true),
        Field::new("s", DataType::Utf8, true),
    ]));
    let batch = |a: Vec<Option<i64>>, s: Vec<Option<&str>>| {
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(a)) as ArrayRef,
                Arc::new(StringArray::from(s)) as ArrayRef,
            ],
        )
    };
    let partitions = vec![
        vec![
            batch(vec![Some(1)], vec![Some("x")])?,
            batch(vec![Some(1), Some(2)], vec![Some("y"), Some("z")])?,
        ],
        vec![batch(vec![None], vec![None])?],
    ];
    ctx.register_table("t", Arc::new(MemTable::try_new(schema, partitions)?))?;
    Ok(())
}

#[tokio::test]
async fn subquery_correlated_exists_multi_partition() -> Result<()> {
    let ctx = SessionContext::new();
    register_multi_partition_table(&ctx)?;

    let sql = "SELECT a, s FROM t t1 WHERE EXISTS (SELECT 1 FROM t t2 WHERE t2.a = t1.a) ORDER BY a, s";
    let actual = execute_to_batches(&ctx, sql).await;

    let expected = [
        "+---+---+",
        "| a | s |",
        "+---+---+",
        "| 1 | x |",
        "| 1 | y |",
        "| 2 | z |",
        "+---+---+",
    ];
    assert_batches_eq!(expected, &actual);

    Ok(())
}

#[tokio::test]
async fn subquery_correlated_scalar_multi_partition() -> Result<()> {
    let ctx = SessionContext::new();
    register_multi_partition_table(&ctx)?;

    let sql = "SELECT a, s, (SELECT max(t2.s) FROM t t2 WHERE t2.a = t1.a) AS m FROM t t1 ORDER BY a, s";
    let actual = execute_to_batches(&ctx, sql).await;

    let expected = [
        "+---+---+---+",
        "| a | s | m |",
        "+---+---+---+",
        "| 1 | x | y |",
        "| 1 | y | y |",
        "| 2 | z | z |",
        "|   |   |   |",
        "+---+---+---+",
    ];
    assert_batches_eq!(expected, &actual);

    Ok(())
}

/// The subquery side yields several batches (one per input partition/batch);
/// a matching row in a batch other than the first must still be seen.
#[tokio::test]
async fn subquery_correlated_match_in_later_batch() -> Result<()> {
    let ctx = SessionContext::new();
    register_multi_partition_table(&ctx)?;

    let sql = "SELECT a, s, \
        EXISTS (SELECT 1 FROM t t2 WHERE t2.s = t1.s) AS e, \
        (SELECT t2.a FROM t t2 WHERE t2.s = t1.s) AS v \
        FROM t t1 ORDER BY a, s";
    let actual = execute_to_batches(&ctx, sql).await;

    let expected = [
        "+---+---+-------+---+",
        "| a | s | e     | v |",
        "+---+---+-------+---+",
        "| 1 | x | true  | 1 |",
        "| 1 | y | true  | 1 |",
        "| 2 | z | true  | 2 |",
        "|   |   | false |   |",
        "+---+---+-------+---+",
    ];
    assert_batches_eq!(expected, &actual);

    Ok(())
}

/// Rows of a scalar sub query may come in different batches; more than one row
/// in total is an error even when each batch holds only one.
#[tokio::test]
async fn subquery_correlated_scalar_rows_in_several_batches() -> Result<()> {
    let ctx = SessionContext::new();
    register_multi_partition_table(&ctx)?;

    // For a = 2 the sub query matches (2,'z') and (NULL,NULL), which sit in
    // different partitions
    let sql = "SELECT a, \
        (SELECT t2.s FROM t t2 WHERE t2.a = t1.a OR t2.a IS NULL) AS v \
        FROM t t1 WHERE t1.a = 2";
    let err = plan_and_collect(&ctx, sql).await.unwrap_err();
    assert_contains!(
        err.to_string(),
        "Sub query should return no more than one row"
    );

    Ok(())
}

#[tokio::test]
async fn subquery_correlated_any_values_in_several_batches() -> Result<()> {
    let ctx = SessionContext::new();
    register_multi_partition_table(&ctx)?;

    let sql = "SELECT a, s FROM t t1 \
        WHERE 'y' = ANY(SELECT t2.s FROM t t2 WHERE t2.a = t1.a) ORDER BY a, s";
    let actual = execute_to_batches(&ctx, sql).await;

    let expected = [
        "+---+---+",
        "| a | s |",
        "+---+---+",
        "| 1 | x |",
        "| 1 | y |",
        "+---+---+",
    ];
    assert_batches_eq!(expected, &actual);

    Ok(())
}

/// Outer partitions run concurrently and share the cursor the correlated sub
/// query reads the outer row from; one partition must not move the cursor of
/// another one (that read past the end of a shorter batch and panicked).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn subquery_correlated_concurrent_partitions() -> Result<()> {
    let ctx = SessionContext::new();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true)]));
    let batch = |a: Vec<Option<i64>>| {
        RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(a))])
    };
    let partitions = (0..8)
        .map(|p| {
            Ok(vec![
                batch((0..10).map(|i| Some(p * 10 + i)).collect())?,
                batch(vec![Some(p * 10)])?,
            ])
        })
        .collect::<Result<Vec<_>>>()?;
    ctx.register_table("t", Arc::new(MemTable::try_new(schema, partitions)?))?;

    let sql = "SELECT count(*) AS n, sum(a) AS s FROM t t1 \
        WHERE EXISTS (SELECT 1 FROM t t2 WHERE t2.a = t1.a)";
    let expected = [
        "+----+------+",
        "| n  | s    |",
        "+----+------+",
        "| 88 | 3440 |",
        "+----+------+",
    ];
    for _ in 0..20 {
        let actual = execute_to_batches(&ctx, sql).await;
        assert_batches_eq!(expected, &actual);
    }

    Ok(())
}
