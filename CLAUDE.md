# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fork of Apache Arrow DataFusion, an extensible query execution framework written in Rust that uses Apache Arrow as its in-memory format. This fork is maintained by Cube and includes custom extensions and optimizations.

## Key Commands

### Building
```bash
cargo build                          # Build the project
cargo build --release                # Build with optimizations
cargo build -p datafusion            # Build specific package
```

### Testing
```bash
# Setup test data (required before first test run)
git submodule init
git submodule update
export PARQUET_TEST_DATA=$(pwd)/parquet-testing/data/
export ARROW_TEST_DATA=$(pwd)/testing/data/

# Run tests
cargo test                           # Run all tests
cargo test -p datafusion             # Test specific package
cargo test test_name                 # Run specific test
cargo test -- --nocapture           # Show println! output during tests
```

### Formatting and Linting
```bash
cargo fmt                            # Format code
cargo fmt --check                    # Check formatting without changes
cargo clippy                         # Run linter
```

### Benchmarks
```bash
cargo bench                          # Run benchmarks
cargo bench -p datafusion            # Run datafusion benchmarks
```

## Architecture Overview

### Core Components

1. **Logical Planning** (`datafusion/src/logical_plan/`)
   - `LogicalPlan`: Represents logical query plans (SELECT, JOIN, etc.)
   - `Expr`: Expression trees for filters, projections, aggregations
   - `DFSchema`: Schema representation with field metadata
   - SQL parsing and planning in `datafusion/src/sql/`

2. **Physical Planning** (`datafusion/src/physical_plan/`)
   - `ExecutionPlan`: Physical execution operators
   - Expression implementations for actual computation
   - Aggregate functions with `Accumulator` trait
   - Custom operators for hash joins, sorts, aggregations

3. **Execution** (`datafusion/src/execution/`)
   - `ExecutionContext`: Main entry point for query execution
   - DataFrame API for programmatic query building
   - Manages memory, concurrency, and resource limits

4. **Optimizer** (`datafusion/src/optimizer/`)
   - Rule-based optimizer with passes like:
     - Predicate pushdown
     - Projection pushdown
     - Constant folding
     - Join reordering

5. **Cube Extensions** (`datafusion/src/cube_ext/`)
   - Custom operators and functions specific to Cube's fork
   - Performance optimizations including:
     - `GroupsAccumulator` for efficient grouped aggregation
     - `GroupsAccumulatorFlatAdapter` for flattened group values
     - Specialized join and aggregation implementations

### Key Design Patterns

- **Visitor Pattern**: Used extensively for traversing and transforming plans
- **Async/Await**: All execution is async using Tokio runtime
- **Arrow Arrays**: All data processing uses Arrow columnar format
- **Stream Processing**: Results are produced as async streams of RecordBatches

### Adding New Functionality

**Scalar Functions**: Implement in appropriate module under `physical_plan/`, register in `physical_plan/functions.rs`

**Aggregate Functions**: Create `Accumulator` implementation, register in `physical_plan/aggregates.rs`

**Optimizer Rules**: Implement `OptimizerRule` trait, add to optimizer pipeline

**Physical Operators**: Implement `ExecutionPlan` trait with proper partitioning and execution

## Important Notes

- This is a Cube fork with custom modifications - Ballista components are disabled
- The `cube_ext` module contains Cube-specific extensions and optimizations
- Performance-critical paths often have specialized implementations for primitive types
- Always run tests with proper test data environment variables set