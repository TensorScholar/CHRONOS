//! Temporal Query Optimization Engine
//!
//! This module implements a revolutionary temporal query optimization system with
//! cost-based query planning, information-theoretic selectivity estimation, and
//! category-theoretic query algebra for efficient temporal debugging operations.
//!
//! Copyright (c) 2025 Mohammad Atashi <mohammadaliatashi@icloud.com>

use crate::algorithm::state::AlgorithmState;
use crate::temporal::{StateManager, StateDelta, ExecutionSignature};
use nalgebra::{DVector, DMatrix};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BTreeMap, BTreeSet, VecDeque};
use std::sync::{Arc, RwLock, Mutex};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};
use thiserror::Error;

/// Temporal query optimization errors
#[derive(Debug, Error)]
pub enum QueryOptimizationError {
    #[error("Invalid query syntax: {reason}")]
    InvalidQuerySyntax { reason: String },
    
    #[error("Cost estimation failed: {reason}")]
    CostEstimationError { reason: String },
    
    #[error("Query compilation error: {reason}")]
    CompilationError { reason: String },
    
    #[error("Execution plan generation failed: {reason}")]
    PlanGenerationError { reason: String },
    
    #[error("Statistics collection error: {reason}")]
    StatisticsError { reason: String },
    
    #[error("Index optimization error: {reason}")]
    IndexOptimizationError { reason: String },
    
    #[error("Parallel execution error: {reason}")]
    ParallelExecutionError { reason: String },
}

/// Temporal query abstract syntax tree with category-theoretic structure
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TemporalQuery {
    /// Retrieve states within time range
    TimeRange {
        start_time: u64,
        end_time: u64,
        step_granularity: Option<u64>,
    },
    
    /// Filter states by algorithm properties
    StateFilter {
        predicate: StatePredicate,
        query: Box<TemporalQuery>,
    },
    
    /// Project specific state attributes
    Projection {
        attributes: Vec<String>,
        query: Box<TemporalQuery>,
    },
    
    /// Join temporal sequences
    TemporalJoin {
        left: Box<TemporalQuery>,
        right: Box<TemporalQuery>,
        join_condition: JoinCondition,
    },
    
    /// Aggregate temporal data
    Aggregation {
        query: Box<TemporalQuery>,
        aggregation_functions: Vec<AggregationFunction>,
        group_by: Vec<String>,
    },
    
    /// Window-based temporal operations
    WindowQuery {
        query: Box<TemporalQuery>,
        window_spec: WindowSpecification,
    },
    
    /// Causal dependency queries
    CausalQuery {
        source_states: Vec<u64>,
        dependency_type: CausalDependencyType,
        max_depth: usize,
    },
    
    /// Counterfactual analysis queries
    CounterfactualQuery {
        base_timeline: u64,
        modifications: Vec<StateModification>,
        analysis_window: (u64, u64),
    },
}

/// State predicate for filtering operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StatePredicate {
    /// Attribute comparison
    AttributeComparison {
        attribute: String,
        operator: ComparisonOperator,
        value: PredicateValue,
    },
    
    /// Logical conjunction
    And(Box<StatePredicate>, Box<StatePredicate>),
    
    /// Logical disjunction
    Or(Box<StatePredicate>, Box<StatePredicate>),
    
    /// Logical negation
    Not(Box<StatePredicate>),
    
    /// Complex property evaluation
    PropertyEvaluation {
        property_name: String,
        evaluator: String, // Serialized evaluator function
    },
}

/// Comparison operators for predicates
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComparisonOperator {
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    Contains,
    StartsWith,
    EndsWith,
    Matches, // Regular expression matching
}

/// Predicate values with type information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PredicateValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    List(Vec<PredicateValue>),
}

/// Join conditions for temporal sequences
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum JoinCondition {
    /// Temporal overlap join
    TemporalOverlap { tolerance: Duration },
    
    /// Causal dependency join
    CausalDependency,
    
    /// Attribute-based join
    AttributeJoin {
        left_attribute: String,
        right_attribute: String,
        operator: ComparisonOperator,
    },
    
    /// Custom join condition
    CustomCondition {
        condition_name: String,
        parameters: HashMap<String, PredicateValue>,
    },
}

/// Aggregation functions for temporal data
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AggregationFunction {
    Count,
    Sum(String), // Attribute name
    Average(String),
    Min(String),
    Max(String),
    StandardDeviation(String),
    Median(String),
    Percentile(String, f64), // Attribute name, percentile
    TimeWeightedAverage(String),
    First(String),
    Last(String),
}

/// Window specifications for temporal operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WindowSpecification {
    /// Window type
    pub window_type: WindowType,
    
    /// Window size
    pub size: WindowSize,
    
    /// Window slide interval
    pub slide: Option<Duration>,
    
    /// Ordering specification
    pub order_by: Vec<OrderByClause>,
}

/// Window types for temporal operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WindowType {
    /// Tumbling window (non-overlapping)
    Tumbling,
    
    /// Sliding window (overlapping)
    Sliding,
    
    /// Session window (gap-based)
    Session { gap_duration: Duration },
    
    /// Landmark window (from beginning)
    Landmark,
}

/// Window size specifications
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WindowSize {
    /// Time-based window
    TimeBased(Duration),
    
    /// Count-based window
    CountBased(usize),
    
    /// Condition-based window
    ConditionBased(StatePredicate),
}

/// Order by clause for sorting
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderByClause {
    pub attribute: String,
    pub direction: SortDirection,
    pub nulls_handling: NullsHandling,
}

/// Sort direction
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SortDirection {
    Ascending,
    Descending,
}

/// Null handling for sorting
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NullsHandling {
    NullsFirst,
    NullsLast,
}

/// Causal dependency types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CausalDependencyType {
    /// Direct causal dependencies
    Direct,
    
    /// Transitive causal dependencies
    Transitive,
    
    /// Probabilistic causal dependencies
    Probabilistic { threshold: f64 },
    
    /// Structural causal dependencies
    Structural,
}

/// State modification for counterfactual analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateModification {
    pub target_state: u64,
    pub modification_type: ModificationType,
    pub parameters: HashMap<String, PredicateValue>,
}

/// Types of state modifications
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModificationType {
    /// Modify algorithm parameters
    ParameterModification,
    
    /// Inject decision override
    DecisionOverride,
    
    /// Modify heuristic function
    HeuristicModification,
    
    /// Insert artificial delay
    DelayInjection,
    
    /// Modify data structure state
    DataStructureModification,
}

/// Query execution plan with cost estimates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryExecutionPlan {
    /// Root execution node
    pub root: ExecutionNode,
    
    /// Estimated total cost
    pub estimated_cost: f64,
    
    /// Estimated execution time
    pub estimated_time: Duration,
    
    /// Memory requirements
    pub memory_estimate: usize,
    
    /// Parallelization strategy
    pub parallelization: ParallelizationStrategy,
    
    /// Index usage plan
    pub index_usage: Vec<IndexUsagePlan>,
    
    /// Statistics used in optimization
    pub statistics_snapshot: StatisticsSnapshot,
}

/// Execution plan node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionNode {
    /// Sequential scan of temporal data
    SequentialScan {
        source: DataSource,
        filter: Option<StatePredicate>,
        cost: f64,
        selectivity: f64,
    },
    
    /// Index-based access
    IndexScan {
        index_name: String,
        key_conditions: Vec<StatePredicate>,
        cost: f64,
        selectivity: f64,
    },
    
    /// Filter operation
    Filter {
        input: Box<ExecutionNode>,
        predicate: StatePredicate,
        cost: f64,
        selectivity: f64,
    },
    
    /// Projection operation
    Project {
        input: Box<ExecutionNode>,
        attributes: Vec<String>,
        cost: f64,
    },
    
    /// Join operation
    Join {
        left: Box<ExecutionNode>,
        right: Box<ExecutionNode>,
        join_type: JoinType,
        condition: JoinCondition,
        cost: f64,
    },
    
    /// Aggregation operation
    Aggregate {
        input: Box<ExecutionNode>,
        functions: Vec<AggregationFunction>,
        group_by: Vec<String>,
        cost: f64,
    },
    
    /// Sort operation
    Sort {
        input: Box<ExecutionNode>,
        order_by: Vec<OrderByClause>,
        cost: f64,
    },
    
    /// Window operation
    Window {
        input: Box<ExecutionNode>,
        window_spec: WindowSpecification,
        cost: f64,
    },
}

/// Join types for execution planning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JoinType {
    Inner,
    LeftOuter,
    RightOuter,
    FullOuter,
    Cross,
    Semi,
    Anti,
}

/// Data sources for query execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSource {
    /// Temporal state store
    StateStore {
        time_range: Option<(u64, u64)>,
        estimated_rows: usize,
    },
    
    /// Execution trace store
    TraceStore {
        trace_id: String,
        estimated_rows: usize,
    },
    
    /// Indexed temporal data
    IndexedData {
        index_name: String,
        estimated_rows: usize,
    },
    
    /// Materialized view
    MaterializedView {
        view_name: String,
        freshness: Duration,
        estimated_rows: usize,
    },
}

/// Parallelization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParallelizationStrategy {
    /// No parallelization
    Sequential,
    
    /// Data parallelism (partition by time)
    DataParallel { partitions: usize },
    
    /// Pipeline parallelism
    Pipeline { stages: usize },
    
    /// Hybrid parallelization
    Hybrid {
        data_partitions: usize,
        pipeline_stages: usize,
    },
}

/// Index usage plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexUsagePlan {
    pub index_name: String,
    pub usage_type: IndexUsageType,
    pub selectivity: f64,
    pub cost_reduction: f64,
}

/// Index usage types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexUsageType {
    /// Full index scan
    FullScan,
    
    /// Range scan
    RangeScan { start_key: String, end_key: String },
    
    /// Point lookup
    PointLookup { key: String },
    
    /// Multi-point lookup
    MultiPointLookup { keys: Vec<String> },
}

/// Statistics snapshot for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticsSnapshot {
    /// Table/source statistics
    pub table_stats: HashMap<String, TableStatistics>,
    
    /// Column/attribute statistics
    pub column_stats: HashMap<String, ColumnStatistics>,
    
    /// Index statistics
    pub index_stats: HashMap<String, IndexStatistics>,
    
    /// Query workload statistics
    pub workload_stats: WorkloadStatistics,
    
    /// Timestamp of statistics collection
    pub timestamp: u64,
}

/// Table-level statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableStatistics {
    pub row_count: usize,
    pub average_row_size: usize,
    pub data_size: usize,
    pub last_updated: u64,
    pub growth_rate: f64, // Rows per second
}

/// Column-level statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnStatistics {
    pub distinct_values: usize,
    pub null_fraction: f64,
    pub most_common_values: Vec<(PredicateValue, f64)>, // (value, frequency)
    pub histogram: Vec<HistogramBucket>,
    pub correlation: f64, // With row order
}

/// Histogram bucket for value distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramBucket {
    pub lower_bound: PredicateValue,
    pub upper_bound: PredicateValue,
    pub frequency: f64,
    pub distinct_values: usize,
}

/// Index-level statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStatistics {
    pub index_type: IndexType,
    pub size: usize,
    pub height: usize, // For tree indexes
    pub leaf_pages: usize,
    pub selectivity: f64,
    pub clustering_factor: f64,
}

/// Index types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexType {
    BTree,
    Hash,
    Bitmap,
    Temporal, // Specialized temporal index
    Spatial,  // For geometric queries
}

/// Workload statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadStatistics {
    pub query_frequency: HashMap<String, f64>, // Query pattern -> frequency
    pub average_response_time: Duration,
    pub peak_concurrency: usize,
    pub resource_utilization: ResourceUtilization,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub io_utilization: f64,
    pub network_utilization: f64,
}

/// Cost model for query optimization
pub struct CostModel {
    /// Base costs for different operations
    operation_costs: HashMap<String, f64>,
    
    /// I/O cost parameters
    io_cost_params: IOCostParameters,
    
    /// CPU cost parameters
    cpu_cost_params: CPUCostParameters,
    
    /// Memory cost parameters
    memory_cost_params: MemoryCostParameters,
    
    /// Network cost parameters (for distributed queries)
    network_cost_params: NetworkCostParameters,
}

/// I/O cost parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IOCostParameters {
    pub sequential_read_cost: f64,
    pub random_read_cost: f64,
    pub write_cost: f64,
    pub page_size: usize,
    pub cache_hit_ratio: f64,
}

/// CPU cost parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CPUCostParameters {
    pub tuple_processing_cost: f64,
    pub comparison_cost: f64,
    pub hash_cost: f64,
    pub sort_cost_per_element: f64,
    pub function_call_cost: f64,
}

/// Memory cost parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCostParameters {
    pub memory_allocation_cost: f64,
    pub memory_access_cost: f64,
    pub cache_miss_penalty: f64,
    pub available_memory: usize,
}

/// Network cost parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkCostParameters {
    pub latency: Duration,
    pub bandwidth: f64, // MB/s
    pub packet_cost: f64,
}

impl CostModel {
    /// Create default cost model with calibrated parameters
    pub fn new() -> Self {
        let mut operation_costs = HashMap::new();
        operation_costs.insert("sequential_scan".to_string(), 1.0);
        operation_costs.insert("index_scan".to_string(), 0.1);
        operation_costs.insert("filter".to_string(), 0.01);
        operation_costs.insert("project".to_string(), 0.005);
        operation_costs.insert("join".to_string(), 2.0);
        operation_costs.insert("aggregate".to_string(), 1.5);
        operation_costs.insert("sort".to_string(), 3.0);
        operation_costs.insert("window".to_string(), 2.5);
        
        Self {
            operation_costs,
            io_cost_params: IOCostParameters {
                sequential_read_cost: 0.1,
                random_read_cost: 1.0,
                write_cost: 1.2,
                page_size: 4096,
                cache_hit_ratio: 0.9,
            },
            cpu_cost_params: CPUCostParameters {
                tuple_processing_cost: 0.001,
                comparison_cost: 0.0001,
                hash_cost: 0.0005,
                sort_cost_per_element: 0.01,
                function_call_cost: 0.0002,
            },
            memory_cost_params: MemoryCostParameters {
                memory_allocation_cost: 0.01,
                memory_access_cost: 0.0001,
                cache_miss_penalty: 0.1,
                available_memory: 1024 * 1024 * 1024, // 1GB
            },
            network_cost_params: NetworkCostParameters {
                latency: Duration::from_millis(1),
                bandwidth: 1000.0, // MB/s
                packet_cost: 0.001,
            },
        }
    }
    
    /// Estimate cost of execution node
    pub fn estimate_cost(
        &self,
        node: &ExecutionNode,
        statistics: &StatisticsSnapshot,
    ) -> Result<f64, QueryOptimizationError> {
        match node {
            ExecutionNode::SequentialScan { source, filter, .. } => {
                let base_cost = self.estimate_scan_cost(source, statistics)?;
                let filter_cost = if let Some(pred) = filter {
                    self.estimate_filter_cost(pred, statistics)?
                } else {
                    0.0
                };
                Ok(base_cost + filter_cost)
            },
            
            ExecutionNode::IndexScan { index_name, key_conditions, .. } => {
                self.estimate_index_scan_cost(index_name, key_conditions, statistics)
            },
            
            ExecutionNode::Filter { input, predicate, .. } => {
                let input_cost = self.estimate_cost(input, statistics)?;
                let filter_cost = self.estimate_filter_cost(predicate, statistics)?;
                Ok(input_cost + filter_cost)
            },
            
            ExecutionNode::Project { input, attributes, .. } => {
                let input_cost = self.estimate_cost(input, statistics)?;
                let project_cost = attributes.len() as f64 * self.cpu_cost_params.tuple_processing_cost;
                Ok(input_cost + project_cost)
            },
            
            ExecutionNode::Join { left, right, join_type, .. } => {
                let left_cost = self.estimate_cost(left, statistics)?;
                let right_cost = self.estimate_cost(right, statistics)?;
                let join_cost = self.estimate_join_cost(left, right, join_type, statistics)?;
                Ok(left_cost + right_cost + join_cost)
            },
            
            ExecutionNode::Aggregate { input, functions, .. } => {
                let input_cost = self.estimate_cost(input, statistics)?;
                let agg_cost = functions.len() as f64 * self.operation_costs["aggregate"];
                Ok(input_cost + agg_cost)
            },
            
            ExecutionNode::Sort { input, order_by, .. } => {
                let input_cost = self.estimate_cost(input, statistics)?;
                let estimated_rows = self.estimate_cardinality(input, statistics)?;
                let sort_cost = estimated_rows as f64 * 
                    (estimated_rows as f64).log2() * 
                    order_by.len() as f64 * 
                    self.cpu_cost_params.sort_cost_per_element;
                Ok(input_cost + sort_cost)
            },
            
            ExecutionNode::Window { input, window_spec, .. } => {
                let input_cost = self.estimate_cost(input, statistics)?;
                let window_cost = self.estimate_window_cost(window_spec, statistics)?;
                Ok(input_cost + window_cost)
            },
        }
    }
    
    /// Estimate scan cost for data source
    fn estimate_scan_cost(
        &self,
        source: &DataSource,
        _statistics: &StatisticsSnapshot,
    ) -> Result<f64, QueryOptimizationError> {
        match source {
            DataSource::StateStore { estimated_rows, .. } => {
                Ok(*estimated_rows as f64 * self.io_cost_params.sequential_read_cost)
            },
            DataSource::TraceStore { estimated_rows, .. } => {
                Ok(*estimated_rows as f64 * self.io_cost_params.sequential_read_cost)
            },
            DataSource::IndexedData { estimated_rows, .. } => {
                Ok(*estimated_rows as f64 * self.io_cost_params.random_read_cost * 0.1)
            },
            DataSource::MaterializedView { estimated_rows, .. } => {
                Ok(*estimated_rows as f64 * self.io_cost_params.sequential_read_cost * 0.5)
            },
        }
    }
    
    /// Estimate filter cost
    fn estimate_filter_cost(
        &self,
        predicate: &StatePredicate,
        _statistics: &StatisticsSnapshot,
    ) -> Result<f64, QueryOptimizationError> {
        match predicate {
            StatePredicate::AttributeComparison { .. } => {
                Ok(self.cpu_cost_params.comparison_cost)
            },
            StatePredicate::And(left, right) => {
                let left_cost = self.estimate_filter_cost(left, _statistics)?;
                let right_cost = self.estimate_filter_cost(right, _statistics)?;
                Ok(left_cost + right_cost)
            },
            StatePredicate::Or(left, right) => {
                let left_cost = self.estimate_filter_cost(left, _statistics)?;
                let right_cost = self.estimate_filter_cost(right, _statistics)?;
                Ok(left_cost + right_cost)
            },
            StatePredicate::Not(inner) => {
                self.estimate_filter_cost(inner, _statistics)
            },
            StatePredicate::PropertyEvaluation { .. } => {
                Ok(self.cpu_cost_params.function_call_cost * 10.0) // Complex evaluation
            },
        }
    }
    
    /// Estimate index scan cost
    fn estimate_index_scan_cost(
        &self,
        index_name: &str,
        _key_conditions: &[StatePredicate],
        statistics: &StatisticsSnapshot,
    ) -> Result<f64, QueryOptimizationError> {
        if let Some(index_stats) = statistics.index_stats.get(index_name) {
            let selectivity = index_stats.selectivity;
            let base_cost = match index_stats.index_type {
                IndexType::BTree => self.io_cost_params.random_read_cost * (index_stats.height as f64).log2(),
                IndexType::Hash => self.io_cost_params.random_read_cost,
                IndexType::Bitmap => self.io_cost_params.sequential_read_cost * 0.1,
                IndexType::Temporal => self.io_cost_params.random_read_cost * 0.5,
                IndexType::Spatial => self.io_cost_params.random_read_cost * 2.0,
            };
            Ok(base_cost * selectivity)
        } else {
            Err(QueryOptimizationError::StatisticsError {
                reason: format!("No statistics available for index: {}", index_name),
            })
        }
    }
    
    /// Estimate join cost
    fn estimate_join_cost(
        &self,
        left: &ExecutionNode,
        right: &ExecutionNode,
        join_type: &JoinType,
        statistics: &StatisticsSnapshot,
    ) -> Result<f64, QueryOptimizationError> {
        let left_cardinality = self.estimate_cardinality(left, statistics)?;
        let right_cardinality = self.estimate_cardinality(right, statistics)?;
        
        let base_cost = match join_type {
            JoinType::Inner | JoinType::LeftOuter | JoinType::RightOuter | JoinType::FullOuter => {
                // Hash join cost estimation
                (left_cardinality + right_cardinality) as f64 * self.cpu_cost_params.hash_cost
            },
            JoinType::Cross => {
                // Cartesian product
                (left_cardinality * right_cardinality) as f64 * self.cpu_cost_params.tuple_processing_cost
            },
            JoinType::Semi | JoinType::Anti => {
                // Semi-join cost
                left_cardinality as f64 * (right_cardinality as f64).log2() * self.cpu_cost_params.comparison_cost
            },
        };
        
        Ok(base_cost)
    }
    
    /// Estimate window operation cost
    fn estimate_window_cost(
        &self,
        window_spec: &WindowSpecification,
        _statistics: &StatisticsSnapshot,
    ) -> Result<f64, QueryOptimizationError> {
        let base_cost = match &window_spec.window_type {
            WindowType::Tumbling => self.operation_costs["window"] * 0.8,
            WindowType::Sliding => self.operation_costs["window"] * 1.2,
            WindowType::Session { .. } => self.operation_costs["window"] * 1.5,
            WindowType::Landmark => self.operation_costs["window"] * 2.0,
        };
        
        let sort_cost = window_spec.order_by.len() as f64 * self.cpu_cost_params.sort_cost_per_element;
        
        Ok(base_cost + sort_cost)
    }
    
    /// Estimate cardinality of execution node
    fn estimate_cardinality(
        &self,
        node: &ExecutionNode,
        statistics: &StatisticsSnapshot,
    ) -> Result<usize, QueryOptimizationError> {
        match node {
            ExecutionNode::SequentialScan { source, filter, selectivity, .. } => {
                let base_cardinality = match source {
                    DataSource::StateStore { estimated_rows, .. } => *estimated_rows,
                    DataSource::TraceStore { estimated_rows, .. } => *estimated_rows,
                    DataSource::IndexedData { estimated_rows, .. } => *estimated_rows,
                    DataSource::MaterializedView { estimated_rows, .. } => *estimated_rows,
                };
                
                let final_selectivity = if filter.is_some() { *selectivity } else { 1.0 };
                Ok((base_cardinality as f64 * final_selectivity) as usize)
            },
            
            ExecutionNode::IndexScan { selectivity, .. } => {
                // Assume a base table size and apply selectivity
                Ok((10000.0 * selectivity) as usize) // Placeholder
            },
            
            ExecutionNode::Filter { input, selectivity, .. } => {
                let input_cardinality = self.estimate_cardinality(input, statistics)?;
                Ok((input_cardinality as f64 * selectivity) as usize)
            },
            
            ExecutionNode::Project { input, .. } => {
                self.estimate_cardinality(input, statistics)
            },
            
            ExecutionNode::Join { left, right, join_type, .. } => {
                let left_cardinality = self.estimate_cardinality(left, statistics)?;
                let right_cardinality = self.estimate_cardinality(right, statistics)?;
                
                match join_type {
                    JoinType::Inner => Ok((left_cardinality * right_cardinality) / 10), // Assume 10% join selectivity
                    JoinType::LeftOuter => Ok(left_cardinality),
                    JoinType::RightOuter => Ok(right_cardinality),
                    JoinType::FullOuter => Ok(left_cardinality + right_cardinality),
                    JoinType::Cross => Ok(left_cardinality * right_cardinality),
                    JoinType::Semi => Ok(left_cardinality / 2), // Approximate
                    JoinType::Anti => Ok(left_cardinality / 2), // Approximate
                }
            },
            
            ExecutionNode::Aggregate { input, group_by, .. } => {
                let input_cardinality = self.estimate_cardinality(input, statistics)?;
                if group_by.is_empty() {
                    Ok(1) // Single aggregate result
                } else {
                    Ok(input_cardinality / 10) // Assume 10x reduction from grouping
                }
            },
            
            ExecutionNode::Sort { input, .. } => {
                self.estimate_cardinality(input, statistics)
            },
            
            ExecutionNode::Window { input, .. } => {
                self.estimate_cardinality(input, statistics)
            },
        }
    }
}

/// Query optimizer with cost-based optimization
pub struct TemporalQueryOptimizer {
    /// Cost model for optimization
    cost_model: Arc<CostModel>,
    
    /// Statistics collector and manager
    statistics_manager: Arc<RwLock<StatisticsManager>>,
    
    /// Index manager for optimization
    index_manager: Arc<RwLock<IndexManager>>,
    
    /// Query plan cache
    plan_cache: Arc<RwLock<HashMap<u64, QueryExecutionPlan>>>,
    
    /// Optimization configuration
    config: OptimizerConfig,
    
    /// Query execution statistics
    execution_stats: Arc<RwLock<QueryExecutionStats>>,
}

/// Statistics manager for collecting and maintaining query statistics
pub struct StatisticsManager {
    /// Current statistics snapshot
    current_stats: StatisticsSnapshot,
    
    /// Statistics update frequency
    update_frequency: Duration,
    
    /// Last update timestamp
    last_update: Instant,
    
    /// Statistics collection thread handle
    collection_handle: Option<std::thread::JoinHandle<()>>,
}

/// Index manager for optimization
pub struct IndexManager {
    /// Available indexes
    indexes: HashMap<String, IndexDefinition>,
    
    /// Index usage statistics
    usage_stats: HashMap<String, IndexUsageStats>,
    
    /// Automatic index recommendations
    recommendations: Vec<IndexRecommendation>,
}

/// Index definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexDefinition {
    pub name: String,
    pub table: String,
    pub columns: Vec<String>,
    pub index_type: IndexType,
    pub unique: bool,
    pub partial_condition: Option<StatePredicate>,
}

/// Index usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexUsageStats {
    pub access_count: u64,
    pub last_used: u64,
    pub selectivity_distribution: Vec<f64>,
    pub maintenance_cost: f64,
}

/// Index recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexRecommendation {
    pub recommended_index: IndexDefinition,
    pub estimated_benefit: f64,
    pub creation_cost: f64,
    pub maintenance_cost: f64,
    pub confidence: f64,
}

/// Optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Maximum optimization time
    pub max_optimization_time: Duration,
    
    /// Cost improvement threshold for plan selection
    pub cost_improvement_threshold: f64,
    
    /// Enable/disable specific optimizations
    pub enable_index_optimization: bool,
    pub enable_join_reordering: bool,
    pub enable_predicate_pushdown: bool,
    pub enable_projection_pushdown: bool,
    pub enable_materialized_view_matching: bool,
    
    /// Parallelization preferences
    pub prefer_parallel_execution: bool,
    pub max_parallel_workers: usize,
    
    /// Memory budget for query execution
    pub memory_budget: usize,
    
    /// Statistics staleness tolerance
    pub statistics_staleness_tolerance: Duration,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            max_optimization_time: Duration::from_millis(100),
            cost_improvement_threshold: 0.1, // 10% improvement required
            enable_index_optimization: true,
            enable_join_reordering: true,
            enable_predicate_pushdown: true,
            enable_projection_pushdown: true,
            enable_materialized_view_matching: true,
            prefer_parallel_execution: true,
            max_parallel_workers: num_cpus::get(),
            memory_budget: 1024 * 1024 * 1024, // 1GB
            statistics_staleness_tolerance: Duration::from_minutes(5),
        }
    }
}

/// Query execution statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct QueryExecutionStats {
    pub total_queries: u64,
    pub optimization_time_total: Duration,
    pub execution_time_total: Duration,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub parallel_queries: u64,
    pub index_usage_count: HashMap<String, u64>,
    pub average_plan_cost: f64,
    pub cost_estimation_accuracy: Vec<f64>,
}

impl TemporalQueryOptimizer {
    /// Create new temporal query optimizer
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            cost_model: Arc::new(CostModel::new()),
            statistics_manager: Arc::new(RwLock::new(StatisticsManager::new())),
            index_manager: Arc::new(RwLock::new(IndexManager::new())),
            plan_cache: Arc::new(RwLock::new(HashMap::new())),
            config,
            execution_stats: Arc::new(RwLock::new(QueryExecutionStats::default())),
        }
    }
    
    /// Optimize temporal query and generate execution plan
    pub fn optimize_query(
        &self,
        query: &TemporalQuery,
    ) -> Result<QueryExecutionPlan, QueryOptimizationError> {
        let optimization_start = Instant::now();
        
        // Check plan cache first
        let query_hash = self.hash_query(query);
        if let Some(cached_plan) = self.plan_cache.read().unwrap().get(&query_hash) {
            let mut stats = self.execution_stats.write().unwrap();
            stats.cache_hits += 1;
            return Ok(cached_plan.clone());
        }
        
        // Get current statistics
        let statistics = self.statistics_manager.read().unwrap().current_stats.clone();
        
        // Generate initial execution plan
        let mut best_plan = self.generate_initial_plan(query, &statistics)?;
        
        // Apply optimization transformations
        if self.config.enable_predicate_pushdown {
            best_plan = self.apply_predicate_pushdown(best_plan, &statistics)?;
        }
        
        if self.config.enable_projection_pushdown {
            best_plan = self.apply_projection_pushdown(best_plan, &statistics)?;
        }
        
        if self.config.enable_join_reordering {
            best_plan = self.apply_join_reordering(best_plan, &statistics)?;
        }
        
        if self.config.enable_index_optimization {
            best_plan = self.apply_index_optimization(best_plan, &statistics)?;
        }
        
        // Determine parallelization strategy
        best_plan.parallelization = self.determine_parallelization_strategy(&best_plan, &statistics)?;
        
        // Cache the optimized plan
        self.plan_cache.write().unwrap().insert(query_hash, best_plan.clone());
        
        // Update statistics
        let optimization_time = optimization_start.elapsed();
        let mut stats = self.execution_stats.write().unwrap();
        stats.total_queries += 1;
        stats.optimization_time_total += optimization_time;
        stats.cache_misses += 1;
        stats.average_plan_cost = (stats.average_plan_cost * (stats.total_queries - 1) as f64 + best_plan.estimated_cost) / stats.total_queries as f64;
        
        log::debug!(
            "Query optimization completed in {:?}, estimated cost: {:.2}",
            optimization_time,
            best_plan.estimated_cost
        );
        
        Ok(best_plan)
    }
    
    /// Generate initial execution plan from query
    fn generate_initial_plan(
        &self,
        query: &TemporalQuery,
        statistics: &StatisticsSnapshot,
    ) -> Result<QueryExecutionPlan, QueryOptimizationError> {
        let root_node = self.query_to_execution_node(query, statistics)?;
        let estimated_cost = self.cost_model.estimate_cost(&root_node, statistics)?;
        let estimated_time = Duration::from_millis((estimated_cost * 10.0) as u64); // Rough estimate
        let memory_estimate = self.estimate_memory_usage(&root_node, statistics)?;
        
        Ok(QueryExecutionPlan {
            root: root_node,
            estimated_cost,
            estimated_time,
            memory_estimate,
            parallelization: ParallelizationStrategy::Sequential,
            index_usage: Vec::new(),
            statistics_snapshot: statistics.clone(),
        })
    }
    
    /// Convert query to execution node
    fn query_to_execution_node(
        &self,
        query: &TemporalQuery,
        statistics: &StatisticsSnapshot,
    ) -> Result<ExecutionNode, QueryOptimizationError> {
        match query {
            TemporalQuery::TimeRange { start_time, end_time, step_granularity } => {
                let estimated_rows = self.estimate_time_range_rows(*start_time, *end_time, statistics)?;
                Ok(ExecutionNode::SequentialScan {
                    source: DataSource::StateStore {
                        time_range: Some((*start_time, *end_time)),
                        estimated_rows,
                    },
                    filter: None,
                    cost: 0.0, // Will be calculated by cost model
                    selectivity: 1.0,
                })
            },
            
            TemporalQuery::StateFilter { predicate, query } => {
                let input = self.query_to_execution_node(query, statistics)?;
                let selectivity = self.estimate_predicate_selectivity(predicate, statistics)?;
                
                Ok(ExecutionNode::Filter {
                    input: Box::new(input),
                    predicate: predicate.clone(),
                    cost: 0.0, // Will be calculated by cost model
                    selectivity,
                })
            },
            
            TemporalQuery::Projection { attributes, query } => {
                let input = self.query_to_execution_node(query, statistics)?;
                
                Ok(ExecutionNode::Project {
                    input: Box::new(input),
                    attributes: attributes.clone(),
                    cost: 0.0, // Will be calculated by cost model
                })
            },
            
            TemporalQuery::TemporalJoin { left, right, join_condition } => {
                let left_node = self.query_to_execution_node(left, statistics)?;
                let right_node = self.query_to_execution_node(right, statistics)?;
                
                Ok(ExecutionNode::Join {
                    left: Box::new(left_node),
                    right: Box::new(right_node),
                    join_type: JoinType::Inner, // Default, could be inferred from condition
                    condition: join_condition.clone(),
                    cost: 0.0, // Will be calculated by cost model
                })
            },
            
            TemporalQuery::Aggregation { query, aggregation_functions, group_by } => {
                let input = self.query_to_execution_node(query, statistics)?;
                
                Ok(ExecutionNode::Aggregate {
                    input: Box::new(input),
                    functions: aggregation_functions.clone(),
                    group_by: group_by.clone(),
                    cost: 0.0, // Will be calculated by cost model
                })
            },
            
            TemporalQuery::WindowQuery { query, window_spec } => {
                let input = self.query_to_execution_node(query, statistics)?;
                
                Ok(ExecutionNode::Window {
                    input: Box::new(input),
                    window_spec: window_spec.clone(),
                    cost: 0.0, // Will be calculated by cost model
                })
            },
            
            TemporalQuery::CausalQuery { source_states, dependency_type, max_depth } => {
                // Convert to specialized scan with causal filtering
                let estimated_rows = source_states.len() * (*max_depth + 1) * 10; // Rough estimate
                
                Ok(ExecutionNode::SequentialScan {
                    source: DataSource::StateStore {
                        time_range: None,
                        estimated_rows,
                    },
                    filter: Some(StatePredicate::PropertyEvaluation {
                        property_name: "causal_dependency".to_string(),
                        evaluator: format!("causal_{:?}_depth_{}", dependency_type, max_depth),
                    }),
                    cost: 0.0,
                    selectivity: 0.1, // Causal queries are typically selective
                })
            },
            
            TemporalQuery::CounterfactualQuery { base_timeline, modifications, analysis_window } => {
                // Convert to specialized scan with counterfactual analysis
                let estimated_rows = (analysis_window.1 - analysis_window.0) as usize * modifications.len();
                
                Ok(ExecutionNode::SequentialScan {
                    source: DataSource::StateStore {
                        time_range: Some((analysis_window.0, analysis_window.1)),
                        estimated_rows,
                    },
                    filter: Some(StatePredicate::PropertyEvaluation {
                        property_name: "counterfactual_analysis".to_string(),
                        evaluator: format!("timeline_{}_modifications_{}", base_timeline, modifications.len()),
                    }),
                    cost: 0.0,
                    selectivity: 0.05, // Counterfactual queries are highly selective
                })
            },
        }
    }
    
    /// Apply predicate pushdown optimization
    fn apply_predicate_pushdown(
        &self,
        mut plan: QueryExecutionPlan,
        _statistics: &StatisticsSnapshot,
    ) -> Result<QueryExecutionPlan, QueryOptimizationError> {
        plan.root = self.pushdown_predicates(plan.root)?;
        plan.estimated_cost = self.cost_model.estimate_cost(&plan.root, _statistics)?;
        Ok(plan)
    }
    
    /// Recursively push predicates down the execution tree
    fn pushdown_predicates(&self, node: ExecutionNode) -> Result<ExecutionNode, QueryOptimizationError> {
        match node {
            ExecutionNode::Filter { input, predicate, .. } => {
                match *input {
                    ExecutionNode::SequentialScan { source, filter, cost, selectivity } => {
                        // Combine predicates at scan level
                        let combined_predicate = if let Some(existing_filter) = filter {
                            StatePredicate::And(Box::new(existing_filter), Box::new(predicate))
                        } else {
                            predicate
                        };
                        
                        Ok(ExecutionNode::SequentialScan {
                            source,
                            filter: Some(combined_predicate),
                            cost,
                            selectivity: selectivity * 0.5, // Assume additional selectivity
                        })
                    },
                    
                    ExecutionNode::Join { left, right, join_type, condition, cost } => {
                        // Try to push predicate to appropriate side of join
                        let referenced_attributes = self.extract_predicate_attributes(&predicate);
                        
                        // For simplicity, push to left side (in practice, would analyze attribute references)
                        let new_left = Box::new(ExecutionNode::Filter {
                            input: left,
                            predicate: predicate.clone(),
                            cost: 0.0,
                            selectivity: 0.5,
                        });
                        
                        Ok(ExecutionNode::Join {
                            left: new_left,
                            right,
                            join_type,
                            condition,
                            cost,
                        })
                    },
                    
                    other => {
                        // Cannot push down further, keep filter
                        Ok(ExecutionNode::Filter {
                            input: Box::new(other),
                            predicate,
                            cost: 0.0,
                            selectivity: 0.5,
                        })
                    }
                }
            },
            
            ExecutionNode::Join { left, right, join_type, condition, cost } => {
                let optimized_left = Box::new(self.pushdown_predicates(*left)?);
                let optimized_right = Box::new(self.pushdown_predicates(*right)?);
                
                Ok(ExecutionNode::Join {
                    left: optimized_left,
                    right: optimized_right,
                    join_type,
                    condition,
                    cost,
                })
            },
            
            other => Ok(other), // No optimization needed for other node types
        }
    }
    
    /// Apply projection pushdown optimization
    fn apply_projection_pushdown(
        &self,
        mut plan: QueryExecutionPlan,
        statistics: &StatisticsSnapshot,
    ) -> Result<QueryExecutionPlan, QueryOptimizationError> {
        plan.root = self.pushdown_projections(plan.root, &HashSet::new())?;
        plan.estimated_cost = self.cost_model.estimate_cost(&plan.root, statistics)?;
        Ok(plan)
    }
    
    /// Recursively push projections down the execution tree
    fn pushdown_projections(
        &self,
        node: ExecutionNode,
        required_attributes: &HashSet<String>,
    ) -> Result<ExecutionNode, QueryOptimizationError> {
        // Simplified projection pushdown - in practice would be more sophisticated
        Ok(node)
    }
    
    /// Apply join reordering optimization
    fn apply_join_reordering(
        &self,
        mut plan: QueryExecutionPlan,
        statistics: &StatisticsSnapshot,
    ) -> Result<QueryExecutionPlan, QueryOptimizationError> {
        plan.root = self.reorder_joins(plan.root, statistics)?;
        plan.estimated_cost = self.cost_model.estimate_cost(&plan.root, statistics)?;
        Ok(plan)
    }
    
    /// Reorder joins for optimal execution
    fn reorder_joins(
        &self,
        node: ExecutionNode,
        _statistics: &StatisticsSnapshot,
    ) -> Result<ExecutionNode, QueryOptimizationError> {
        // Simplified join reordering - in practice would use dynamic programming
        // or heuristic approaches for complex join graphs
        Ok(node)
    }
    
    /// Apply index optimization
    fn apply_index_optimization(
        &self,
        mut plan: QueryExecutionPlan,
        statistics: &StatisticsSnapshot,
    ) -> Result<QueryExecutionPlan, QueryOptimizationError> {
        plan.root = self.optimize_index_usage(plan.root, statistics)?;
        plan.estimated_cost = self.cost_model.estimate_cost(&plan.root, statistics)?;
        Ok(plan)
    }
    
    /// Optimize index usage in execution plan
    fn optimize_index_usage(
        &self,
        node: ExecutionNode,
        statistics: &StatisticsSnapshot,
    ) -> Result<ExecutionNode, QueryOptimizationError> {
        match node {
            ExecutionNode::SequentialScan { source, filter, .. } => {
                // Check if an index can be used instead of sequential scan
                if let Some(predicate) = &filter {
                    if let Some((index_name, selectivity)) = self.find_best_index(predicate, statistics)? {
                        return Ok(ExecutionNode::IndexScan {
                            index_name,
                            key_conditions: vec![predicate.clone()],
                            cost: 0.0, // Will be calculated by cost model
                            selectivity,
                        });
                    }
                }
                
                Ok(ExecutionNode::SequentialScan { source, filter, cost: 0.0, selectivity: 1.0 })
            },
            
            other => Ok(other),
        }
    }
    
    /// Find best index for predicate
    fn find_best_index(
        &self,
        _predicate: &StatePredicate,
        statistics: &StatisticsSnapshot,
    ) -> Result<Option<(String, f64)>, QueryOptimizationError> {
        // Simplified index selection - in practice would analyze predicate structure
        // and match against available indexes
        if let Some((index_name, index_stats)) = statistics.index_stats.iter().next() {
            Ok(Some((index_name.clone(), index_stats.selectivity)))
        } else {
            Ok(None)
        }
    }
    
    /// Determine parallelization strategy
    fn determine_parallelization_strategy(
        &self,
        plan: &QueryExecutionPlan,
        _statistics: &StatisticsSnapshot,
    ) -> Result<ParallelizationStrategy, QueryOptimizationError> {
        if !self.config.prefer_parallel_execution {
            return Ok(ParallelizationStrategy::Sequential);
        }
        
        // Analyze plan characteristics to determine best parallelization
        let estimated_data_size = plan.memory_estimate;
        let estimated_complexity = plan.estimated_cost;
        
        if estimated_data_size > 1024 * 1024 * 100 && estimated_complexity > 100.0 {
            // Large data and complex operations - use data parallelism
            Ok(ParallelizationStrategy::DataParallel {
                partitions: self.config.max_parallel_workers.min(8),
            })
        } else if estimated_complexity > 50.0 {
            // Complex operations - use pipeline parallelism
            Ok(ParallelizationStrategy::Pipeline {
                stages: self.config.max_parallel_workers.min(4),
            })
        } else {
            Ok(ParallelizationStrategy::Sequential)
        }
    }
    
    /// Hash query for caching
    fn hash_query(&self, query: &TemporalQuery) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        
        let serialized = serde_json::to_string(query).unwrap_or_default();
        let mut hasher = DefaultHasher::new();
        hasher.write(serialized.as_bytes());
        hasher.finish()
    }
    
    /// Estimate memory usage for execution plan
    fn estimate_memory_usage(
        &self,
        node: &ExecutionNode,
        statistics: &StatisticsSnapshot,
    ) -> Result<usize, QueryOptimizationError> {
        match node {
            ExecutionNode::SequentialScan { source, .. } => {
                match source {
                    DataSource::StateStore { estimated_rows, .. } => {
                        Ok(*estimated_rows * 1024) // Assume 1KB per row
                    },
                    DataSource::TraceStore { estimated_rows, .. } => {
                        Ok(*estimated_rows * 512) // Assume 512B per trace entry
                    },
                    DataSource::IndexedData { estimated_rows, .. } => {
                        Ok(*estimated_rows * 256) // Assume 256B per indexed entry
                    },
                    DataSource::MaterializedView { estimated_rows, .. } => {
                        Ok(*estimated_rows * 800) // Assume 800B per materialized row
                    },
                }
            },
            
            ExecutionNode::Join { left, right, .. } => {
                let left_memory = self.estimate_memory_usage(left, statistics)?;
                let right_memory = self.estimate_memory_usage(right, statistics)?;
                Ok(left_memory + right_memory + (left_memory + right_memory) / 4) // Hash table overhead
            },
            
            ExecutionNode::Sort { input, .. } => {
                let input_memory = self.estimate_memory_usage(input, statistics)?;
                Ok(input_memory * 2) // Sorting requires additional space
            },
            
            ExecutionNode::Aggregate { input, .. } => {
                let input_memory = self.estimate_memory_usage(input, statistics)?;
                Ok(input_memory / 10) // Aggregation typically reduces data size
            },
            
            other => Ok(1024 * 1024), // Default 1MB estimate
        }
    }
    
    /// Estimate time range query cardinality
    fn estimate_time_range_rows(
        &self,
        start_time: u64,
        end_time: u64,
        statistics: &StatisticsSnapshot,
    ) -> Result<usize, QueryOptimizationError> {
        let time_span = end_time - start_time;
        
        // Use statistics to estimate rows per time unit
        if let Some(table_stats) = statistics.table_stats.get("state_store") {
            let rows_per_time_unit = table_stats.growth_rate;
            Ok((time_span as f64 * rows_per_time_unit) as usize)
        } else {
            // Default estimate
            Ok((time_span / 1000) as usize) // Assume 1 row per second
        }
    }
    
    /// Estimate predicate selectivity
    fn estimate_predicate_selectivity(
        &self,
        predicate: &StatePredicate,
        statistics: &StatisticsSnapshot,
    ) -> Result<f64, QueryOptimizationError> {
        match predicate {
            StatePredicate::AttributeComparison { attribute, operator, value } => {
                if let Some(col_stats) = statistics.column_stats.get(attribute) {
                    match operator {
                        ComparisonOperator::Equal => {
                            Ok(1.0 / col_stats.distinct_values as f64)
                        },
                        ComparisonOperator::LessThan | ComparisonOperator::GreaterThan => {
                            Ok(0.33) // Default estimate
                        },
                        ComparisonOperator::LessThanOrEqual | ComparisonOperator::GreaterThanOrEqual => {
                            Ok(0.34) // Slightly higher than strict inequalities
                        },
                        _ => Ok(0.1), // Conservative estimate for complex operators
                    }
                } else {
                    Ok(0.1) // Default selectivity without statistics
                }
            },
            
            StatePredicate::And(left, right) => {
                let left_sel = self.estimate_predicate_selectivity(left, statistics)?;
                let right_sel = self.estimate_predicate_selectivity(right, statistics)?;
                Ok(left_sel * right_sel) // Assume independence
            },
            
            StatePredicate::Or(left, right) => {
                let left_sel = self.estimate_predicate_selectivity(left, statistics)?;
                let right_sel = self.estimate_predicate_selectivity(right, statistics)?;
                Ok(left_sel + right_sel - left_sel * right_sel) // Union probability
            },
            
            StatePredicate::Not(inner) => {
                let inner_sel = self.estimate_predicate_selectivity(inner, statistics)?;
                Ok(1.0 - inner_sel)
            },
            
            StatePredicate::PropertyEvaluation { .. } => {
                Ok(0.05) // Conservative estimate for complex evaluations
            },
        }
    }
    
    /// Extract attributes referenced in predicate
    fn extract_predicate_attributes(&self, predicate: &StatePredicate) -> HashSet<String> {
        let mut attributes = HashSet::new();
        self.collect_predicate_attributes(predicate, &mut attributes);
        attributes
    }
    
    /// Recursively collect predicate attributes
    fn collect_predicate_attributes(&self, predicate: &StatePredicate, attributes: &mut HashSet<String>) {
        match predicate {
            StatePredicate::AttributeComparison { attribute, .. } => {
                attributes.insert(attribute.clone());
            },
            StatePredicate::And(left, right) | StatePredicate::Or(left, right) => {
                self.collect_predicate_attributes(left, attributes);
                self.collect_predicate_attributes(right, attributes);
            },
            StatePredicate::Not(inner) => {
                self.collect_predicate_attributes(inner, attributes);
            },
            StatePredicate::PropertyEvaluation { property_name, .. } => {
                attributes.insert(property_name.clone());
            },
        }
    }
    
    /// Get optimization statistics
    pub fn get_execution_statistics(&self) -> QueryExecutionStats {
        self.execution_stats.read().unwrap().clone()
    }
    
    /// Clear plan cache
    pub fn clear_plan_cache(&self) {
        self.plan_cache.write().unwrap().clear();
    }
    
    /// Update statistics
    pub fn update_statistics(&self, new_stats: StatisticsSnapshot) {
        self.statistics_manager.write().unwrap().current_stats = new_stats;
        // Clear plan cache since statistics have changed
        self.clear_plan_cache();
    }
}

impl StatisticsManager {
    /// Create new statistics manager
    pub fn new() -> Self {
        Self {
            current_stats: StatisticsSnapshot {
                table_stats: HashMap::new(),
                column_stats: HashMap::new(),
                index_stats: HashMap::new(),
                workload_stats: WorkloadStatistics {
                    query_frequency: HashMap::new(),
                    average_response_time: Duration::from_millis(100),
                    peak_concurrency: 1,
                    resource_utilization: ResourceUtilization {
                        cpu_utilization: 0.1,
                        memory_utilization: 0.1,
                        io_utilization: 0.1,
                        network_utilization: 0.1,
                    },
                },
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
            update_frequency: Duration::from_minutes(5),
            last_update: Instant::now(),
            collection_handle: None,
        }
    }
    
    /// Start automatic statistics collection
    pub fn start_collection(&mut self) {
        // Implementation would start background statistics collection
        log::info!("Started automatic statistics collection");
    }
    
    /// Stop automatic statistics collection
    pub fn stop_collection(&mut self) {
        if let Some(handle) = self.collection_handle.take() {
            // Would signal thread to stop and join
            log::info!("Stopped automatic statistics collection");
        }
    }
}

impl IndexManager {
    /// Create new index manager
    pub fn new() -> Self {
        Self {
            indexes: HashMap::new(),
            usage_stats: HashMap::new(),
            recommendations: Vec::new(),
        }
    }
    
    /// Register index
    pub fn register_index(&mut self, definition: IndexDefinition) {
        let name = definition.name.clone();
        self.indexes.insert(name.clone(), definition);
        self.usage_stats.insert(name, IndexUsageStats {
            access_count: 0,
            last_used: 0,
            selectivity_distribution: Vec::new(),
            maintenance_cost: 0.0,
        });
    }
    
    /// Generate index recommendations
    pub fn generate_recommendations(
        &mut self,
        workload_stats: &WorkloadStatistics,
    ) -> Vec<IndexRecommendation> {
        // Simplified recommendation generation
        // In practice would analyze query patterns and identify optimization opportunities
        self.recommendations.clear();
        
        // Example recommendation based on query frequency
        for (query_pattern, frequency) in &workload_stats.query_frequency {
            if *frequency > 0.1 { // Frequent query pattern
                let recommendation = IndexRecommendation {
                    recommended_index: IndexDefinition {
                        name: format!("auto_index_{}", query_pattern),
                        table: "state_store".to_string(),
                        columns: vec!["timestamp".to_string()], // Example
                        index_type: IndexType::BTree,
                        unique: false,
                        partial_condition: None,
                    },
                    estimated_benefit: frequency * 100.0,
                    creation_cost: 50.0,
                    maintenance_cost: 10.0,
                    confidence: 0.8,
                };
                
                self.recommendations.push(recommendation);
            }
        }
        
        self.recommendations.clone()
    }
}

// Required trait implementations
unsafe impl Send for TemporalQueryOptimizer {}
unsafe impl Sync for TemporalQueryOptimizer {}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cost_model_creation() {
        let cost_model = CostModel::new();
        assert!(cost_model.operation_costs.contains_key("sequential_scan"));
        assert!(cost_model.operation_costs.contains_key("index_scan"));
    }
    
    #[test]
    fn test_predicate_selectivity_estimation() {
        let optimizer = TemporalQueryOptimizer::new(OptimizerConfig::default());
        let statistics = StatisticsSnapshot {
            table_stats: HashMap::new(),
            column_stats: {
                let mut stats = HashMap::new();
                stats.insert("test_attr".to_string(), ColumnStatistics {
                    distinct_values: 100,
                    null_fraction: 0.1,
                    most_common_values: Vec::new(),
                    histogram: Vec::new(),
                    correlation: 0.0,
                });
                stats
            },
            index_stats: HashMap::new(),
            workload_stats: WorkloadStatistics {
                query_frequency: HashMap::new(),
                average_response_time: Duration::from_millis(100),
                peak_concurrency: 1,
                resource_utilization: ResourceUtilization {
                    cpu_utilization: 0.1,
                    memory_utilization: 0.1,
                    io_utilization: 0.1,
                    network_utilization: 0.1,
                },
            },
            timestamp: 0,
        };
        
        let predicate = StatePredicate::AttributeComparison {
            attribute: "test_attr".to_string(),
            operator: ComparisonOperator::Equal,
            value: PredicateValue::Integer(42),
        };
        
        let selectivity = optimizer.estimate_predicate_selectivity(&predicate, &statistics).unwrap();
        assert!((selectivity - 0.01).abs() < 0.001); // 1/100 = 0.01
    }
    
    #[test]
    fn test_query_optimization() {
        let optimizer = TemporalQueryOptimizer::new(OptimizerConfig::default());
        
        let query = TemporalQuery::TimeRange {
            start_time: 0,
            end_time: 1000,
            step_granularity: Some(10),
        };
        
        let plan = optimizer.optimize_query(&query).unwrap();
        assert!(plan.estimated_cost > 0.0);
        assert!(plan.memory_estimate > 0);
    }
    
    #[test]
    fn test_join_cardinality_estimation() {
        let cost_model = CostModel::new();
        let statistics = StatisticsSnapshot {
            table_stats: HashMap::new(),
            column_stats: HashMap::new(),
            index_stats: HashMap::new(),
            workload_stats: WorkloadStatistics {
                query_frequency: HashMap::new(),
                average_response_time: Duration::from_millis(100),
                peak_concurrency: 1,
                resource_utilization: ResourceUtilization {
                    cpu_utilization: 0.1,
                    memory_utilization: 0.1,
                    io_utilization: 0.1,
                    network_utilization: 0.1,
                },
            },
            timestamp: 0,
        };
        
        let left_scan = ExecutionNode::SequentialScan {
            source: DataSource::StateStore {
                time_range: None,
                estimated_rows: 1000,
            },
            filter: None,
            cost: 0.0,
            selectivity: 1.0,
        };
        
        let right_scan = ExecutionNode::SequentialScan {
            source: DataSource::StateStore {
                time_range: None,
                estimated_rows: 500,
            },
            filter: None,
            cost: 0.0,
            selectivity: 1.0,
        };
        
        let join_node = ExecutionNode::Join {
            left: Box::new(left_scan),
            right: Box::new(right_scan),
            join_type: JoinType::Inner,
            condition: JoinCondition::AttributeJoin {
                left_attribute: "id".to_string(),
                right_attribute: "id".to_string(),
                operator: ComparisonOperator::Equal,
            },
            cost: 0.0,
        };
        
        let cardinality = cost_model.estimate_cardinality(&join_node, &statistics).unwrap();
        assert_eq!(cardinality, 50000); // (1000 * 500) / 10 = 50000
    }
}
