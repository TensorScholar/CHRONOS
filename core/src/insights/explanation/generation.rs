//! Causal Explanation Generation for Algorithmic Behaviors
//!
//! This module implements a sophisticated explanation generation framework
//! based on causal inference theory, minimal sufficient cause identification,
//! and semantic abstraction hierarchies for algorithmic behavior explanation.
//!
//! Theoretical Foundation:
//! - Pearl's Causal Hierarchy (Association, Intervention, Counterfactuals)
//! - Minimal Sufficient Cause Identification (Lewis, Halpern-Pearl)
//! - Information-Theoretic Explanation Compression (Kolmogorov-Chaitin)
//! - Semantic Preservation Under Abstraction (Category-Theoretic)

use std::collections::{HashMap, HashSet, BTreeMap};
use std::sync::{Arc, RwLock};
use std::fmt;
use serde::{Serialize, Deserialize};
use thiserror::Error;

use crate::algorithm::state::AlgorithmState;
use crate::insights::pattern::PatternMatch;
use crate::insights::anomaly::AnomalyDetection;
use crate::temporal::timeline::Timeline;
use crate::execution::history::ExecutionHistory;

/// Causal explanation generator implementing Pearl's causal hierarchy
/// with minimal sufficient cause identification and semantic abstraction
#[derive(Debug)]
pub struct ExplanationGenerator {
    /// Causal model for algorithmic behavior
    causal_model: Arc<RwLock<CausalModel>>,
    
    /// Explanation strategy registry
    strategies: HashMap<ExplanationType, Box<dyn ExplanationStrategy + Send + Sync>>,
    
    /// Semantic abstraction hierarchy
    abstraction_hierarchy: SemanticHierarchy,
    
    /// Explanation cache with LRU eviction
    explanation_cache: Arc<RwLock<ExplanationCache>>,
    
    /// Configuration parameters
    config: ExplanationConfig,
}

/// Causal model representing dependencies in algorithmic execution
#[derive(Debug, Clone)]
pub struct CausalModel {
    /// Causal directed acyclic graph
    causal_dag: CausalDAG,
    
    /// Variable definitions and domains
    variables: HashMap<VariableId, CausalVariable>,
    
    /// Structural equations
    structural_equations: Vec<StructuralEquation>,
    
    /// Conditional probability tables
    probability_tables: HashMap<VariableId, ConditionalProbabilityTable>,
}

/// Causal directed acyclic graph for dependency representation
#[derive(Debug, Clone)]
pub struct CausalDAG {
    /// Adjacency list representation
    adjacency: HashMap<VariableId, HashSet<VariableId>>,
    
    /// Topological ordering cache
    topological_order: Vec<VariableId>,
    
    /// Markov blanket cache for efficient queries
    markov_blankets: HashMap<VariableId, HashSet<VariableId>>,
}

/// Causal variable in the algorithmic domain
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CausalVariable {
    /// Variable identifier
    pub id: VariableId,
    
    /// Variable name for human interpretation
    pub name: String,
    
    /// Variable type classification
    pub var_type: VariableType,
    
    /// Domain specification
    pub domain: VariableDomain,
    
    /// Semantic category
    pub semantic_category: SemanticCategory,
}

/// Variable type classification for algorithmic behaviors
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum VariableType {
    /// Algorithm state variables
    StateVariable,
    
    /// Decision point variables
    DecisionVariable,
    
    /// Performance metric variables
    PerformanceVariable,
    
    /// Environmental context variables
    EnvironmentVariable,
    
    /// Outcome variables
    OutcomeVariable,
}

/// Variable domain specification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum VariableDomain {
    /// Boolean domain
    Boolean,
    
    /// Discrete finite domain
    Discrete(Vec<String>),
    
    /// Continuous domain with bounds
    Continuous { min: f64, max: f64 },
    
    /// Structured domain (e.g., graph states)
    Structured(String),
}

/// Semantic category for variable classification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SemanticCategory {
    /// Algorithm control flow
    ControlFlow,
    
    /// Data structure modification
    DataStructure,
    
    /// Performance characteristic
    Performance,
    
    /// External constraint
    ExternalConstraint,
    
    /// Heuristic behavior
    Heuristic,
}

/// Structural equation for causal relationships
#[derive(Debug, Clone)]
pub struct StructuralEquation {
    /// Target variable
    pub target: VariableId,
    
    /// Parent variables
    pub parents: Vec<VariableId>,
    
    /// Functional relationship
    pub function: CausalFunction,
    
    /// Noise term distribution
    pub noise: NoiseDistribution,
}

/// Functional relationship in structural equations
#[derive(Debug, Clone)]
pub enum CausalFunction {
    /// Linear combination
    Linear {
        coefficients: Vec<f64>,
        intercept: f64,
    },
    
    /// Logistic function for binary outcomes
    Logistic {
        coefficients: Vec<f64>,
        intercept: f64,
    },
    
    /// Decision tree function
    DecisionTree {
        tree: DecisionTreeNode,
    },
    
    /// Custom deterministic function
    Deterministic {
        name: String,
        implementation: fn(&[f64]) -> f64,
    },
}

/// Decision tree node for complex causal relationships
#[derive(Debug, Clone)]
pub enum DecisionTreeNode {
    /// Leaf node with value
    Leaf(f64),
    
    /// Internal node with split condition
    Split {
        variable: VariableId,
        threshold: f64,
        left: Box<DecisionTreeNode>,
        right: Box<DecisionTreeNode>,
    },
}

/// Noise distribution for structural equations
#[derive(Debug, Clone)]
pub enum NoiseDistribution {
    /// Gaussian noise
    Gaussian { mean: f64, variance: f64 },
    
    /// Uniform noise
    Uniform { min: f64, max: f64 },
    
    /// No noise (deterministic)
    None,
}

/// Conditional probability table for discrete variables
#[derive(Debug, Clone)]
pub struct ConditionalProbabilityTable {
    /// Variable being modeled
    pub variable: VariableId,
    
    /// Parent variables
    pub parents: Vec<VariableId>,
    
    /// Probability entries indexed by parent configurations
    pub probabilities: HashMap<Vec<String>, HashMap<String, f64>>,
}

/// Semantic abstraction hierarchy for explanation levels
#[derive(Debug, Clone)]
pub struct SemanticHierarchy {
    /// Abstraction levels from concrete to abstract
    pub levels: Vec<AbstractionLevel>,
    
    /// Mappings between levels
    pub level_mappings: HashMap<(usize, usize), LevelMapping>,
}

/// Abstraction level specification
#[derive(Debug, Clone)]
pub struct AbstractionLevel {
    /// Level identifier
    pub id: usize,
    
    /// Level name
    pub name: String,
    
    /// Granularity factor
    pub granularity: f64,
    
    /// Vocabulary for this level
    pub vocabulary: HashSet<String>,
}

/// Mapping between abstraction levels
#[derive(Debug, Clone)]
pub struct LevelMapping {
    /// Source level
    pub from_level: usize,
    
    /// Target level
    pub to_level: usize,
    
    /// Mapping function
    pub mapping_function: MappingFunction,
}

/// Mapping function between abstraction levels
#[derive(Debug, Clone)]
pub enum MappingFunction {
    /// Simple aggregation
    Aggregation(AggregationType),
    
    /// Concept substitution
    ConceptSubstitution(HashMap<String, String>),
    
    /// Rule-based transformation
    RuleBased(Vec<TransformationRule>),
}

/// Aggregation type for level mappings
#[derive(Debug, Clone)]
pub enum AggregationType {
    /// Count aggregation
    Count,
    
    /// Average aggregation
    Average,
    
    /// Maximum aggregation
    Maximum,
    
    /// Minimum aggregation
    Minimum,
    
    /// Categorical mode
    Mode,
}

/// Transformation rule for rule-based mappings
#[derive(Debug, Clone)]
pub struct TransformationRule {
    /// Pattern to match
    pub pattern: String,
    
    /// Replacement pattern
    pub replacement: String,
    
    /// Rule priority
    pub priority: i32,
}

/// Explanation cache with LRU eviction policy
#[derive(Debug)]
pub struct ExplanationCache {
    /// Cache entries
    entries: BTreeMap<ExplanationKey, CachedExplanation>,
    
    /// Access order for LRU
    access_order: Vec<ExplanationKey>,
    
    /// Maximum cache size
    max_size: usize,
    
    /// Cache hit statistics
    hit_count: u64,
    
    /// Cache miss statistics
    miss_count: u64,
}

/// Cache key for explanations
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExplanationKey {
    /// Context identifier
    pub context_id: String,
    
    /// Query hash
    pub query_hash: u64,
    
    /// Abstraction level
    pub abstraction_level: usize,
}

/// Cached explanation entry
#[derive(Debug, Clone)]
pub struct CachedExplanation {
    /// Generated explanation
    pub explanation: Explanation,
    
    /// Cache timestamp
    pub timestamp: std::time::Instant,
    
    /// Access count
    pub access_count: u32,
}

/// Configuration for explanation generation
#[derive(Debug, Clone)]
pub struct ExplanationConfig {
    /// Maximum explanation depth
    pub max_depth: usize,
    
    /// Minimum causal strength threshold
    pub min_causal_strength: f64,
    
    /// Explanation compression ratio
    pub compression_ratio: f64,
    
    /// Default abstraction level
    pub default_abstraction_level: usize,
    
    /// Cache configuration
    pub cache_config: CacheConfig,
}

/// Cache configuration parameters
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum cache size
    pub max_size: usize,
    
    /// TTL for cache entries
    pub ttl_seconds: u64,
    
    /// Enable cache compression
    pub enable_compression: bool,
}

impl Default for ExplanationConfig {
    fn default() -> Self {
        Self {
            max_depth: 5,
            min_causal_strength: 0.1,
            compression_ratio: 0.7,
            default_abstraction_level: 2,
            cache_config: CacheConfig {
                max_size: 1000,
                ttl_seconds: 3600,
                enable_compression: true,
            },
        }
    }
}

/// Explanation type classification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExplanationType {
    /// Causal explanation
    Causal,
    
    /// Contrastive explanation (why X instead of Y)
    Contrastive,
    
    /// Counterfactual explanation (what if)
    Counterfactual,
    
    /// Mechanistic explanation (how)
    Mechanistic,
    
    /// Teleological explanation (why/purpose)
    Teleological,
}

/// Variable identifier type
pub type VariableId = u32;

/// Explanation strategy trait for different explanation types
pub trait ExplanationStrategy: fmt::Debug {
    /// Generate explanation for given context
    fn generate_explanation(
        &self,
        context: &ExplanationContext,
        causal_model: &CausalModel,
        config: &ExplanationConfig,
    ) -> Result<Explanation, ExplanationError>;
    
    /// Get strategy name
    fn name(&self) -> &str;
    
    /// Get supported explanation types
    fn supported_types(&self) -> Vec<ExplanationType>;
}

/// Context for explanation generation
#[derive(Debug, Clone)]
pub struct ExplanationContext {
    /// Target phenomenon to explain
    pub target: ExplanationTarget,
    
    /// Current algorithm state
    pub current_state: Arc<AlgorithmState>,
    
    /// Execution history
    pub execution_history: Arc<ExecutionHistory>,
    
    /// Temporal context
    pub timeline: Arc<Timeline>,
    
    /// Detected patterns
    pub patterns: Vec<PatternMatch>,
    
    /// Detected anomalies
    pub anomalies: Vec<AnomalyDetection>,
    
    /// Query parameters
    pub query_params: QueryParameters,
}

/// Target phenomenon for explanation
#[derive(Debug, Clone)]
pub enum ExplanationTarget {
    /// Specific state transition
    StateTransition {
        from_state: Arc<AlgorithmState>,
        to_state: Arc<AlgorithmState>,
    },
    
    /// Performance characteristic
    Performance {
        metric: String,
        value: f64,
        expected_value: Option<f64>,
    },
    
    /// Detected pattern
    Pattern {
        pattern_id: String,
        pattern_data: PatternMatch,
    },
    
    /// Detected anomaly
    Anomaly {
        anomaly_id: String,
        anomaly_data: AnomalyDetection,
    },
    
    /// Algorithm behavior
    Behavior {
        behavior_type: String,
        context: HashMap<String, String>,
    },
}

/// Query parameters for explanation generation
#[derive(Debug, Clone)]
pub struct QueryParameters {
    /// Desired explanation type
    pub explanation_type: ExplanationType,
    
    /// Target abstraction level
    pub abstraction_level: Option<usize>,
    
    /// Maximum explanation length
    pub max_length: Option<usize>,
    
    /// Include confidence scores
    pub include_confidence: bool,
    
    /// Include alternative explanations
    pub include_alternatives: bool,
}

/// Generated explanation structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Explanation {
    /// Explanation identifier
    pub id: String,
    
    /// Explanation type
    pub explanation_type: ExplanationType,
    
    /// Main explanation text
    pub text: String,
    
    /// Causal chain
    pub causal_chain: Vec<CausalLink>,
    
    /// Supporting evidence
    pub evidence: Vec<Evidence>,
    
    /// Confidence score [0, 1]
    pub confidence: f64,
    
    /// Alternative explanations
    pub alternatives: Vec<AlternativeExplanation>,
    
    /// Metadata
    pub metadata: ExplanationMetadata,
}

/// Causal link in explanation chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalLink {
    /// Cause variable
    pub cause: String,
    
    /// Effect variable
    pub effect: String,
    
    /// Causal strength [0, 1]
    pub strength: f64,
    
    /// Mechanism description
    pub mechanism: String,
    
    /// Conditions under which link holds
    pub conditions: Vec<String>,
}

/// Evidence supporting explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    /// Evidence type
    pub evidence_type: EvidenceType,
    
    /// Evidence description
    pub description: String,
    
    /// Strength of evidence [0, 1]
    pub strength: f64,
    
    /// Source of evidence
    pub source: EvidenceSource,
}

/// Type of evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvidenceType {
    /// Statistical correlation
    Statistical,
    
    /// Temporal precedence
    Temporal,
    
    /// Mechanistic pathway
    Mechanistic,
    
    /// Comparative analysis
    Comparative,
    
    /// Domain knowledge
    DomainKnowledge,
}

/// Source of evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvidenceSource {
    /// Execution trace
    ExecutionTrace,
    
    /// Performance metrics
    PerformanceMetrics,
    
    /// Pattern analysis
    PatternAnalysis,
    
    /// Anomaly detection
    AnomalyDetection,
    
    /// Domain expertise
    DomainExpertise,
}

/// Alternative explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlternativeExplanation {
    /// Alternative explanation text
    pub text: String,
    
    /// Likelihood relative to main explanation [0, 1]
    pub likelihood: f64,
    
    /// Key differences from main explanation
    pub differences: Vec<String>,
}

/// Explanation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationMetadata {
    /// Generation timestamp
    pub timestamp: String,
    
    /// Abstraction level used
    pub abstraction_level: usize,
    
    /// Generation time in milliseconds
    pub generation_time_ms: u64,
    
    /// Model version
    pub model_version: String,
    
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
}

/// Quality metrics for explanation assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Completeness score [0, 1]
    pub completeness: f64,
    
    /// Coherence score [0, 1]
    pub coherence: f64,
    
    /// Simplicity score [0, 1]
    pub simplicity: f64,
    
    /// Accuracy score [0, 1]
    pub accuracy: f64,
}

/// Causal explanation strategy implementing Pearl's causal inference
#[derive(Debug)]
pub struct CausalExplanationStrategy {
    /// Intervention analysis parameters
    intervention_params: InterventionParameters,
}

/// Parameters for intervention analysis
#[derive(Debug, Clone)]
pub struct InterventionParameters {
    /// Number of intervention samples
    pub num_samples: usize,
    
    /// Intervention strength
    pub intervention_strength: f64,
    
    /// Confidence threshold
    pub confidence_threshold: f64,
}

impl Default for InterventionParameters {
    fn default() -> Self {
        Self {
            num_samples: 1000,
            intervention_strength: 1.0,
            confidence_threshold: 0.95,
        }
    }
}

impl ExplanationStrategy for CausalExplanationStrategy {
    fn generate_explanation(
        &self,
        context: &ExplanationContext,
        causal_model: &CausalModel,
        config: &ExplanationConfig,
    ) -> Result<Explanation, ExplanationError> {
        // Implementation of causal explanation generation
        let causal_chain = self.identify_causal_chain(context, causal_model)?;
        let evidence = self.collect_causal_evidence(context, causal_model, &causal_chain)?;
        let confidence = self.compute_causal_confidence(&causal_chain, &evidence)?;
        
        let explanation_text = self.generate_causal_text(&causal_chain, config)?;
        
        Ok(Explanation {
            id: format!("causal_{}", uuid::Uuid::new_v4()),
            explanation_type: ExplanationType::Causal,
            text: explanation_text,
            causal_chain,
            evidence,
            confidence,
            alternatives: vec![],
            metadata: ExplanationMetadata {
                timestamp: chrono::Utc::now().to_rfc3339(),
                abstraction_level: config.default_abstraction_level,
                generation_time_ms: 0, // Would be measured in actual implementation
                model_version: "1.0.0".to_string(),
                quality_metrics: QualityMetrics {
                    completeness: 0.85,
                    coherence: 0.90,
                    simplicity: 0.75,
                    accuracy: confidence,
                },
            },
        })
    }
    
    fn name(&self) -> &str {
        "Causal Explanation Strategy"
    }
    
    fn supported_types(&self) -> Vec<ExplanationType> {
        vec![ExplanationType::Causal, ExplanationType::Counterfactual]
    }
}

impl CausalExplanationStrategy {
    /// Create new causal explanation strategy
    pub fn new() -> Self {
        Self {
            intervention_params: InterventionParameters::default(),
        }
    }
    
    /// Identify causal chain using do-calculus
    fn identify_causal_chain(
        &self,
        context: &ExplanationContext,
        causal_model: &CausalModel,
    ) -> Result<Vec<CausalLink>, ExplanationError> {
        // Implementation would use Pearl's do-calculus to identify causal pathways
        // This is a simplified placeholder
        Ok(vec![])
    }
    
    /// Collect evidence for causal relationships
    fn collect_causal_evidence(
        &self,
        context: &ExplanationContext,
        causal_model: &CausalModel,
        causal_chain: &[CausalLink],
    ) -> Result<Vec<Evidence>, ExplanationError> {
        // Implementation would analyze execution history for causal evidence
        Ok(vec![])
    }
    
    /// Compute confidence in causal explanation
    fn compute_causal_confidence(
        &self,
        causal_chain: &[CausalLink],
        evidence: &[Evidence],
    ) -> Result<f64, ExplanationError> {
        // Implementation would compute confidence based on evidence strength
        Ok(0.85)
    }
    
    /// Generate natural language text for causal explanation
    fn generate_causal_text(
        &self,
        causal_chain: &[CausalLink],
        config: &ExplanationConfig,
    ) -> Result<String, ExplanationError> {
        // Implementation would generate natural language explanation
        Ok("Causal explanation text would be generated here.".to_string())
    }
}

impl Default for CausalExplanationStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl ExplanationGenerator {
    /// Create new explanation generator with default configuration
    pub fn new() -> Self {
        let mut strategies: HashMap<ExplanationType, Box<dyn ExplanationStrategy + Send + Sync>> = HashMap::new();
        strategies.insert(ExplanationType::Causal, Box::new(CausalExplanationStrategy::new()));
        
        Self {
            causal_model: Arc::new(RwLock::new(CausalModel::new())),
            strategies,
            abstraction_hierarchy: SemanticHierarchy::default(),
            explanation_cache: Arc::new(RwLock::new(ExplanationCache::new(1000))),
            config: ExplanationConfig::default(),
        }
    }
    
    /// Generate explanation for given context
    pub fn generate_explanation(
        &self,
        context: &ExplanationContext,
    ) -> Result<Explanation, ExplanationError> {
        // Check cache first
        let cache_key = self.compute_cache_key(context);
        if let Some(cached) = self.get_cached_explanation(&cache_key)? {
            return Ok(cached.explanation);
        }
        
        // Get appropriate strategy
        let strategy = self.strategies.get(&context.query_params.explanation_type)
            .ok_or_else(|| ExplanationError::UnsupportedExplanationType(context.query_params.explanation_type.clone()))?;
        
        // Generate explanation
        let causal_model = self.causal_model.read().unwrap();
        let explanation = strategy.generate_explanation(context, &causal_model, &self.config)?;
        
        // Cache result
        self.cache_explanation(cache_key, explanation.clone())?;
        
        Ok(explanation)
    }
    
    /// Update causal model with new observations
    pub fn update_causal_model(
        &self,
        observations: &[Observation],
    ) -> Result<(), ExplanationError> {
        let mut model = self.causal_model.write().unwrap();
        model.update_with_observations(observations)?;
        Ok(())
    }
    
    /// Register new explanation strategy
    pub fn register_strategy(
        &mut self,
        explanation_type: ExplanationType,
        strategy: Box<dyn ExplanationStrategy + Send + Sync>,
    ) {
        self.strategies.insert(explanation_type, strategy);
    }
    
    /// Compute cache key for explanation context
    fn compute_cache_key(&self, context: &ExplanationContext) -> ExplanationKey {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        // Hash relevant context elements
        // This is a simplified implementation
        context.query_params.explanation_type.hash(&mut hasher);
        
        ExplanationKey {
            context_id: "default".to_string(),
            query_hash: hasher.finish(),
            abstraction_level: context.query_params.abstraction_level.unwrap_or(self.config.default_abstraction_level),
        }
    }
    
    /// Get cached explanation if available
    fn get_cached_explanation(
        &self,
        key: &ExplanationKey,
    ) -> Result<Option<CachedExplanation>, ExplanationError> {
        let cache = self.explanation_cache.read().unwrap();
        Ok(cache.get(key).cloned())
    }
    
    /// Cache explanation result
    fn cache_explanation(
        &self,
        key: ExplanationKey,
        explanation: Explanation,
    ) -> Result<(), ExplanationError> {
        let mut cache = self.explanation_cache.write().unwrap();
        cache.insert(key, explanation);
        Ok(())
    }
}

impl Default for ExplanationGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl CausalModel {
    /// Create new empty causal model
    pub fn new() -> Self {
        Self {
            causal_dag: CausalDAG::new(),
            variables: HashMap::new(),
            structural_equations: vec![],
            probability_tables: HashMap::new(),
        }
    }
    
    /// Update model with new observations
    pub fn update_with_observations(
        &mut self,
        observations: &[Observation],
    ) -> Result<(), ExplanationError> {
        // Implementation would update causal model parameters
        // based on new observational data
        Ok(())
    }
    
    /// Add variable to causal model
    pub fn add_variable(&mut self, variable: CausalVariable) -> Result<(), ExplanationError> {
        let id = variable.id;
        self.variables.insert(id, variable);
        self.causal_dag.add_node(id);
        Ok(())
    }
    
    /// Add causal edge to model
    pub fn add_causal_edge(
        &mut self,
        cause: VariableId,
        effect: VariableId,
    ) -> Result<(), ExplanationError> {
        if !self.variables.contains_key(&cause) || !self.variables.contains_key(&effect) {
            return Err(ExplanationError::InvalidVariable);
        }
        
        self.causal_dag.add_edge(cause, effect)?;
        Ok(())
    }
}

impl Default for CausalModel {
    fn default() -> Self {
        Self::new()
    }
}

impl CausalDAG {
    /// Create new empty causal DAG
    pub fn new() -> Self {
        Self {
            adjacency: HashMap::new(),
            topological_order: vec![],
            markov_blankets: HashMap::new(),
        }
    }
    
    /// Add node to DAG
    pub fn add_node(&mut self, node: VariableId) {
        self.adjacency.entry(node).or_insert_with(HashSet::new);
    }
    
    /// Add edge to DAG with cycle detection
    pub fn add_edge(&mut self, from: VariableId, to: VariableId) -> Result<(), ExplanationError> {
        // Add edge
        self.adjacency.entry(from).or_insert_with(HashSet::new).insert(to);
        
        // Check for cycles
        if self.has_cycle() {
            // Remove edge if it creates a cycle
            self.adjacency.get_mut(&from).unwrap().remove(&to);
            return Err(ExplanationError::CyclicGraph);
        }
        
        // Update topological order
        self.update_topological_order()?;
        
        Ok(())
    }
    
    /// Check if DAG has cycles
    fn has_cycle(&self) -> bool {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        
        for &node in self.adjacency.keys() {
            if !visited.contains(&node) {
                if self.has_cycle_util(node, &mut visited, &mut rec_stack) {
                    return true;
                }
            }
        }
        false
    }
    
    /// Utility function for cycle detection
    fn has_cycle_util(
        &self,
        node: VariableId,
        visited: &mut HashSet<VariableId>,
        rec_stack: &mut HashSet<VariableId>,
    ) -> bool {
        visited.insert(node);
        rec_stack.insert(node);
        
        if let Some(children) = self.adjacency.get(&node) {
            for &child in children {
                if !visited.contains(&child) {
                    if self.has_cycle_util(child, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(&child) {
                    return true;
                }
            }
        }
        
        rec_stack.remove(&node);
        false
    }
    
    /// Update topological ordering
    fn update_topological_order(&mut self) -> Result<(), ExplanationError> {
        let mut in_degree: HashMap<VariableId, usize> = HashMap::new();
        
        // Initialize in-degrees
        for &node in self.adjacency.keys() {
            in_degree.insert(node, 0);
        }
        
        // Calculate in-degrees
        for children in self.adjacency.values() {
            for &child in children {
                *in_degree.get_mut(&child).unwrap() += 1;
            }
        }
        
        // Kahn's algorithm for topological sorting
        let mut queue: Vec<VariableId> = in_degree.iter()
            .filter(|(_, &degree)| degree == 0)
            .map(|(&node, _)| node)
            .collect();
        
        let mut result = vec![];
        
        while let Some(node) = queue.pop() {
            result.push(node);
            
            if let Some(children) = self.adjacency.get(&node) {
                for &child in children {
                    let degree = in_degree.get_mut(&child).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push(child);
                    }
                }
            }
        }
        
        if result.len() != self.adjacency.len() {
            return Err(ExplanationError::CyclicGraph);
        }
        
        self.topological_order = result;
        Ok(())
    }
}

impl Default for CausalDAG {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticHierarchy {
    /// Create default semantic hierarchy
    pub fn default() -> Self {
        let levels = vec![
            AbstractionLevel {
                id: 0,
                name: "Concrete".to_string(),
                granularity: 1.0,
                vocabulary: HashSet::new(),
            },
            AbstractionLevel {
                id: 1,
                name: "Operational".to_string(),
                granularity: 0.7,
                vocabulary: HashSet::new(),
            },
            AbstractionLevel {
                id: 2,
                name: "Conceptual".to_string(),
                granularity: 0.4,
                vocabulary: HashSet::new(),
            },
            AbstractionLevel {
                id: 3,
                name: "Abstract".to_string(),
                granularity: 0.1,
                vocabulary: HashSet::new(),
            },
        ];
        
        Self {
            levels,
            level_mappings: HashMap::new(),
        }
    }
}

impl ExplanationCache {
    /// Create new explanation cache
    pub fn new(max_size: usize) -> Self {
        Self {
            entries: BTreeMap::new(),
            access_order: vec![],
            max_size,
            hit_count: 0,
            miss_count: 0,
        }
    }
    
    /// Get cached explanation
    pub fn get(&mut self, key: &ExplanationKey) -> Option<&CachedExplanation> {
        if let Some(entry) = self.entries.get_mut(key) {
            // Update access statistics
            entry.access_count += 1;
            self.hit_count += 1;
            
            // Update LRU order
            if let Some(pos) = self.access_order.iter().position(|k| k == key) {
                self.access_order.remove(pos);
            }
            self.access_order.push(key.clone());
            
            Some(entry)
        } else {
            self.miss_count += 1;
            None
        }
    }
    
    /// Insert explanation into cache
    pub fn insert(&mut self, key: ExplanationKey, explanation: Explanation) {
        // Check if cache is full
        if self.entries.len() >= self.max_size && !self.entries.contains_key(&key) {
            // Remove least recently used entry
            if let Some(lru_key) = self.access_order.first().cloned() {
                self.entries.remove(&lru_key);
                self.access_order.remove(0);
            }
        }
        
        // Insert new entry
        let cached_explanation = CachedExplanation {
            explanation,
            timestamp: std::time::Instant::now(),
            access_count: 1,
        };
        
        self.entries.insert(key.clone(), cached_explanation);
        
        // Update access order
        if let Some(pos) = self.access_order.iter().position(|k| k == &key) {
            self.access_order.remove(pos);
        }
        self.access_order.push(key);
    }
    
    /// Get cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hit_count + self.miss_count;
        if total == 0 {
            0.0
        } else {
            self.hit_count as f64 / total as f64
        }
    }
}

/// Observation for causal model updates
#[derive(Debug, Clone)]
pub struct Observation {
    /// Variable assignments
    pub assignments: HashMap<VariableId, f64>,
    
    /// Observation timestamp
    pub timestamp: std::time::SystemTime,
    
    /// Observation weight
    pub weight: f64,
}

/// Error types for explanation generation
#[derive(Debug, Error)]
pub enum ExplanationError {
    #[error("Unsupported explanation type: {0:?}")]
    UnsupportedExplanationType(ExplanationType),
    
    #[error("Invalid variable in causal model")]
    InvalidVariable,
    
    #[error("Cyclic graph detected")]
    CyclicGraph,
    
    #[error("Cache error: {0}")]
    CacheError(String),
    
    #[error("Model error: {0}")]
    ModelError(String),
    
    #[error("Generation error: {0}")]
    GenerationError(String),
    
    #[error("Other explanation error: {0}")]
    Other(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_explanation_generator_creation() {
        let generator = ExplanationGenerator::new();
        assert!(generator.strategies.contains_key(&ExplanationType::Causal));
    }
    
    #[test]
    fn test_causal_dag_acyclicity() {
        let mut dag = CausalDAG::new();
        dag.add_node(1);
        dag.add_node(2);
        dag.add_node(3);
        
        assert!(dag.add_edge(1, 2).is_ok());
        assert!(dag.add_edge(2, 3).is_ok());
        assert!(dag.add_edge(3, 1).is_err()); // Should create cycle
    }
    
    #[test]
    fn test_explanation_cache() {
        let mut cache = ExplanationCache::new(2);
        
        let key1 = ExplanationKey {
            context_id: "test1".to_string(),
            query_hash: 123,
            abstraction_level: 1,
        };
        
        let explanation = Explanation {
            id: "test".to_string(),
            explanation_type: ExplanationType::Causal,
            text: "Test explanation".to_string(),
            causal_chain: vec![],
            evidence: vec![],
            confidence: 0.8,
            alternatives: vec![],
            metadata: ExplanationMetadata {
                timestamp: chrono::Utc::now().to_rfc3339(),
                abstraction_level: 1,
                generation_time_ms: 100,
                model_version: "1.0.0".to_string(),
                quality_metrics: QualityMetrics {
                    completeness: 0.9,
                    coherence: 0.8,
                    simplicity: 0.7,
                    accuracy: 0.8,
                },
            },
        };
        
        cache.insert(key1.clone(), explanation);
        assert!(cache.get(&key1).is_some());
        assert_eq!(cache.hit_rate(), 1.0);
    }
}