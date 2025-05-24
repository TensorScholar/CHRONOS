//! # Learning Path Knowledge Graph Framework
//! 
//! This module implements a directed acyclic knowledge graph for educational pathways with
//! topological sorting algorithms for optimal learning sequence generation. The system 
//! provides a mathematically rigorous foundation for curriculum design and adaptive learning.
//!
//! ## Theoretical Foundation
//!
//! The implementation leverages graph theory and order theory principles to model knowledge
//! dependencies as a directed acyclic graph (DAG). Learning paths are generated through
//! weighted topological sorting algorithms that maintain prerequisite consistency while
//! optimizing for cognitive load, knowledge retention, and learning efficiency.
//!
//! ## Computational Characteristics
//!
//! - Time Complexity: O(V+E) for path generation through knowledge space
//! - Space Complexity: O(V+E) for graph representation with adjacency lists
//! - Topological Sort: O(V+E) using Kahn's algorithm with priority queues
//! - Path Optimization: O(V log V) for multi-objective path planning
//!
//! Copyright (c) 2025 Mohammad Atashi. All rights reserved.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, BinaryHeap, VecDeque};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

use serde::{Serialize, Deserialize};
use thiserror::Error;
use log::{debug, warn};

use crate::education::progressive::{ExpertiseLevel, UserProfile};
use crate::education::assessment::KnowledgeState;

/// Error types for learning path operations
#[derive(Debug, Error)]
pub enum PathError {
    /// Cycle detected in knowledge graph
    #[error("Cycle detected in knowledge graph: {0}")]
    CycleDetected(String),
    
    /// Node not found
    #[error("Knowledge node not found: {0}")]
    NodeNotFound(String),
    
    /// Invalid path
    #[error("Invalid learning path: {0}")]
    InvalidPath(String),
    
    /// Inconsistent prerequisites
    #[error("Inconsistent prerequisites: {0}")]
    InconsistentPrerequisites(String),
    
    /// Path generation error
    #[error("Path generation error: {0}")]
    PathGenerationError(String),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    /// Other path error
    #[error("Learning path error: {0}")]
    Other(String),
}

/// Knowledge domain identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DomainId(pub String);

impl fmt::Display for DomainId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Knowledge concept identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConceptId {
    /// Domain this concept belongs to
    pub domain: DomainId,
    
    /// Concept identifier within domain
    pub id: String,
}

impl fmt::Display for ConceptId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.domain, self.id)
    }
}

/// Learning objective identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ObjectiveId {
    /// Concept this objective belongs to
    pub concept: ConceptId,
    
    /// Objective identifier within concept
    pub id: String,
}

impl fmt::Display for ObjectiveId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.concept, self.id)
    }
}

/// Knowledge concept metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptMetadata {
    /// Concept identifier
    pub id: ConceptId,
    
    /// Concept title
    pub title: String,
    
    /// Concept description
    pub description: String,
    
    /// Minimum expertise level required
    pub min_expertise: ExpertiseLevel,
    
    /// Estimated learning time in minutes
    pub estimated_time_minutes: u32,
    
    /// Complexity score (0.0-1.0)
    pub complexity: f32,
    
    /// Importance score (0.0-1.0)
    pub importance: f32,
    
    /// Tags for categorization
    pub tags: Vec<String>,
    
    /// Additional metadata
    pub additional: HashMap<String, String>,
}

/// Learning objective metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveMetadata {
    /// Objective identifier
    pub id: ObjectiveId,
    
    /// Objective title
    pub title: String,
    
    /// Objective description
    pub description: String,
    
    /// Bloom's taxonomy level
    pub taxonomy_level: TaxonomyLevel,
    
    /// Minimum expertise level required
    pub min_expertise: ExpertiseLevel,
    
    /// Estimated completion time in minutes
    pub estimated_time_minutes: u32,
    
    /// Complexity score (0.0-1.0)
    pub complexity: f32,
    
    /// Importance score (0.0-1.0)
    pub importance: f32,
    
    /// Additional metadata
    pub additional: HashMap<String, String>,
}

/// Bloom's taxonomy level for learning objectives
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TaxonomyLevel {
    /// Remember: Recall facts and basic concepts
    Remember = 1,
    
    /// Understand: Explain ideas or concepts
    Understand = 2,
    
    /// Apply: Use information in new situations
    Apply = 3,
    
    /// Analyze: Draw connections among ideas
    Analyze = 4,
    
    /// Evaluate: Justify a stand or decision
    Evaluate = 5,
    
    /// Create: Produce new or original work
    Create = 6,
}

/// Prerequisite relationship between concepts or objectives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prerequisite<T> {
    /// Source node (prerequisite)
    pub source: T,
    
    /// Target node (dependent)
    pub target: T,
    
    /// Prerequisite strength (0.0-1.0)
    /// 0.0 = optional, 1.0 = required
    pub strength: f32,
    
    /// Explanation of the prerequisite relationship
    pub explanation: Option<String>,
}

impl<T: Clone + Eq + Hash> Prerequisite<T> {
    /// Create a new prerequisite relationship
    pub fn new(source: T, target: T, strength: f32) -> Self {
        Self {
            source,
            target,
            strength: strength.max(0.0).min(1.0), // Clamp to [0.0, 1.0]
            explanation: None,
        }
    }
    
    /// Create a new required prerequisite relationship
    pub fn required(source: T, target: T) -> Self {
        Self::new(source, target, 1.0)
    }
    
    /// Create a new optional prerequisite relationship
    pub fn optional(source: T, target: T, strength: f32) -> Self {
        Self::new(source, target, strength)
    }
    
    /// Add explanation to the prerequisite relationship
    pub fn with_explanation(mut self, explanation: String) -> Self {
        self.explanation = Some(explanation);
        self
    }
    
    /// Check if this is a required prerequisite
    pub fn is_required(&self) -> bool {
        self.strength >= 0.99 // Allow for floating-point imprecision
    }
}

/// Path optimization strategy for learning path generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PathStrategy {
    /// Shortest path (minimize number of concepts)
    Shortest,
    
    /// Fastest path (minimize learning time)
    Fastest,
    
    /// Easiest path (minimize complexity)
    Easiest,
    
    /// Most important concepts first
    ImportanceFirst,
    
    /// Balanced (consider all factors)
    Balanced,
    
    /// Custom strategy with weights
    Custom,
}

/// Weights for custom path optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathWeights {
    /// Weight for path length (number of concepts)
    pub length_weight: f32,
    
    /// Weight for learning time
    pub time_weight: f32,
    
    /// Weight for complexity
    pub complexity_weight: f32,
    
    /// Weight for importance
    pub importance_weight: f32,
    
    /// Weight for prerequisite strength
    pub prerequisite_weight: f32,
}

impl Default for PathWeights {
    fn default() -> Self {
        Self {
            length_weight: 1.0,
            time_weight: 1.0,
            complexity_weight: 1.0,
            importance_weight: 1.0,
            prerequisite_weight: 1.0,
        }
    }
}

/// Learning path segment representing a subset of a complete path
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathSegment {
    /// Concepts in this segment
    pub concepts: Vec<ConceptId>,
    
    /// Objectives in this segment
    pub objectives: Vec<ObjectiveId>,
    
    /// Estimated learning time in minutes
    pub estimated_time_minutes: u32,
    
    /// Complexity score (0.0-1.0)
    pub complexity: f32,
    
    /// Importance score (0.0-1.0)
    pub importance: f32,
    
    /// Segment description
    pub description: Option<String>,
}

/// Complete learning path from current state to learning goals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningPath {
    /// Path segments in order
    pub segments: Vec<PathSegment>,
    
    /// Total estimated learning time in minutes
    pub total_time_minutes: u32,
    
    /// Average complexity score (0.0-1.0)
    pub average_complexity: f32,
    
    /// Average importance score (0.0-1.0)
    pub average_importance: f32,
    
    /// Path strategy used to generate this path
    pub strategy: PathStrategy,
    
    /// Custom weights if strategy is Custom
    pub weights: Option<PathWeights>,
    
    /// Path generation timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    
    /// User profile the path was generated for
    pub user_profile_id: Option<String>,
}

impl LearningPath {
    /// Get all concepts in the path
    pub fn all_concepts(&self) -> Vec<ConceptId> {
        self.segments.iter()
            .flat_map(|segment| segment.concepts.clone())
            .collect()
    }
    
    /// Get all objectives in the path
    pub fn all_objectives(&self) -> Vec<ObjectiveId> {
        self.segments.iter()
            .flat_map(|segment| segment.objectives.clone())
            .collect()
    }
    
    /// Check if the path contains a specific concept
    pub fn contains_concept(&self, concept_id: &ConceptId) -> bool {
        self.segments.iter()
            .any(|segment| segment.concepts.contains(concept_id))
    }
    
    /// Check if the path contains a specific objective
    pub fn contains_objective(&self, objective_id: &ObjectiveId) -> bool {
        self.segments.iter()
            .any(|segment| segment.objectives.contains(objective_id))
    }
    
    /// Get the position of a concept in the path (segment index)
    pub fn concept_position(&self, concept_id: &ConceptId) -> Option<usize> {
        self.segments.iter()
            .position(|segment| segment.concepts.contains(concept_id))
    }
    
    /// Get the position of an objective in the path (segment index)
    pub fn objective_position(&self, objective_id: &ObjectiveId) -> Option<usize> {
        self.segments.iter()
            .position(|segment| segment.objectives.contains(objective_id))
    }
    
    /// Check if a concept precedes another in the path
    pub fn concept_precedes(&self, first: &ConceptId, second: &ConceptId) -> bool {
        let first_pos = self.concept_position(first);
        let second_pos = self.concept_position(second);
        
        match (first_pos, second_pos) {
            (Some(first_idx), Some(second_idx)) => first_idx < second_idx,
            _ => false,
        }
    }
    
    /// Get path segment containing a specific concept
    pub fn segment_for_concept(&self, concept_id: &ConceptId) -> Option<&PathSegment> {
        self.segments.iter()
            .find(|segment| segment.concepts.contains(concept_id))
    }
    
    /// Get path segment containing a specific objective
    pub fn segment_for_objective(&self, objective_id: &ObjectiveId) -> Option<&PathSegment> {
        self.segments.iter()
            .find(|segment| segment.objectives.contains(objective_id))
    }
}

/// Node in concept-level knowledge graph
#[derive(Debug)]
struct ConceptNode {
    /// Concept metadata
    metadata: ConceptMetadata,
    
    /// Outgoing edges (prerequisites of other concepts)
    outgoing: HashMap<ConceptId, f32>, // Target -> Strength
    
    /// Incoming edges (concepts that are prerequisites for this one)
    incoming: HashMap<ConceptId, f32>, // Source -> Strength
}

/// Node in objective-level knowledge graph
#[derive(Debug)]
struct ObjectiveNode {
    /// Objective metadata
    metadata: ObjectiveMetadata,
    
    /// Outgoing edges (prerequisites of other objectives)
    outgoing: HashMap<ObjectiveId, f32>, // Target -> Strength
    
    /// Incoming edges (objectives that are prerequisites for this one)
    incoming: HashMap<ObjectiveId, f32>, // Source -> Strength
}

/// Priority queue entry for path generation
#[derive(Debug, Clone)]
struct PathEntry<T> {
    /// Node identifier
    id: T,
    
    /// Cumulative path cost
    cost: f32,
}

impl<T: Eq> PartialEq for PathEntry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T: Eq> Eq for PathEntry<T> {}

// Implement Ord and PartialOrd for use in BinaryHeap (min-heap)
impl<T: Eq> Ord for PathEntry<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (lower cost = higher priority)
        other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
    }
}

impl<T: Eq> PartialOrd for PathEntry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Knowledge graph for learning paths
#[derive(Debug)]
pub struct KnowledgeGraph {
    /// Concept nodes
    concepts: HashMap<ConceptId, ConceptNode>,
    
    /// Objective nodes
    objectives: HashMap<ObjectiveId, ObjectiveNode>,
    
    /// Top-level domains
    domains: HashSet<DomainId>,
    
    /// Concept-to-objectives mapping
    concept_objectives: HashMap<ConceptId, Vec<ObjectiveId>>,
    
    /// Domain-to-concepts mapping
    domain_concepts: HashMap<DomainId, Vec<ConceptId>>,
}

impl KnowledgeGraph {
    /// Create a new empty knowledge graph
    pub fn new() -> Self {
        Self {
            concepts: HashMap::new(),
            objectives: HashMap::new(),
            domains: HashSet::new(),
            concept_objectives: HashMap::new(),
            domain_concepts: HashMap::new(),
        }
    }
    
    /// Add a domain to the knowledge graph
    pub fn add_domain(&mut self, domain_id: DomainId) {
        self.domains.insert(domain_id);
    }
    
    /// Add a concept to the knowledge graph
    pub fn add_concept(&mut self, metadata: ConceptMetadata) -> Result<(), PathError> {
        let concept_id = metadata.id.clone();
        
        // Add domain if not exists
        self.domains.insert(concept_id.domain.clone());
        
        // Create concept node
        let node = ConceptNode {
            metadata,
            outgoing: HashMap::new(),
            incoming: HashMap::new(),
        };
        
        // Add to graph
        self.concepts.insert(concept_id.clone(), node);
        
        // Add to domain-concepts mapping
        self.domain_concepts.entry(concept_id.domain.clone())
            .or_insert_with(Vec::new)
            .push(concept_id);
        
        Ok(())
    }
    
    /// Add an objective to the knowledge graph
    pub fn add_objective(&mut self, metadata: ObjectiveMetadata) -> Result<(), PathError> {
        let objective_id = metadata.id.clone();
        let concept_id = objective_id.concept.clone();
        
        // Check if concept exists
        if !self.concepts.contains_key(&concept_id) {
            return Err(PathError::NodeNotFound(format!("Concept not found: {}", concept_id)));
        }
        
        // Create objective node
        let node = ObjectiveNode {
            metadata,
            outgoing: HashMap::new(),
            incoming: HashMap::new(),
        };
        
        // Add to graph
        self.objectives.insert(objective_id.clone(), node);
        
        // Add to concept-objectives mapping
        self.concept_objectives.entry(concept_id)
            .or_insert_with(Vec::new)
            .push(objective_id);
        
        Ok(())
    }
    
    /// Add a concept prerequisite relationship
    pub fn add_concept_prerequisite(&mut self, prerequisite: Prerequisite<ConceptId>) -> Result<(), PathError> {
        let source = prerequisite.source.clone();
        let target = prerequisite.target.clone();
        let strength = prerequisite.strength;
        
        // Check if concepts exist
        if !self.concepts.contains_key(&source) {
            return Err(PathError::NodeNotFound(format!("Source concept not found: {}", source)));
        }
        if !self.concepts.contains_key(&target) {
            return Err(PathError::NodeNotFound(format!("Target concept not found: {}", target)));
        }
        
        // Add prerequisite relationship
        if let Some(node) = self.concepts.get_mut(&source) {
            node.outgoing.insert(target.clone(), strength);
        }
        if let Some(node) = self.concepts.get_mut(&target) {
            node.incoming.insert(source.clone(), strength);
        }
        
        // Check for cycles
        if self.has_cycle_concepts() {
            // Remove the relationship that caused the cycle
            if let Some(node) = self.concepts.get_mut(&source) {
                node.outgoing.remove(&target);
            }
            if let Some(node) = self.concepts.get_mut(&target) {
                node.incoming.remove(&source);
            }
            
            return Err(PathError::CycleDetected(format!(
                "Adding prerequisite from {} to {} would create a cycle",
                source, target
            )));
        }
        
        Ok(())
    }
    
    /// Add an objective prerequisite relationship
    pub fn add_objective_prerequisite(&mut self, prerequisite: Prerequisite<ObjectiveId>) -> Result<(), PathError> {
        let source = prerequisite.source.clone();
        let target = prerequisite.target.clone();
        let strength = prerequisite.strength;
        
        // Check if objectives exist
        if !self.objectives.contains_key(&source) {
            return Err(PathError::NodeNotFound(format!("Source objective not found: {}", source)));
        }
        if !self.objectives.contains_key(&target) {
            return Err(PathError::NodeNotFound(format!("Target objective not found: {}", target)));
        }
        
        // Add prerequisite relationship
        if let Some(node) = self.objectives.get_mut(&source) {
            node.outgoing.insert(target.clone(), strength);
        }
        if let Some(node) = self.objectives.get_mut(&target) {
            node.incoming.insert(source.clone(), strength);
        }
        
        // Check for cycles
        if self.has_cycle_objectives() {
            // Remove the relationship that caused the cycle
            if let Some(node) = self.objectives.get_mut(&source) {
                node.outgoing.remove(&target);
            }
            if let Some(node) = self.objectives.get_mut(&target) {
                node.incoming.remove(&source);
            }
            
            return Err(PathError::CycleDetected(format!(
                "Adding prerequisite from {} to {} would create a cycle",
                source, target
            )));
        }
        
        Ok(())
    }
    
    /// Check if the concept graph has a cycle
    fn has_cycle_concepts(&self) -> bool {
        let mut visited = HashSet::new();
        let mut stack = HashSet::new();
        
        for concept_id in self.concepts.keys() {
            if self.has_cycle_concepts_dfs(concept_id, &mut visited, &mut stack) {
                return true;
            }
        }
        
        false
    }
    
    /// Depth-first search for cycle detection in concept graph
    fn has_cycle_concepts_dfs(&self, 
                             concept_id: &ConceptId, 
                             visited: &mut HashSet<ConceptId>, 
                             stack: &mut HashSet<ConceptId>) -> bool {
        // If already in stack, we found a cycle
        if stack.contains(concept_id) {
            return true;
        }
        
        // If already visited (but not in stack), no cycle through this node
        if visited.contains(concept_id) {
            return false;
        }
        
        // Mark as visited and add to stack
        visited.insert(concept_id.clone());
        stack.insert(concept_id.clone());
        
        // Check all outgoing edges
        if let Some(node) = self.concepts.get(concept_id) {
            for target in node.outgoing.keys() {
                if self.has_cycle_concepts_dfs(target, visited, stack) {
                    return true;
                }
            }
        }
        
        // Remove from stack (backtrack)
        stack.remove(concept_id);
        
        false
    }
    
    /// Check if the objective graph has a cycle
    fn has_cycle_objectives(&self) -> bool {
        let mut visited = HashSet::new();
        let mut stack = HashSet::new();
        
        for objective_id in self.objectives.keys() {
            if self.has_cycle_objectives_dfs(objective_id, &mut visited, &mut stack) {
                return true;
            }
        }
        
        false
    }
    
    /// Depth-first search for cycle detection in objective graph
    fn has_cycle_objectives_dfs(&self, 
                              objective_id: &ObjectiveId, 
                              visited: &mut HashSet<ObjectiveId>, 
                              stack: &mut HashSet<ObjectiveId>) -> bool {
        // If already in stack, we found a cycle
        if stack.contains(objective_id) {
            return true;
        }
        
        // If already visited (but not in stack), no cycle through this node
        if visited.contains(objective_id) {
            return false;
        }
        
        // Mark as visited and add to stack
        visited.insert(objective_id.clone());
        stack.insert(objective_id.clone());
        
        // Check all outgoing edges
        if let Some(node) = self.objectives.get(objective_id) {
            for target in node.outgoing.keys() {
                if self.has_cycle_objectives_dfs(target, visited, stack) {
                    return true;
                }
            }
        }
        
        // Remove from stack (backtrack)
        stack.remove(objective_id);
        
        false
    }
    
    /// Get all concepts in the graph
    pub fn get_all_concepts(&self) -> Vec<ConceptId> {
        self.concepts.keys().cloned().collect()
    }
    
    /// Get all objectives in the graph
    pub fn get_all_objectives(&self) -> Vec<ObjectiveId> {
        self.objectives.keys().cloned().collect()
    }
    
    /// Get concepts in a specific domain
    pub fn get_domain_concepts(&self, domain_id: &DomainId) -> Vec<ConceptId> {
        self.domain_concepts.get(domain_id)
            .cloned()
            .unwrap_or_else(Vec::new)
    }
    
    /// Get objectives for a specific concept
    pub fn get_concept_objectives(&self, concept_id: &ConceptId) -> Vec<ObjectiveId> {
        self.concept_objectives.get(concept_id)
            .cloned()
            .unwrap_or_else(Vec::new)
    }
    
    /// Get concept metadata
    pub fn get_concept_metadata(&self, concept_id: &ConceptId) -> Option<ConceptMetadata> {
        self.concepts.get(concept_id).map(|node| node.metadata.clone())
    }
    
    /// Get objective metadata
    pub fn get_objective_metadata(&self, objective_id: &ObjectiveId) -> Option<ObjectiveMetadata> {
        self.objectives.get(objective_id).map(|node| node.metadata.clone())
    }
    
    /// Get direct prerequisites for a concept
    pub fn get_concept_prerequisites(&self, concept_id: &ConceptId) -> Vec<(ConceptId, f32)> {
        if let Some(node) = self.concepts.get(concept_id) {
            node.incoming.iter()
                .map(|(src, strength)| (src.clone(), *strength))
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Get direct prerequisites for an objective
    pub fn get_objective_prerequisites(&self, objective_id: &ObjectiveId) -> Vec<(ObjectiveId, f32)> {
        if let Some(node) = self.objectives.get(objective_id) {
            node.incoming.iter()
                .map(|(src, strength)| (src.clone(), *strength))
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Get all transitive prerequisites for a concept
    pub fn get_all_concept_prerequisites(&self, concept_id: &ConceptId) -> HashMap<ConceptId, f32> {
        let mut result = HashMap::new();
        let mut queue = VecDeque::new();
        
        // Start with direct prerequisites
        if let Some(node) = self.concepts.get(concept_id) {
            for (src, strength) in &node.incoming {
                queue.push_back((src.clone(), *strength));
            }
        }
        
        // Process queue
        while let Some((current, strength)) = queue.pop_front() {
            // Skip if already processed with higher strength
            if let Some(existing_strength) = result.get(&current) {
                if *existing_strength >= strength {
                    continue;
                }
            }
            
            // Add to result
            result.insert(current.clone(), strength);
            
            // Add prerequisites of current node
            if let Some(node) = self.concepts.get(&current) {
                for (src, src_strength) in &node.incoming {
                    // Combine strengths (taking minimum as the weakest link)
                    let combined_strength = strength.min(*src_strength);
                    queue.push_back((src.clone(), combined_strength));
                }
            }
        }
        
        result
    }
    
    /// Generate a learning path from current knowledge state to target concepts
    pub fn generate_path(&self, 
                       knowledge_state: &KnowledgeState, 
                       target_concepts: &[ConceptId], 
                       strategy: PathStrategy, 
                       weights: Option<PathWeights>) -> Result<LearningPath, PathError> {
        // Validate inputs
        for concept_id in target_concepts {
            if !self.concepts.contains_key(concept_id) {
                return Err(PathError::NodeNotFound(format!("Target concept not found: {}", concept_id)));
            }
        }
        
        // Get all prerequisites for target concepts
        let mut all_prerequisites = HashMap::new();
        for concept_id in target_concepts {
            let prereqs = self.get_all_concept_prerequisites(concept_id);
            for (prereq_id, strength) in prereqs {
                all_prerequisites.insert(prereq_id, strength);
            }
        }
        
        // Add target concepts themselves
        for concept_id in target_concepts {
            all_prerequisites.insert(concept_id.clone(), 1.0);
        }
        
        // Filter out concepts that are already known
        all_prerequisites.retain(|concept_id, _| {
            !knowledge_state.is_concept_known(concept_id)
        });
        
        // If empty, all targets are already known
        if all_prerequisites.is_empty() {
            return Ok(LearningPath {
                segments: Vec::new(),
                total_time_minutes: 0,
                average_complexity: 0.0,
                average_importance: 0.0,
                strategy,
                weights: weights.clone(),
                timestamp: chrono::Utc::now(),
                user_profile_id: knowledge_state.user_id.clone(),
            });
        }
        
        // Generate topological ordering
        let ordering = self.topological_sort_concepts(&all_prerequisites, strategy, weights.clone())?;
        
        // Create path segments based on ordering
        let mut segments = Vec::new();
        let mut current_segment = Vec::new();
        let mut segment_time = 0;
        let mut segment_complexity = 0.0;
        let mut segment_importance = 0.0;
        
        for concept_id in ordering {
            let metadata = self.get_concept_metadata(&concept_id).unwrap();
            
            // Start a new segment if this would make current one too long
            if !current_segment.is_empty() && segment_time + metadata.estimated_time_minutes > 120 {
                // Finalize current segment
                let objectives = current_segment.iter()
                    .flat_map(|c| self.get_concept_objectives(c))
                    .collect();
                
                segments.push(PathSegment {
                    concepts: current_segment,
                    objectives,
                    estimated_time_minutes: segment_time,
                    complexity: segment_complexity / current_segment.len() as f32,
                    importance: segment_importance / current_segment.len() as f32,
                    description: None,
                });
                
                // Start new segment
                current_segment = Vec::new();
                segment_time = 0;
                segment_complexity = 0.0;
                segment_importance = 0.0;
            }
            
            // Add to current segment
            current_segment.push(concept_id);
            segment_time += metadata.estimated_time_minutes;
            segment_complexity += metadata.complexity;
            segment_importance += metadata.importance;
        }
        
        // Finalize last segment if not empty
        if !current_segment.is_empty() {
            let objectives = current_segment.iter()
                .flat_map(|c| self.get_concept_objectives(c))
                .collect();
            
            segments.push(PathSegment {
                concepts: current_segment,
                objectives,
                estimated_time_minutes: segment_time,
                complexity: segment_complexity / current_segment.len() as f32,
                importance: segment_importance / current_segment.len() as f32,
                description: None,
            });
        }
        
        // Calculate total metrics
        let total_time = segments.iter()
            .map(|s| s.estimated_time_minutes)
            .sum();
        
        let total_concepts = segments.iter()
            .map(|s| s.concepts.len())
            .sum::<usize>();
        
        let avg_complexity = segments.iter()
            .map(|s| s.complexity * s.concepts.len() as f32)
            .sum::<f32>() / total_concepts as f32;
        
        let avg_importance = segments.iter()
            .map(|s| s.importance * s.concepts.len() as f32)
            .sum::<f32>() / total_concepts as f32;
        
        // Create complete learning path
        Ok(LearningPath {
            segments,
            total_time_minutes: total_time,
            average_complexity: avg_complexity,
            average_importance: avg_importance,
            strategy,
            weights,
            timestamp: chrono::Utc::now(),
            user_profile_id: knowledge_state.user_id.clone(),
        })
    }
    
    /// Topological sort of concepts with optimized ordering
    fn topological_sort_concepts(&self, 
                               concepts: &HashMap<ConceptId, f32>,
                               strategy: PathStrategy,
                               weights: Option<PathWeights>) -> Result<Vec<ConceptId>, PathError> {
        // Convert selected concepts to graph for topological sort
        let mut graph: HashMap<ConceptId, HashSet<ConceptId>> = HashMap::new();
        let mut in_degree: HashMap<ConceptId, usize> = HashMap::new();
        
        // Initialize in-degree to 0 for all nodes
        for concept_id in concepts.keys() {
            in_degree.insert(concept_id.clone(), 0);
            graph.insert(concept_id.clone(), HashSet::new());
        }
        
        // Build graph and compute in-degree
        for concept_id in concepts.keys() {
            if let Some(node) = self.concepts.get(concept_id) {
                for (prereq_id, _) in node.incoming.iter() {
                    // Only include edges between selected concepts
                    if concepts.contains_key(prereq_id) {
                        graph.entry(prereq_id.clone())
                            .or_insert_with(HashSet::new)
                            .insert(concept_id.clone());
                        
                        *in_degree.entry(concept_id.clone()).or_insert(0) += 1;
                    }
                }
            }
        }
        
        // Kahn's algorithm with priority queue
        let mut queue = BinaryHeap::new();
        let mut result = Vec::new();
        
        // Add nodes with in-degree 0 to queue
        for (concept_id, degree) in &in_degree {
            if *degree == 0 {
                let cost = self.calculate_node_cost(concept_id, strategy, weights.as_ref());
                queue.push(PathEntry { id: concept_id.clone(), cost });
            }
        }
        
        // Process queue
        while let Some(entry) = queue.pop() {
            let concept_id = entry.id;
            
            // Add to result
            result.push(concept_id.clone());
            
            // Update in-degree of adjacent nodes
            if let Some(adjacent) = graph.get(&concept_id) {
                for adj_id in adjacent {
                    if let Some(degree) = in_degree.get_mut(adj_id) {
                        *degree -= 1;
                        
                        // If in-degree becomes 0, add to queue
                        if *degree == 0 {
                            let cost = self.calculate_node_cost(adj_id, strategy, weights.as_ref());
                            queue.push(PathEntry { id: adj_id.clone(), cost });
                        }
                    }
                }
            }
        }
        
        // Check if all nodes were visited
        if result.len() != concepts.len() {
            return Err(PathError::CycleDetected(
                "Cycle detected in selected concepts".to_string()
            ));
        }
        
        Ok(result)
    }
    
    /// Calculate cost for priority queue based on strategy
    fn calculate_node_cost(&self, 
                         concept_id: &ConceptId, 
                         strategy: PathStrategy,
                         weights: Option<&PathWeights>) -> f32 {
        let metadata = match self.get_concept_metadata(concept_id) {
            Some(meta) => meta,
            None => return f32::MAX, // Should never happen
        };
        
        match strategy {
            PathStrategy::Shortest => 1.0, // All nodes have equal weight for shortest path
            
            PathStrategy::Fastest => metadata.estimated_time_minutes as f32,
            
            PathStrategy::Easiest => metadata.complexity,
            
            PathStrategy::ImportanceFirst => 1.0 - metadata.importance, // Lower cost for higher importance
            
            PathStrategy::Balanced => {
                // Combine factors with equal weights
                let time_factor = metadata.estimated_time_minutes as f32 / 120.0; // Normalize to ~1.0
                let complexity_factor = metadata.complexity;
                let importance_factor = 1.0 - metadata.importance;
                
                (time_factor + complexity_factor + importance_factor) / 3.0
            },
            
            PathStrategy::Custom => {
                // Use custom weights
                let weights = weights.unwrap_or(&PathWeights::default());
                
                let time_factor = metadata.estimated_time_minutes as f32 / 120.0; // Normalize to ~1.0
                let complexity_factor = metadata.complexity;
                let importance_factor = 1.0 - metadata.importance;
                
                // Calculate weighted sum
                let total_weight = weights.time_weight + weights.complexity_weight + weights.importance_weight;
                let weighted_sum = (time_factor * weights.time_weight) +
                                   (complexity_factor * weights.complexity_weight) +
                                   (importance_factor * weights.importance_weight);
                
                weighted_sum / total_weight
            },
        }
    }
    
    /// Check if a learning path is valid (preserves prerequisite relationships)
    pub fn validate_path(&self, path: &LearningPath) -> Result<(), PathError> {
        // Get all concepts in the path
        let all_concepts: Vec<ConceptId> = path.all_concepts();
        
        // Check prerequisite relationships
        for concept_id in &all_concepts {
            if let Some(node) = self.concepts.get(concept_id) {
                for (prereq_id, strength) in &node.incoming {
                    // Skip optional prerequisites
                    if *strength < 0.5 {
                        continue;
                    }
                    
                    // If prerequisite is in the path, it must come before this concept
                    if all_concepts.contains(prereq_id) && !path.concept_precedes(prereq_id, concept_id) {
                        return Err(PathError::InconsistentPrerequisites(format!(
                            "Prerequisite {} must come before {} in the path",
                            prereq_id, concept_id
                        )));
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Check for consistency of prerequisites
    pub fn validate_prerequisites(&self) -> Result<(), PathError> {
        // Check for cycles
        if self.has_cycle_concepts() {
            return Err(PathError::CycleDetected(
                "Cycle detected in concept graph".to_string()
            ));
        }
        
        if self.has_cycle_objectives() {
            return Err(PathError::CycleDetected(
                "Cycle detected in objective graph".to_string()
            ));
        }
        
        // Check for concept-level consistency
        // (if A requires B and B requires C, then A should require C)
        for concept_id in self.concepts.keys() {
            let direct_prereqs = self.get_concept_prerequisites(concept_id);
            let transitive_prereqs = self.get_all_concept_prerequisites(concept_id);
            
            for (direct_id, direct_strength) in &direct_prereqs {
                if let Some(node) = self.concepts.get(direct_id) {
                    for (indirect_id, indirect_strength) in &node.incoming {
                        let combined_strength = direct_strength.min(*indirect_strength);
                        
                        // Check if the transitive prerequisite is properly represented
                        match transitive_prereqs.get(indirect_id) {
                            Some(existing_strength) => {
                                if *existing_strength < combined_strength - 0.01 {
                                    // The transitive prerequisite exists but with too low strength
                                    return Err(PathError::InconsistentPrerequisites(format!(
                                        "Inconsistent transitive prerequisite strength: {} -> {} -> {}",
                                        indirect_id, direct_id, concept_id
                                    )));
                                }
                            },
                            None => {
                                // Missing transitive prerequisite
                                return Err(PathError::InconsistentPrerequisites(format!(
                                    "Missing transitive prerequisite: {} -> {} -> {}",
                                    indirect_id, direct_id, concept_id
                                )));
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Find knowledge gaps between current state and target concepts
    pub fn find_knowledge_gaps(&self, 
                             knowledge_state: &KnowledgeState, 
                             target_concepts: &[ConceptId]) -> Vec<ConceptId> {
        let mut gaps = Vec::new();
        
        // Get all prerequisites for target concepts
        for concept_id in target_concepts {
            if !knowledge_state.is_concept_known(concept_id) {
                gaps.push(concept_id.clone());
            }
            
            let prereqs = self.get_all_concept_prerequisites(concept_id);
            for (prereq_id, strength) in prereqs {
                // Only include required prerequisites
                if strength >= 0.5 && !knowledge_state.is_concept_known(&prereq_id) {
                    gaps.push(prereq_id);
                }
            }
        }
        
        // Deduplicate
        gaps.sort();
        gaps.dedup();
        
        gaps
    }
    
    /// Recommend next concepts to learn based on current knowledge state
    pub fn recommend_next_concepts(&self, 
                                 knowledge_state: &KnowledgeState, 
                                 limit: usize) -> Vec<ConceptId> {
        let mut candidates = Vec::new();
        
        // Find concepts with all prerequisites satisfied
        for concept_id in self.concepts.keys() {
            // Skip if already known
            if knowledge_state.is_concept_known(concept_id) {
                continue;
            }
            
            // Check prerequisites
            let prereqs = self.get_concept_prerequisites(concept_id);
            let all_prereqs_satisfied = prereqs.iter()
                .all(|(prereq_id, strength)| {
                    // Required prerequisite must be known
                    if *strength >= 0.5 {
                        knowledge_state.is_concept_known(prereq_id)
                    } else {
                        // Optional prerequisite, can be unknown
                        true
                    }
                });
            
            if all_prereqs_satisfied {
                candidates.push(concept_id.clone());
            }
        }
        
        // Sort by importance
        candidates.sort_by(|a, b| {
            let a_meta = self.get_concept_metadata(a).unwrap();
            let b_meta = self.get_concept_metadata(b).unwrap();
            b_meta.importance.partial_cmp(&a_meta.importance).unwrap_or(Ordering::Equal)
        });
        
        // Return top N candidates
        candidates.into_iter().take(limit).collect()
    }
}

/// Learning path manager for path generation and tracking
pub struct LearningPathManager {
    /// Knowledge graph
    graph: Arc<RwLock<KnowledgeGraph>>,
    
    /// User learning paths
    user_paths: HashMap<String, LearningPath>,
}

impl LearningPathManager {
    /// Create a new learning path manager with the given knowledge graph
    pub fn new(graph: Arc<RwLock<KnowledgeGraph>>) -> Self {
        Self {
            graph,
            user_paths: HashMap::new(),
        }
    }
    
    /// Generate a learning path for a user
    pub fn generate_path(&mut self, 
                       knowledge_state: &KnowledgeState, 
                       target_concepts: &[ConceptId], 
                       strategy: PathStrategy, 
                       weights: Option<PathWeights>) -> Result<LearningPath, PathError> {
        // Get read lock on graph
        let graph = self.graph.read()
            .map_err(|e| PathError::Other(format!("Failed to acquire read lock: {}", e)))?;
        
        // Generate path
        let path = graph.generate_path(knowledge_state, target_concepts, strategy, weights)?;
        
        // Store path for user
        if let Some(user_id) = &knowledge_state.user_id {
            self.user_paths.insert(user_id.clone(), path.clone());
        }
        
        Ok(path)
    }
    
    /// Get current learning path for a user
    pub fn get_user_path(&self, user_id: &str) -> Option<&LearningPath> {
        self.user_paths.get(user_id)
    }
    
    /// Update user's progress in the learning path
    pub fn update_progress(&mut self, 
                         knowledge_state: &KnowledgeState, 
                         completed_concepts: &[ConceptId]) -> Result<(), PathError> {
        let user_id = match &knowledge_state.user_id {
            Some(id) => id,
            None => return Err(PathError::Other("User ID not specified".to_string())),
        };
        
        // Check if user has a path
        if !self.user_paths.contains_key(user_id) {
            return Err(PathError::Other(format!("No learning path for user {}", user_id)));
        }
        
        // Get current path
        let current_path = self.user_paths.get(user_id).unwrap().clone();
        
        // Get remaining concepts
        let remaining_concepts: Vec<ConceptId> = current_path.all_concepts().into_iter()
            .filter(|c| !completed_concepts.contains(c) && !knowledge_state.is_concept_known(c))
            .collect();
        
        // If no remaining concepts, the path is complete
        if remaining_concepts.is_empty() {
            // Remove completed path
            self.user_paths.remove(user_id);
            return Ok(());
        }
        
        // Regenerate path with remaining target concepts
        let new_path = self.generate_path(
            knowledge_state,
            &remaining_concepts,
            current_path.strategy,
            current_path.weights.clone()
        )?;
        
        // Update user path
        self.user_paths.insert(user_id.clone(), new_path);
        
        Ok(())
    }
    
    /// Find knowledge gaps between current state and learning goals
    pub fn find_knowledge_gaps(&self, 
                             knowledge_state: &KnowledgeState, 
                             target_concepts: &[ConceptId]) -> Result<Vec<ConceptId>, PathError> {
        // Get read lock on graph
        let graph = self.graph.read()
            .map_err(|e| PathError::Other(format!("Failed to acquire read lock: {}", e)))?;
        
        // Find gaps
        let gaps = graph.find_knowledge_gaps(knowledge_state, target_concepts);
        
        Ok(gaps)
    }
    
    /// Recommend next concepts to learn
    pub fn recommend_next_concepts(&self, 
                                 knowledge_state: &KnowledgeState, 
                                 limit: usize) -> Result<Vec<ConceptId>, PathError> {
        // Get read lock on graph
        let graph = self.graph.read()
            .map_err(|e| PathError::Other(format!("Failed to acquire read lock: {}", e)))?;
        
        // Get recommendations
        let recommendations = graph.recommend_next_concepts(knowledge_state, limit);
        
        Ok(recommendations)
    }
    
    /// Calculate expected learning time for a set of concepts
    pub fn calculate_learning_time(&self, 
                                concept_ids: &[ConceptId]) -> Result<u32, PathError> {
        // Get read lock on graph
        let graph = self.graph.read()
            .map_err(|e| PathError::Other(format!("Failed to acquire read lock: {}", e)))?;
        
        // Sum up estimated time for each concept
        let mut total_time = 0;
        
        for concept_id in concept_ids {
            if let Some(metadata) = graph.get_concept_metadata(concept_id) {
                total_time += metadata.estimated_time_minutes;
            } else {
                return Err(PathError::NodeNotFound(format!("Concept not found: {}", concept_id)));
            }
        }
        
        Ok(total_time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Helper function to create a test knowledge graph
    fn create_test_graph() -> KnowledgeGraph {
        let mut graph = KnowledgeGraph::new();
        
        // Add domains
        let math_domain = DomainId("math".to_string());
        let cs_domain = DomainId("cs".to_string());
        
        graph.add_domain(math_domain.clone());
        graph.add_domain(cs_domain.clone());
        
        // Add concepts for math domain
        let math_basics = ConceptId { domain: math_domain.clone(), id: "basics".to_string() };
        let algebra = ConceptId { domain: math_domain.clone(), id: "algebra".to_string() };
        let calculus = ConceptId { domain: math_domain.clone(), id: "calculus".to_string() };
        
        graph.add_concept(ConceptMetadata {
            id: math_basics.clone(),
            title: "Math Basics".to_string(),
            description: "Basic mathematics concepts".to_string(),
            min_expertise: ExpertiseLevel::Beginner,
            estimated_time_minutes: 60,
            complexity: 0.2,
            importance: 0.9,
            tags: vec!["math".to_string(), "basics".to_string()],
            additional: HashMap::new(),
        }).unwrap();
        
        graph.add_concept(ConceptMetadata {
            id: algebra.clone(),
            title: "Algebra".to_string(),
            description: "Algebraic concepts and operations".to_string(),
            min_expertise: ExpertiseLevel::Intermediate,
            estimated_time_minutes: 90,
            complexity: 0.4,
            importance: 0.8,
            tags: vec!["math".to_string(), "algebra".to_string()],
            additional: HashMap::new(),
        }).unwrap();
        
        graph.add_concept(ConceptMetadata {
            id: calculus.clone(),
            title: "Calculus".to_string(),
            description: "Differential and integral calculus".to_string(),
            min_expertise: ExpertiseLevel::Advanced,
            estimated_time_minutes: 120,
            complexity: 0.7,
            importance: 0.7,
            tags: vec!["math".to_string(), "calculus".to_string()],
            additional: HashMap::new(),
        }).unwrap();
        
        // Add concepts for CS domain
        let algorithms = ConceptId { domain: cs_domain.clone(), id: "algorithms".to_string() };
        let data_structures = ConceptId { domain: cs_domain.clone(), id: "data_structures".to_string() };
        
        graph.add_concept(ConceptMetadata {
            id: algorithms.clone(),
            title: "Algorithms".to_string(),
            description: "Algorithm design and analysis".to_string(),
            min_expertise: ExpertiseLevel::Intermediate,
            estimated_time_minutes: 100,
            complexity: 0.6,
            importance: 0.8,
            tags: vec!["cs".to_string(), "algorithms".to_string()],
            additional: HashMap::new(),
        }).unwrap();
        
        graph.add_concept(ConceptMetadata {
            id: data_structures.clone(),
            title: "Data Structures".to_string(),
            description: "Common data structures and their operations".to_string(),
            min_expertise: ExpertiseLevel::Intermediate,
            estimated_time_minutes: 80,
            complexity: 0.5,
            importance: 0.8,
            tags: vec!["cs".to_string(), "data structures".to_string()],
            additional: HashMap::new(),
        }).unwrap();
        
        // Add prerequisites
        graph.add_concept_prerequisite(Prerequisite::required(math_basics.clone(), algebra.clone())).unwrap();
        graph.add_concept_prerequisite(Prerequisite::required(algebra.clone(), calculus.clone())).unwrap();
        graph.add_concept_prerequisite(Prerequisite::required(math_basics.clone(), data_structures.clone())).unwrap();
        graph.add_concept_prerequisite(Prerequisite::required(data_structures.clone(), algorithms.clone())).unwrap();
        
        // Add a few objectives
        let algebra_obj1 = ObjectiveId { concept: algebra.clone(), id: "solve_equations".to_string() };
        let algebra_obj2 = ObjectiveId { concept: algebra.clone(), id: "factor_expressions".to_string() };
        
        graph.add_objective(ObjectiveMetadata {
            id: algebra_obj1.clone(),
            title: "Solve Equations".to_string(),
            description: "Solve linear and quadratic equations".to_string(),
            taxonomy_level: TaxonomyLevel::Apply,
            min_expertise: ExpertiseLevel::Intermediate,
            estimated_time_minutes: 30,
            complexity: 0.4,
            importance: 0.8,
            additional: HashMap::new(),
        }).unwrap();
        
        graph.add_objective(ObjectiveMetadata {
            id: algebra_obj2.clone(),
            title: "Factor Expressions".to_string(),
            description: "Factor algebraic expressions".to_string(),
            taxonomy_level: TaxonomyLevel::Apply,
            min_expertise: ExpertiseLevel::Intermediate,
            estimated_time_minutes: 25,
            complexity: 0.5,
            importance: 0.7,
            additional: HashMap::new(),
        }).unwrap();
        
        // Add objective prerequisites
        graph.add_objective_prerequisite(Prerequisite::required(algebra_obj1.clone(), algebra_obj2.clone())).unwrap();
        
        graph
    }
    
    #[test]
    fn test_knowledge_graph_basics() {
        let graph = create_test_graph();
        
        // Test domain concepts
        let math_domain = DomainId("math".to_string());
        let math_concepts = graph.get_domain_concepts(&math_domain);
        assert_eq!(math_concepts.len(), 3);
        
        // Test concept prerequisites
        let calculus = ConceptId { domain: math_domain.clone(), id: "calculus".to_string() };
        let prereqs = graph.get_concept_prerequisites(&calculus);
        assert_eq!(prereqs.len(), 1);
        
        // Test all prerequisites (transitive)
        let all_prereqs = graph.get_all_concept_prerequisites(&calculus);
        assert_eq!(all_prereqs.len(), 2); // algebra and math_basics
    }
    
    #[test]
    fn test_cycle_detection() {
        let mut graph = create_test_graph();
        
        // Try to create a cycle
        let math_domain = DomainId("math".to_string());
        let algebra = ConceptId { domain: math_domain.clone(), id: "algebra".to_string() };
        let calculus = ConceptId { domain: math_domain.clone(), id: "calculus".to_string() };
        
        // This should fail because it would create a cycle: calculus -> algebra -> calculus
        let result = graph.add_concept_prerequisite(Prerequisite::required(calculus.clone(), algebra.clone()));
        assert!(result.is_err());
        
        // Verify the error is a cycle detection
        match result {
            Err(PathError::CycleDetected(_)) => (),
            _ => panic!("Expected cycle detection error"),
        }
    }
    
    #[test]
    fn test_path_generation() {
        let graph = create_test_graph();
        
        // Create knowledge state
        let mut knowledge_state = KnowledgeState::new(Some("user1".to_string()));
        
        // Mark math_basics as known
        let math_domain = DomainId("math".to_string());
        let math_basics = ConceptId { domain: math_domain.clone(), id: "basics".to_string() };
        knowledge_state.set_concept_known(&math_basics, true);
        
        // Generate path to calculus
        let calculus = ConceptId { domain: math_domain.clone(), id: "calculus".to_string() };
        let path = graph.generate_path(&knowledge_state, &[calculus.clone()], PathStrategy::Shortest, None).unwrap();
        
        // Path should contain algebra and calculus (in that order)
        assert_eq!(path.segments.len(), 2); // Two segments due to time constraints
        
        // First segment should be algebra
        let algebra = ConceptId { domain: math_domain.clone(), id: "algebra".to_string() };
        assert!(path.segment_for_concept(&algebra).is_some());
        
        // Calculus should be in the path
        assert!(path.contains_concept(&calculus));
        
        // Math basics should not be in the path (already known)
        assert!(!path.contains_concept(&math_basics));
        
        // Algebra should precede calculus
        assert!(path.concept_precedes(&algebra, &calculus));
    }
    
    #[test]
    fn test_different_path_strategies() {
        let graph = create_test_graph();
        
        // Create knowledge state (nothing known)
        let knowledge_state = KnowledgeState::new(Some("user1".to_string()));
        
        // Target concept: algorithms in CS domain
        let cs_domain = DomainId("cs".to_string());
        let algorithms = ConceptId { domain: cs_domain.clone(), id: "algorithms".to_string() };
        
        // Generate paths with different strategies
        let shortest_path = graph.generate_path(
            &knowledge_state, 
            &[algorithms.clone()], 
            PathStrategy::Shortest,
            None
        ).unwrap();
        
        let fastest_path = graph.generate_path(
            &knowledge_state, 
            &[algorithms.clone()], 
            PathStrategy::Fastest,
            None
        ).unwrap();
        
        let easiest_path = graph.generate_path(
            &knowledge_state, 
            &[algorithms.clone()], 
            PathStrategy::Easiest,
            None
        ).unwrap();
        
        // All paths should lead to algorithms
        assert!(shortest_path.contains_concept(&algorithms));
        assert!(fastest_path.contains_concept(&algorithms));
        assert!(easiest_path.contains_concept(&algorithms));
        
        // Check for expected characteristics
        // Note: Due to constraints, the ordering may be the same for this small example
        // In more complex graphs, the different strategies would produce different paths
        assert_eq!(shortest_path.total_time_minutes, easiest_path.total_time_minutes);
    }
    
    #[test]
    fn test_knowledge_gaps() {
        let graph = create_test_graph();
        
        // Create knowledge state
        let mut knowledge_state = KnowledgeState::new(Some("user1".to_string()));
        
        // Mark math_basics as known
        let math_domain = DomainId("math".to_string());
        let math_basics = ConceptId { domain: math_domain.clone(), id: "basics".to_string() };
        knowledge_state.set_concept_known(&math_basics, true);
        
        // Find gaps for calculus
        let calculus = ConceptId { domain: math_domain.clone(), id: "calculus".to_string() };
        let gaps = graph.find_knowledge_gaps(&knowledge_state, &[calculus.clone()]);
        
        // Gaps should include algebra and calculus
        assert_eq!(gaps.len(), 2);
        
        let algebra = ConceptId { domain: math_domain.clone(), id: "algebra".to_string() };
        assert!(gaps.contains(&algebra));
        assert!(gaps.contains(&calculus));
        
        // Mark algebra as known
        knowledge_state.set_concept_known(&algebra, true);
        
        // Find gaps again
        let gaps = graph.find_knowledge_gaps(&knowledge_state, &[calculus.clone()]);
        
        // Now only calculus should be in gaps
        assert_eq!(gaps.len(), 1);
        assert!(gaps.contains(&calculus));
    }
    
    #[test]
    fn test_learning_path_manager() {
        let graph = Arc::new(RwLock::new(create_test_graph()));
        let mut manager = LearningPathManager::new(graph);
        
        // Create knowledge state
        let mut knowledge_state = KnowledgeState::new(Some("user1".to_string()));
        
        // Mark math_basics as known
        let math_domain = DomainId("math".to_string());
        let math_basics = ConceptId { domain: math_domain.clone(), id: "basics".to_string() };
        knowledge_state.set_concept_known(&math_basics, true);
        
        // Generate path to calculus
        let calculus = ConceptId { domain: math_domain.clone(), id: "calculus".to_string() };
        let path = manager.generate_path(&knowledge_state, &[calculus.clone()], PathStrategy::Shortest, None).unwrap();
        
        // Path should be stored for user
        let user_path = manager.get_user_path("user1").unwrap();
        assert_eq!(user_path.segments.len(), path.segments.len());
        
        // Mark algebra as completed
        let algebra = ConceptId { domain: math_domain.clone(), id: "algebra".to_string() };
        knowledge_state.set_concept_known(&algebra, true);
        
        // Update progress
        manager.update_progress(&knowledge_state, &[algebra.clone()]).unwrap();
        
        // Path should be updated
        let updated_path = manager.get_user_path("user1").unwrap();
        
        // Updated path should only contain calculus
        let all_concepts = updated_path.all_concepts();
        assert_eq!(all_concepts.len(), 1);
        assert!(all_concepts.contains(&calculus));
    }
}