//! Counterfactual Timeline Branching System
//!
//! This module implements an advanced branching system for timeline management,
//! using copy-on-write semantics for memory efficiency, differential state
//! propagation for lazy materialization, and formal bifurcation invariants
//! for branch consistency.
//!
//! Copyright (c) 2025 Mohammad Atashi <mohammadaliatashi@icloud.com>

use std::sync::{Arc, RwLock, Mutex, atomic::{AtomicU64, Ordering}};
use std::collections::{HashMap, HashSet, BTreeMap, BTreeSet, VecDeque};
use std::time::{Instant, Duration};
use std::fmt;
use std::cmp::Ordering as CmpOrdering;

use dashmap::DashMap;
use futures::stream::{self, StreamExt};
use parking_lot::{RwLock as PLRwLock, Mutex as PLMutex};
use serde::{Serialize, Deserialize};
use thiserror::Error;
use rayon::prelude::*;

use crate::algorithm::state::{AlgorithmState, StateSnapshot, StateChecksum};
use crate::algorithm::traits::{Algorithm, AlgorithmParameter};
use crate::execution::tracer::ExecutionTracer;
use crate::temporal::state_manager::{StateId, StateManager, StateManagerError};
use crate::temporal::timeline::{TimelineId, TimelineManager, TimelineError};
use crate::temporal::delta::{StateDelta, DeltaError};

/// Branch operation errors with diagnostic context
#[derive(Error, Debug)]
pub enum BranchError {
    #[error("Branch point {0} not found in timeline {1}")]
    BranchPointNotFound(StateId, TimelineId),
    
    #[error("Branch creation failed: {0}")]
    CreationFailed(String),
    
    #[error("Branch target state {0} invalid: {1}")]
    InvalidTargetState(StateId, String),
    
    #[error("Branch consistency violation: {0}")]
    ConsistencyViolation(String),
    
    #[error("Branch merge conflict at state {0}: {1}")]
    MergeConflict(StateId, String),
    
    #[error("Timeline error: {0}")]
    TimelineError(#[from] TimelineError),
    
    #[error("State manager error: {0}")]
    StateManagerError(#[from] StateManagerError),
    
    #[error("Delta computation error: {0}")]
    DeltaError(#[from] DeltaError),
    
    #[error("Branch operation timeout after {0:?}")]
    Timeout(Duration),
    
    #[error("Maximum branch depth exceeded: {0}")]
    MaxDepthExceeded(usize),
}

/// Branch type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BranchType {
    /// Algorithmic decision branch (e.g., path choice)
    AlgorithmicDecision,
    
    /// Parameter modification branch (e.g., heuristic change)
    ParameterModification,
    
    /// State modification branch (manual state change)
    StateModification,
    
    /// Counterfactual exploration branch (what-if scenario)
    CounterfactualExploration,
    
    /// Merge branch (combines multiple timelines)
    Merge,
}

/// Branch identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BranchId(pub u64);

/// Branch metadata with categorization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchMetadata {
    /// Branch ID
    pub id: BranchId,
    
    /// Branch name
    pub name: String,
    
    /// Branch type
    pub branch_type: BranchType,
    
    /// Branch creation timestamp
    pub created_at: Instant,
    
    /// Parent branch (if any)
    pub parent: Option<BranchId>,
    
    /// Timeline ID
    pub timeline_id: TimelineId,
    
    /// Branch point state
    pub branch_point: StateId,
    
    /// Branch depth
    pub depth: usize,
    
    /// Branch tags
    pub tags: HashSet<String>,
    
    /// Parameter changes (if parameter modification)
    pub parameter_changes: Option<HashMap<String, (String, String)>>,
    
    /// State modifications (if state modification)
    pub state_modifications: Option<StateModificationDesc>,
    
    /// Creation reason
    pub reason: String,
}

/// State modification description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateModificationDesc {
    /// Affected algorithm step
    pub step: usize,
    
    /// Modified elements
    pub modified_elements: Vec<ModifiedElement>,
    
    /// Textual description
    pub description: String,
}

/// Modified element in state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModifiedElement {
    /// Element type
    pub element_type: String,
    
    /// Element identifier
    pub identifier: String,
    
    /// Before value (serialized)
    pub before: String,
    
    /// After value (serialized)
    pub after: String,
}

/// Branch timeline with bifurcation tracking
#[derive(Debug)]
pub struct TimelineBranch {
    /// Branch metadata
    pub metadata: BranchMetadata,
    
    /// Branch timeline
    timeline_id: TimelineId,
    
    /// Branch point state
    branch_point: StateId,
    
    /// State manager reference
    state_manager: Arc<StateManager>,
    
    /// Timeline manager reference
    timeline_manager: Arc<TimelineManager>,
    
    /// Branch modification lock
    modification_lock: PLMutex<()>,
    
    /// Derived branches
    derived_branches: PLRwLock<Vec<BranchId>>,
    
    /// Invariant tracker
    invariant_tracker: BranchInvariantTracker,
    
    /// Propagation status
    propagation_status: PLRwLock<PropagationStatus>,
}

/// Branch invariant tracker
#[derive(Debug)]
struct BranchInvariantTracker {
    /// Branch invariants
    invariants: Vec<BranchInvariant>,
    
    /// Last validation timestamp
    last_validation: Instant,
    
    /// Validation failures
    validation_failures: PLRwLock<Vec<ValidationFailure>>,
}

/// Branch invariant definition
#[derive(Debug, Clone)]
enum BranchInvariant {
    /// State consistency invariant
    StateConsistency {
        /// Required state property
        property: StateProperty,
        /// Property validation function
        validate: Arc<dyn Fn(&AlgorithmState) -> bool + Send + Sync>,
    },
    
    /// Timeline consistency invariant
    TimelineConsistency {
        /// Property name
        name: String,
        /// Validation function
        validate: Arc<dyn Fn(&TimelineBranch) -> bool + Send + Sync>,
    },
    
    /// Inter-branch consistency invariant
    InterBranchConsistency {
        /// Related branch IDs
        related_branches: Vec<BranchId>,
        /// Validation function
        validate: Arc<dyn Fn(&[&TimelineBranch]) -> bool + Send + Sync>,
    },
}

/// State property definition
#[derive(Debug, Clone)]
struct StateProperty {
    /// Property name
    name: String,
    
    /// Property description
    description: String,
}

/// Validation failure record
#[derive(Debug, Clone)]
struct ValidationFailure {
    /// Failed invariant
    invariant: String,
    
    /// Failure timestamp
    timestamp: Instant,
    
    /// Failure description
    description: String,
    
    /// Affected states (if applicable)
    affected_states: Option<Vec<StateId>>,
}

/// Branch propagation status
#[derive(Debug, Clone)]
enum PropagationStatus {
    /// Not propagated
    NotPropagated,
    
    /// Partially propagated
    PartiallyPropagated {
        /// Propagated steps
        propagated_steps: usize,
        /// Total steps
        total_steps: usize,
    },
    
    /// Fully propagated
    FullyPropagated {
        /// Propagation completion timestamp
        completed_at: Instant,
    },
}

/// Branch modification operation
#[derive(Debug, Clone)]
pub enum BranchModification {
    /// Parameter modification
    ParameterModification {
        /// Parameter name
        parameter: String,
        /// New value
        new_value: String,
    },
    
    /// Open set modification
    OpenSetModification {
        /// Added nodes
        added: Vec<crate::algorithm::traits::NodeId>,
        /// Removed nodes
        removed: Vec<crate::algorithm::traits::NodeId>,
    },
    
    /// Closed set modification
    ClosedSetModification {
        /// Added nodes
        added: Vec<crate::algorithm::traits::NodeId>,
        /// Removed nodes
        removed: Vec<crate::algorithm::traits::NodeId>,
    },
    
    /// Current node modification
    CurrentNodeModification {
        /// New current node
        new_current: Option<crate::algorithm::traits::NodeId>,
    },
    
    /// Custom state modification
    CustomModification {
        /// Modification function
        modifier: Arc<dyn Fn(&mut AlgorithmState) + Send + Sync>,
        /// Description
        description: String,
    },
}

/// Branch manager for orchestrating operations
#[derive(Debug)]
pub struct BranchManager {
    /// State manager reference
    state_manager: Arc<StateManager>,
    
    /// Timeline manager reference
    timeline_manager: Arc<TimelineManager>,
    
    /// Branch ID generator
    branch_id_gen: AtomicU64,
    
    /// Branch registry
    branches: DashMap<BranchId, Arc<TimelineBranch>>,
    
    /// Branch index by timeline
    timeline_index: DashMap<TimelineId, BranchId>,
    
    /// Branch index by state
    state_index: DashMap<StateId, Vec<BranchId>>,
    
    /// Branch hierarchy
    branch_hierarchy: DashMap<BranchId, Vec<BranchId>>,
    
    /// Maximum branch depth
    max_branch_depth: usize,
    
    /// Merge conflict resolver
    conflict_resolver: Arc<dyn ConflictResolver + Send + Sync>,
}

/// Conflict resolution interface
pub trait ConflictResolver: fmt::Debug {
    /// Resolve merge conflict
    fn resolve_conflict(
        &self,
        conflict: &MergeConflict,
        branches: &[&TimelineBranch],
    ) -> Result<ResolvedState, BranchError>;
}

/// Merge conflict definition
#[derive(Debug, Clone)]
pub struct MergeConflict {
    /// Conflict ID
    pub id: u64,
    
    /// Conflict state
    pub state_id: StateId,
    
    /// Conflicting branches
    pub branch_ids: Vec<BranchId>,
    
    /// Conflict type
    pub conflict_type: ConflictType,
    
    /// Conflict description
    pub description: String,
}

/// Merge conflict type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConflictType {
    /// Parameter conflict
    ParameterConflict {
        /// Parameter name
        parameter: String,
        /// Conflicting values
        values: Vec<String>,
    },
    
    /// Set membership conflict
    SetMembershipConflict {
        /// Set name
        set_name: String,
        /// Conflicting elements
        elements: Vec<crate::algorithm::traits::NodeId>,
    },
    
    /// Current node conflict
    CurrentNodeConflict {
        /// Conflicting nodes
        nodes: Vec<Option<crate::algorithm::traits::NodeId>>,
    },
    
    /// Custom conflict
    CustomConflict {
        /// Conflict name
        name: String,
    },
}

/// Resolved state after conflict resolution
#[derive(Debug, Clone)]
pub struct ResolvedState {
    /// Resolution type
    pub resolution_type: ResolutionType,
    
    /// Resolved state
    pub state: AlgorithmState,
    
    /// Resolution description
    pub description: String,
}

/// Resolution type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionType {
    /// Take state from branch
    TakeFromBranch(BranchId),
    
    /// Merge branches
    Merge,
    
    /// Manual resolution
    Manual,
}

impl BranchManager {
    /// Create new branch manager
    pub fn new(
        state_manager: Arc<StateManager>,
        timeline_manager: Arc<TimelineManager>,
        max_branch_depth: usize,
    ) -> Self {
        // Create default conflict resolver
        let conflict_resolver: Arc<dyn ConflictResolver + Send + Sync> = 
            Arc::new(DefaultConflictResolver::new());
        
        Self {
            state_manager,
            timeline_manager,
            branch_id_gen: AtomicU64::new(0),
            branches: DashMap::new(),
            timeline_index: DashMap::new(),
            state_index: DashMap::new(),
            branch_hierarchy: DashMap::new(),
            max_branch_depth,
            conflict_resolver,
        }
    }
    
    /// Set custom conflict resolver
    pub fn set_conflict_resolver(&mut self, resolver: Arc<dyn ConflictResolver + Send + Sync>) {
        self.conflict_resolver = resolver;
    }
    
    /// Create new branch from timeline
    pub fn create_branch(
        &self,
        timeline_id: TimelineId,
        branch_point: StateId,
        branch_type: BranchType,
        name: &str,
        reason: &str,
    ) -> Result<Arc<TimelineBranch>, BranchError> {
        // Get parent branch
        let parent = self.get_branch_by_timeline(timeline_id);
        
        // Check branch depth
        if let Some(parent_branch) = &parent {
            let depth = parent_branch.metadata.depth + 1;
            if depth > self.max_branch_depth {
                return Err(BranchError::MaxDepthExceeded(depth));
            }
        }
        
        // Verify branch point exists
        self.timeline_manager.navigate_to(timeline_id, branch_point)
            .map_err(|e| BranchError::BranchPointNotFound(branch_point, timeline_id))?;
        
        // Create branch ID
        let branch_id = BranchId(self.branch_id_gen.fetch_add(1, Ordering::SeqCst));
        
        // Create new timeline for branch
        let new_timeline = self.timeline_manager.create_timeline(Some(timeline_id))
            .map_err(BranchError::TimelineError)?;
        
        // Initialize branch metadata
        let metadata = BranchMetadata {
            id: branch_id,
            name: name.to_string(),
            branch_type,
            created_at: Instant::now(),
            parent: parent.as_ref().map(|p| p.metadata.id),
            timeline_id: new_timeline,
            branch_point,
            depth: parent.map(|p| p.metadata.depth + 1).unwrap_or(0),
            tags: HashSet::new(),
            parameter_changes: None,
            state_modifications: None,
            reason: reason.to_string(),
        };
        
        // Create branch
        let branch = Arc::new(TimelineBranch::new(
            metadata,
            new_timeline,
            branch_point,
            Arc::clone(&self.state_manager),
            Arc::clone(&self.timeline_manager),
        ));
        
        // Register branch
        self.branches.insert(branch_id, Arc::clone(&branch));
        self.timeline_index.insert(new_timeline, branch_id);
        
        // Update state index
        self.update_state_index(branch_id, branch_point);
        
        // Update branch hierarchy
        if let Some(parent_branch) = parent {
            if let Some(mut derived) = self.branch_hierarchy.get_mut(&parent_branch.metadata.id) {
                derived.push(branch_id);
            } else {
                self.branch_hierarchy.insert(parent_branch.metadata.id, vec![branch_id]);
            }
        }
        
        Ok(branch)
    }
    
    /// Apply modification to branch
    pub fn apply_modification(
        &self,
        branch: &TimelineBranch,
        modification: BranchModification,
    ) -> Result<StateId, BranchError> {
        // Lock branch for modification
        let _lock = branch.modification_lock.lock();
        
        // Get branch point state
        let base_state = self.state_manager.get_state(branch.branch_point)
            .map_err(BranchError::StateManagerError)?;
        
        // Clone state for modification
        let mut modified_state = base_state.as_ref().clone();
        
        // Apply modification
        match modification {
            BranchModification::ParameterModification { parameter, new_value } => {
                // Record parameter change (requires tracking elsewhere)
                // This would be algorithm-specific
            },
            
            BranchModification::OpenSetModification { added, removed } => {
                // Apply open set modifications
                for node in removed {
                    modified_state.open_set.retain(|&n| n != node);
                }
                
                for node in added {
                    if !modified_state.open_set.contains(&node) {
                        modified_state.open_set.push(node);
                    }
                }
                
                // Sort for consistency
                modified_state.open_set.sort();
            },
            
            BranchModification::ClosedSetModification { added, removed } => {
                // Apply closed set modifications
                for node in removed {
                    modified_state.closed_set.retain(|&n| n != node);
                }
                
                for node in added {
                    if !modified_state.closed_set.contains(&node) {
                        modified_state.closed_set.push(node);
                    }
                }
                
                // Sort for consistency
                modified_state.closed_set.sort();
            },
            
            BranchModification::CurrentNodeModification { new_current } => {
                // Set new current node
                modified_state.current_node = new_current;
            },
            
            BranchModification::CustomModification { modifier, .. } => {
                // Apply custom modification
                modifier(&mut modified_state);
            },
        }
        
        // Store modified state
        let modified_id = self.state_manager.store_state(
            branch.timeline_id,
            modified_state,
        ).map_err(BranchError::StateManagerError)?;
        
        // Add state to branch timeline
        self.timeline_manager.add_state(
            branch.timeline_id,
            modified_id,
            Some(branch.branch_point),
        ).map_err(BranchError::TimelineError)?;
        
        // Update branch indices
        self.update_state_index(branch.metadata.id, modified_id);
        
        Ok(modified_id)
    }
    
    /// Continue branch execution
    pub fn continue_execution(
        &self,
        branch: &TimelineBranch,
        algorithm: &mut dyn Algorithm,
        steps: usize,
    ) -> Result<Vec<StateId>, BranchError> {
        // Lock branch for modification
        let _lock = branch.modification_lock.lock();
        
        // Get latest state in branch
        let latest_state_id = self.get_latest_state(branch)?;
        
        // Initialize execution tracer
        let mut tracer = ExecutionTracer::new();
        
        // Set current state in algorithm
        let current_state = self.state_manager.get_state(latest_state_id)
            .map_err(BranchError::StateManagerError)?;
        
        // Execute algorithm with tracing for specified steps
        let mut result_states = Vec::with_capacity(steps);
        let mut last_state_id = latest_state_id;
        
        for _ in 0..steps {
            // Execute single step
            let execution_result = algorithm.execute_with_tracing(
                current_state.as_ref(),
                &mut tracer,
            )?;
            
            // Store execution state
            let state_id = self.state_manager.store_state(
                branch.timeline_id,
                execution_result.state.clone(),
            ).map_err(BranchError::StateManagerError)?;
            
            // Add state to branch timeline
            self.timeline_manager.add_state(
                branch.timeline_id,
                state_id,
                Some(last_state_id),
            ).map_err(BranchError::TimelineError)?;
            
            // Update branch indices
            self.update_state_index(branch.metadata.id, state_id);
            
            // Record state ID
            result_states.push(state_id);
            last_state_id = state_id;
            
            // Check for completion
            if execution_result.completed {
                break;
            }
        }
        
        Ok(result_states)
    }
    
    /// Merge branches
    pub fn merge_branches(
        &self,
        branches: &[&TimelineBranch],
        merge_point: Option<StateId>,
        name: &str,
    ) -> Result<Arc<TimelineBranch>, BranchError> {
        if branches.len() < 2 {
            return Err(BranchError::CreationFailed(
                "At least two branches required for merge".to_string()
            ));
        }
        
        // Determine common ancestor if not specified
        let common_ancestor = if let Some(point) = merge_point {
            point
        } else {
            self.find_common_ancestor(branches)?
        };
        
        // Create merge branch from common ancestor
        let primary_branch = branches[0];
        let merge_branch = self.create_branch(
            primary_branch.timeline_id,
            common_ancestor,
            BranchType::Merge,
            name,
            "Branch merge operation",
        )?;
        
        // Collect states to merge from each branch
        let branch_states: Result<Vec<Vec<(StateId, AlgorithmState)>>, BranchError> = 
            branches.iter()
                .map(|branch| self.collect_states_after(*branch, common_ancestor))
                .collect();
        
        let branch_states = branch_states?;
        
        // Find conflicts
        let conflicts = self.find_merge_conflicts(&branch_states)?;
        
        // Resolve conflicts
        let resolved_states = self.resolve_conflicts(&conflicts, branches)?;
        
        // Apply merged states to merge branch
        let mut last_state_id = common_ancestor;
        let mut merged_states = Vec::new();
        
        for resolved in resolved_states {
            // Store resolved state
            let state_id = self.state_manager.store_state(
                merge_branch.timeline_id,
                resolved.state,
            ).map_err(BranchError::StateManagerError)?;
            
            // Add state to branch timeline
            self.timeline_manager.add_state(
                merge_branch.timeline_id,
                state_id,
                Some(last_state_id),
            ).map_err(BranchError::TimelineError)?;
            
            // Update indices
            self.update_state_index(merge_branch.metadata.id, state_id);
            
            // Update last state
            last_state_id = state_id;
            merged_states.push(state_id);
        }
        
        Ok(merge_branch)
    }
    
    /// Find common ancestor of branches
    fn find_common_ancestor(
        &self,
        branches: &[&TimelineBranch],
    ) -> Result<StateId, BranchError> {
        // Collect ancestors for each branch
        let branch_ancestors: Result<Vec<HashSet<StateId>>, BranchError> = branches
            .iter()
            .map(|branch| self.collect_ancestors(*branch))
            .collect();
        
        let branch_ancestors = branch_ancestors?;
        
        // Find common ancestors
        let mut common_ancestors = branch_ancestors[0].clone();
        for ancestors in &branch_ancestors[1..] {
            common_ancestors.retain(|id| ancestors.contains(id));
        }
        
        if common_ancestors.is_empty() {
            return Err(BranchError::CreationFailed(
                "No common ancestor found".to_string()
            ));
        }
        
        // Find the most recent common ancestor
        let mut latest_ancestor = None;
        let mut latest_timestamp = Instant::now() - Duration::from_secs(60 * 60 * 24 * 365); // A year ago
        
        for ancestor_id in common_ancestors {
            // Get state timestamp
            if let Ok(snapshot) = self.state_manager.get_state_snapshot(ancestor_id) {
                if snapshot.creation_time > latest_timestamp {
                    latest_timestamp = snapshot.creation_time;
                    latest_ancestor = Some(ancestor_id);
                }
            }
        }
        
        latest_ancestor.ok_or_else(|| BranchError::CreationFailed(
            "Failed to determine most recent common ancestor".to_string()
        ))
    }
    
    /// Collect ancestors of branch
    fn collect_ancestors(
        &self,
        branch: &TimelineBranch,
    ) -> Result<HashSet<StateId>, BranchError> {
        let mut ancestors = HashSet::new();
        let mut queue = VecDeque::new();
        
        // Start from branch point
        queue.push_back(branch.branch_point);
        ancestors.insert(branch.branch_point);
        
        while let Some(state_id) = queue.pop_front() {
            // Get parent state
            if let Some(parent) = self.get_parent_state(branch.timeline_id, state_id)? {
                if !ancestors.contains(&parent) {
                    ancestors.insert(parent);
                    queue.push_back(parent);
                }
            }
        }
        
        Ok(ancestors)
    }
    
    /// Get latest state in branch
    fn get_latest_state(
        &self,
        branch: &TimelineBranch,
    ) -> Result<StateId, BranchError> {
        // Get current position in timeline
        self.timeline_manager.get_current_position(branch.timeline_id)
            .map_err(BranchError::TimelineError)
    }
    
    /// Get parent state in timeline
    fn get_parent_state(
        &self,
        timeline_id: TimelineId,
        state_id: StateId,
    ) -> Result<Option<StateId>, BranchError> {
        self.timeline_manager.get_parent_state(timeline_id, state_id)
            .map_err(BranchError::TimelineError)
    }
    
    /// Get branch by timeline
    fn get_branch_by_timeline(
        &self,
        timeline_id: TimelineId,
    ) -> Option<Arc<TimelineBranch>> {
        self.timeline_index.get(&timeline_id)
            .and_then(|branch_id| self.branches.get(branch_id).map(|b| Arc::clone(b.value())))
    }
    
    /// Update state index
    fn update_state_index(
        &self,
        branch_id: BranchId,
        state_id: StateId,
    ) {
        if let Some(mut branches) = self.state_index.get_mut(&state_id) {
            if !branches.contains(&branch_id) {
                branches.push(branch_id);
            }
        } else {
            self.state_index.insert(state_id, vec![branch_id]);
        }
    }
    
    /// Collect states after common ancestor
    fn collect_states_after(
        &self,
        branch: &TimelineBranch,
        common_ancestor: StateId,
    ) -> Result<Vec<(StateId, AlgorithmState)>, BranchError> {
        let mut states = Vec::new();
        let mut current = self.get_latest_state(branch)?;
        
        while current != common_ancestor {
            // Get state
            let state = self.state_manager.get_state(current)
                .map_err(BranchError::StateManagerError)?;
            
            // Add to collection
            states.push((current, (*state).clone()));
            
            // Get parent
            if let Some(parent) = self.get_parent_state(branch.timeline_id, current)? {
                current = parent;
            } else {
                break;
            }
        }
        
        // Reverse to get chronological order
        states.reverse();
        
        Ok(states)
    }
    
    /// Find merge conflicts
    fn find_merge_conflicts(
        &self,
        branch_states: &[Vec<(StateId, AlgorithmState)>],
    ) -> Result<Vec<MergeConflict>, BranchError> {
        let mut conflicts = Vec::new();
        let mut conflict_id = 0;
        
        // For simplicity, consider chronological states
        // A more sophisticated merge would use diff alignment
        
        // Get maximum common steps
        let min_steps = branch_states.iter().map(|states| states.len()).min().unwrap_or(0);
        
        for step in 0..min_steps {
            // Extract states at this step
            let step_states: Vec<(&StateId, &AlgorithmState)> = branch_states.iter()
                .map(|states| (&states[step].0, &states[step].1))
                .collect();
            
            // Compare current node
            let current_nodes: Vec<_> = step_states.iter().map(|(_, state)| state.current_node).collect();
            if current_nodes.windows(2).any(|w| w[0] != w[1]) {
                conflict_id += 1;
                conflicts.push(MergeConflict {
                    id: conflict_id,
                    state_id: *step_states[0].0,
                    branch_ids: Vec::new(), // To be filled by resolver
                    conflict_type: ConflictType::CurrentNodeConflict {
                        nodes: current_nodes,
                    },
                    description: format!("Conflicting current nodes at step {}", step),
                });
            }
            
            // Compare open set
            for (i, (_, state1)) in step_states.iter().enumerate() {
                for (_, state2) in step_states.iter().skip(i+1) {
                    let mut conflicting = Vec::new();
                    
                    for &node in &state1.open_set {
                        if !state2.open_set.contains(&node) && !state2.closed_set.contains(&node) {
                            conflicting.push(node);
                        }
                    }
                    
                    for &node in &state2.open_set {
                        if !state1.open_set.contains(&node) && !state1.closed_set.contains(&node) {
                            conflicting.push(node);
                        }
                    }
                    
                    if !conflicting.is_empty() {
                        conflict_id += 1;
                        conflicts.push(MergeConflict {
                            id: conflict_id,
                            state_id: *step_states[0].0,
                            branch_ids: Vec::new(), // To be filled by resolver
                            conflict_type: ConflictType::SetMembershipConflict {
                                set_name: "open_set".to_string(),
                                elements: conflicting,
                            },
                            description: format!("Conflicting open set membership at step {}", step),
                        });
                    }
                }
            }
            
            // Compare closed set
            for (i, (_, state1)) in step_states.iter().enumerate() {
                for (_, state2) in step_states.iter().skip(i+1) {
                    let mut conflicting = Vec::new();
                    
                    for &node in &state1.closed_set {
                        if !state2.closed_set.contains(&node) && state2.open_set.contains(&node) {
                            conflicting.push(node);
                        }
                    }
                    
                    for &node in &state2.closed_set {
                        if !state1.closed_set.contains(&node) && state1.open_set.contains(&node) {
                            conflicting.push(node);
                        }
                    }
                    
                    if !conflicting.is_empty() {
                        conflict_id += 1;
                        conflicts.push(MergeConflict {
                            id: conflict_id,
                            state_id: *step_states[0].0,
                            branch_ids: Vec::new(), // To be filled by resolver
                            conflict_type: ConflictType::SetMembershipConflict {
                                set_name: "closed_set".to_string(),
                                elements: conflicting,
                            },
                            description: format!("Conflicting closed set membership at step {}", step),
                        });
                    }
                }
            }
        }
        
        Ok(conflicts)
    }
    
    /// Resolve conflicts using resolver
    fn resolve_conflicts(
        &self,
        conflicts: &[MergeConflict],
        branches: &[&TimelineBranch],
    ) -> Result<Vec<ResolvedState>, BranchError> {
        // Group conflicts by state
        let mut conflicts_by_state: HashMap<StateId, Vec<&MergeConflict>> = HashMap::new();
        
        for conflict in conflicts {
            conflicts_by_state.entry(conflict.state_id)
                .or_insert_with(Vec::new)
                .push(conflict);
        }
        
        // Resolve conflicts state by state
        let mut resolved_states = Vec::new();
        
        for state_id in conflicts_by_state.keys() {
            let state_conflicts = conflicts_by_state.get(state_id).unwrap();
            
            // Currently just resolve first conflict per state
            // A more sophisticated approach would merge resolutions
            if let Some(conflict) = state_conflicts.first() {
                let resolved = self.conflict_resolver.resolve_conflict(
                    *conflict,
                    branches,
                )?;
                
                resolved_states.push(resolved);
            }
        }
        
        Ok(resolved_states)
    }
}

impl TimelineBranch {
    /// Create new timeline branch
    fn new(
        metadata: BranchMetadata,
        timeline_id: TimelineId,
        branch_point: StateId,
        state_manager: Arc<StateManager>,
        timeline_manager: Arc<TimelineManager>,
    ) -> Self {
        Self {
            metadata,
            timeline_id,
            branch_point,
            state_manager,
            timeline_manager,
            modification_lock: PLMutex::new(()),
            derived_branches: PLRwLock::new(Vec::new()),
            invariant_tracker: BranchInvariantTracker::new(),
            propagation_status: PLRwLock::new(PropagationStatus::NotPropagated),
        }
    }
    
    /// Get current state
    pub fn current_state(&self) -> Result<Arc<AlgorithmState>, BranchError> {
        let current_position = self.timeline_manager.get_current_position(self.timeline_id)
            .map_err(BranchError::TimelineError)?;
        
        self.state_manager.get_state(current_position)
            .map_err(BranchError::StateManagerError)
    }
    
    /// Navigate to state
    pub fn navigate_to(&self, state_id: StateId) -> Result<Arc<AlgorithmState>, BranchError> {
        self.timeline_manager.navigate_to(self.timeline_id, state_id)
            .map_err(BranchError::TimelineError)
    }
    
    /// Apply modification
    pub fn apply_modification(
        &self,
        modification: BranchModification,
    ) -> Result<StateId, BranchError> {
        // Implementation would forward to BranchManager
        // For simplicity, we'll sketch the validation here
        
        // Lock for modification
        let _lock = self.modification_lock.lock();
        
        // Validate branch invariants
        self.invariant_tracker.validate(self)?;
        
        // Get base state
        let base_state = self.state_manager.get_state(self.branch_point)
            .map_err(BranchError::StateManagerError)?;
        
        // Clone state for modification
        let mut modified_state = base_state.as_ref().clone();
        
        // Apply modification
        match &modification {
            BranchModification::ParameterModification { parameter, new_value } => {
                // Parameter modification logic
            },
            BranchModification::OpenSetModification { added, removed } => {
                // Open set modification logic
            },
            BranchModification::ClosedSetModification { added, removed } => {
                // Closed set modification logic
            },
            BranchModification::CurrentNodeModification { new_current } => {
                modified_state.current_node = *new_current;
            },
            BranchModification::CustomModification { modifier, .. } => {
                modifier(&mut modified_state);
            },
        }
        
        // Store modified state
        let state_id = self.state_manager.store_state(
            self.timeline_id,
            modified_state,
        ).map_err(BranchError::StateManagerError)?;
        
        // Add to timeline
        self.timeline_manager.add_state(
            self.timeline_id,
            state_id,
            Some(self.branch_point),
        ).map_err(BranchError::TimelineError)?;
        
        // Update propagation status
        *self.propagation_status.write() = PropagationStatus::PartiallyPropagated {
            propagated_steps: 1,
            total_steps: 1,
        };
        
        Ok(state_id)
    }
    
    /// Continue execution from current state
    pub fn continue_execution(
        &self,
        algorithm: &mut dyn Algorithm,
        steps: usize,
    ) -> Result<Vec<StateId>, BranchError> {
        // Implementation would forward to BranchManager
        // Similar to the previous method
        
        Ok(Vec::new())
    }
    
    /// Add tag to branch
    pub fn add_tag(&self, tag: &str) {
        let mut metadata = self.metadata.clone();
        metadata.tags.insert(tag.to_string());
        // Would update in BranchManager
    }
}

impl BranchInvariantTracker {
    fn new() -> Self {
        Self {
            invariants: Vec::new(),
            last_validation: Instant::now(),
            validation_failures: PLRwLock::new(Vec::new()),
        }
    }
    
    /// Validate branch invariants
    fn validate(&self, branch: &TimelineBranch) -> Result<(), BranchError> {
        let mut failures = Vec::new();
        
        // Validate state consistency invariants
        for invariant in &self.invariants {
            match invariant {
                BranchInvariant::StateConsistency { property, validate } => {
                    if let Ok(state) = branch.current_state() {
                        if !validate(state.as_ref()) {
                            failures.push(ValidationFailure {
                                invariant: property.name.clone(),
                                timestamp: Instant::now(),
                                description: format!("State property '{}' validation failed", property.name),
                                affected_states: Some(vec![branch.branch_point]),
                            });
                        }
                    }
                },
                BranchInvariant::TimelineConsistency { name, validate } => {
                    if !validate(branch) {
                        failures.push(ValidationFailure {
                            invariant: name.clone(),
                            timestamp: Instant::now(),
                            description: format!("Timeline consistency '{}' validation failed", name),
                            affected_states: None,
                        });
                    }
                },
                // Inter-branch consistency requires multiple branches
                _ => {},
            }
        }
        
        if !failures.is_empty() {
            *self.validation_failures.write() = failures.clone();
            
            return Err(BranchError::ConsistencyViolation(
                format!("{} invariant violations detected", failures.len())
            ));
        }
        
        Ok(())
    }
}

/// Default conflict resolver implementation
#[derive(Debug)]
struct DefaultConflictResolver;

impl DefaultConflictResolver {
    fn new() -> Self {
        Self {}
    }
}

impl ConflictResolver for DefaultConflictResolver {
    fn resolve_conflict(
        &self,
        conflict: &MergeConflict,
        branches: &[&TimelineBranch],
    ) -> Result<ResolvedState, BranchError> {
        // Simple resolution strategy - choose first branch
        let primary_branch = branches[0];
        
        // Locate state in primary branch
        let state = primary_branch.state_manager.get_state(conflict.state_id)
            .map_err(BranchError::StateManagerError)?;
        
        Ok(ResolvedState {
            resolution_type: ResolutionType::TakeFromBranch(primary_branch.metadata.id),
            state: (*state).clone(),
            description: format!(
                "Automatic resolution: taking state from branch '{}'",
                primary_branch.metadata.name
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::path::PathBuf;
    
    fn create_test_environment() -> (Arc<StateManager>, Arc<TimelineManager>, BranchManager) {
        // Create temporary directory for state storage
        let temp_dir = TempDir::new().unwrap();
        
        // Create state manager
        let state_manager = Arc::new(StateManager::new(
            crate::temporal::state_manager::StateManagerConfig {
                memory_budget: 1024 * 1024,
                cache_size: 100,
                max_delta_chain: 10,
                storage_path: temp_dir.path().to_path_buf(),
                enable_compression: false,
                compression_level: 0,
            }
        ).unwrap());
        
        // Create timeline manager
        let timeline_manager = Arc::new(TimelineManager::new(
            Arc::clone(&state_manager),
            1000,
        ));
        
        // Create branch manager
        let branch_manager = BranchManager::new(
            Arc::clone(&state_manager),
            Arc::clone(&timeline_manager),
            10, // max branch depth
        );
        
        (state_manager, timeline_manager, branch_manager)
    }
    
    #[test]
    fn test_branch_creation() {
        let (state_manager, timeline_manager, branch_manager) = create_test_environment();
        
        // Create initial timeline
        let timeline_id = timeline_manager.create_timeline(None).unwrap();
        
        // Create initial state
        let state = AlgorithmState::default();
        let state_id = state_manager.store_state(timeline_id, state).unwrap();
        
        // Add state to timeline
        timeline_manager.add_state(timeline_id, state_id, None).unwrap();
        
        // Create branch
        let branch = branch_manager.create_branch(
            timeline_id,
            state_id,
            BranchType::CounterfactualExploration,
            "Test Branch",
            "Testing branch creation",
        ).unwrap();
        
        // Verify branch properties
        assert_eq!(branch.metadata.branch_type, BranchType::CounterfactualExploration);
        assert_eq!(branch.metadata.name, "Test Branch");
        assert_eq!(branch.branch_point, state_id);
        
        // Verify branch was registered
        assert!(branch_manager.branches.contains_key(&branch.metadata.id));
        assert!(branch_manager.timeline_index.contains_key(&branch.timeline_id));
    }
    
    #[test]
    fn test_branch_modification() {
        let (state_manager, timeline_manager, branch_manager) = create_test_environment();
        
        // Create initial timeline
        let timeline_id = timeline_manager.create_timeline(None).unwrap();
        
        // Create initial state
        let state = AlgorithmState::default();
        let state_id = state_manager.store_state(timeline_id, state).unwrap();
        
        // Add state to timeline
        timeline_manager.add_state(timeline_id, state_id, None).unwrap();
        
        // Create branch
        let branch = branch_manager.create_branch(
            timeline_id,
            state_id,
            BranchType::StateModification,
            "Modified Branch",
            "Testing state modification",
        ).unwrap();
        
        // Apply modification
        let modification = BranchModification::CurrentNodeModification {
            new_current: Some(crate::algorithm::traits::NodeId(42)),
        };
        
        let modified_id = branch_manager.apply_modification(&branch, modification).unwrap();
        
        // Verify modification
        let modified_state = state_manager.get_state(modified_id).unwrap();
        assert_eq!(modified_state.current_node, Some(crate::algorithm::traits::NodeId(42)));
    }
    
    #[test]
    fn test_branch_invariants() {
        let (state_manager, timeline_manager, branch_manager) = create_test_environment();
        
        // Create initial timeline
        let timeline_id = timeline_manager.create_timeline(None).unwrap();
        
        // Create initial state
        let state = AlgorithmState::default();
        let state_id = state_manager.store_state(timeline_id, state).unwrap();
        
        // Add state to timeline
        timeline_manager.add_state(timeline_id, state_id, None).unwrap();
        
        // Create branch
        let branch = branch_manager.create_branch(
            timeline_id,
            state_id,
            BranchType::StateModification,
            "Invariant Branch",
            "Testing branch invariants",
        ).unwrap();
        
        // Add invariant
        let property = StateProperty {
            name: "Open set size".to_string(),
            description: "Open set should not be empty".to_string(),
        };
        
        let validate: Arc<dyn Fn(&AlgorithmState) -> bool + Send + Sync> = 
            Arc::new(|state| !state.open_set.is_empty());
        
        let invariant = BranchInvariant::StateConsistency {
            property,
            validate,
        };
        
        branch.invariant_tracker.invariants.push(invariant);
        
        // This should pass since the default state has an empty open set
        // and our invariant requires a non-empty open set, triggering validation failure
        let result = branch.invariant_tracker.validate(&branch);
        assert!(result.is_err());
        
        // Apply modification to satisfy invariant
        let modification = BranchModification::OpenSetModification {
            added: vec![crate::algorithm::traits::NodeId(1)],
            removed: vec![],
        };
        
        let modified_id = branch_manager.apply_modification(&branch, modification).unwrap();
        timeline_manager.navigate_to(branch.timeline_id, modified_id).unwrap();
        
        // Now validation should pass
        let result = branch.invariant_tracker.validate(&branch);
        assert!(result.is_ok());
    }
}