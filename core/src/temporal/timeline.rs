//! Timeline Management System
//!
//! This module implements an advanced timeline management system using B-tree
//! indexing for O(log n) access, immutable spine architecture for consistency,
//! and MVCC semantics for concurrent operations.
//!
//! Copyright (c) 2025 Mohammad Atashi <mohammadaliatashi@icloud.com>

use std::sync::{Arc, RwLock, atomic::{AtomicU64, AtomicBool, Ordering}};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::{Instant, Duration};
use std::marker::PhantomData;
use std::cmp::Ordering as StdOrdering;

use dashmap::DashMap;
use parking_lot::RwLock as ParkingRwLock;
use serde::{Serialize, Deserialize};
use thiserror::Error;
use rayon::prelude::*;

use crate::algorithm::state::{AlgorithmState, StateSnapshot};
use crate::temporal::state_manager::{StateId, StateManager, StateLocation};
use crate::temporal::delta::StateDelta;

/// Timeline management errors with diagnostic context
#[derive(Error, Debug)]
pub enum TimelineError {
    #[error("State {0} not found in timeline {1}")]
    StateNotFound(StateId, TimelineId),
    
    #[error("Concurrent modification detected at state {0}")]
    ConcurrentModificationError(StateId),
    
    #[error("Branch point not found at state {0}")]
    BranchPointNotFound(StateId),
    
    #[error("Invalid navigation: current={0}, target={1}")]
    InvalidNavigation(StateId, StateId),
    
    #[error("Timeline capacity exceeded: {0} states")]
    CapacityExceeded(usize),
    
    #[error("MVCC conflict: read version {0}, write version {1}")]
    MVCCConflict(u64, u64),
}

/// Unique timeline identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TimelineId(pub u64);

/// State transition record with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    /// Source state ID
    pub from_state: StateId,
    /// Target state ID
    pub to_state: StateId,
    /// Transition timestamp
    pub timestamp: Instant,
    /// Transition type
    pub transition_type: TransitionType,
    /// Delta between states
    pub delta: Option<Arc<StateDelta>>,
}

/// Types of state transitions
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TransitionType {
    /// Normal algorithmic step
    Algorithmic,
    /// Branch creation
    Branch,
    /// State modification
    Modification,
    /// Merge from another timeline
    Merge,
}

/// Branch point in timeline execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchPoint {
    /// Branch point state ID
    pub state_id: StateId,
    /// Parent timeline ID
    pub parent_timeline: TimelineId,
    /// Child timelines created from this point
    pub child_timelines: Vec<TimelineId>,
    /// Branch creation timestamp
    pub timestamp: Instant,
    /// Branch point metadata
    pub metadata: BranchMetadata,
}

/// Branch point metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchMetadata {
    /// Reason for branching (decision point, state modification, etc.)
    pub reason: BranchReason,
    /// Algorithm step at branch point
    pub algorithm_step: usize,
    /// Decision context
    pub decision_context: HashMap<String, String>,
}

/// Reasons for timeline branching
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BranchReason {
    /// Algorithm decision point
    DecisionPoint,
    /// User-initiated state modification
    StateModification,
    /// Counterfactual exploration
    CounterfactualExploration,
    /// Heuristic function change
    HeuristicChange,
}

/// MVCC (Multi-Version Concurrency Control) state access
#[derive(Debug, Clone)]
struct MVCCAccess {
    /// State ID being accessed
    state_id: StateId,
    /// Access timestamp
    timestamp: Instant,
    /// Access type
    access_type: AccessType,
    /// Version number
    version: u64,
}

/// Access type for MVCC
#[derive(Debug, Clone, Copy)]
enum AccessType {
    Read,
    Write,
}

/// B-tree node for timeline indexing
#[derive(Debug, Clone)]
struct TimelineNode {
    /// State IDs in this node
    state_ids: Vec<StateId>,
    /// State metadata
    metadata: Vec<StateMetadata>,
    /// Child node pointers
    children: Vec<Arc<TimelineNode>>,
    /// Is this a leaf node?
    is_leaf: bool,
}

/// Metadata for timeline states
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StateMetadata {
    /// Position in execution sequence
    position: usize,
    /// Algorithm step number
    step: usize,
    /// Is this a branch point?
    is_branch_point: bool,
    /// Parent state ID
    parent: Option<StateId>,
    /// Creation timestamp
    timestamp: Instant,
}

/// Immutable spine for timeline consistency
#[derive(Debug, Clone)]
struct TimelineSpine {
    /// Root node of the B-tree
    root: Arc<TimelineNode>,
    /// Timeline metadata
    metadata: TimelineMetadata,
    /// Version number for MVCC
    version: u64,
}

/// Timeline metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TimelineMetadata {
    /// Timeline ID
    timeline_id: TimelineId,
    /// Parent timeline (if any)
    parent: Option<TimelineId>,
    /// Creation timestamp
    created_at: Instant,
    /// Total states in timeline
    state_count: usize,
    /// Branch points
    branch_points: Vec<StateId>,
}

/// Advanced timeline manager with B-tree indexing
pub struct TimelineManager {
    /// State manager reference
    state_manager: Arc<StateManager>,
    
    /// Timeline ID generator
    timeline_id_gen: AtomicU64,
    
    /// Active timelines
    timelines: DashMap<TimelineId, Arc<ParkingRwLock<TimelineSpine>>>,
    
    /// Branch point index
    branch_point_index: DashMap<StateId, BranchPoint>,
    
    /// State transition history
    transition_history: DashMap<(StateId, StateId), StateTransition>,
    
    /// MVCC access tracker
    mvcc_tracker: Arc<ParkingRwLock<HashMap<StateId, VecDeque<MVCCAccess>>>>,
    
    /// Current positions in timelines
    timeline_positions: DashMap<TimelineId, StateId>,
    
    /// Branch point detector
    branch_detector: BranchPointDetector,
    
    /// Timeline capacity limit
    max_timeline_capacity: usize,
}

/// Branch point detection system
struct BranchPointDetector {
    /// Decision pattern matcher
    pattern_matcher: DecisionPatternMatcher,
    
    /// Heuristic change detector
    heuristic_detector: HeuristicChangeDetector,
    
    /// Algorithm state analyzer
    state_analyzer: StateAnalyzer,
}

/// Pattern matcher for decision points
struct DecisionPatternMatcher {
    /// Known decision patterns
    patterns: Vec<DecisionPattern>,
    
    /// Pattern matching cache
    match_cache: DashMap<StateId, Vec<DecisionPattern>>,
}

/// Decision pattern definition
#[derive(Debug, Clone)]
struct DecisionPattern {
    /// Pattern identifier
    id: u32,
    /// Pattern description
    description: String,
    /// Matching criteria
    criteria: MatchingCriteria,
}

/// Matching criteria for decision patterns
#[derive(Debug, Clone)]
struct MatchingCriteria {
    /// State conditions
    state_conditions: Vec<StateCondition>,
    /// Transition requirements
    transition_requirements: Vec<TransitionRequirement>,
}

impl TimelineManager {
    /// Create new timeline manager with advanced indexing
    pub fn new(
        state_manager: Arc<StateManager>,
        max_timeline_capacity: usize,
    ) -> Self {
        Self {
            state_manager,
            timeline_id_gen: AtomicU64::new(0),
            timelines: DashMap::new(),
            branch_point_index: DashMap::new(),
            transition_history: DashMap::new(),
            mvcc_tracker: Arc::new(ParkingRwLock::new(HashMap::new())),
            timeline_positions: DashMap::new(),
            branch_detector: BranchPointDetector::new(),
            max_timeline_capacity,
        }
    }
    
    /// Create new timeline with immutable spine
    pub fn create_timeline(
        &self,
        parent: Option<TimelineId>,
    ) -> Result<TimelineId, TimelineError> {
        let timeline_id = TimelineId(self.timeline_id_gen.fetch_add(1, Ordering::SeqCst));
        
        // Create immutable spine
        let spine = TimelineSpine {
            root: Arc::new(TimelineNode::new_leaf()),
            metadata: TimelineMetadata {
                timeline_id,
                parent,
                created_at: Instant::now(),
                state_count: 0,
                branch_points: Vec::new(),
            },
            version: 0,
        };
        
        // Store timeline
        self.timelines.insert(timeline_id, Arc::new(ParkingRwLock::new(spine)));
        
        Ok(timeline_id)
    }
    
    /// Add state to timeline with O(log n) insertion
    pub fn add_state(
        &self,
        timeline_id: TimelineId,
        state_id: StateId,
        parent_state: Option<StateId>,
    ) -> Result<(), TimelineError> {
        let timeline = self.timelines.get(&timeline_id)
            .ok_or(TimelineError::StateNotFound(state_id, timeline_id))?;
        
        let mut spine = timeline.write();
        
        // Create new state metadata
        let metadata = StateMetadata {
            position: spine.metadata.state_count,
            step: self.get_algorithm_step(state_id)?,
            is_branch_point: false,
            parent: parent_state,
            timestamp: Instant::now(),
        };
        
        // Insert into B-tree with copy-on-write
        let new_root = self.insert_into_btree(
            Arc::clone(&spine.root),
            state_id,
            metadata.clone(),
        )?;
        
        // Create new immutable spine
        let new_spine = TimelineSpine {
            root: new_root,
            metadata: TimelineMetadata {
                timeline_id,
                parent: spine.metadata.parent,
                created_at: spine.metadata.created_at,
                state_count: spine.metadata.state_count + 1,
                branch_points: spine.metadata.branch_points.clone(),
            },
            version: spine.version + 1,
        };
        
        *spine = new_spine;
        
        // Update current position
        self.timeline_positions.insert(timeline_id, state_id);
        
        // Check for branch point
        if let Some(parent) = parent_state {
            self.check_branch_point(timeline_id, parent, state_id)?;
        }
        
        Ok(())
    }
    
    /// Navigate to specific state with O(log n) access
    pub fn navigate_to(
        &self,
        timeline_id: TimelineId,
        target_state: StateId,
    ) -> Result<Arc<AlgorithmState>, TimelineError> {
        let timeline = self.timelines.get(&timeline_id)
            .ok_or(TimelineError::StateNotFound(target_state, timeline_id))?;
        
        let spine = timeline.read();
        
        // Find state in B-tree
        let (metadata, path) = self.find_in_btree(&spine.root, target_state)
            .ok_or(TimelineError::StateNotFound(target_state, timeline_id))?;
        
        // Track MVCC access
        self.track_mvcc_access(target_state, AccessType::Read, spine.version)?;
        
        // Get state from state manager
        let state = self.state_manager.get_state(target_state)
            .map_err(|_| TimelineError::StateNotFound(target_state, timeline_id))?;
        
        // Update timeline position
        self.timeline_positions.insert(timeline_id, target_state);
        
        Ok(state)
    }
    
    /// Find next state in timeline
    pub fn next_state(
        &self,
        timeline_id: TimelineId,
    ) -> Result<Option<StateId>, TimelineError> {
        let current_state = self.timeline_positions.get(&timeline_id)
            .ok_or(TimelineError::InvalidNavigation(StateId(0), StateId(0)))?;
        
        let timeline = self.timelines.get(&timeline_id)
            .ok_or(TimelineError::StateNotFound(*current_state, timeline_id))?;
        
        let spine = timeline.read();
        
        // Find next state in B-tree order
        let next = self.find_next_in_btree(&spine.root, *current_state);
        
        Ok(next)
    }
    
    /// Find previous state in timeline
    pub fn previous_state(
        &self,
        timeline_id: TimelineId,
    ) -> Result<Option<StateId>, TimelineError> {
        let current_state = self.timeline_positions.get(&timeline_id)
            .ok_or(TimelineError::InvalidNavigation(StateId(0), StateId(0)))?;
        
        let timeline = self.timelines.get(&timeline_id)
            .ok_or(TimelineError::StateNotFound(*current_state, timeline_id))?;
        
        let spine = timeline.read();
        
        // Find previous state in B-tree order
        let prev = self.find_previous_in_btree(&spine.root, *current_state);
        
        Ok(prev)
    }
    
    /// Create branch at current position
    pub fn create_branch(
        &self,
        source_timeline: TimelineId,
        reason: BranchReason,
    ) -> Result<TimelineId, TimelineError> {
        let current_state = self.timeline_positions.get(&source_timeline)
            .ok_or(TimelineError::InvalidNavigation(StateId(0), StateId(0)))?;
        
        // Create new timeline
        let new_timeline = self.create_timeline(Some(source_timeline))?;
        
        // Copy states up to branch point
        let states_to_copy = self.collect_states_up_to(*current_state, source_timeline)?;
        
        for (state_id, parent) in states_to_copy {
            self.add_state(new_timeline, state_id, parent)?;
        }
        
        // Create branch point
        let branch_point = BranchPoint {
            state_id: *current_state,
            parent_timeline: source_timeline,
            child_timelines: vec![new_timeline],
            timestamp: Instant::now(),
            metadata: BranchMetadata {
                reason,
                algorithm_step: self.get_algorithm_step(*current_state)?,
                decision_context: HashMap::new(),
            },
        };
        
        self.branch_point_index.insert(*current_state, branch_point);
        
        Ok(new_timeline)
    }
    
    /// Detect branch points automatically
    pub fn detect_branch_points(
        &self,
        timeline_id: TimelineId,
    ) -> Result<Vec<BranchPoint>, TimelineError> {
        let timeline = self.timelines.get(&timeline_id)
            .ok_or(TimelineError::StateNotFound(StateId(0), timeline_id))?;
        
        let spine = timeline.read();
        
        // Collect all states in timeline
        let states = self.collect_all_states(&spine.root);
        
        // Analyze states for branch points
        let potential_branches = self.branch_detector.analyze_states(&states)?;
        
        // Filter and sort branch points
        let branch_points: Vec<BranchPoint> = potential_branches
            .into_iter()
            .filter(|bp| self.validate_branch_point(bp))
            .collect();
        
        Ok(branch_points)
    }
    
    /// Track MVCC access for concurrency control
    fn track_mvcc_access(
        &self,
        state_id: StateId,
        access_type: AccessType,
        version: u64,
    ) -> Result<(), TimelineError> {
        let mut tracker = self.mvcc_tracker.write();
        
        let accesses = tracker.entry(state_id).or_insert_with(VecDeque::new);
        
        // Check for conflicts
        if let Some(last_access) = accesses.back() {
            match (last_access.access_type, access_type) {
                (AccessType::Write, AccessType::Read) => {
                    if last_access.version > version {
                        return Err(TimelineError::MVCCConflict(version, last_access.version));
                    }
                },
                (AccessType::Write, AccessType::Write) => {
                    return Err(TimelineError::ConcurrentModificationError(state_id));
                },
                _ => {}
            }
        }
        
        // Add new access
        accesses.push_back(MVCCAccess {
            state_id,
            timestamp: Instant::now(),
            access_type,
            version,
        });
        
        // Cleanup old accesses
        while accesses.len() > 100 {
            accesses.pop_front();
        }
        
        Ok(())
    }
    
    /// Insert state into B-tree with copy-on-write
    fn insert_into_btree(
        &self,
        node: Arc<TimelineNode>,
        state_id: StateId,
        metadata: StateMetadata,
    ) -> Result<Arc<TimelineNode>, TimelineError> {
        if node.is_leaf {
            // Create new leaf node with inserted state
            let mut new_node = (*node).clone();
            let insert_pos = new_node.state_ids.binary_search(&state_id)
                .unwrap_or_else(|e| e);
            
            new_node.state_ids.insert(insert_pos, state_id);
            new_node.metadata.insert(insert_pos, metadata);
            
            // Split if necessary
            if new_node.state_ids.len() > MAX_BTREE_NODE_SIZE {
                let (left, right) = self.split_leaf_node(new_node)?;
                let parent = TimelineNode::new_internal(vec![left, right]);
                Ok(Arc::new(parent))
            } else {
                Ok(Arc::new(new_node))
            }
        } else {
            // Find child to insert into
            let child_index = self.find_child_index(&node, state_id);
            let new_child = self.insert_into_btree(
                Arc::clone(&node.children[child_index]),
                state_id,
                metadata,
            )?;
            
            // Create new internal node with updated child
            let mut new_node = (*node).clone();
            new_node.children[child_index] = new_child;
            
            Ok(Arc::new(new_node))
        }
    }
    
    /// Find state in B-tree with path tracking
    fn find_in_btree(
        &self,
        node: &TimelineNode,
        state_id: StateId,
    ) -> Option<(StateMetadata, Vec<usize>)> {
        if node.is_leaf {
            // Binary search in leaf node
            node.state_ids.binary_search(&state_id)
                .ok()
                .map(|index| (node.metadata[index].clone(), vec![index]))
        } else {
            // Search in internal node
            let child_index = self.find_child_index(node, state_id);
            self.find_in_btree(&node.children[child_index], state_id)
                .map(|(metadata, mut path)| {
                    path.push(child_index);
                    (metadata, path)
                })
        }
    }
    
    /// Check if state is a branch point
    fn check_branch_point(
        &self,
        timeline_id: TimelineId,
        parent_state: StateId,
        current_state: StateId,
    ) -> Result<(), TimelineError> {
        let parent = self.state_manager.get_state(parent_state)?;
        let current = self.state_manager.get_state(current_state)?;
        
        // Detect decision pattern
        if self.branch_detector.is_decision_point(&parent, &current) {
            let branch_point = BranchPoint {
                state_id: current_state,
                parent_timeline: timeline_id,
                child_timelines: Vec::new(),
                timestamp: Instant::now(),
                metadata: BranchMetadata {
                    reason: BranchReason::DecisionPoint,
                    algorithm_step: current.step,
                    decision_context: HashMap::new(),
                },
            };
            
            self.branch_point_index.insert(current_state, branch_point);
            
            // Update timeline metadata
            let timeline = self.timelines.get(&timeline_id).unwrap();
            let mut spine = timeline.write();
            
            let mut new_metadata = spine.metadata.clone();
            new_metadata.branch_points.push(current_state);
            
            let new_spine = TimelineSpine {
                root: Arc::clone(&spine.root),
                metadata: new_metadata,
                version: spine.version + 1,
            };
            
            *spine = new_spine;
        }
        
        Ok(())
    }
}

impl BranchPointDetector {
    fn new() -> Self {
        Self {
            pattern_matcher: DecisionPatternMatcher::new(),
            heuristic_detector: HeuristicChangeDetector::new(),
            state_analyzer: StateAnalyzer::new(),
        }
    }
    
    /// Analyze states for potential branch points
    fn analyze_states(
        &self,
        states: &[(StateId, StateMetadata)],
    ) -> Result<Vec<BranchPoint>, TimelineError> {
        let mut branch_points = Vec::new();
        
        // Parallel analysis of state transitions
        let transitions: Vec<_> = states
            .par_windows(2)
            .map(|window| {
                let (from_id, from_meta) = &window[0];
                let (to_id, to_meta) = &window[1];
                
                self.analyze_transition(*from_id, *to_id, from_meta, to_meta)
            })
            .collect();
        
        // Identify branch points from transitions
        for transition in transitions {
            if let Some(branch_point) = transition {
                branch_points.push(branch_point);
            }
        }
        
        Ok(branch_points)
    }
    
    /// Detect if state transition is a decision point
    fn is_decision_point(
        &self,
        parent: &AlgorithmState,
        current: &AlgorithmState,
    ) -> bool {
        // Analyze state changes for decision patterns
        let open_set_changes = self.analyze_set_changes(&parent.open_set, &current.open_set);
        let closed_set_changes = self.analyze_set_changes(&parent.closed_set, &current.closed_set);
        
        // Check for decision pattern matches
        self.pattern_matcher.matches_decision_pattern(
            &open_set_changes,
            &closed_set_changes,
            parent.current_node,
            current.current_node,
        )
    }
    
    /// Analyze transition for branch point potential
    fn analyze_transition(
        &self,
        from_id: StateId,
        to_id: StateId,
        from_meta: &StateMetadata,
        to_meta: &StateMetadata,
    ) -> Option<BranchPoint> {
        // Check for heuristic changes
        if self.heuristic_detector.detect_change(from_id, to_id) {
            return Some(self.create_branch_point(to_id, BranchReason::HeuristicChange));
        }
        
        // Check for state modifications
        if to_meta.step > from_meta.step + 1 {
            return Some(self.create_branch_point(to_id, BranchReason::StateModification));
        }
        
        None
    }
}

impl TimelineNode {
    fn new_leaf() -> Self {
        Self {
            state_ids: Vec::with_capacity(MAX_BTREE_NODE_SIZE),
            metadata: Vec::with_capacity(MAX_BTREE_NODE_SIZE),
            children: Vec::new(),
            is_leaf: true,
        }
    }
    
    fn new_internal(children: Vec<Arc<TimelineNode>>) -> Self {
        Self {
            state_ids: Vec::with_capacity(MAX_BTREE_NODE_SIZE),
            metadata: Vec::with_capacity(MAX_BTREE_NODE_SIZE),
            children,
            is_leaf: false,
        }
    }
}

const MAX_BTREE_NODE_SIZE: usize = 64; // Optimized for cache line size

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_timeline_navigation() {
        let temp_dir = TempDir::new().unwrap();
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
        
        let timeline_manager = TimelineManager::new(state_manager.clone(), 1000);
        let timeline = timeline_manager.create_timeline(None).unwrap();
        
        // Create state chain
        let mut state_ids = Vec::new();
        for i in 0..10 {
            let state = AlgorithmState::with_step(i);
            let state_id = state_manager.store_state(timeline, state).unwrap();
            
            let parent = if i > 0 { Some(state_ids[i-1]) } else { None };
            timeline_manager.add_state(timeline, state_id, parent).unwrap();
            state_ids.push(state_id);
        }
        
        // Test navigation
        for i in 0..10 {
            let state = timeline_manager.navigate_to(timeline, state_ids[i]).unwrap();
            assert_eq!(state.step, i);
        }
        
        // Test next/previous
        timeline_manager.navigate_to(timeline, state_ids[5]).unwrap();
        let next = timeline_manager.next_state(timeline).unwrap();
        assert_eq!(next, Some(state_ids[6]));
        
        let prev = timeline_manager.previous_state(timeline).unwrap();
        assert_eq!(prev, Some(state_ids[4]));
    }
    
    #[test]
    fn test_branch_creation() {
        let temp_dir = TempDir::new().unwrap();
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
        
        let timeline_manager = TimelineManager::new(state_manager.clone(), 1000);
        let timeline = timeline_manager.create_timeline(None).unwrap();
        
        // Create initial states
        let mut state_ids = Vec::new();
        for i in 0..5 {
            let state = AlgorithmState::with_step(i);
            let state_id = state_manager.store_state(timeline, state).unwrap();
            
            let parent = if i > 0 { Some(state_ids[i-1]) } else { None };
            timeline_manager.add_state(timeline, state_id, parent).unwrap();
            state_ids.push(state_id);
        }
        
        // Create branch
        timeline_manager.navigate_to(timeline, state_ids[2]).unwrap();
        let branch_timeline = timeline_manager.create_branch(
            timeline,
            BranchReason::CounterfactualExploration
        ).unwrap();
        
        // Verify branch point
        let branch_point = timeline_manager.branch_point_index.get(&state_ids[2]).unwrap();
        assert_eq!(branch_point.parent_timeline, timeline);
        assert_eq!(branch_point.child_timelines, vec![branch_timeline]);
    }
    
    #[test]
    fn test_btree_operations() {
        let node = TimelineNode::new_leaf();
        let timeline_manager = TimelineManager::new(
            Arc::new(StateManager::new(
                crate::temporal::state_manager::StateManagerConfig {
                    memory_budget: 1024 * 1024,
                    cache_size: 100,
                    max_delta_chain: 10,
                    storage_path: PathBuf::from("/tmp"),
                    enable_compression: false,
                    compression_level: 0,
                }
            ).unwrap()),
            1000
        );
        
        // Insert states
        let mut root = Arc::new(node);
        for i in 0..100 {
            let state_id = StateId(i);
            let metadata = StateMetadata {
                position: i as usize,
                step: i as usize,
                is_branch_point: false,
                parent: None,
                timestamp: Instant::now(),
            };
            
            root = timeline_manager.insert_into_btree(root, state_id, metadata).unwrap();
        }
        
        // Test search
        for i in 0..100 {
            let state_id = StateId(i);
            let result = timeline_manager.find_in_btree(&root, state_id);
            assert!(result.is_some());
            
            let (metadata, _) = result.unwrap();
            assert_eq!(metadata.step, i as usize);
        }
    }
    
    #[test]
    fn test_mvcc_concurrency() {
        let temp_dir = TempDir::new().unwrap();
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
        
        let timeline_manager = TimelineManager::new(state_manager.clone(), 1000);
        let timeline = timeline_manager.create_timeline(None).unwrap();
        
        let state_id = StateId(1);
        
        // Test concurrent access
        timeline_manager.track_mvcc_access(state_id, AccessType::Read, 1).unwrap();
        timeline_manager.track_mvcc_access(state_id, AccessType::Read, 1).unwrap();
        
        // Test write conflict
        timeline_manager.track_mvcc_access(state_id, AccessType::Write, 2).unwrap();
        let result = timeline_manager.track_mvcc_access(state_id, AccessType::Write, 2);
        assert!(matches!(result, Err(TimelineError::ConcurrentModificationError(_))));
    }
}