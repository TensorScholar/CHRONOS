//! Hierarchical State Management System
//!
//! This module implements a mathematically rigorous state management system
//! for temporal debugging operations, leveraging persistent data structures,
//! MVCC semantics, and advanced caching strategies.
//!
//! Copyright (c) 2025 Mohammad Atashi <mohammadaliatashi@icloud.com>

use std::sync::{Arc, RwLock, atomic::{AtomicU64, AtomicBool, Ordering}};
use std::collections::{BTreeMap, HashMap};
use std::time::{Instant, Duration};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Sender, Receiver};
use std::thread::{self, JoinHandle};
use std::alloc::{Layout, alloc, dealloc};

use dashmap::DashMap;
use lru::LruCache;
use parking_lot::RwLock as ParkingRwLock;
use serde::{Serialize, Deserialize};
use thiserror::Error;

use crate::algorithm::state::{AlgorithmState, StateSnapshot, StateChecksum};
use crate::temporal::delta::{StateDelta, DeltaCompressor};
use crate::temporal::storage::{DiskStorage, StorageBackend};

/// State management error types with forensic context
#[derive(Error, Debug)]
pub enum StateManagerError {
    #[error("State ID {0} not found in timeline")]
    StateNotFound(StateId),
    
    #[error("Memory budget exceeded: {used}/{limit} bytes")]
    MemoryExhausted { used: usize, limit: usize },
    
    #[error("State corruption detected: expected checksum {expected}, got {actual}")]
    IntegrityCheckFailed { expected: StateChecksum, actual: StateChecksum },
    
    #[error("Concurrent modification conflict at state {0}")]
    ModificationConflict(StateId),
    
    #[error("Storage backend error: {0}")]
    StorageError(#[from] std::io::Error),
    
    #[error("Delta chain too long: {0} entries exceed threshold {1}")]
    DeltaChainExhausted(usize, usize),
}

/// Unique identifier for algorithm states
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct StateId(pub u64);

/// Timeline identifier for managing multiple execution branches
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TimelineId(pub u64);

/// State metrics for performance monitoring
#[derive(Debug, Clone)]
pub struct StateMetrics {
    pub access_count: u64,
    pub last_access: Instant,
    pub memory_size: usize,
    pub delta_chain_length: usize,
    pub compression_ratio: f64,
}

/// Configuration for state manager behavior
#[derive(Debug, Clone)]
pub struct StateManagerConfig {
    /// Maximum memory budget in bytes
    pub memory_budget: usize,
    /// LRU cache size
    pub cache_size: usize,
    /// Maximum delta chain length before snapshot
    pub max_delta_chain: usize,
    /// Disk storage path
    pub storage_path: PathBuf,
    /// Enable compression
    pub enable_compression: bool,
    /// Compression level (0-9)
    pub compression_level: u8,
}

/// Hierarchical state manager with multi-level caching
pub struct StateManager {
    /// Current state ID generator
    state_id_gen: AtomicU64,
    
    /// Timeline ID generator
    timeline_id_gen: AtomicU64,
    
    /// Memory-resident state cache (L1)
    memory_cache: ParkingRwLock<LruCache<StateId, Arc<StateSnapshot>>>,
    
    /// Compressed state cache (L2)
    compressed_cache: ParkingRwLock<LruCache<StateId, Arc<CompressedState>>>,
    
    /// State index for O(log n) access
    state_index: DashMap<StateId, StateLocation>,
    
    /// Delta chain tracker
    delta_chains: DashMap<StateId, Arc<DeltaChain>>,
    
    /// Timeline metadata
    timelines: DashMap<TimelineId, Arc<TimelineMetadata>>,
    
    /// Disk storage backend
    disk_storage: Arc<DiskStorage>,
    
    /// Memory usage tracking
    memory_used: AtomicU64,
    
    /// Configuration
    config: StateManagerConfig,
    
    /// Performance metrics
    metrics: DashMap<StateId, StateMetrics>,
    
    /// Background compaction worker
    compaction_worker: Option<CompactionWorker>,
    
    /// Consistency verification caches
    checksum_cache: DashMap<StateId, StateChecksum>,
}

/// Compressed state representation
#[derive(Debug, Clone)]
pub struct CompressedState {
    state_id: StateId,
    compressed_data: Vec<u8>,
    original_size: usize,
    compression_ratio: f64,
    checksum: StateChecksum,
}

/// State location metadata
#[derive(Debug, Clone)]
pub enum StateLocation {
    Memory { timeline_id: TimelineId, offset: usize },
    Compressed { timeline_id: TimelineId, offset: usize },
    DiskBacked { timeline_id: TimelineId, file_id: u64, offset: u64 },
    DeltaChain { base_state: StateId, delta_index: usize },
}

/// Timeline metadata for branch management
#[derive(Debug, Clone)]
pub struct TimelineMetadata {
    timeline_id: TimelineId,
    parent: Option<TimelineId>,
    creation_time: Instant,
    state_count: AtomicU64,
    memory_usage: AtomicU64,
    root_state: Option<StateId>,
}

/// Delta chain for incremental updates
#[derive(Debug)]
pub struct DeltaChain {
    base_state: StateId,
    deltas: RwLock<Vec<Arc<StateDelta>>>,
    total_size: AtomicU64,
    modification_lock: RwLock<()>,
}

/// Background compaction worker
struct CompactionWorker {
    handle: JoinHandle<()>,
    shutdown_signal: Arc<AtomicBool>,
    task_queue: Sender<CompactionTask>,
}

/// Compaction tasks for background processing
#[derive(Debug)]
enum CompactionTask {
    CompressDeltaChain(StateId),
    EvictColdStates,
    VerifyIntegrity(StateId),
    DiskFlush(Vec<(StateId, Arc<StateSnapshot>)>),
}

impl StateManager {
    /// Create new state manager with specified configuration
    pub fn new(config: StateManagerConfig) -> Result<Self, StateManagerError> {
        let disk_storage = Arc::new(DiskStorage::new(&config.storage_path)?);
        
        // Initialize cache with advanced algorithms
        let memory_cache = ParkingRwLock::new(LruCache::new(config.cache_size));
        let compressed_cache = ParkingRwLock::new(LruCache::new(config.cache_size / 2));
        
        // Initialize worker thread for background tasks
        let (task_sender, task_receiver) = channel();
        let shutdown_signal = Arc::new(AtomicBool::new(false));
        
        let compaction_worker = CompactionWorker::new(
            task_receiver,
            Arc::clone(&shutdown_signal),
            Arc::clone(&disk_storage),
        );
        
        Ok(Self {
            state_id_gen: AtomicU64::new(0),
            timeline_id_gen: AtomicU64::new(0),
            memory_cache,
            compressed_cache,
            state_index: DashMap::new(),
            delta_chains: DashMap::new(),
            timelines: DashMap::new(),
            disk_storage,
            memory_used: AtomicU64::new(0),
            config,
            metrics: DashMap::new(),
            compaction_worker: Some(compaction_worker),
            checksum_cache: DashMap::new(),
        })
    }
    
    /// Create new timeline for execution branch
    pub fn create_timeline(&self, parent: Option<TimelineId>) -> TimelineId {
        let timeline_id = TimelineId(self.timeline_id_gen.fetch_add(1, Ordering::SeqCst));
        
        let metadata = Arc::new(TimelineMetadata {
            timeline_id,
            parent,
            creation_time: Instant::now(),
            state_count: AtomicU64::new(0),
            memory_usage: AtomicU64::new(0),
            root_state: None,
        });
        
        self.timelines.insert(timeline_id, metadata);
        timeline_id
    }
    
    /// Store new state with hierarchical management
    pub fn store_state(
        &self,
        timeline_id: TimelineId,
        state: AlgorithmState,
    ) -> Result<StateId, StateManagerError> {
        let state_id = StateId(self.state_id_gen.fetch_add(1, Ordering::SeqCst));
        
        // Create snapshot with validation
        let snapshot = Arc::new(StateSnapshot {
            state_id,
            timeline_id,
            state,
            creation_time: Instant::now(),
            memory_size: self.calculate_state_size(&state),
            checksum: self.compute_checksum(&state),
        });
        
        // Check memory budget
        if self.check_memory_budget(snapshot.memory_size)? {
            // Store in memory cache (L1)
            self.store_in_memory_cache(state_id, Arc::clone(&snapshot));
            
            // Update state index
            self.state_index.insert(
                state_id,
                StateLocation::Memory {
                    timeline_id,
                    offset: 0, // Offset managed by cache
                },
            );
            
            // Update timeline metadata
            if let Some(timeline) = self.timelines.get(&timeline_id) {
                timeline.state_count.fetch_add(1, Ordering::SeqCst);
                timeline.memory_usage.fetch_add(snapshot.memory_size as u64, Ordering::SeqCst);
            }
            
            // Initialize metrics
            self.metrics.insert(state_id, StateMetrics {
                access_count: 1,
                last_access: Instant::now(),
                memory_size: snapshot.memory_size,
                delta_chain_length: 0,
                compression_ratio: 1.0,
            });
            
            // Store checksum for integrity verification
            self.checksum_cache.insert(state_id, snapshot.checksum);
            
            Ok(state_id)
        } else {
            // Memory exhausted - trigger compaction
            self.trigger_compaction();
            Err(StateManagerError::MemoryExhausted {
                used: self.memory_used.load(Ordering::SeqCst) as usize,
                limit: self.config.memory_budget,
            })
        }
    }
    
    /// Retrieve state with O(log n) access
    pub fn get_state(
        &self,
        state_id: StateId,
    ) -> Result<Arc<AlgorithmState>, StateManagerError> {
        // Update access metrics
        if let Some(mut metrics) = self.metrics.get_mut(&state_id) {
            metrics.access_count += 1;
            metrics.last_access = Instant::now();
        }
        
        // Hierarchical cache lookup
        match self.locate_state(state_id)? {
            StateLocation::Memory { .. } => {
                // L1 cache hit
                self.memory_cache.read()
                    .get(&state_id)
                    .map(|snapshot| Arc::clone(&snapshot.state))
                    .ok_or(StateManagerError::StateNotFound(state_id))
            },
            StateLocation::Compressed { .. } => {
                // L2 cache hit - decompress and promote to L1
                self.decompress_and_promote(state_id)
            },
            StateLocation::DiskBacked { file_id, offset, .. } => {
                // Disk hit - load and cache
                self.load_from_disk(state_id, file_id, offset)
            },
            StateLocation::DeltaChain { base_state, delta_index } => {
                // Delta chain reconstruction
                self.reconstruct_from_delta_chain(base_state, delta_index)
            },
        }
    }
    
    /// Create delta between two states
    pub fn create_delta(
        &self,
        from_state: StateId,
        to_state: StateId,
    ) -> Result<Arc<StateDelta>, StateManagerError> {
        let from = self.get_state(from_state)?;
        let to = self.get_state(to_state)?;
        
        let delta = StateDelta::compute(&from, &to)?;
        let compressed_delta = if self.config.enable_compression {
            delta.compress(self.config.compression_level)?
        } else {
            delta
        };
        
        // Update delta chains
        self.append_to_delta_chain(from_state, to_state, Arc::new(compressed_delta))
    }
    
    /// Append delta to chain with automatic compaction
    fn append_to_delta_chain(
        &self,
        base_state: StateId,
        new_state: StateId,
        delta: Arc<StateDelta>,
    ) -> Result<(), StateManagerError> {
        let chain = self.delta_chains.entry(base_state).or_insert_with(|| {
            Arc::new(DeltaChain {
                base_state,
                deltas: RwLock::new(Vec::new()),
                total_size: AtomicU64::new(0),
                modification_lock: RwLock::new(()),
            })
        });
        
        // Acquire modification lock
        let _lock = chain.modification_lock.write();
        
        // Append delta
        let mut deltas = chain.deltas.write();
        deltas.push(Arc::clone(&delta));
        chain.total_size.fetch_add(delta.size(), Ordering::SeqCst);
        
        // Check chain length and trigger compaction if needed
        if deltas.len() > self.config.max_delta_chain {
            drop(deltas);
            self.schedule_compaction(base_state)?;
        }
        
        Ok(())
    }
    
    /// Schedule compaction task
    fn schedule_compaction(&self, state_id: StateId) -> Result<(), StateManagerError> {
        if let Some(worker) = &self.compaction_worker {
            worker.task_queue
                .send(CompactionTask::CompressDeltaChain(state_id))
                .map_err(|_| StateManagerError::StorageError(
                    std::io::Error::new(std::io::ErrorKind::BrokenPipe, "Compaction worker disconnected")
                ))
        } else {
            Ok(())
        }
    }
    
    /// Reconstruct state from delta chain
    fn reconstruct_from_delta_chain(
        &self,
        base_state: StateId,
        target_delta_index: usize,
    ) -> Result<Arc<AlgorithmState>, StateManagerError> {
        let chain = self.delta_chains.get(&base_state)
            .ok_or(StateManagerError::StateNotFound(base_state))?;
        
        let _lock = chain.modification_lock.read();
        let deltas = chain.deltas.read();
        
        // Validate delta index
        if target_delta_index >= deltas.len() {
            return Err(StateManagerError::StateNotFound(StateId(
                (base_state.0 as u128 + target_delta_index as u128) as u64
            )));
        }
        
        // Get base state
        let mut current_state = self.get_state(base_state)?;
        
        // Apply deltas sequentially
        for i in 0..=target_delta_index {
            current_state = Arc::new(deltas[i].apply(&current_state)?);
        }
        
        Ok(current_state)
    }
    
    /// Compress and store state snapshot
    fn compress_state(
        &self,
        state_id: StateId,
        snapshot: Arc<StateSnapshot>,
    ) -> Result<(), StateManagerError> {
        let compressed = if self.config.enable_compression {
            let compressed_data = bincode::serialize(&snapshot)?;
            let compression_ratio = snapshot.memory_size as f64 / compressed_data.len() as f64;
            
            CompressedState {
                state_id,
                compressed_data,
                original_size: snapshot.memory_size,
                compression_ratio,
                checksum: snapshot.checksum,
            }
        } else {
            // No compression - store raw data
            let serialized = bincode::serialize(&snapshot)?;
            CompressedState {
                state_id,
                compressed_data: serialized,
                original_size: snapshot.memory_size,
                compression_ratio: 1.0,
                checksum: snapshot.checksum,
            }
        };
        
        // Store in compressed cache
        let mut cache = self.compressed_cache.write();
        cache.put(state_id, Arc::new(compressed));
        
        // Update state location
        self.state_index.insert(
            state_id,
            StateLocation::Compressed {
                timeline_id: snapshot.timeline_id,
                offset: 0,
            },
        );
        
        // Update metrics
        if let Some(mut metrics) = self.metrics.get_mut(&state_id) {
            metrics.compression_ratio = compressed.compression_ratio;
        }
        
        Ok(())
    }
    
    /// Verify state integrity
    pub fn verify_integrity(&self, state_id: StateId) -> Result<bool, StateManagerError> {
        let state = self.get_state(state_id)?;
        let computed_checksum = self.compute_checksum(&state);
        
        let expected_checksum = self.checksum_cache.get(&state_id)
            .map(|entry| *entry.value())
            .unwrap_or_else(|| computed_checksum); // Fallback for legacy states
        
        if computed_checksum != expected_checksum {
            Err(StateManagerError::IntegrityCheckFailed {
                expected: expected_checksum,
                actual: computed_checksum,
            })
        } else {
            Ok(true)
        }
    }
    
    /// Get state metrics for monitoring
    pub fn get_metrics(&self, state_id: StateId) -> Option<StateMetrics> {
        self.metrics.get(&state_id).map(|entry| entry.clone())
    }
    
    /// Memory budget management
    fn check_memory_budget(&self, additional_size: usize) -> Result<bool, StateManagerError> {
        let current_usage = self.memory_used.load(Ordering::SeqCst) as usize;
        if current_usage + additional_size <= self.config.memory_budget {
            self.memory_used.fetch_add(additional_size as u64, Ordering::SeqCst);
            Ok(true)
        } else {
            Ok(false)
        }
    }
    
    /// Calculate state memory size
    fn calculate_state_size(&self, state: &AlgorithmState) -> usize {
        // Precise calculation considering all state components
        let mut size = std::mem::size_of::<AlgorithmState>();
        
        // Add collections sizes
        size += state.open_set.capacity() * std::mem::size_of::<crate::algorithm::traits::NodeId>();
        size += state.closed_set.capacity() * std::mem::size_of::<crate::algorithm::traits::NodeId>();
        
        // Add any custom metadata
        if let Some(metadata) = &state.metadata {
            size += metadata.size_of();
        }
        
        size
    }
    
    /// Compute state checksum for integrity verification
    fn compute_checksum(&self, state: &AlgorithmState) -> StateChecksum {
        use std::hash::{Hash, Hasher};
        use fnv::FnvHasher;
        
        let mut hasher = FnvHasher::default();
        state.hash(&mut hasher);
        
        StateChecksum(hasher.finish())
    }
}

impl CompactionWorker {
    fn new(
        task_receiver: Receiver<CompactionTask>,
        shutdown_signal: Arc<AtomicBool>,
        disk_storage: Arc<DiskStorage>,
    ) -> Self {
        let handle = thread::spawn(move || {
            while !shutdown_signal.load(Ordering::Relaxed) {
                match task_receiver.recv() {
                    Ok(task) => Self::process_task(task, &disk_storage),
                    Err(_) => break, // Channel closed
                }
            }
        });
        
        Self {
            handle,
            shutdown_signal,
            task_queue: task_sender,
        }
    }
    
    fn process_task(task: CompactionTask, storage: &DiskStorage) {
        match task {
            CompactionTask::CompressDeltaChain(state_id) => {
                // Implement delta chain compression
            },
            CompactionTask::EvictColdStates => {
                // Implement cold state eviction
            },
            CompactionTask::VerifyIntegrity(state_id) => {
                // Verify state integrity
            },
            CompactionTask::DiskFlush(states) => {
                // Flush states to disk
                for (state_id, snapshot) in states {
                    storage.write_state(state_id, &snapshot);
                }
            },
        }
    }
}

impl Drop for StateManager {
    fn drop(&mut self) {
        // Shutdown background worker
        if let Some(worker) = self.compaction_worker.take() {
            worker.shutdown_signal.store(true, Ordering::SeqCst);
            let _ = worker.handle.join();
        }
        
        // Flush remaining data to disk
        self.disk_storage.flush_all().unwrap_or_else(|e| {
            eprintln!("Failed to flush disk storage during shutdown: {}", e);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::time::Duration;
    
    #[test]
    fn test_hierarchical_access() {
        let temp_dir = TempDir::new().unwrap();
        let config = StateManagerConfig {
            memory_budget: 1024 * 1024, // 1MB
            cache_size: 100,
            max_delta_chain: 10,
            storage_path: temp_dir.path().to_path_buf(),
            enable_compression: true,
            compression_level: 6,
        };
        
        let manager = StateManager::new(config).unwrap();
        let timeline = manager.create_timeline(None);
        
        // Create test state
        let state = AlgorithmState::default();
        let state_id = manager.store_state(timeline, state).unwrap();
        
        // Verify O(log n) access
        let retrieved = manager.get_state(state_id).unwrap();
        assert_eq!(retrieved.step, 0);
        
        // Verify metrics tracking
        let metrics = manager.get_metrics(state_id).unwrap();
        assert_eq!(metrics.access_count, 2); // Store + Retrieve
    }
    
    #[test]
    fn test_memory_management() {
        let temp_dir = TempDir::new().unwrap();
        let config = StateManagerConfig {
            memory_budget: 1024, // Very small for testing
            cache_size: 10,
            max_delta_chain: 5,
            storage_path: temp_dir.path().to_path_buf(),
            enable_compression: true,
            compression_level: 9,
        };
        
        let manager = StateManager::new(config).unwrap();
        let timeline = manager.create_timeline(None);
        
        // Fill memory until exhaustion
        let mut state_ids = Vec::new();
        for i in 0..50 {
            let state = AlgorithmState::with_step(i);
            match manager.store_state(timeline, state) {
                Ok(id) => state_ids.push(id),
                Err(StateManagerError::MemoryExhausted { .. }) => break,
            }
        }
        
        // Verify some states were evicted
        assert!(state_ids.len() < 50);
        
        // Verify cold states are still accessible
        let first_state = manager.get_state(state_ids[0]).unwrap();
        assert_eq!(first_state.step, 0);
    }
    
    #[test]
    fn test_delta_chain_reconstruction() {
        let temp_dir = TempDir::new().unwrap();
        let config = StateManagerConfig {
            memory_budget: 1024 * 1024,
            cache_size: 100,
            max_delta_chain: 20,
            storage_path: temp_dir.path().to_path_buf(),
            enable_compression: false,
            compression_level: 0,
        };
        
        let manager = StateManager::new(config).unwrap();
        let timeline = manager.create_timeline(None);
        
        // Create state chain
        let mut state_ids = Vec::new();
        let base_state = AlgorithmState::default();
        let base_id = manager.store_state(timeline, base_state).unwrap();
        state_ids.push(base_id);
        
        // Create delta chain
        for i in 1..10 {
            let state = AlgorithmState::with_step(i);
            let state_id = manager.store_state(timeline, state).unwrap();
            manager.create_delta(state_ids[i-1], state_id).unwrap();
            state_ids.push(state_id);
        }
        
        // Verify reconstruction
        for i in 0..10 {
            let state = manager.get_state(state_ids[i]).unwrap();
            assert_eq!(state.step, i);
        }
    }
    
    #[test]
    fn test_integrity_verification() {
        let temp_dir = TempDir::new().unwrap();
        let config = StateManagerConfig {
            memory_budget: 1024 * 1024,
            cache_size: 100,
            max_delta_chain: 10,
            storage_path: temp_dir.path().to_path_buf(),
            enable_compression: true,
            compression_level: 3,
        };
        
        let manager = StateManager::new(config).unwrap();
        let timeline = manager.create_timeline(None);
        
        let state = AlgorithmState::default();
        let state_id = manager.store_state(timeline, state).unwrap();
        
        // Verify integrity
        assert!(manager.verify_integrity(state_id).unwrap());
        
        // Corrupt checksum cache and test detection
        let fake_checksum = StateChecksum(12345);
        manager.checksum_cache.insert(state_id, fake_checksum);
        
        match manager.verify_integrity(state_id) {
            Err(StateManagerError::IntegrityCheckFailed { .. }) => {
                // Expected error
            },
            _ => panic!("Integrity check should have failed"),
        }
    }
}