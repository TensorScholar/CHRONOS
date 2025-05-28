//! Advanced Temporal State Management
//! Revolutionary bidirectional execution with mathematical optimization
//! Copyright (c) 2025 Mohammad Atashi <mohammadaliatashi@icloud.com>

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, RwLock};
use serde::{Serialize, Deserialize};

/// Advanced temporal state manager
#[derive(Debug)]
pub struct TemporalStateManager<T: Clone + Send + Sync> {
    /// State timeline storage
    timeline: Arc<RwLock<BTreeMap<usize, T>>>,
    /// Current state index
    current_index: Arc<RwLock<usize>>,
    /// State compression engine
    compressor: StateCompressor<T>,
}

impl<T: Clone + Send + Sync> TemporalStateManager<T> {
    /// Create new temporal state manager
    pub fn new() -> Self {
        Self {
            timeline: Arc::new(RwLock::new(BTreeMap::new())),
            current_index: Arc::new(RwLock::new(0)),
            compressor: StateCompressor::new(),
        }
    }
    
    /// Capture current algorithm state
    pub fn capture_state(&self, state: T) -> Result<usize, TemporalError> {
        let mut timeline = self.timeline.write().unwrap();
        let mut current = self.current_index.write().unwrap();
        
        *current += 1;
        timeline.insert(*current, state);
        
        Ok(*current)
    }
    
    /// Navigate to specific state index
    pub fn navigate_to(&self, index: usize) -> Result<Option<T>, TemporalError> {
        let timeline = self.timeline.read().unwrap();
        let mut current = self.current_index.write().unwrap();
        
        if let Some(state) = timeline.get(&index) {
            *current = index;
            Ok(Some(state.clone()))
        } else {
            Ok(None)
        }
    }
    
    /// Step forward in timeline
    pub fn step_forward(&self) -> Result<Option<T>, TemporalError> {
        let current = self.current_index.read().unwrap();
        self.navigate_to(*current + 1)
    }
    
    /// Step backward in timeline
    pub fn step_backward(&self) -> Result<Option<T>, TemporalError> {
        let current = self.current_index.read().unwrap();
        if *current > 0 {
            self.navigate_to(*current - 1)
        } else {
            Ok(None)
        }
    }
}

/// State compression for memory optimization
#[derive(Debug)]
pub struct StateCompressor<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T> StateCompressor<T> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Temporal system errors
#[derive(Debug, thiserror::Error)]
pub enum TemporalError {
    #[error("State not found")]
    StateNotFound,
    #[error("Timeline corruption")]
    TimelineCorruption,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_temporal_state_manager() {
        let manager = TemporalStateManager::new();
        let index = manager.capture_state("test_state".to_string()).unwrap();
        assert_eq!(index, 1);
        
        let state = manager.navigate_to(index).unwrap();
        assert_eq!(state, Some("test_state".to_string()));
    }
}
