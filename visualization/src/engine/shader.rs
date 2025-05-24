//! Advanced shader management system with runtime composition capabilities
//!
//! This module implements a sophisticated shader management system with
//! support for shader variants, preprocessing, and dynamic compilation.
//! Leverages categorical composition for shader transformations and
//! runtime optimization for performance-critical rendering operations.
//!
//! Copyright (c) 2025 Mohammad Atashi <mohammadaliatashi@icloud.com>

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use wgpu::{Device, ShaderModule, ShaderModuleDescriptor, ShaderSource};
use thiserror::Error;
use log::{debug, info, warn};

/// Shader stages following the graphics pipeline model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderStage {
    /// Vertex processing stage
    Vertex,
    /// Fragment (pixel) processing stage
    Fragment,
    /// Compute processing stage
    Compute,
}

impl std::fmt::Display for ShaderStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vertex => write!(f, "Vertex"),
            Self::Fragment => write!(f, "Fragment"),
            Self::Compute => write!(f, "Compute"),
        }
    }
}

/// Shader variant identifier with applied transformations
///
/// Represents a specific configuration of a shader with
/// preprocessor definitions and optimization settings.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShaderVariant {
    /// Base shader identifier
    pub shader_id: String,
    
    /// Shader stage
    pub stage: ShaderStage,
    
    /// Applied preprocessor definitions
    pub defines: Vec<(String, String)>,
    
    /// Optimization level
    pub optimization_level: OptimizationLevel,
}

impl ShaderVariant {
    /// Create a new shader variant
    pub fn new(shader_id: &str, stage: ShaderStage) -> Self {
        Self {
            shader_id: shader_id.to_string(),
            stage,
            defines: Vec::new(),
            optimization_level: OptimizationLevel::Default,
        }
    }
    
    /// Add a preprocessor definition
    pub fn with_define(mut self, key: &str, value: &str) -> Self {
        self.defines.push((key.to_string(), value.to_string()));
        self
    }
    
    /// Set optimization level
    pub fn with_optimization(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }
    
    /// Generate a unique key for this variant
    fn cache_key(&self) -> String {
        let mut key = format!("{}:{}", self.shader_id, self.stage);
        
        // Sort defines for deterministic key generation
        let mut defines = self.defines.clone();
        defines.sort_by(|(a, _), (b, _)| a.cmp(b));
        
        for (k, v) in defines {
            key.push_str(&format!(":{}{}", k, v));
        }
        
        key.push_str(&format!(":{:?}", self.optimization_level));
        
        key
    }
}

/// Shader optimization levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptimizationLevel {
    /// No optimization
    None,
    /// Default optimization
    Default,
    /// Aggressive optimization
    Aggressive,
    /// Size optimization
    Size,
}

/// Shader source definition
#[derive(Debug, Clone)]
pub struct ShaderSource {
    /// Shader ID
    id: String,
    
    /// Source code
    source: String,
    
    /// Entry points for each stage
    entry_points: HashMap<ShaderStage, String>,
    
    /// Required features
    required_features: wgpu::Features,
    
    /// Metadata
    metadata: HashMap<String, String>,
}

impl ShaderSource {
    /// Create a new shader source
    pub fn new(id: &str, source: &str) -> Self {
        let mut entry_points = HashMap::new();
        entry_points.insert(ShaderStage::Vertex, "vs_main".to_string());
        entry_points.insert(ShaderStage::Fragment, "fs_main".to_string());
        entry_points.insert(ShaderStage::Compute, "cs_main".to_string());
        
        Self {
            id: id.to_string(),
            source: source.to_string(),
            entry_points,
            required_features: wgpu::Features::empty(),
            metadata: HashMap::new(),
        }
    }
    
    /// Set entry point for a shader stage
    pub fn with_entry_point(mut self, stage: ShaderStage, entry_point: &str) -> Self {
        self.entry_points.insert(stage, entry_point.to_string());
        self
    }
    
    /// Set required features
    pub fn with_required_features(mut self, features: wgpu::Features) -> Self {
        self.required_features = features;
        self
    }
    
    /// Add metadata
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
    
    /// Get entry point for a shader stage
    pub fn entry_point(&self, stage: ShaderStage) -> Option<&str> {
        self.entry_points.get(&stage).map(|s| s.as_str())
    }
}

/// Compiled shader module with metadata
#[derive(Debug)]
pub struct CompiledShader {
    /// Shader module
    module: Arc<wgpu::ShaderModule>,
    
    /// Entry points
    entry_points: HashMap<ShaderStage, String>,
    
    /// Compilation timestamp
    timestamp: std::time::Instant,
    
    /// Variant used for compilation
    variant: ShaderVariant,
}

impl CompiledShader {
    /// Create a new compiled shader
    fn new(
        module: wgpu::ShaderModule,
        entry_points: HashMap<ShaderStage, String>,
        variant: ShaderVariant,
    ) -> Self {
        Self {
            module: Arc::new(module),
            entry_points,
            timestamp: std::time::Instant::now(),
            variant,
        }
    }
    
    /// Get shader module
    pub fn module(&self) -> Arc<wgpu::ShaderModule> {
        self.module.clone()
    }
    
    /// Get entry point for a shader stage
    pub fn entry_point(&self, stage: ShaderStage) -> Option<&str> {
        self.entry_points.get(&stage).map(|s| s.as_str())
    }
    
    /// Get age of compiled shader
    pub fn age(&self) -> std::time::Duration {
        self.timestamp.elapsed()
    }
}

/// Advanced shader manager with preprocessing and caching
///
/// Implements a sophisticated shader management system with
/// support for shader variants, preprocessing, and dynamic compilation.
pub struct ShaderManager {
    /// Graphics device
    device: Arc<wgpu::Device>,
    
    /// Shader sources
    sources: HashMap<String, ShaderSource>,
    
    /// Compiled shader cache
    compiled_cache: HashMap<String, CompiledShader>,
    
    /// Include paths for shader preprocessing
    include_paths: Vec<PathBuf>,
    
    /// Preprocessor defines available to all shaders
    global_defines: HashMap<String, String>,
    
    /// Shader compilation statistics
    stats: ShaderStatistics,
}

/// Shader compilation statistics
#[derive(Debug, Default)]
struct ShaderStatistics {
    /// Number of compilation requests
    compilation_requests: usize,
    
    /// Number of cache hits
    cache_hits: usize,
    
    /// Total compilation time
    total_compilation_time: std::time::Duration,
    
    /// Maximum compilation time
    max_compilation_time: std::time::Duration,
}

impl ShaderManager {
    /// Create a new shader manager
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self {
            device,
            sources: HashMap::new(),
            compiled_cache: HashMap::new(),
            include_paths: vec![],
            global_defines: HashMap::new(),
            stats: ShaderStatistics::default(),
        }
    }
    
    /// Add an include path for shader preprocessing
    pub fn add_include_path<P: AsRef<Path>>(&mut self, path: P) {
        let path = path.as_ref().to_path_buf();
        if !self.include_paths.contains(&path) {
            self.include_paths.push(path);
        }
    }
    
    /// Add a global preprocessor definition
    pub fn add_global_define(&mut self, key: &str, value: &str) {
        self.global_defines.insert(key.to_string(), value.to_string());
    }
    
    /// Register a shader source
    pub fn register_shader(&mut self, source: ShaderSource) -> Result<(), ShaderError> {
        // Extract includes to validate them
        let includes = self.extract_includes(&source.source)?;
        
        // Ensure all includes exist
        for include in includes {
            let include_path = self.resolve_include(&include)?;
            debug!("Resolved include: {} -> {}", include, include_path.display());
        }
        
        self.sources.insert(source.id.clone(), source);
        Ok(())
    }
    
    /// Register a shader from source code
    pub fn register_shader_source(&mut self, id: &str, source: &str) -> Result<(), ShaderError> {
        let shader_source = ShaderSource::new(id, source);
        self.register_shader(shader_source)
    }
    
    /// Register a shader from file
    pub fn register_shader_from_file<P: AsRef<Path>>(&mut self, id: &str, path: P) -> Result<(), ShaderError> {
        let source = std::fs::read_to_string(&path)
            .map_err(|e| ShaderError::IoError(format!("Failed to read shader file: {}: {}", path.as_ref().display(), e)))?;
        
        self.register_shader_source(id, &source)
    }
    
    /// Get a compiled shader
    pub fn get_shader(&mut self, variant: &ShaderVariant) -> Result<CompiledShader, ShaderError> {
        self.stats.compilation_requests += 1;
        
        let cache_key = variant.cache_key();
        
        // Check cache first
        if let Some(compiled) = self.compiled_cache.get(&cache_key) {
            self.stats.cache_hits += 1;
            return Ok(compiled.clone());
        }
        
        // Get shader source
        let source = self.sources.get(&variant.shader_id)
            .ok_or_else(|| ShaderError::ShaderNotFound(variant.shader_id.clone()))?;
        
        // Check if device supports required features
        if !self.device.features().contains(source.required_features) {
            return Err(ShaderError::UnsupportedFeatures(source.required_features));
        }
        
        // Process shader source
        let start_time = std::time::Instant::now();
        let processed_source = self.preprocess_shader(&source.source, &variant.defines)?;
        
        // Create shader module
        let shader_desc = wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{} {}", variant.stage, variant.shader_id)),
            source: wgpu::ShaderSource::Wgsl(processed_source.into()),
        };
        
        let shader_module = self.device.create_shader_module(shader_desc);
        
        // Create compiled shader
        let compiled = CompiledShader::new(
            shader_module,
            source.entry_points.clone(),
            variant.clone(),
        );
        
        // Update statistics
        let compilation_time = start_time.elapsed();
        self.stats.total_compilation_time += compilation_time;
        if compilation_time > self.stats.max_compilation_time {
            self.stats.max_compilation_time = compilation_time;
        }
        
        // Cache compiled shader
        self.compiled_cache.insert(cache_key, compiled.clone());
        
        info!("Compiled shader: {}:{} in {:?}", variant.shader_id, variant.stage, compilation_time);
        
        Ok(compiled)
    }
    
    /// Preprocess shader source with includes and defines
    fn preprocess_shader(&self, source: &str, defines: &[(String, String)]) -> Result<String, ShaderError> {
        // First pass: process includes recursively
        let source_with_includes = self.process_includes(source)?;
        
        // Second pass: process defines
        let mut processed = source_with_includes;
        
        // Apply global defines first
        for (key, value) in &self.global_defines {
            processed = self.apply_define(&processed, key, value);
        }
        
        // Apply variant defines
        for (key, value) in defines {
            processed = self.apply_define(&processed, key, value);
        }
        
        // Apply conditional compilation
        processed = self.process_conditionals(&processed)?;
        
        Ok(processed)
    }
    
    /// Process includes recursively
    fn process_includes(&self, source: &str) -> Result<String, ShaderError> {
        let mut result = String::new();
        let mut lines = source.lines();
        
        while let Some(line) = lines.next() {
            if let Some(include_path) = self.parse_include_directive(line) {
                // Resolve include path
                let full_path = self.resolve_include(&include_path)?;
                
                // Read include file
                let include_source = std::fs::read_to_string(&full_path)
                    .map_err(|e| ShaderError::IoError(format!("Failed to read include file: {}: {}", full_path.display(), e)))?;
                
                // Process includes in the included file recursively
                let processed_include = self.process_includes(&include_source)?;
                
                // Add processed include
                result.push_str(&processed_include);
                result.push('\n');
            } else {
                result.push_str(line);
                result.push('\n');
            }
        }
        
        Ok(result)
    }
    
    /// Parse include directive from a line
    fn parse_include_directive(&self, line: &str) -> Option<String> {
        let line = line.trim();
        
        if line.starts_with("#include") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let path = parts[1];
                if (path.starts_with('"') && path.ends_with('"')) || 
                   (path.starts_with('<') && path.ends_with('>')) {
                    return Some(path[1..path.len()-1].to_string());
                }
            }
        }
        
        None
    }
    
    /// Extract all includes from shader source
    fn extract_includes(&self, source: &str) -> Result<HashSet<String>, ShaderError> {
        let mut includes = HashSet::new();
        
        for line in source.lines() {
            if let Some(include_path) = self.parse_include_directive(line) {
                includes.insert(include_path);
            }
        }
        
        Ok(includes)
    }
    
    /// Resolve include path
    fn resolve_include(&self, include_path: &str) -> Result<PathBuf, ShaderError> {
        for path in &self.include_paths {
            let full_path = path.join(include_path);
            if full_path.exists() {
                return Ok(full_path);
            }
        }
        
        Err(ShaderError::IncludeNotFound(include_path.to_string()))
    }
    
    /// Apply preprocessor definition to shader source
    fn apply_define(&self, source: &str, key: &str, value: &str) -> String {
        let define_pattern = format!("#{} {}", key, value);
        let mut result = String::new();
        
        for line in source.lines() {
            if line.trim().starts_with(&format!("#define {}", key)) {
                result.push_str(&format!("#define {} {}\n", key, value));
            } else {
                result.push_str(line);
                result.push('\n');
            }
        }
        
        result
    }
    
    /// Process conditional compilation directives
    fn process_conditionals(&self, source: &str) -> Result<String, ShaderError> {
        // This is a simplified implementation
        // A full implementation would need to handle nested conditionals, etc.
        
        let mut result = String::new();
        let mut include_lines = true;
        
        for line in source.lines() {
            let trimmed = line.trim();
            
            if trimmed.starts_with("#ifdef") {
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 2 {
                    let define = parts[1];
                    include_lines = self.global_defines.contains_key(define);
                }
            } else if trimmed.starts_with("#ifndef") {
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 2 {
                    let define = parts[1];
                    include_lines = !self.global_defines.contains_key(define);
                }
            } else if trimmed.starts_with("#else") {
                include_lines = !include_lines;
            } else if trimmed.starts_with("#endif") {
                include_lines = true;
            } else if include_lines {
                result.push_str(line);
                result.push('\n');
            }
        }
        
        Ok(result)
    }
    
    /// Get shader compilation statistics
    pub fn get_statistics(&self) -> ShaderStatisticsReport {
        ShaderStatisticsReport {
            compilation_requests: self.stats.compilation_requests,
            cache_hits: self.stats.cache_hits,
            cache_hit_rate: if self.stats.compilation_requests > 0 {
                (self.stats.cache_hits as f64) / (self.stats.compilation_requests as f64)
            } else {
                0.0
            },
            total_compilation_time: self.stats.total_compilation_time,
            max_compilation_time: self.stats.max_compilation_time,
            shader_count: self.sources.len(),
            compiled_shader_count: self.compiled_cache.len(),
        }
    }
    
    /// Clear compiled shader cache
    pub fn clear_cache(&mut self) {
        self.compiled_cache.clear();
    }
}

/// Shader compilation statistics report
#[derive(Debug, Clone)]
pub struct ShaderStatisticsReport {
    /// Number of compilation requests
    pub compilation_requests: usize,
    
    /// Number of cache hits
    pub cache_hits: usize,
    
    /// Cache hit rate (0.0-1.0)
    pub cache_hit_rate: f64,
    
    /// Total compilation time
    pub total_compilation_time: std::time::Duration,
    
    /// Maximum compilation time
    pub max_compilation_time: std::time::Duration,
    
    /// Number of registered shaders
    pub shader_count: usize,
    
    /// Number of compiled shaders in cache
    pub compiled_shader_count: usize,
}

/// Shader-related error types
#[derive(Debug, Error)]
pub enum ShaderError {
    /// Shader not found
    #[error("Shader not found: {0}")]
    ShaderNotFound(String),
    
    /// IO error
    #[error("IO error: {0}")]
    IoError(String),
    
    /// Include not found
    #[error("Include not found: {0}")]
    IncludeNotFound(String),
    
    /// Preprocessing error
    #[error("Preprocessing error: {0}")]
    PreprocessingError(String),
    
    /// Unsupported features
    #[error("Shader requires unsupported features: {0:?}")]
    UnsupportedFeatures(wgpu::Features),
    
    /// Compilation error
    #[error("Shader compilation error: {0}")]
    CompilationError(String),
    
    /// Other error
    #[error("Shader error: {0}")]
    Other(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_shader_variant_cache_key() {
        let variant1 = ShaderVariant::new("test", ShaderStage::Vertex)
            .with_define("A", "1")
            .with_define("B", "2");
            
        let variant2 = ShaderVariant::new("test", ShaderStage::Vertex)
            .with_define("B", "2")
            .with_define("A", "1");
            
        // Defines should be sorted, so keys should be equal
        assert_eq!(variant1.cache_key(), variant2.cache_key());
        
        let variant3 = variant1.with_optimization(OptimizationLevel::Aggressive);
        assert_ne!(variant1.cache_key(), variant3.cache_key());
    }
    
    #[test]
    fn test_extract_includes() {
        let manager = ShaderManager::new(Arc::new(unsafe { wgpu::Device::new(wgpu::DeviceDescriptor {
            label: None,
            features: wgpu::Features::empty(),
            limits: wgpu::Limits::default(),
        }) }));
        
        let source = r#"
        #include "common.wgsl"
        #include <math.wgsl>
        
        fn main() {
            // Some code
        }
        "#;
        
        let includes = manager.extract_includes(source).unwrap();
        assert_eq!(includes.len(), 2);
        assert!(includes.contains("common.wgsl"));
        assert!(includes.contains("math.wgsl"));
    }
}