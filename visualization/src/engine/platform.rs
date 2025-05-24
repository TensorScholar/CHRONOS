//! Cross-platform rendering system with automatic feature adaptation
//!
//! This module provides a comprehensive abstraction layer for graphics rendering
//! across multiple platforms, enabling consistent visualization with automatic
//! feature negotiation and graceful fallback strategies.
//!
//! Copyright (c) 2025 Mohammad Atashi <mohammadaliatashi@icloud.com>

use std::sync::Arc;
use wgpu::{Adapter, Device, Instance, Queue, Surface, SurfaceConfiguration};
use wgpu::util::DeviceExt;
use winit::window::Window;
use raw_window_handle::HasRawWindowHandle;
use thiserror::Error;
use log::{info, warn, debug};

/// Platform capability detection and feature negotiation system
///
/// Provides runtime detection of graphics capabilities with automatic
/// feature negotiation and graceful fallback strategies.
#[derive(Debug, Clone)]
pub struct PlatformCapabilities {
    /// Maximum texture dimensions
    pub max_texture_dimension: u32,
    
    /// Maximum compute workgroup size
    pub max_compute_workgroup_size: [u32; 3],
    
    /// Maximum uniform buffer binding size
    pub max_uniform_buffer_binding_size: u64,
    
    /// Maximum storage buffer binding size
    pub max_storage_buffer_binding_size: u64,
    
    /// Available features
    pub features: wgpu::Features,
    
    /// Supported texture formats
    pub texture_formats: Vec<wgpu::TextureFormat>,
    
    /// Adapter info
    pub adapter_info: wgpu::AdapterInfo,
    
    /// Preferred texture format
    pub preferred_texture_format: wgpu::TextureFormat,
    
/// Platform type
    pub platform_type: PlatformType,
    
    /// Backend type
    pub backend_type: wgpu::BackendBit,
    
    /// Supported sample count for anti-aliasing
    pub supported_sample_count: u32,
    
    /// Has discrete GPU
    pub has_discrete_gpu: bool,
}

/// Platform type identification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlatformType {
    /// Windows platform
    Windows,
    
    /// macOS platform
    MacOS,
    
    /// Linux platform
    Linux,
    
    /// Web platform
    Web,
    
    /// Android platform
    Android,
    
    /// iOS platform
    IOS,
    
    /// Other platform
    Other,
}

impl PlatformType {
    /// Detect platform type at runtime
    pub fn detect() -> Self {
        #[cfg(target_os = "windows")]
        return PlatformType::Windows;
        
        #[cfg(target_os = "macos")]
        return PlatformType::MacOS;
        
        #[cfg(target_os = "linux")]
        return PlatformType::Linux;
        
        #[cfg(target_arch = "wasm32")]
        return PlatformType::Web;
        
        #[cfg(target_os = "android")]
        return PlatformType::Android;
        
        #[cfg(target_os = "ios")]
        return PlatformType::IOS;
        
        PlatformType::Other
    }
    
    /// Check if the platform is desktop (Windows, macOS, Linux)
    pub fn is_desktop(&self) -> bool {
        matches!(self, Self::Windows | Self::MacOS | Self::Linux)
    }
    
    /// Check if the platform is mobile (Android, iOS)
    pub fn is_mobile(&self) -> bool {
        matches!(self, Self::Android | Self::IOS)
    }
    
    /// Check if the platform is web
    pub fn is_web(&self) -> bool {
        matches!(self, Self::Web)
    }
}

/// Platform-specific configuration for rendering
#[derive(Debug, Clone)]
pub struct PlatformConfig {
    /// Requested adapter options
    pub adapter_options: wgpu::RequestAdapterOptions<'static>,
    
    /// Requested device options
    pub device_options: wgpu::DeviceDescriptor<'static>,
    
    /// Surface configuration
    pub surface_config: Option<wgpu::SurfaceConfiguration>,
    
    /// Required features
    pub required_features: wgpu::Features,
    
    /// Required limits
    pub required_limits: wgpu::Limits,
    
    /// Preferred backend
    pub preferred_backend: Option<wgpu::BackendBit>,
    
    /// Preferred power preference
    pub power_preference: wgpu::PowerPreference,
    
    /// Force software rendering
    pub force_software: bool,
    
    /// Enable validation
    pub enable_validation: bool,
    
    /// High dynamic range (HDR)
    pub enable_hdr: bool,
    
    /// Multi-sampling anti-aliasing sample count
    pub sample_count: u32,
}

impl Default for PlatformConfig {
    fn default() -> Self {
        // Create default adapter options
        let adapter_options = wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        };

        // Create default device options
        let device_options = wgpu::DeviceDescriptor {
            label: Some("Chronos Rendering Device"),
            features: wgpu::Features::empty(),
            limits: wgpu::Limits::default(),
        };
        
        Self {
            adapter_options,
            device_options,
            surface_config: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            preferred_backend: None,
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_software: false,
            enable_validation: cfg!(debug_assertions),
            enable_hdr: false,
            sample_count: 1,
        }
    }
}

/// Cross-platform rendering system with automatic fallback
///
/// Provides a unified interface for rendering across different platforms,
/// with automatic feature negotiation and graceful fallback strategies.
pub struct RenderingPlatform {
    /// WGPU instance
    instance: wgpu::Instance,
    
    /// Selected graphics adapter
    adapter: Arc<wgpu::Adapter>,
    
    /// Graphics device
    device: Arc<wgpu::Device>,
    
    /// Command queue
    queue: Arc<wgpu::Queue>,
    
    /// Surface for window rendering (if available)
    surface: Option<wgpu::Surface>,
    
    /// Surface configuration (if available)
    surface_config: Option<wgpu::SurfaceConfiguration>,
    
    /// Platform capabilities
    capabilities: PlatformCapabilities,
    
    /// Platform configuration
    config: PlatformConfig,
}

impl RenderingPlatform {
    /// Create a new platform for offscreen rendering
    pub async fn new_headless(config: PlatformConfig) -> Result<Self, PlatformError> {
        // Create instance with selected backends
        let instance = Self::create_instance(&config)?;
        
        // Request adapter
        let adapter = Self::request_adapter(&instance, &config, None).await?;
        let adapter = Arc::new(adapter);
        
        // Create device and queue
        let (device, queue) = Self::create_device_and_queue(&adapter, &config).await?;
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        
        // Get platform capabilities
        let capabilities = Self::detect_capabilities(&adapter, &device, None)?;
        
        // Log platform information
        info!("Created headless rendering platform");
        info!("Adapter: {}", capabilities.adapter_info.name);
        info!("Backend: {:?}", capabilities.backend_type);
        info!("Platform: {:?}", capabilities.platform_type);
        
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            surface: None,
            surface_config: None,
            capabilities,
            config,
        })
    }
    
    /// Create a new platform for window rendering
    pub async fn new_with_window(window: &Window, config: PlatformConfig) -> Result<Self, PlatformError> {
        // Create instance with selected backends
        let instance = Self::create_instance(&config)?;
        
        // Create surface for window
        let surface = unsafe { instance.create_surface(window) };
        
        // Create adapter options with surface
        let mut adapter_options = config.adapter_options.clone();
        adapter_options.compatible_surface = Some(&surface);
        
        // Request adapter
        let adapter = Self::request_adapter(&instance, &config, Some(&surface)).await?;
        let adapter = Arc::new(adapter);
        
        // Create device and queue
        let (device, queue) = Self::create_device_and_queue(&adapter, &config).await?;
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        
        // Get platform capabilities
        let capabilities = Self::detect_capabilities(&adapter, &device, Some(&surface))?;
        
        // Configure surface
        let (width, height) = window.inner_size().into();
        let surface_config = Self::configure_surface(
            &surface,
            &adapter,
            width,
            height,
            &capabilities,
            &config,
        )?;
        
        // Log platform information
        info!("Created window rendering platform");
        info!("Adapter: {}", capabilities.adapter_info.name);
        info!("Backend: {:?}", capabilities.backend_type);
        info!("Platform: {:?}", capabilities.platform_type);
        info!("Surface format: {:?}", surface_config.format);
        info!("Surface size: {}x{}", surface_config.width, surface_config.height);
        
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            surface: Some(surface),
            surface_config: Some(surface_config),
            capabilities,
            config,
        })
    }
    
    /// Create WGPU instance
    fn create_instance(config: &PlatformConfig) -> Result<wgpu::Instance, PlatformError> {
        // Determine backends to use
        let backends = if let Some(preferred) = config.preferred_backend {
            preferred
        } else {
            // Use all available backends by default
            wgpu::Backends::all()
        };
        
        // Create instance with validation if enabled
        let instance_descriptor = wgpu::InstanceDescriptor {
            backends,
            dx12_shader_compiler: wgpu::Dx12Compiler::default(),
        };
        
        let instance = wgpu::Instance::new(instance_descriptor);
        
        Ok(instance)
    }
    
    /// Request adapter with appropriate options
    async fn request_adapter(
        instance: &wgpu::Instance,
        config: &PlatformConfig,
        surface: Option<&wgpu::Surface>,
    ) -> Result<wgpu::Adapter, PlatformError> {
        // Create adapter options
        let mut adapter_options = wgpu::RequestAdapterOptions {
            power_preference: config.power_preference,
            force_fallback_adapter: config.force_software,
            compatible_surface: surface,
        };
        
        // Request adapter with options
        let adapter = instance.request_adapter(&adapter_options)
            .await
            .ok_or_else(|| PlatformError::NoCompatibleAdapter)?;
        
        // Verify required features
        let adapter_features = adapter.features();
        let missing_features = config.required_features - adapter_features;
        
        if !missing_features.is_empty() {
            warn!("Adapter missing required features: {:?}", missing_features);
            
            // If force_fallback is not enabled, try again with fallback adapter
            if !config.force_software {
                warn!("Trying fallback adapter");
                adapter_options.force_fallback_adapter = true;
                
                let fallback_adapter = instance.request_adapter(&adapter_options)
                    .await
                    .ok_or_else(|| PlatformError::UnsupportedFeatures(missing_features))?;
                
                let fallback_features = fallback_adapter.features();
                let fallback_missing = config.required_features - fallback_features;
                
                if !fallback_missing.is_empty() {
                    return Err(PlatformError::UnsupportedFeatures(fallback_missing));
                }
                
                return Ok(fallback_adapter);
            }
            
            return Err(PlatformError::UnsupportedFeatures(missing_features));
        }
        
        Ok(adapter)
    }
    
    /// Create device and queue
    async fn create_device_and_queue(
        adapter: &wgpu::Adapter,
        config: &PlatformConfig,
    ) -> Result<(wgpu::Device, wgpu::Queue), PlatformError> {
        // Create device descriptor
        let mut device_descriptor = wgpu::DeviceDescriptor {
            label: config.device_options.label.clone(),
            required_features: config.required_features,
            required_limits: config.required_limits.clone(),
        };
        
        // Get device and queue
        let (device, queue) = adapter.request_device(&device_descriptor, None)
            .await
            .map_err(|e| PlatformError::DeviceCreationFailed(e.to_string()))?;
        
        Ok((device, queue))
    }
    
    /// Configure surface
    fn configure_surface(
        surface: &wgpu::Surface,
        adapter: &wgpu::Adapter,
        width: u32,
        height: u32,
        capabilities: &PlatformCapabilities,
        config: &PlatformConfig,
    ) -> Result<wgpu::SurfaceConfiguration, PlatformError> {
        // Get preferred format
        let format = capabilities.preferred_texture_format;
        
        // Determine present mode
        let present_mode = if config.enable_hdr && 
           adapter.get_texture_format_features(format).flags.contains(wgpu::TextureFormatFeatureFlags::HDR) {
            wgpu::PresentMode::Fifo
        } else {
            wgpu::PresentMode::Mailbox
        };
        
        // Create surface configuration
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: width.max(1),
            height: height.max(1),
            present_mode,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![format],
        };
        
        // Configure the surface
        surface.configure(adapter.device(), &surface_config);
        
        Ok(surface_config)
    }
    
    /// Detect platform capabilities
    fn detect_capabilities(
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        surface: Option<&wgpu::Surface>,
    ) -> Result<PlatformCapabilities, PlatformError> {
        // Get adapter info
        let adapter_info = adapter.get_info();
        
        // Get platform type
        let platform_type = PlatformType::detect();
        
        // Get adapter features
        let features = adapter.features();
        
        // Get adapter limits
        let limits = adapter.limits();
        
        // Get preferred texture format
        let preferred_texture_format = if let Some(surface) = surface {
            surface.get_capabilities(adapter).formats[0]
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };
        
        // Get supported texture formats
        let texture_formats = if let Some(surface) = surface {
            surface.get_capabilities(adapter).formats
        } else {
            // Default formats if no surface
            vec![
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::TextureFormat::Bgra8Unorm,
                wgpu::TextureFormat::Rgba8UnormSrgb,
                wgpu::TextureFormat::Bgra8UnormSrgb,
            ]
        };
        
        // Determine maximum supported sample count
        let max_sample_count = if features.contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES) {
            let mut max_count = 1;
            for &count in &[1, 2, 4, 8, 16] {
                let sample_flags = adapter.get_texture_format_features(preferred_texture_format)
                    .flags;
                
                if count > 1 && !sample_flags.contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X2) {
                    break;
                }
                
                max_count = count;
            }
            max_count
        } else {
            // Conservative default
            4
        };
        
        // Determine if discrete GPU
        let has_discrete_gpu = matches!(
            adapter_info.device_type,
            wgpu::DeviceType::DiscreteGpu
        );
        
        Ok(PlatformCapabilities {
            max_texture_dimension: limits.max_texture_dimension_2d,
            max_compute_workgroup_size: limits.max_compute_workgroup_size,
            max_uniform_buffer_binding_size: limits.max_uniform_buffer_binding_size,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            features,
            texture_formats,
            adapter_info,
            preferred_texture_format,
            platform_type,
            backend_type: adapter_info.backend.into(),
            supported_sample_count: max_sample_count,
            has_discrete_gpu,
        })
    }
    
    /// Get platform capabilities
    pub fn capabilities(&self) -> &PlatformCapabilities {
        &self.capabilities
    }
    
    /// Get device reference
    pub fn device(&self) -> Arc<wgpu::Device> {
        self.device.clone()
    }
    
    /// Get queue reference
    pub fn queue(&self) -> Arc<wgpu::Queue> {
        self.queue.clone()
    }
    
    /// Get adapter reference
    pub fn adapter(&self) -> Arc<wgpu::Adapter> {
        self.adapter.clone()
    }
    
    /// Check if platform has a surface
    pub fn has_surface(&self) -> bool {
        self.surface.is_some()
    }
    
    /// Resize surface (if available)
    pub fn resize_surface(&mut self, width: u32, height: u32) -> Result<(), PlatformError> {
        if let (Some(surface), Some(mut config)) = (self.surface.as_ref(), self.surface_config.clone()) {
            // Update configuration
            config.width = width.max(1);
            config.height = height.max(1);
            
            // Configure surface
            surface.configure(&self.device, &config);
            
            // Update stored configuration
            self.surface_config = Some(config);
            
            Ok(())
        } else {
            Err(PlatformError::NoSurface)
        }
    }
    
    /// Get current surface texture (if available)
    pub fn current_surface_texture(&self) -> Result<wgpu::SurfaceTexture, PlatformError> {
        if let Some(surface) = &self.surface {
            surface.get_current_texture()
                .map_err(|_| PlatformError::SurfaceLost)
        } else {
            Err(PlatformError::NoSurface)
        }
    }
    
    /// Create texture view from surface texture
    pub fn create_view(&self, texture: &wgpu::SurfaceTexture) -> wgpu::TextureView {
        texture.texture.create_view(&wgpu::TextureViewDescriptor::default())
    }
    
    /// Check if platform supports a feature
    pub fn supports_feature(&self, feature: wgpu::Features) -> bool {
        self.capabilities.features.contains(feature)
    }
    
    /// Create a render pipeline with appropriate features
    pub fn create_render_pipeline(
        &self,
        layout: &wgpu::PipelineLayout,
        vertex_shader: &wgpu::ShaderModule,
        fragment_shader: &wgpu::ShaderModule,
        vertex_buffers: &[wgpu::VertexBufferLayout],
        color_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
        sample_count: u32,
    ) -> wgpu::RenderPipeline {
        // Create render pipeline descriptor
        let pipeline_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Chronos Render Pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: vertex_shader,
                entry_point: "main",
                buffers: vertex_buffers,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
                format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: fragment_shader,
                entry_point: "main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
        };
        
        // Create render pipeline
        self.device.create_render_pipeline(&pipeline_descriptor)
    }
}

/// Platform-specific error types
#[derive(Debug, Error)]
pub enum PlatformError {
    #[error("No compatible graphics adapter found")]
    NoCompatibleAdapter,
    
    #[error("Failed to create device: {0}")]
    DeviceCreationFailed(String),
    
    #[error("Failed to create surface: {0}")]
    SurfaceCreationFailed(String),
    
    #[error("No surface available for window rendering")]
    NoSurface,
    
    #[error("Required features not supported: {0:?}")]
    UnsupportedFeatures(wgpu::Features),
    
    #[error("Required limits not supported")]
    UnsupportedLimits,
    
    #[error("Surface lost or invalid")]
    SurfaceLost,
    
    #[error("Other platform error: {0}")]
    Other(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_platform_type_detection() {
        // Test platform type detection
        let platform_type = PlatformType::detect();
        
        // Platform type should be detected correctly
        #[cfg(target_os = "windows")]
        assert_eq!(platform_type, PlatformType::Windows);
        
        #[cfg(target_os = "macos")]
        assert_eq!(platform_type, PlatformType::MacOS);
        
        #[cfg(target_os = "linux")]
        assert_eq!(platform_type, PlatformType::Linux);
        
        #[cfg(target_arch = "wasm32")]
        assert_eq!(platform_type, PlatformType::Web);
        
        // Test platform type checks
        if cfg!(target_os = "windows") || cfg!(target_os = "macos") || cfg!(target_os = "linux") {
            assert!(platform_type.is_desktop());
            assert!(!platform_type.is_mobile());
            assert!(!platform_type.is_web());
        } else if cfg!(target_os = "android") || cfg!(target_os = "ios") {
            assert!(!platform_type.is_desktop());
            assert!(platform_type.is_mobile());
            assert!(!platform_type.is_web());
        } else if cfg!(target_arch = "wasm32") {
            assert!(!platform_type.is_desktop());
            assert!(!platform_type.is_mobile());
            assert!(platform_type.is_web());
        }
    }
    
    #[test]
    fn test_platform_config_defaults() {
        // Create default platform config
        let config = PlatformConfig::default();
        
        // Verify default values
        assert_eq!(config.power_preference, wgpu::PowerPreference::HighPerformance);
        assert!(!config.force_software);
        assert_eq!(config.enable_validation, cfg!(debug_assertions));
        assert!(!config.enable_hdr);
        assert_eq!(config.sample_count, 1);
    }
}