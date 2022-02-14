use std::error::Error;
use std::ffi::CString;

use ash::Device;
use ash::vk::{Extent2D, RenderPass, PipelineVertexInputStateCreateInfo, PipelineLayoutCreateInfo, Pipeline, PipelineLayout, ShaderModuleCreateInfo, PipelineShaderStageCreateInfo, ShaderStageFlags, PipelineInputAssemblyStateCreateInfo, PrimitiveTopology, Viewport, Rect2D, Offset2D, PipelineViewportStateCreateInfo, PipelineRasterizationStateCreateInfo, FrontFace, CullModeFlags, PolygonMode, PipelineMultisampleStateCreateInfo, SampleCountFlags, PipelineColorBlendAttachmentState, BlendFactor, BlendOp, ColorComponentFlags, PipelineColorBlendStateCreateInfo, GraphicsPipelineCreateInfo, PipelineCache, PipelineDepthStencilStateCreateInfo, CompareOp};
use vk_shader_macros::include_glsl;

pub struct VulkanPipeline {

}

/* What goes into a pipeline?
 * shaders stages, including a source file and entry name
 * the topology mode
 * rasterization state
 * 
 */
impl VulkanPipeline {
    pub fn new(device: &Device, extent: Extent2D, render_pass: &RenderPass, msaa_count: SampleCountFlags,
        vertex_input_state: PipelineVertexInputStateCreateInfo,
        pipeline_layout_state: PipelineLayoutCreateInfo) -> Result<(Pipeline, PipelineLayout), Box<dyn Error>> {
        let vertex_module = {
            let create_info = ShaderModuleCreateInfo::builder()
                .code(include_glsl!("./shaders/shader.vert"));
            unsafe { device.create_shader_module(&create_info, None)? }
        };
        let fragment_module = {
            let create_info = ShaderModuleCreateInfo::builder()
                .code(include_glsl!("./shaders/shader.frag"));
            unsafe { device.create_shader_module(&create_info, None)? }
        };
        let shader_entry = CString::new("main")?;
        let vertex_stage = PipelineShaderStageCreateInfo::builder()
            .stage(ShaderStageFlags::VERTEX)
            .module(vertex_module)
            .name(&shader_entry);
        let fragment_stage = PipelineShaderStageCreateInfo::builder()
            .stage(ShaderStageFlags::FRAGMENT)
            .module(fragment_module)
            .name(&shader_entry);
        let input_assembly_state = PipelineInputAssemblyStateCreateInfo::builder()
            .topology(PrimitiveTopology::TRIANGLE_LIST);
        let viewports = [ Viewport {
                x: 0.0,
                y: 0.0,
                width: extent.width as f32,
                height: extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0
        } ];
        let scissors = [ Rect2D {
            offset: Offset2D { x: 0, y: 0 },
            extent
        } ];
        let viewport_state = PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors);
        let rasterization_state = PipelineRasterizationStateCreateInfo::builder()
            .line_width(1.0)
            .front_face(FrontFace::CLOCKWISE)
            .cull_mode(CullModeFlags::BACK)
            .polygon_mode(PolygonMode::FILL);
        let multisample_state = PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(msaa_count);
        let color_blend_attachments = [ PipelineColorBlendAttachmentState::builder()
            .blend_enable(true)
            .src_color_blend_factor(BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(BlendOp::ADD)
            .src_alpha_blend_factor(BlendFactor::SRC_ALPHA)
            .dst_alpha_blend_factor(BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(BlendOp::ADD)
            .color_write_mask(ColorComponentFlags::RGBA)
            .build()
        ];
        let color_blend_state = PipelineColorBlendStateCreateInfo::builder()
            .attachments(&color_blend_attachments);
        let depth_stencil_state = PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);
        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_state, None)? };
        let graphics_pipeline = {
            let create_infos = [ GraphicsPipelineCreateInfo::builder()
                .stages(&[*vertex_stage, *fragment_stage])
                .vertex_input_state(&vertex_input_state)
                .input_assembly_state(&input_assembly_state)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterization_state)
                .multisample_state(&multisample_state)
                .color_blend_state(&color_blend_state)
                .depth_stencil_state(&depth_stencil_state)
                .layout(pipeline_layout)
                .render_pass(*render_pass)
                .subpass(0)
                .build()
            ];
            unsafe { device.create_graphics_pipelines(PipelineCache::null(), &create_infos, None).map_err(|t| t.1)? }
        }[0];

        // Clean up shader modules
        unsafe {
            device.destroy_shader_module(vertex_module, None);
            device.destroy_shader_module(fragment_module, None);
        }
        
        Ok ((graphics_pipeline, pipeline_layout))
    }
}
