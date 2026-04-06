// clang-format off
#include "mpr_materials.hpp"

#include <stdexcept>

#include "mpr_pipelines.hpp"
#include "mpr_engine.hpp"

// clang-format on

namespace mp {
void GLTFMetallicRoughness::build_pipelines(Engine& engine) {
  VkShaderModule meshVertShader;
  if (!load_shader_module("../../src/compiled_shaders/mesh.vertex.spv",
                          engine.m_device, &meshVertShader)) {
    throw std::runtime_error("Failed to load mesh.vertex.spv");
  }
  VkShaderModule meshFragShader;
  if (!load_shader_module("../../src/compiled_shaders/mesh.pixel.spv",
                          engine.m_device, &meshFragShader)) {
    throw std::runtime_error("Failed to load mesh.pixel.spv");
  }

  VkShaderModule transparentVertexShader;
  if (!load_shader_module("../../src/compiled_shaders/wboit_forward.vertex.spv",
                          engine.m_device, &transparentVertexShader)) {
    throw std::runtime_error("Failed to load mesh.vertex.spv");
  }
  VkShaderModule transparentFragmentShader;
  if (!load_shader_module("../../src/compiled_shaders/wboit_forward.pixel.spv",
                          engine.m_device, &transparentFragmentShader)) {
    throw std::runtime_error("Failed to load mesh.pixel.spv");
  }

  {
    materialLayout =
        DescriptorSetLayoutBuilder()
            .add_binding(
                0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10000,
                VK_SHADER_STAGE_FRAGMENT_BIT)
            .add_binding(1, VK_DESCRIPTOR_TYPE_SAMPLER, 10000,
                         VK_SHADER_STAGE_FRAGMENT_BIT)
            .add_binding(2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 10000,
                         VK_SHADER_STAGE_FRAGMENT_BIT)
            .build(engine.m_device,
                   VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT);
  }
  descriptors =
      DescriptorBuffer(engine.m_device, materialLayout,
                       DescriptorBufferProperties::query(engine.m_chosenGpu));
  descriptors.create_buffer([&engine](const std::size_t allocSize,
                                      const VkBufferUsageFlags bufferUsage) {
    return engine.create_buffer(allocSize, bufferUsage,
                                VMA_MEMORY_USAGE_CPU_ONLY);
  });
  const VkDescriptorSetLayout layouts[]{materialLayout};

  {
    const VkPushConstantRange pushConstantRange{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0,
        .size = static_cast<std::uint32_t>(sizeof(GBufferPassPushConstants)),
    };
    const VkPipelineLayoutCreateInfo layoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = std::size(layouts),
        .pSetLayouts = layouts,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstantRange,

    };
    VkPipelineLayout layout;
    vkCreatePipelineLayout(engine.m_device, &layoutCreateInfo, nullptr,
                           &layout) >>
        chk;
    opaquePipeline.pipelineLayout = layout;
    alphaTestedPipeline.pipelineLayout = layout;
  }

  {
    const VkPushConstantRange pushConstantRange{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = 0,
        .size = static_cast<std::uint32_t>(sizeof(OITForwardPassPushConstants)),
    };
    const VkPipelineLayoutCreateInfo layoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = std::size(layouts),
        .pSetLayouts = layouts,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstantRange,

    };
    VkPipelineLayout layout;
    vkCreatePipelineLayout(engine.m_device, &layoutCreateInfo, nullptr,
                           &layout) >>
        chk;
    transparentPipeline.pipelineLayout = layout;
  }

  {
    PipelineBuilder pipelineBuilder;
    pipelineBuilder.pipelineLayout = opaquePipeline.pipelineLayout;
    pipelineBuilder.add_shader(meshVertShader, VK_SHADER_STAGE_VERTEX_BIT);
    pipelineBuilder.add_shader(meshFragShader, VK_SHADER_STAGE_FRAGMENT_BIT);
    pipelineBuilder.enable_depth_test(false, VK_COMPARE_OP_EQUAL);
    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.add_color_attachment_format(
        engine.m_frameData.at(0).gBuffer.normal.imageFormat);
    pipelineBuilder.add_color_attachment_format(
        engine.m_frameData.at(0).gBuffer.diffuse.imageFormat);
    pipelineBuilder.add_color_attachment_format(
        engine.m_frameData.at(0).gBuffer.specular.imageFormat);
    pipelineBuilder.set_depth_format(
        engine.m_frameData.at(0).depthImage.imageFormat);
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_BACK_BIT,
                                  VK_FRONT_FACE_COUNTER_CLOCKWISE);

    pipelineBuilder.set_multisampling_none();
    pipelineBuilder.colorBlends.insert(
        pipelineBuilder.colorBlends.begin(),
        pipelineBuilder.colorAttachmentFormats.size(),
        {
            .blendEnable = VK_FALSE,
            .colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        });
    opaquePipeline.pipeline = pipelineBuilder.build_pipeline(
        engine.m_device, VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT);
    alphaTestedPipeline.pipeline = pipelineBuilder.build_pipeline(
        engine.m_device, VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT);
  }

  {
    PipelineBuilder pipelineBuilder;
    pipelineBuilder.pipelineLayout = transparentPipeline.pipelineLayout;
    pipelineBuilder.add_shader(transparentVertexShader,
                               VK_SHADER_STAGE_VERTEX_BIT);
    pipelineBuilder.add_shader(transparentFragmentShader,
                               VK_SHADER_STAGE_FRAGMENT_BIT);
    pipelineBuilder.enable_depth_test(false, VK_COMPARE_OP_LESS_OR_EQUAL);
    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.add_color_attachment_format(
        engine.m_frameData.at(0).oitAccImage.imageFormat);
    pipelineBuilder.add_color_attachment_format(
        engine.m_frameData.at(0).oitRevealImage.imageFormat);
    pipelineBuilder.set_depth_format(
        engine.m_frameData.at(0).depthImage.imageFormat);
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_BACK_BIT,
                                  VK_FRONT_FACE_COUNTER_CLOCKWISE);
    pipelineBuilder.set_multisampling_none();
    pipelineBuilder.colorBlends.push_back(
        {.blendEnable = VK_TRUE,
         .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
         .dstColorBlendFactor = VK_BLEND_FACTOR_ONE,
         .colorBlendOp = VK_BLEND_OP_ADD,
         .srcAlphaBlendFactor =
             VK_BLEND_FACTOR_ONE,  // consistent with color (premultiplied)
         .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
         .alphaBlendOp = VK_BLEND_OP_ADD,
         .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT |
                           VK_COLOR_COMPONENT_A_BIT});

    pipelineBuilder.colorBlends.push_back({
        .blendEnable = VK_TRUE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_ZERO,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor =
            VK_BLEND_FACTOR_ZERO,  // revealage is R8, but Vulkan still requires
                                   // alpha setup
        .dstAlphaBlendFactor =
            VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,  // or ONE_MINUS_SRC_COLOR if
                                                  // treating alpha as color;
                                                  // but format is R8
        .alphaBlendOp = VK_BLEND_OP_ADD,
        .colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT  // only red channel used in R8 format
    });

    transparentPipeline.pipeline = pipelineBuilder.build_pipeline(
        engine.m_device, VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT);
  }
  vkDestroyShaderModule(engine.m_device, meshVertShader, nullptr);
  vkDestroyShaderModule(engine.m_device, meshFragShader, nullptr);
  vkDestroyShaderModule(engine.m_device, transparentVertexShader, nullptr);
  vkDestroyShaderModule(engine.m_device, transparentFragmentShader, nullptr);
}

void GLTFMetallicRoughness::clear_resources(Engine& engine) {
  vkDestroyPipeline(engine.m_device, opaquePipeline.pipeline, nullptr);
  vkDestroyPipeline(engine.m_device, alphaTestedPipeline.pipeline, nullptr);
  vkDestroyPipeline(engine.m_device, transparentPipeline.pipeline, nullptr);
  vkDestroyPipelineLayout(engine.m_device, transparentPipeline.pipelineLayout,
                          nullptr);
  // alphaTestedPipeline shares layout with opaquePipeline — destroy once
  vkDestroyPipelineLayout(engine.m_device, opaquePipeline.pipelineLayout,
                          nullptr);
  vkDestroyDescriptorSetLayout(engine.m_device, materialLayout, nullptr);
  engine.destroy_buffer(descriptors.get_buffer());
}

std::uint32_t GLTFMetallicRoughness::write_uniform_buffer(
    const VkDeviceAddress uniformBuffer) {
  descriptors.write_uniform_buffer(0, currentMaterialOffset, uniformBuffer,
                                   sizeof(MaterialConstants));
  return currentMaterialOffset++;
}

std::uint32_t GLTFMetallicRoughness::write_sampler(const VkSampler sampler) {
  descriptors.write_sampler(1, currentSamplerOffset, sampler);
  return currentSamplerOffset++;
}

std::uint32_t GLTFMetallicRoughness::write_texture(
    const VkImageView imageView) {
  descriptors.write_sampled_image(2, currentTextureOffset, imageView,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  return currentTextureOffset++;
}

MaterialPipeline* GLTFMetallicRoughness::select_pipeline(
    const MaterialPass pass) {
  if (pass == MaterialPass::Opaque) {
    return &opaquePipeline;
  }
  if (pass == MaterialPass::AlphaTested) {
    return &alphaTestedPipeline;
  }
  if (pass == MaterialPass::Transparent) {
    return &transparentPipeline;
  }
  return nullptr;
}

}  // namespace mp