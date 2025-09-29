#include "mpr_pipelines.hpp"

#include <fstream>

#include "mpr_error_check.hpp"
#include "mpr_init_vk_stucts.hpp"

namespace mp {
bool load_shader_module(const char* filePath, VkDevice device,
                        VkShaderModule* outShaderModule) {
  std::ifstream file(filePath, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    return false;
  }

  const std::size_t fileSize = static_cast<std::size_t>(file.tellg());

  std::vector<std::uint32_t> buffer(fileSize / sizeof(std::uint32_t));

  file.seekg(0);

  file.read(reinterpret_cast<char*>(buffer.data()), fileSize);

  if (!file.good()) return false;

  file.close();

  const VkShaderModuleCreateInfo createInfo{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .codeSize = buffer.size() * sizeof(std::uint32_t),
      .pCode = buffer.data(),
  };

  VkShaderModule shaderModule;
  if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) !=
      VK_SUCCESS) {
    return false;
  }
  *outShaderModule = shaderModule;
  return true;
}

void PipelineBuilder::clear() {
  inputAssembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};

  rasterizer = {.sType =
                    VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};

  colorBlend = {};

  multisampling = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};

  pipelineLayout = {};

  depthStencil = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};

  renderInfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};

  shaderStages.clear();
}

VkPipeline PipelineBuilder::build_pipeline(const VkDevice device) {
  constexpr VkPipelineViewportStateCreateInfo viewportState{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .pNext = nullptr,
      .viewportCount = 1,
      .scissorCount = 1,
  };

  const VkPipelineColorBlendStateCreateInfo colorBlendState{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .pNext = nullptr,
      .logicOpEnable = VK_FALSE,
      .logicOp = VK_LOGIC_OP_COPY,
      .attachmentCount = 1,
      .pAttachments = &colorBlend,
  };

  constexpr VkPipelineVertexInputStateCreateInfo vertexInputInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
  };

  constexpr VkDynamicState dynamicState[]{VK_DYNAMIC_STATE_VIEWPORT,
                                          VK_DYNAMIC_STATE_SCISSOR};

  const VkPipelineDynamicStateCreateInfo dynamicStateCreateInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = std::size(dynamicState),
      .pDynamicStates = dynamicState,
  };
  const VkGraphicsPipelineCreateInfo createInfo{
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &renderInfo,
      .stageCount = static_cast<std::uint32_t>(shaderStages.size()),
      .pStages = shaderStages.data(),
      .pVertexInputState = &vertexInputInfo,
      .pInputAssemblyState = &inputAssembly,
      .pViewportState = &viewportState,
      .pRasterizationState = &rasterizer,
      .pMultisampleState = &multisampling,
      .pDepthStencilState = &depthStencil,
      .pColorBlendState = &colorBlendState,
      .pDynamicState = &dynamicStateCreateInfo,
      .layout = pipelineLayout,

  };

  assert(shaderStages.size() >= 2);
  VkPipeline pipeline{};
  vkCreateGraphicsPipelines(device, nullptr, 1, &createInfo, nullptr,
                            &pipeline) >>
      chk;

  return pipeline;
}

void PipelineBuilder::add_shader(const VkShaderModule shaderModule,
                                 const VkShaderStageFlagBits shaderType,
                                 const char* entryPoint) {
  shaderStages.emplace_back(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                            nullptr, 0, shaderType, shaderModule, entryPoint,
                            nullptr);
}

void PipelineBuilder::set_input_topology(const VkPrimitiveTopology topology) {
  inputAssembly.topology = topology;
  inputAssembly.primitiveRestartEnable = VK_FALSE;
}

void PipelineBuilder::set_polygon_mode(VkPolygonMode polygonMode) {
  rasterizer.polygonMode = polygonMode;
  rasterizer.lineWidth = 1.0f;
}

void PipelineBuilder::set_cull_mode(const VkCullModeFlags cullMode,
                                    const VkFrontFace frontFace) {
  rasterizer.cullMode = cullMode;
  rasterizer.frontFace = frontFace;
}

void PipelineBuilder::set_multisampling_none() {
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  multisampling.minSampleShading = 1.0f;
  multisampling.pSampleMask = nullptr;
  multisampling.alphaToCoverageEnable = VK_FALSE;
  multisampling.alphaToOneEnable = VK_FALSE;
}


void PipelineBuilder::set_color_attachment_format(const VkFormat format) {
  colorAttachmentFormat = format;

  renderInfo.colorAttachmentCount = 1;
  renderInfo.pColorAttachmentFormats = &colorAttachmentFormat;
}

void PipelineBuilder::set_depth_format(const VkFormat format) {
  renderInfo.depthAttachmentFormat = format;
}

void PipelineBuilder::disable_depth_test() {
  depthStencil.depthTestEnable = VK_FALSE;
  depthStencil.depthWriteEnable = VK_FALSE;
  depthStencil.depthCompareOp = VK_COMPARE_OP_NEVER;
  depthStencil.depthBoundsTestEnable = VK_FALSE;
  depthStencil.stencilTestEnable = VK_FALSE;
  depthStencil.front = {};
  depthStencil.back = {};
  depthStencil.minDepthBounds = 0.f;
  depthStencil.maxDepthBounds = 1.f;
}

void PipelineBuilder::enable_depth_test(const bool enableDepthWrite, const VkCompareOp depthCompareOp) {
  depthStencil.depthTestEnable = VK_TRUE;
  depthStencil.depthWriteEnable = enableDepthWrite;
  depthStencil.depthCompareOp = depthCompareOp;
  depthStencil.depthBoundsTestEnable = VK_FALSE;
  depthStencil.stencilTestEnable = VK_FALSE;
  depthStencil.front = {};
  depthStencil.back = {};
  depthStencil.minDepthBounds = 0.f;
  depthStencil.maxDepthBounds = 1.f;
}

void PipelineBuilder::disable_blending() {
  colorBlend.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  colorBlend.blendEnable = VK_FALSE;
}

void PipelineBuilder::enable_blending_additive() {
  colorBlend.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  colorBlend.blendEnable = VK_TRUE;
  colorBlend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  colorBlend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
  colorBlend.colorBlendOp = VK_BLEND_OP_ADD;
  colorBlend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  colorBlend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  colorBlend.alphaBlendOp = VK_BLEND_OP_ADD;
}

void PipelineBuilder::enable_blending_alpha() {
  colorBlend.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  colorBlend.blendEnable = VK_TRUE;
  colorBlend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  colorBlend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  colorBlend.colorBlendOp = VK_BLEND_OP_ADD;
  colorBlend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  colorBlend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  colorBlend.alphaBlendOp = VK_BLEND_OP_ADD;
}
}  // namespace mp