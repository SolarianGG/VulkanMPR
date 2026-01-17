#pragma once

#include "mpr_types.hpp"

namespace mp {
[[nodiscard]] bool load_shader_module(const char* filePath, VkDevice device,
                                      VkShaderModule* outShaderModule);

struct PipelineBuilder {
  std::vector<VkPipelineShaderStageCreateInfo> shaderStages;

  VkPipelineInputAssemblyStateCreateInfo inputAssembly;
  VkPipelineRasterizationStateCreateInfo rasterizer;
  VkPipelineMultisampleStateCreateInfo multisampling;
  VkPipelineLayout pipelineLayout;
  VkPipelineDepthStencilStateCreateInfo depthStencil;
  VkPipelineRenderingCreateInfo renderInfo;

  std::vector<VkPipelineColorBlendAttachmentState> colorBlends;
  std::vector<
  VkFormat> colorAttachmentFormats;

  PipelineBuilder() { clear(); }

  void clear();

  VkPipeline build_pipeline(VkDevice device, VkPipelineCreateFlags pipelineCreateFlags = 0);

  void add_shader(const VkShaderModule shaderModule,
                  const VkShaderStageFlagBits shaderType,
                  const char* entryPoint = "main");

  void clear_shaders() { shaderStages.clear(); }

  void set_input_topology(VkPrimitiveTopology topology);

  void set_polygon_mode(VkPolygonMode polygonMode);

  void set_cull_mode(VkCullModeFlags cullMode, VkFrontFace frontFace);

  void set_multisampling_none();

  void add_color_attachment_format(VkFormat format);

  void set_depth_format(VkFormat format);

  void disable_depth_test();
  void enable_depth_test(const bool enableDepthWrite,
                         const VkCompareOp depthCompareOp);

};

}  // namespace mp