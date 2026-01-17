#pragma once

#include "mpr_types.hpp"
#include "mpr_descriptors.hpp"

namespace mp {

class Engine;


struct GLTFMetallicRoughness {
  MaterialPipeline opaquePipeline;
  MaterialPipeline transparentPipeline;

  VkDescriptorSetLayout materialLayout;
  DescriptorBuffer descriptors;
  std::uint32_t currentMaterialOffset = 0;
  std::uint32_t currentSamplerOffset = 0;
  std::uint32_t currentTextureOffset = 0;

  struct MaterialConstants {
    glm::vec4 colorFactors;
    glm::vec4 metalRoughFactors;
    float alphaCutoff;
    float padding[3];
  };

  void build_pipelines(Engine& engine);
  void clear_resources(Engine& engine);

  std::uint32_t write_uniform_buffer(VkDeviceAddress uniformBuffer);
  std::uint32_t write_sampler(VkSampler sampler);
  std::uint32_t write_texture(VkImageView imageView);
  MaterialPipeline* select_pipeline(const MaterialPass pass);
};


}