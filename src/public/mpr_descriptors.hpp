#pragma once

#include "mpr_types.hpp"

namespace mp {
struct DescriptorLayoutBuilder {
  std::vector<VkDescriptorSetLayoutBinding> bindings;

  void add_binding(const std::uint32_t binding, VkDescriptorType type);
  void clear() { bindings.clear(); }
  [[nodiscard]]
  VkDescriptorSetLayout build(VkDevice device, VkShaderStageFlags shaderStages,
                              void* pNext = nullptr,
                              VkDescriptorSetLayoutCreateFlags flags = 0);
};

struct DescriptorAllocator {
  struct PoolSizeRatio {
    VkDescriptorType type;
    float ratio;
  };

  VkDescriptorPool pool;

  void init_pool(VkDevice device, std::uint32_t maxSets,
                 std::span<PoolSizeRatio> poolRatios);
  void clear_descriptors(VkDevice device);
  void destroy_pool(VkDevice device);

  [[nodiscard]]
  VkDescriptorSet allocate(VkDevice device, VkDescriptorSetLayout layout);
};

}  // namespace mp