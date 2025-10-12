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

class DescriptorAllocatorGrowable {
 public:
  struct PoolSizeRatio {
    VkDescriptorType type;
    float ratio;
  };

  DescriptorAllocatorGrowable() = default;
  void init(VkDevice device, std::uint32_t initialSets,
            std::span<const PoolSizeRatio> poolSizeRatios);
  void clear_pools(VkDevice device);
  void destroy_pools(VkDevice device);

  [[nodiscard]]
  VkDescriptorSet allocate(VkDevice device, VkDescriptorSetLayout layout,
                           void* pNext = nullptr);

 private:
  VkDescriptorPool get_pool(VkDevice device);
  static VkDescriptorPool create_pool(VkDevice device, std::uint32_t setCount,
                                      std::span<const PoolSizeRatio> poolSizeRatios);

  std::vector<PoolSizeRatio> m_poolSizeRatios;
  std::vector<VkDescriptorPool> m_fullPools;
  std::vector<VkDescriptorPool> m_readyPools;
  std::uint32_t m_setsPerPool{};
};

struct DescriptorWriter {
  std::deque<VkDescriptorImageInfo> descriptorImageInfos;
  std::deque<VkDescriptorBufferInfo> descriptorBufferInfos;
  std::vector<VkWriteDescriptorSet> writes;

  void write_image(int binding, VkImageView image, VkSampler sampler,
                   VkImageLayout layout, VkDescriptorType descriptorType);
  void write_buffer(int binding, VkBuffer buffer, std::size_t size,
                    std::size_t offset, VkDescriptorType descriptorType);

  void clear();

  void update_set(VkDevice device, VkDescriptorSet descriptorSet);
};

}  // namespace mp