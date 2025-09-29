#include "mpr_descriptors.hpp"

#include "mpr_error_check.hpp"

namespace mp {
void DescriptorLayoutBuilder::add_binding(const std::uint32_t binding,
                                          VkDescriptorType type) {
  bindings.emplace_back(binding, type, 1, 0, nullptr);
}

VkDescriptorSetLayout DescriptorLayoutBuilder::build(
    VkDevice device, VkShaderStageFlags shaderStages, void* pNext,
    VkDescriptorSetLayoutCreateFlags flags) {
  for (auto& binding : bindings) {
    binding.stageFlags |= shaderStages;
  }

  const VkDescriptorSetLayoutCreateInfo createInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .pNext = pNext,
      .flags = flags,
      .bindingCount = static_cast<std::uint32_t>(bindings.size()),
      .pBindings = bindings.data(),
  };

  VkDescriptorSetLayout layout{};
  vkCreateDescriptorSetLayout(device, &createInfo, nullptr, &layout) >> chk;

  return layout;
}

void DescriptorAllocator::init_pool(VkDevice device, std::uint32_t maxSets,
                                    std::span<PoolSizeRatio> poolRatios) {
  std::vector<VkDescriptorPoolSize> poolSizes;
  for (const auto& ratio : poolRatios) {
    poolSizes.emplace_back(ratio.type,
                           static_cast<std::uint32_t>(ratio.ratio * maxSets));
  }

  const VkDescriptorPoolCreateInfo createInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .maxSets = maxSets,
      .poolSizeCount = static_cast<std::uint32_t>(poolSizes.size()),
      .pPoolSizes = poolSizes.data(),
  };
  vkCreateDescriptorPool(device, &createInfo, nullptr, &pool) >> chk;
}

void DescriptorAllocator::clear_descriptors(VkDevice device) {
  vkResetDescriptorPool(device, pool, 0) >> chk;
}

void DescriptorAllocator::destroy_pool(VkDevice device) {
  vkDestroyDescriptorPool(device, pool, nullptr);
}

VkDescriptorSet DescriptorAllocator::allocate(VkDevice device,
                                              VkDescriptorSetLayout layout) {
  const VkDescriptorSetAllocateInfo allocateInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .pNext = nullptr,
      .descriptorPool = pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &layout,
  };

  VkDescriptorSet descSet;
  vkAllocateDescriptorSets(device, &allocateInfo, &descSet) >> chk;
  return descSet;
}
}  // namespace mp