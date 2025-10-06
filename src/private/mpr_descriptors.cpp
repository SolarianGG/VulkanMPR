#include "mpr_descriptors.hpp"

#include "mpr_error_check.hpp"

namespace mp {
void DescriptorLayoutBuilder::add_binding(const std::uint32_t binding,
                                          VkDescriptorType type) {
  bindings.emplace_back(binding, type, 1, 0, nullptr);
}

VkDescriptorSetLayout DescriptorLayoutBuilder::build(
    const VkDevice device, const VkShaderStageFlags shaderStages, void* pNext,
    const VkDescriptorSetLayoutCreateFlags flags) {
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

void DescriptorAllocatorGrowable::init(
    const VkDevice device, const std::uint32_t initialSets,
    std::span<const PoolSizeRatio> poolSizeRatios) {
  m_poolSizeRatios.clear();
  m_poolSizeRatios.append_range(poolSizeRatios);

  const VkDescriptorPool newPool =
      create_pool(device, initialSets, poolSizeRatios);

  m_setsPerPool = static_cast<std::uint32_t>(initialSets * 1.5);

  m_readyPools.push_back(newPool);
}

void DescriptorAllocatorGrowable::clear_pools(const VkDevice device) {
  for (const auto p : m_readyPools) {
    vkResetDescriptorPool(device, p, 0) >> chk;
  }

  for (const auto p : m_fullPools) {
    vkResetDescriptorPool(device, p, 0) >> chk;
    m_readyPools.push_back(p);
  }

  m_fullPools.clear();
}

void DescriptorAllocatorGrowable::destroy_pools(const VkDevice device) {
  for (const auto p : m_readyPools) {
    vkDestroyDescriptorPool(device, p, nullptr);
  }
  m_readyPools.clear();
  for (const auto p : m_fullPools) {
    vkDestroyDescriptorPool(device, p, nullptr);
  }
  m_fullPools.clear();
}

VkDescriptorSet DescriptorAllocatorGrowable::allocate(
    const VkDevice device, VkDescriptorSetLayout layout, void* pNext) {
  VkDescriptorPool pool = get_pool(device);

  VkDescriptorSetAllocateInfo allocateInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .pNext = pNext,
      .descriptorPool = pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &layout,
  };
  VkDescriptorSet descSet;
  const VkResult res =
      vkAllocateDescriptorSets(device, &allocateInfo, &descSet);
  if (res == VK_ERROR_OUT_OF_POOL_MEMORY || res == VK_ERROR_FRAGMENTED_POOL) {
    m_fullPools.push_back(pool);

    pool = get_pool(device);
    allocateInfo.descriptorPool = pool;
    vkAllocateDescriptorSets(device, &allocateInfo, &descSet) >> chk;
  }

  m_readyPools.push_back(pool);
  return descSet;
}

VkDescriptorPool DescriptorAllocatorGrowable::get_pool(const VkDevice device) {
  VkDescriptorPool newPool;
  if (!m_readyPools.empty()) {
    newPool = m_readyPools.back();
    m_readyPools.pop_back();
  } else {
    newPool = create_pool(device, m_setsPerPool, m_poolSizeRatios);

    m_setsPerPool = std::min<uint32_t>(
        static_cast<std::uint32_t>(m_setsPerPool * 1.5), 4092);
  }
  return newPool;
}

VkDescriptorPool DescriptorAllocatorGrowable::create_pool(
    const VkDevice device, const std::uint32_t setCount,
    std::span<const PoolSizeRatio> poolSizeRatios) {
  std::vector<VkDescriptorPoolSize> poolSizes;
  for (const auto& [type, ratio] : poolSizeRatios) {
    poolSizes.emplace_back(type, static_cast<std::uint32_t>(ratio * setCount));
  }

  const VkDescriptorPoolCreateInfo poolCreateInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .maxSets = setCount,
      .poolSizeCount = static_cast<std::uint32_t>(poolSizes.size()),
      .pPoolSizes = poolSizes.data(),
  };
  VkDescriptorPool newDescriptorPool;
  vkCreateDescriptorPool(device, &poolCreateInfo, nullptr,
                         &newDescriptorPool) >>
      chk;
  return newDescriptorPool;
}

void DescriptorWriter::write_image(int binding, VkImageView image,
                                   VkSampler sampler, VkImageLayout layout,
                                   VkDescriptorType descriptorType) {
  auto& imageInfo = descriptorImageInfos.emplace_back(sampler, image, layout);

  writes.emplace_back(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, nullptr,
                      binding, 0, 1, descriptorType, &imageInfo, nullptr,
                      nullptr);
}

void DescriptorWriter::write_buffer(int binding, VkBuffer buffer,
                                    std::size_t size, std::size_t offset,
                                    VkDescriptorType descriptorType) {
  assert(descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
         descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
         descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
         descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC);
  auto& bufferInfo = descriptorBufferInfos.emplace_back(buffer, offset, size);
  writes.emplace_back(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, nullptr,
                      binding, 0, 1, descriptorType, nullptr, &bufferInfo,
                      nullptr);
}
void DescriptorWriter::clear() {
  descriptorImageInfos.clear();
  descriptorBufferInfos.clear();
  writes.clear();
}

void DescriptorWriter::update_set(const VkDevice device,
                                  const VkDescriptorSet descriptorSet) {
  for (auto& write : writes) {
    write.dstSet = descriptorSet;
  }
  vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()),
                         writes.data(), 0, nullptr);
}
}  // namespace mp