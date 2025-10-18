#pragma once
#include "mpr_types.hpp"

#include "mpr_error_check.hpp"

namespace mp {
struct DescriptorBinding {
  uint32_t binding;
  VkDescriptorType type;
  uint32_t count;
  VkShaderStageFlags stageFlags;
};

class DescriptorSetLayoutBuilder {
 public:
  DescriptorSetLayoutBuilder& add_binding(const uint32_t binding,
                                          const VkDescriptorType type,
                                          const uint32_t count,
                                          const VkShaderStageFlags stageFlags) {
    m_bindings.push_back({binding, type, count, stageFlags});
    return *this;
  }

  VkDescriptorSetLayout build(
      const VkDevice device, const VkDescriptorSetLayoutCreateFlags flags = 0) {
    std::vector<VkDescriptorSetLayoutBinding> vkBindings;
    vkBindings.reserve(m_bindings.size());

    for (const auto& binding : m_bindings) {
      vkBindings.push_back({
          .binding = binding.binding,
          .descriptorType = binding.type,
          .descriptorCount = binding.count,
          .stageFlags = binding.stageFlags,
      });
    }

    const VkDescriptorSetLayoutCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .flags = flags,
        .bindingCount = static_cast<uint32_t>(vkBindings.size()),
        .pBindings = vkBindings.data()};

    VkDescriptorSetLayout layout;
    vkCreateDescriptorSetLayout(device, &createInfo, nullptr, &layout) >> chk;
    return layout;
  }

  [[nodiscard]] const std::vector<DescriptorBinding>& get_bindings() const {
    return m_bindings;
  }

 private:
  std::vector<DescriptorBinding> m_bindings;
};

class DescriptorBuffer {
 public:
  struct Properties {
    VkDeviceSize uniformBufferDescriptorSize;
    VkDeviceSize storageBufferDescriptorSize;
    VkDeviceSize sampledImageDescriptorSize;
    VkDeviceSize storageImageDescriptorSize;
    VkDeviceSize samplerDescriptorSize;
    VkDeviceSize descriptorBufferOffsetAlignment;
  };

  DescriptorBuffer() = default;

  DescriptorBuffer(const VkDevice device, const VkDescriptorSetLayout layout,
                   const Properties& props, const uint32_t setCount = 1)
      : m_device(device),
        m_layout(layout),
        m_props(props),
        m_setCount(setCount) {
    vkGetDescriptorSetLayoutSizeEXT(m_device, m_layout, &m_layoutSize);

    m_layoutSize =
        align_size(m_layoutSize, m_props.descriptorBufferOffsetAlignment);

    cache_binding_offsets();
  }

  template <typename CreateBufferFunc>
  void create_buffer(CreateBufferFunc&& createFunc) {
    VkDeviceSize totalSize = m_layoutSize * m_setCount;

    m_buffer = createFunc(
        totalSize, VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                       VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT |
                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    const VkBufferDeviceAddressInfo addressInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .buffer = m_buffer.buffer,
    };
    m_deviceAddress = vkGetBufferDeviceAddress(m_device, &addressInfo);
  }

  void write_uniform_buffer(const uint32_t binding, const uint32_t arrayIndex,
                            const VkDeviceAddress bufferAddress,
                            const VkDeviceSize range,
                            const uint32_t setIndex = 0) {
    VkDescriptorAddressInfoEXT addressInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
        .address = bufferAddress,
        .range = range,
        .format = VK_FORMAT_UNDEFINED,
    };

    const VkDescriptorGetInfoEXT getInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .data = {.pUniformBuffer = &addressInfo}};

    write_descriptor(binding, arrayIndex, setIndex, getInfo,
                     m_props.uniformBufferDescriptorSize);
  }

  void write_storage_buffer(const uint32_t binding, const uint32_t arrayIndex,
                            const VkDeviceAddress bufferAddress,
                            const VkDeviceSize range,
                            const uint32_t setIndex = 0) {
    VkDescriptorAddressInfoEXT addressInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
        .address = bufferAddress,
        .range = range,
        .format = VK_FORMAT_UNDEFINED,
    };

    const VkDescriptorGetInfoEXT getInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .data = {.pStorageBuffer = &addressInfo}};

    write_descriptor(binding, arrayIndex, setIndex, getInfo,
                     m_props.storageBufferDescriptorSize);
  }

  void write_sampler(const uint32_t binding, const uint32_t arrayIndex,
                     VkSampler sampler, const uint32_t setIndex = 0) {
    const VkDescriptorGetInfoEXT getInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
        .type = VK_DESCRIPTOR_TYPE_SAMPLER,
        .data = {.pSampler = &sampler}};

    write_descriptor(binding, arrayIndex, setIndex, getInfo,
                     m_props.samplerDescriptorSize);
  }

  void write_sampled_image(const uint32_t binding, const uint32_t arrayIndex,
                           const VkImageView imageView,
                           const VkImageLayout layout,
                           const uint32_t setIndex = 0) {
    VkDescriptorImageInfo imageInfo{.imageView = imageView,
                                    .imageLayout = layout};

    const VkDescriptorGetInfoEXT getInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
        .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        .data = {.pSampledImage = &imageInfo}};

    write_descriptor(binding, arrayIndex, setIndex, getInfo,
                     m_props.sampledImageDescriptorSize);
  }

  void write_storage_image(const uint32_t binding, const uint32_t arrayIndex,
                           const VkImageView imageView,
                           const VkImageLayout layout,
                           const uint32_t setIndex = 0) {
    VkDescriptorImageInfo imageInfo{.imageView = imageView,
                                    .imageLayout = layout};

    const VkDescriptorGetInfoEXT getInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
        .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .data = {.pStorageImage = &imageInfo}};

    write_descriptor(binding, arrayIndex, setIndex, getInfo,
                     m_props.storageImageDescriptorSize);
  }

  [[nodiscard]]
  VkDeviceAddress get_device_address(const uint32_t setIndex = 0) const {
    return m_deviceAddress + (m_layoutSize * setIndex);
  }

  [[nodiscard]]
  VkDeviceSize get_layout_size() const {
    return m_layoutSize;
  }
  [[nodiscard]]
  VkDeviceSize get_total_size() const {
    return m_layoutSize * m_setCount;
  }

  AllocatedBuffer& get_buffer() & { return m_buffer; }

 private:
  void cache_binding_offsets() {
    // TODO: FIX
    // Получаем максимальный binding для определения размера вектора
    uint32_t maxBinding = 0;
    VkDescriptorSetLayoutBinding bindings[32];  // Временный буфер
    uint32_t bindingCount = 0;

    // Здесь нужно получить bindings из layout (в реальности это делается через
    // рефлексию) Для упрощения предполагаем, что bindings последовательные

    for (uint32_t i = 0; i < 32; ++i) {
      VkDeviceSize offset;
      vkGetDescriptorSetLayoutBindingOffsetEXT(m_device, m_layout, i, &offset);

      m_bindingOffsets[i] = offset;
    }
  }

  void write_descriptor(const uint32_t binding, const uint32_t arrayIndex,
                        const uint32_t setIndex,
                        const VkDescriptorGetInfoEXT& getInfo,
                        const VkDeviceSize descriptorSize) {
    const auto it = m_bindingOffsets.find(binding);
    if (it == m_bindingOffsets.end()) {
      return;
    }

    const VkDeviceSize offset =
        calc_offset(it->second, arrayIndex, descriptorSize, setIndex);

    auto* mappedData = static_cast<char*>(m_buffer.allocationInfo.pMappedData);
    vkGetDescriptorEXT(m_device, &getInfo, descriptorSize, mappedData + offset);
  }

  [[nodiscard]]
  VkDeviceSize calc_offset(const VkDeviceSize bindingOffset,
                           const uint32_t arrayIndex,
                           const VkDeviceSize descriptorSize,
                           const uint32_t setIndex) const {
    return (setIndex * m_layoutSize) + bindingOffset +
           (arrayIndex * descriptorSize);
  }

  static VkDeviceSize align_size(const VkDeviceSize size,
                                 const VkDeviceSize alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
  }

  VkDevice m_device;
  VkDescriptorSetLayout m_layout;
  Properties m_props;
  uint32_t m_setCount;

  VkDeviceSize m_layoutSize = 0;
  VkDeviceAddress m_deviceAddress = 0;
  std::unordered_map<uint32_t, VkDeviceSize> m_bindingOffsets;

  AllocatedBuffer m_buffer;
};

class DescriptorBufferProperties {
 public:
  static DescriptorBuffer::Properties query(VkPhysicalDevice physicalDevice) {
    VkPhysicalDeviceDescriptorBufferPropertiesEXT bufferProps{
        .sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT,
    };

    VkPhysicalDeviceProperties2 deviceProps{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &bufferProps,
    };

    vkGetPhysicalDeviceProperties2(physicalDevice, &deviceProps);

    return {
        .uniformBufferDescriptorSize = bufferProps.uniformBufferDescriptorSize,
        .storageBufferDescriptorSize = bufferProps.storageBufferDescriptorSize,
        .sampledImageDescriptorSize = bufferProps.sampledImageDescriptorSize,
        .storageImageDescriptorSize = bufferProps.storageImageDescriptorSize,
        .samplerDescriptorSize = bufferProps.samplerDescriptorSize,
        .descriptorBufferOffsetAlignment =
            bufferProps.descriptorBufferOffsetAlignment,
    };
  }
};

}  // namespace mp