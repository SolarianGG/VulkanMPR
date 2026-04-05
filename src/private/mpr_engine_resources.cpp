// clang-format off
#include "mpr_engine.hpp"

#include <vulkan/utility/vk_format_utils.h>
#include <vulkan/vk_enum_string_helper.h>

#include <format>
#include <vk_mem_alloc.h>

#include "mpr_error_check.hpp"
#include "mpr_image.hpp"
#include "mpr_init_vk_stucts.hpp"
#include "mpr_debug_utils.hpp"
// clang-format on

namespace mp {

void Engine::immediate_submit(
    const std::function<void(VkCommandBuffer)>& function) {
  vkResetFences(m_device, 1, &m_immFence) >> chk;
  {
    constexpr VkCommandBufferBeginInfo cmdBeginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };
    vkBeginCommandBuffer(m_immCommandBuffer, &cmdBeginInfo) >> chk;
  }

  function(m_immCommandBuffer);

  vkEndCommandBuffer(m_immCommandBuffer) >> chk;

  const auto cmdSubmitInfo =
      utils::command_buffer_submit_info(m_immCommandBuffer);

  const auto submitInfo = utils::submit_info(&cmdSubmitInfo, nullptr, nullptr);

  vkQueueSubmit2(m_queue, 1, &submitInfo, m_immFence) >> chk;

  vkWaitForFences(m_device, 1, &m_immFence, true, ~0ull) >> chk;
}

AllocatedBuffer Engine::create_buffer(const std::size_t allocSize,
                                      const VkBufferUsageFlags usageFlags,
                                      const VmaMemoryUsage memoryUsage) {
  const VkBufferCreateInfo bufferCreateInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext = nullptr,
      .size = allocSize,
      .usage = usageFlags,
      .queueFamilyIndexCount = 1,
      .pQueueFamilyIndices = &m_queueFamilyIndex,
  };

  const VmaAllocationCreateInfo allocCreateInfo{
      .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT,
      .usage = memoryUsage,
  };
  AllocatedBuffer alloc{};
  vmaCreateBuffer(m_allocator, &bufferCreateInfo, &allocCreateInfo,
                  &alloc.buffer, &alloc.allocation, &alloc.allocationInfo) >>
      chk;

  return alloc;
}

void Engine::destroy_buffer(const AllocatedBuffer& buffer) {
  vmaDestroyBuffer(m_allocator, buffer.buffer, buffer.allocation);
}

AllocatedImage Engine::create_image(const VkExtent3D extent,
                                    const VkFormat format,
                                    const VkImageUsageFlags imageUsage,
                                    const bool mipMapped) {
  AllocatedImage image;
  image.imageFormat = format;
  image.imageExtent = extent;

  VkImageCreateInfo imageCreateInfo =
      utils::image_create_info(format, imageUsage, extent);

  if (mipMapped) {
    imageCreateInfo.mipLevels =
        utils::calculate_mip_levels(VkExtent2D{extent.width, extent.height});
  }
  const VmaAllocationCreateInfo allocationCreateInfo{
      .usage = VMA_MEMORY_USAGE_GPU_ONLY,
      .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
  };
  vmaCreateImage(m_allocator, &imageCreateInfo, &allocationCreateInfo,
                 &image.image, &image.allocation, nullptr) >>
      chk;

  const VkImageAspectFlags aspectFlags = vkuFormatIsDepthOnly(format)
                                             ? VK_IMAGE_ASPECT_DEPTH_BIT
                                             : VK_IMAGE_ASPECT_COLOR_BIT;
  VkImageViewCreateInfo imageViewCreateInfo =
      utils::image_view_create_info(format, image.image, aspectFlags);
  imageViewCreateInfo.subresourceRange.levelCount = imageCreateInfo.mipLevels;
  vkCreateImageView(m_device, &imageViewCreateInfo, nullptr,
                    &image.imageView) >>
      chk;

  return image;
}

AllocatedImage Engine::create_image(void* data, const VkExtent3D extent,
                                    const VkFormat format,
                                    const VkImageUsageFlags imageUsage,
                                    const bool mipMapped) {
  constexpr auto imagePixelSize = 4;  // NOTE: 8 + 8 + 8 + 8 = 32bits (4 bytes)
  if (!(vkuFormatIs8bit(format) && vkuFormatComponentCount(format) == 4)) {
    throw std::runtime_error(std::format("Unsupported image format for now: {}",
                                         string_VkFormat(format)));
  }
  const auto bufferSize =
      extent.width * extent.height * extent.depth * imagePixelSize;
  const AllocatedImage image =
      create_image(extent, format,
                   imageUsage | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                   mipMapped);

  const AllocatedBuffer stagingBuffer = create_buffer(
      bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

  auto* bufferData =
      static_cast<char*>(stagingBuffer.allocationInfo.pMappedData);
  std::memcpy(bufferData, data, bufferSize);

  immediate_submit([&](const VkCommandBuffer cmd) {
    utils::BarrierBuilder barrierBuilder;
    barrierBuilder.add_image_barrier(
        image.image, VK_PIPELINE_STAGE_2_NONE, 0,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
    barrierBuilder.barrier(cmd);
    VkBufferImageCopy copyRegion;
    copyRegion.imageExtent = extent;
    copyRegion.bufferOffset = 0;
    copyRegion.bufferRowLength = 0;
    copyRegion.bufferImageHeight = 0;
    copyRegion.imageOffset = {};
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageSubresource.mipLevel = 0;

    vkCmdCopyBufferToImage(cmd, stagingBuffer.buffer, image.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                           &copyRegion);

    if (mipMapped) {
      utils::generate_mipmaps(cmd, image.image,
                              VkExtent2D{extent.width, extent.height});
    } else {
      utils::transition_image(cmd, image.image,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
  });
  destroy_buffer(stagingBuffer);
  return image;
}

void Engine::destroy_image(const AllocatedImage& image) {
  vkDestroyImageView(m_device, image.imageView, nullptr);
  vmaDestroyImage(m_allocator, image.image, image.allocation);
}

void Engine::ensure_position_capacity(std::size_t additionalCount) {
  if (m_globalPositionCount + additionalCount <= m_globalPositionCapacity) {
    return;
  }

  std::size_t newCapacity =
      m_globalPositionCapacity == 0 ? 1024 : m_globalPositionCapacity * 2;
  while (m_globalPositionCount + additionalCount > newCapacity) {
    newCapacity *= 2;
  }

  const std::size_t newSize = newCapacity * sizeof(VertexPosition);
  const AllocatedBuffer newBuffer = create_buffer(
      newSize,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      VMA_MEMORY_USAGE_GPU_ONLY);

  if (m_globalPositionCount > 0) {
    immediate_submit([&](VkCommandBuffer cmd) {
      const VkBufferCopy copyRegion{
          .srcOffset = 0,
          .dstOffset = 0,
          .size = m_globalPositionCount * sizeof(VertexPosition),
      };
      vkCmdCopyBuffer(cmd, m_globalPositionBuffer.buffer, newBuffer.buffer, 1,
                      &copyRegion);
    });
    destroy_buffer(m_globalPositionBuffer);
  } else if (m_globalPositionCapacity > 0) {
    destroy_buffer(m_globalPositionBuffer);
  }

  m_globalPositionBuffer = newBuffer;
  mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_BUFFER,
                             reinterpret_cast<uint64_t>(m_globalPositionBuffer.buffer), "Global Position Buffer");
  m_globalPositionCapacity = newCapacity;

  const VkBufferDeviceAddressInfo addrInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
      .buffer = m_globalPositionBuffer.buffer,
  };
  m_globalPositionBufferAddress = vkGetBufferDeviceAddress(m_device, &addrInfo);
}

void Engine::ensure_attributes_capacity(std::size_t additionalCount) {
  if (m_globalAttributesCount + additionalCount <= m_globalAttributesCapacity) {
    return;
  }

  std::size_t newCapacity =
      m_globalAttributesCapacity == 0 ? 1024 : m_globalAttributesCapacity * 2;
  while (m_globalAttributesCount + additionalCount > newCapacity) {
    newCapacity *= 2;
  }

  const std::size_t newSize = newCapacity * sizeof(VertexAttributes);
  const AllocatedBuffer newBuffer = create_buffer(
      newSize,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      VMA_MEMORY_USAGE_GPU_ONLY);

  if (m_globalAttributesCount > 0) {
    immediate_submit([&](VkCommandBuffer cmd) {
      const VkBufferCopy copyRegion{
          .srcOffset = 0,
          .dstOffset = 0,
          .size = m_globalAttributesCount * sizeof(VertexAttributes),
      };
      vkCmdCopyBuffer(cmd, m_globalAttributesBuffer.buffer, newBuffer.buffer, 1,
                      &copyRegion);
    });
    destroy_buffer(m_globalAttributesBuffer);
  } else if (m_globalAttributesCapacity > 0) {
    destroy_buffer(m_globalAttributesBuffer);
  }

  m_globalAttributesBuffer = newBuffer;
  mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_BUFFER,
                             reinterpret_cast<uint64_t>(m_globalAttributesBuffer.buffer), "Global Attributes Buffer");
  m_globalAttributesCapacity = newCapacity;

  const VkBufferDeviceAddressInfo addrInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
      .buffer = m_globalAttributesBuffer.buffer,
  };
  m_globalAttributesBufferAddress = vkGetBufferDeviceAddress(m_device, &addrInfo);
}

void Engine::ensure_index_capacity(std::size_t additionalCount) {
  if (m_globalIndexCount + additionalCount <= m_globalIndexCapacity) {
    return;
  }

  std::size_t newCapacity =
      m_globalIndexCapacity == 0 ? 1024 : m_globalIndexCapacity * 2;
  while (m_globalIndexCount + additionalCount > newCapacity) {
    newCapacity *= 2;
  }

  const std::size_t newSize = newCapacity * sizeof(std::uint32_t);
  const AllocatedBuffer newBuffer = create_buffer(
      newSize,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
          VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      VMA_MEMORY_USAGE_GPU_ONLY);

  if (m_globalIndexCount > 0) {
    immediate_submit([&](VkCommandBuffer cmd) {
      const VkBufferCopy copyRegion{
          .srcOffset = 0,
          .dstOffset = 0,
          .size = m_globalIndexCount * sizeof(std::uint32_t),
      };
      vkCmdCopyBuffer(cmd, m_globalIndexBuffer.buffer, newBuffer.buffer, 1,
                      &copyRegion);
    });
    destroy_buffer(m_globalIndexBuffer);
  } else if (m_globalIndexCapacity > 0) {
    destroy_buffer(m_globalIndexBuffer);
  }

  m_globalIndexBuffer = newBuffer;
  mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_BUFFER,
                             reinterpret_cast<uint64_t>(m_globalIndexBuffer.buffer), "Global Index Buffer");
  m_globalIndexCapacity = newCapacity;
}

}  // namespace mp
