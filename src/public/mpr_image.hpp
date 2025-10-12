#pragma once
#include <vulkan/vulkan.h>

#include <cmath>
#include <deque>
#include <vector>

namespace mp::utils {

inline std::uint32_t calculate_mip_levels(const VkExtent2D extent) {
  return static_cast<std::uint32_t>(
             std::floor(std::log2(std::max(extent.width, extent.height)))) +
         1;
}


void generate_mipmaps(VkCommandBuffer cmd, VkImage image, VkExtent2D extent);
void transition_image(VkCommandBuffer cmd, VkImage image,
                      VkImageLayout currentLayout, VkImageLayout newLayout);

void copy_to_image(VkCommandBuffer cmd, VkImage src, VkImage dest,
                   VkExtent2D srcSize, VkExtent2D destSize);

struct BarrierBuilder {
  std::vector<VkImageMemoryBarrier2> imageBarriers;
  std::vector<VkBufferMemoryBarrier2> bufferBarriers;
  std::vector<VkMemoryBarrier2> memoryBarriers;

  void clear() {
    memoryBarriers.clear();
    bufferBarriers.clear();
    imageBarriers.clear();
  }

  void add_image_barrier(
      VkImage image, VkPipelineStageFlags2 srcStageMask,
      VkAccessFlags2 srcAccessMask, VkPipelineStageFlags2 dstStageMask,
      VkAccessFlags2 dstAccessMask, VkImageLayout oldLayout,
      VkImageLayout newLayout, VkImageSubresourceRange subRange,
      std::uint32_t srcFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      std::uint32_t dstFamilyIndex = VK_QUEUE_FAMILY_IGNORED) {
    imageBarriers.emplace_back(
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, nullptr, srcStageMask,
        srcAccessMask, dstStageMask, dstAccessMask, oldLayout, newLayout,
        srcFamilyIndex, dstFamilyIndex, image, subRange);
  }

  // TODO: Simplify add buffer and memory barrier
  void add_buffer_barrier(const VkBufferMemoryBarrier2& bufferBarrier) {
    bufferBarriers.push_back(bufferBarrier);
  }

  void add_memory_barrier(const VkMemoryBarrier2& memoryBarrier) {
    memoryBarriers.push_back(memoryBarrier);
  }

  void barrier(const VkCommandBuffer cmd,
               const VkDependencyFlags dependencyFlags = 0);
};

}  // namespace mp::utils