#include "mpr_image.hpp"

#include "mpr_init_vk_stucts.hpp"

namespace mp::utils {
void transition_image(const VkCommandBuffer cmd, const VkImage image,
                      const VkImageLayout currentLayout,
                      const VkImageLayout newLayout) {
  VkImageMemoryBarrier2 imageBarrier{
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .dstAccessMask =
          VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
      .oldLayout = currentLayout,
      .newLayout = newLayout,
      .image = image,
  };
  const VkImageAspectFlags aspectMask =
      newLayout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL
          ? VK_IMAGE_ASPECT_DEPTH_BIT
          : VK_IMAGE_ASPECT_COLOR_BIT;
  imageBarrier.subresourceRange = init_subresource_range(aspectMask);
  const VkDependencyInfo dependencyInfo{
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &imageBarrier,

  };
  vkCmdPipelineBarrier2(cmd, &dependencyInfo);
}

void copy_to_image(const VkCommandBuffer cmd, const VkImage src, const VkImage dest,
                   const VkExtent2D srcSize, const VkExtent2D destSize) {
  VkImageBlit2 blitRegion{VK_STRUCTURE_TYPE_IMAGE_BLIT_2};
  blitRegion.srcOffsets[1] = {.x = static_cast<int32_t>(srcSize.width),
                              .y = static_cast<int32_t>(srcSize.height),
                              .z = 1};
  blitRegion.dstOffsets[1] = {.x = static_cast<int32_t>(destSize.width),
                              .y = static_cast<int32_t>(destSize.height),
                              .z = 1};
  blitRegion.srcSubresource = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .mipLevel = 0,
      .baseArrayLayer = 0,
      .layerCount = 1,
  };

  blitRegion.dstSubresource = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .mipLevel = 0,
      .baseArrayLayer = 0,
      .layerCount = 1,
  };
  const VkBlitImageInfo2 blitImageInfo{
      .sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
      .pNext = nullptr,
      .srcImage = src,
      .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,

      .dstImage = dest,
      .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .regionCount = 1,
      .pRegions = &blitRegion,
      .filter = VK_FILTER_LINEAR,
  };

  vkCmdBlitImage2(cmd, &blitImageInfo);
}
}  // namespace mp::utils
