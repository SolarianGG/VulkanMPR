#pragma once
#include <volk.h>

#include <cassert>

namespace mp::utils {
constexpr VkImageSubresourceRange init_subresource_range(
    const VkImageAspectFlags aspectFlags) {
  return {
      .aspectMask = aspectFlags,
      .baseMipLevel = 0,
      .levelCount = VK_REMAINING_MIP_LEVELS,
      .baseArrayLayer = 0,
      .layerCount = VK_REMAINING_ARRAY_LAYERS,
  };
}

constexpr VkSemaphoreSubmitInfo semaphore_submit_info(
    const VkPipelineStageFlags2 stageMask, const VkSemaphore semaphore) {
  return {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .pNext = nullptr,
      .semaphore = semaphore,
      .value = 1,
      .stageMask = stageMask,
      .deviceIndex = 0,
  };
}

constexpr VkCommandBufferSubmitInfo command_buffer_submit_info(
    const VkCommandBuffer cmd) {
  return {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .pNext = nullptr,
      .commandBuffer = cmd,
      .deviceMask = 0,
  };
}

constexpr VkSubmitInfo2 submit_info(
    const VkCommandBufferSubmitInfo* cmd,
    const VkSemaphoreSubmitInfo* waitSemaphoreInfo,
    const VkSemaphoreSubmitInfo* signalSubmitInfo) {
  VkSubmitInfo2 smd{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .pNext = nullptr,
      .flags = 0,
      .pWaitSemaphoreInfos = waitSemaphoreInfo,
      .commandBufferInfoCount = 1,
      .pCommandBufferInfos = cmd,
      .pSignalSemaphoreInfos = signalSubmitInfo,
  };
  smd.waitSemaphoreInfoCount = waitSemaphoreInfo ? 1 : 0;
  smd.signalSemaphoreInfoCount = signalSubmitInfo ? 1 : 0;
  return smd;
}

constexpr VkImageCreateInfo image_create_info(
    const VkFormat format, const VkImageUsageFlags usageFlags,
    const VkExtent3D extent) {
  return {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = nullptr,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = format,
      .extent = extent,
      .mipLevels = 1,
      .arrayLayers = 1,
      // For MSAA, default for us is one, we won't use it
      .samples = VK_SAMPLE_COUNT_1_BIT,
      // optimal tiling, which means that image is stored on the best GPU format
      // use LINEAR for the CPU access
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usageFlags,
  };
}

constexpr VkImageViewCreateInfo image_view_create_info(
    const VkFormat format, const VkImage image,
    const VkImageAspectFlags aspectFlags) {
  return {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
          .pNext = nullptr,
          .image = image,
          .viewType = VK_IMAGE_VIEW_TYPE_2D,
          .format = format,
          .subresourceRange = {
              .aspectMask = aspectFlags,
              .baseMipLevel = 0,
              .levelCount = 1,
              .baseArrayLayer = 0,
              .layerCount = 1,
          }};
}

inline VkRenderingAttachmentInfo attachment_info(const VkImageView view,
                                                 const VkClearValue* clear,
                                                 const VkImageLayout layout) {
  VkRenderingAttachmentInfo attachment{
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .pNext = nullptr,
      .imageView = view,
      .imageLayout = layout,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,

  };
  attachment.loadOp =
      clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
  if (clear) {
    attachment.clearValue = *clear;
  }
  return attachment;
}

inline VkRenderingAttachmentInfo depth_attachment(
    const VkImageView view, const VkImageLayout imageLayout) {
  constexpr VkClearDepthStencilValue depthStencilValue{
      .depth = 0.0f,
  };
  const VkClearValue clearValue{.depthStencil = depthStencilValue};
  return attachment_info(view, &clearValue, imageLayout);
}

inline VkRenderingInfo rendering_info(
    const VkExtent2D swapchainExtent,
    const VkRenderingAttachmentInfo* colorAttachment,
    const VkRenderingAttachmentInfo* depthAttachment) {
  assert(colorAttachment);
  const VkRenderingInfo renderingInfo{
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .pNext = nullptr,
      .renderArea = {.extent = swapchainExtent},
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = colorAttachment,
      .pDepthAttachment = depthAttachment,
      .pStencilAttachment = nullptr,
  };

  return renderingInfo;
}

}  // namespace mp::utils