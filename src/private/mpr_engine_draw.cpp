// clang-format off
#include "mpr_engine.hpp"

#include <imgui.h>
#include <imgui_impl_vulkan.h>

#include <chrono>
#include <ranges>

#include "mpr_error_check.hpp"
#include "mpr_image.hpp"
#include "mpr_init_vk_stucts.hpp"
// clang-format on

namespace cn = std::chrono;
namespace rn = std::ranges;

namespace mp {

void Engine::draw() {
  FrameData& currentFrame = get_current_frame();
  update_scene();
  // Wait if command buffer is in execution on the gpu
  vkWaitForFences(m_device, 1, &currentFrame.fence, true,
                  std::numeric_limits<std::uint64_t>::max()) >>
      chk;
  currentFrame.frameDeletionQueue.flush();

  // Get the current image from the swapchain

  std::uint32_t swapchainImageIndex;
  const VkResult swapchainAcquireRes = vkAcquireNextImageKHR(
      m_device, m_swapchain, std::numeric_limits<std::uint64_t>::max(),
      currentFrame.swapchainSemaphore, nullptr, &swapchainImageIndex);
  if (swapchainAcquireRes == VK_ERROR_OUT_OF_DATE_KHR) {
    m_bSwapchainResizeRequest = true;
    return;
  }
  if (swapchainAcquireRes == VK_SUBOPTIMAL_KHR) {
    m_bSwapchainResizeRequest = true;
  } else {
    swapchainAcquireRes >> chk;
  }
  VkSemaphore& signalSemaphore = m_swapchainSemaphores[swapchainImageIndex];

  // Reset command buffer
  const VkCommandBuffer& cmd = currentFrame.commandBuffer;
  const VkImage& swapchainImage = m_swapchainImages[swapchainImageIndex];

  constexpr VkCommandBufferBeginInfo beginInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .pNext = nullptr,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
      .pInheritanceInfo = nullptr,
  };

  const AllocatedImage& currentDrawingImage = currentFrame.drawImage;
  const AllocatedImage& currentDepthImage = currentFrame.depthImage;
  const auto& gBuffer = currentFrame.gBuffer;
  m_drawExtent.width =
      std::min(currentDrawingImage.imageExtent.width, m_swapchainExtent.width) *
      m_renderScale;
  m_drawExtent.height = std::min(currentDrawingImage.imageExtent.height,
                                 m_swapchainExtent.height) *
                        m_renderScale;

  vkBeginCommandBuffer(cmd, &beginInfo) >> chk;

  utils::BarrierBuilder barrierBuilder;
  m_stats.drawCallCount = 0;
  m_stats.triangleCount = 0;

  vkCmdBindIndexBuffer(cmd, m_globalIndexBuffer.buffer, 0,
                       VK_INDEX_TYPE_UINT32);

  {
    barrierBuilder.add_image_barrier(
        currentFrame.shadowPassDepthImage.image,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
            VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        utils::init_subresource_range(VK_IMAGE_ASPECT_DEPTH_BIT));
    barrierBuilder.barrier(cmd);
  }
  draw_shadow_pass(cmd);

  // GPass
  {
    barrierBuilder.add_image_barrier(
        gBuffer.position.image, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
    barrierBuilder.add_image_barrier(
        gBuffer.normal.image, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
    barrierBuilder.add_image_barrier(
        gBuffer.diffuse.image, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
    barrierBuilder.add_image_barrier(
        gBuffer.specular.image, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
    barrierBuilder.add_image_barrier(
        currentDepthImage.image, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
            VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        utils::init_subresource_range(VK_IMAGE_ASPECT_DEPTH_BIT));
    barrierBuilder.barrier(cmd);
  }

  draw_gBuffer_pass(cmd);

  // Light pass
  {
    barrierBuilder.add_image_barrier(
        gBuffer.position.image, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
    barrierBuilder.add_image_barrier(
        gBuffer.normal.image, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
    barrierBuilder.add_image_barrier(
        gBuffer.diffuse.image, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
    barrierBuilder.add_image_barrier(
        gBuffer.specular.image, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
    barrierBuilder.add_image_barrier(
        currentDrawingImage.image, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_WRITE_BIT_KHR, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL,
        utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
    barrierBuilder.add_image_barrier(
        currentFrame.shadowPassDepthImage.image,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
            VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
        utils::init_subresource_range(VK_IMAGE_ASPECT_DEPTH_BIT));

    barrierBuilder.barrier(cmd);
  }

  draw_light_pass(cmd);

  // forward WBOIT pass

  {
    barrierBuilder.add_image_barrier(
        currentFrame.oitAccImage.image, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
    barrierBuilder.add_image_barrier(
        currentFrame.oitRevealImage.image, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
        0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
    barrierBuilder.barrier(cmd);
  }

  draw_wboit(cmd);

  {
    barrierBuilder.add_image_barrier(
        currentFrame.oitAccImage.image,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
    barrierBuilder.add_image_barrier(
        currentFrame.oitRevealImage.image,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
    barrierBuilder.add_image_barrier(
        currentDrawingImage.image, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_WRITE_BIT_KHR,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
    barrierBuilder.barrier(cmd);
  }

  draw_wboit_composite(cmd);

  {
    barrierBuilder.add_image_barrier(
        currentDrawingImage.image,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_WRITE_BIT_KHR | VK_ACCESS_2_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
        utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
    barrierBuilder.barrier(cmd);
  }

  draw_post(cmd);

  // copy to swapchain
  {
    barrierBuilder.add_image_barrier(
        currentDrawingImage.image, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_WRITE_BIT_KHR | VK_ACCESS_2_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));

    barrierBuilder.add_image_barrier(
        swapchainImage, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
    barrierBuilder.barrier(cmd);
  }

  utils::copy_to_image(cmd, currentDrawingImage.image, swapchainImage,
                       m_drawExtent, m_swapchainExtent);

  // Imgui
  {
    barrierBuilder.add_image_barrier(
        swapchainImage, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,

        utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));

    barrierBuilder.barrier(cmd);
  }

  draw_imgui(cmd, m_swapchainImageViews[swapchainImageIndex]);

  {
    barrierBuilder.add_image_barrier(
        swapchainImage, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));

    barrierBuilder.barrier(cmd);
  }

  vkEndCommandBuffer(currentFrame.commandBuffer) >> chk;

  const auto waitSemaphoreInfo = utils::semaphore_submit_info(
      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
      currentFrame.swapchainSemaphore);

  const auto signalSemaphoreInfo = utils::semaphore_submit_info(
      VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, signalSemaphore);

  const auto cmdInfo = utils::command_buffer_submit_info(cmd);

  const auto renderSubmitInfo =
      utils::submit_info(&cmdInfo, &waitSemaphoreInfo, &signalSemaphoreInfo);
  vkResetFences(m_device, 1, &currentFrame.fence) >> chk;
  vkQueueSubmit2(m_queue, 1, &renderSubmitInfo, currentFrame.fence) >> chk;

  const VkPresentInfoKHR presentInfo{
      .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .pNext = nullptr,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &signalSemaphore,
      .swapchainCount = 1,
      .pSwapchains = &m_swapchain,
      .pImageIndices = &swapchainImageIndex,
      .pResults = nullptr,
  };
  const VkResult swapchainPresentResult =
      vkQueuePresentKHR(m_queue, &presentInfo);
  if (swapchainPresentResult == VK_ERROR_OUT_OF_DATE_KHR ||
      swapchainPresentResult == VK_SUBOPTIMAL_KHR) {
    m_bSwapchainResizeRequest = true;
  } else {
    swapchainPresentResult >> chk;
  }
  ++m_frameNumber;
}

void Engine::draw_shadow_pass(VkCommandBuffer cmd) {
  const auto it = rn::find_if(m_mainDrawContext.lights, [](const auto& light) {
    return light.lightType == 0;
  });
  if (it == m_mainDrawContext.lights.end()) return;
  const auto& light = *it;
  const auto start = cn::steady_clock::now();
  constexpr VkExtent2D shadowPassExtent{2048, 2048};
  auto& currentFrame = get_current_frame();
  const auto depthAttachment =
      utils::depth_attachment(currentFrame.shadowPassDepthImage.imageView,
                              VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
  const auto renderInfo =
      utils::rendering_info(shadowPassExtent, 0, nullptr, &depthAttachment);

  cull_objects(cmd, m_OpaqueSize, 0, light.lightVP);
  vkCmdBeginRendering(cmd, &renderInfo);
  const VkViewport viewport{
      .x = 0,
      .y = static_cast<float>(shadowPassExtent.height),
      .width = static_cast<float>(shadowPassExtent.width),
      .height = -static_cast<float>(shadowPassExtent.height),
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
  };
  vkCmdSetViewport(cmd, 0, 1, &viewport);

  const VkRect2D scissor{
      .extent = shadowPassExtent,
  };
  vkCmdSetScissor(cmd, 0, 1, &scissor);

  const ShadowPassPushConstants shadowPassPushConstants{
      .globalVertexBufferAddr = m_globalVertexBufferAddress,
      .instanceBufferDeviceAddr = currentFrame.instanceBufferAddr,
      .lightVP = light.lightVP};
  draw_meshes(cmd, m_ShadowPassPipelineLayout, m_ShadowPassPipeline,
              m_OpaqueSize, shadowPassPushConstants, VK_SHADER_STAGE_VERTEX_BIT,
              false);

  vkCmdEndRendering(cmd);
  const auto end = cn::steady_clock::now();
  const auto elapsed = cn::duration_cast<cn::milliseconds>(end - start);
  m_stats.shadowPassDrawTime = elapsed.count() / 1000.0f;
}

void Engine::draw_gBuffer_pass(VkCommandBuffer cmd) {
  const auto start = cn::steady_clock::now();
  auto& gBuffer = get_current_frame().gBuffer;
  auto& depthImage = get_current_frame().depthImage;

  // ---
  VkClearValue val{.color = {0.0f, 0.0f, 0.0f, 1.0f}};
  const auto positionAttachment =
      utils::attachment_info(gBuffer.position.imageView, &val,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  const auto normalAttachment = utils::attachment_info(
      gBuffer.normal.imageView, &val, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  const auto diffuseAttachment =
      utils::attachment_info(gBuffer.diffuse.imageView, &val,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  const auto specularAttachment =
      utils::attachment_info(gBuffer.specular.imageView, &val,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  const auto depthAttachment = utils::depth_attachment(
      depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

  VkRenderingAttachmentInfo attachments[]{positionAttachment, normalAttachment,
                                          diffuseAttachment,
                                          specularAttachment};
  const auto renderInfo =
      utils::rendering_info(m_CommonImageExtent2D, std::size(attachments),
                            attachments, &depthAttachment);

  cull_objects(cmd, m_OpaqueSize, 0, m_sceneData.projView);
  vkCmdBeginRendering(cmd, &renderInfo);
  const VkViewport viewport{
      .x = 0,
      .y = static_cast<float>(m_CommonImageExtent2D.height),
      .width = static_cast<float>(m_CommonImageExtent2D.width),
      .height = -static_cast<float>(m_CommonImageExtent2D.height),
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
  };
  vkCmdSetViewport(cmd, 0, 1, &viewport);

  const VkRect2D scissor{
      .extent = m_CommonImageExtent2D,
  };
  vkCmdSetScissor(cmd, 0, 1, &scissor);

  draw_meshes(cmd, m_metalRoughness.opaquePipeline.pipelineLayout,
              m_metalRoughness.opaquePipeline.pipeline, m_OpaqueSize,
              m_GBufferMeshPushConstants, VK_SHADER_STAGE_VERTEX_BIT);

  vkCmdEndRendering(cmd);
  const auto end = cn::steady_clock::now();
  const auto elapsed = cn::duration_cast<cn::milliseconds>(end - start);
  m_stats.gBufferPassTime = elapsed.count() / 1000.0f;
}

void Engine::draw_light_pass(const VkCommandBuffer cmd) {
  const auto start = cn::steady_clock::now();
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_LightPassPipeline);

  const VkDescriptorBufferBindingInfoEXT buffersInfo[]{
      {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
       .pNext = nullptr,
       .address =
           get_current_frame().drawImageDescriptorBuffer.get_device_address(),
       .usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT},
      {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
       .pNext = nullptr,
       .address =
           get_current_frame().lightPassDescriptorBuffer.get_device_address(),
       .usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT}};
  vkCmdBindDescriptorBuffersEXT(cmd, std::size(buffersInfo), buffersInfo);

  const std::uint32_t indices[]{0, 1};
  const VkDeviceSize offsets[]{0, 0};
  vkCmdSetDescriptorBufferOffsetsEXT(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                     m_LightPassPipelineLayout, 0,
                                     std::size(offsets), indices, offsets);

  vkCmdPushConstants(cmd, m_LightPassPipelineLayout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     sizeof(LightPassConstantRange), &m_LightPassConstants);

  vkCmdDispatch(cmd, std::ceil(m_CommonImageExtent2D.width / 16.0f),
                std::ceil(m_CommonImageExtent2D.height / 16.0f), 1);

  const auto end = cn::steady_clock::now();
  const auto elapsed = cn::duration_cast<cn::milliseconds>(end - start);
  m_stats.gBufferLightPassTime = elapsed.count() / 1000.0f;
}

void Engine::draw_wboit(VkCommandBuffer cmd) {
  const auto start = cn::steady_clock::now();
  const auto& currentFrame = get_current_frame();

  const VkClearValue clearAccum{.color = {0.0f, 0.0f, 0.0f, 0.0f}};
  const VkClearValue clearReveal{.color = {1.0f, 1.0f, 1.0f, 1.0f}};
  const VkRenderingAttachmentInfo colorAttachments[]{
      utils::attachment_info(currentFrame.oitAccImage.imageView, &clearAccum,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL),
      utils::attachment_info(currentFrame.oitRevealImage.imageView,
                             &clearReveal,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)};
  const auto depthAttachment =
      utils::depth_attachment(currentFrame.depthImage.imageView,
                              VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, false);
  const auto renderInfo =
      utils::rendering_info(m_CommonImageExtent2D, std::size(colorAttachments),
                            colorAttachments, &depthAttachment);

  cull_objects(cmd, m_TransparentSize, m_OpaqueSize, m_sceneData.projView);
  vkCmdBeginRendering(cmd, &renderInfo);
  draw_meshes(cmd, m_metalRoughness.transparentPipeline.pipelineLayout,
              m_metalRoughness.transparentPipeline.pipeline, m_TransparentSize,
              m_WBOITForwardPassPushConstants,
              VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
  vkCmdEndRendering(cmd);
  const auto end = cn::steady_clock::now();
  const auto elapsed = cn::duration_cast<cn::milliseconds>(end - start);
  m_stats.transparentForwardLightPassTime = elapsed.count() / 1000.0f;
}

void Engine::draw_wboit_composite(VkCommandBuffer cmd) {
  const auto start = cn::steady_clock::now();

  const auto attachment =
      utils::attachment_info(get_current_frame().drawImage.imageView, nullptr,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  const VkRenderingInfo renderingInfo =
      utils::rendering_info(m_CommonImageExtent2D, 1, &attachment, nullptr);
  vkCmdBeginRendering(cmd, &renderingInfo);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    m_WBOITCompositePassPipeline);

  const VkDescriptorBufferBindingInfoEXT buffersInfo[]{
      {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
       .pNext = nullptr,
       .address = get_current_frame()
                      .wboitCompositePassDescBuffer.get_device_address(),
       .usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT}};
  vkCmdBindDescriptorBuffersEXT(cmd, std::size(buffersInfo), buffersInfo);

  const std::uint32_t indices[]{0};
  const VkDeviceSize offsets[]{0};
  vkCmdSetDescriptorBufferOffsetsEXT(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                     m_WBOITCompositePassPipelineLayout, 0,
                                     std::size(offsets), indices, offsets);

  vkCmdDraw(cmd, 6, 1, 0, 0);

  vkCmdEndRendering(cmd);
  const auto end = cn::steady_clock::now();
  const auto elapsed = cn::duration_cast<cn::milliseconds>(end - start);
  m_stats.postProcessPassTime = elapsed.count() / 1000.0f;
}

void Engine::draw_imgui(const VkCommandBuffer cmd,
                        const VkImageView targetImageView) {
  const auto start = cn::steady_clock::now();
  const auto colorAttachment = utils::attachment_info(
      targetImageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  const auto renderingInfo =
      utils::rendering_info(m_swapchainExtent, 1, &colorAttachment, nullptr);

  vkCmdBeginRendering(cmd, &renderingInfo);

  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

  vkCmdEndRendering(cmd);
  const auto end = cn::steady_clock::now();
  const auto elapsed = cn::duration_cast<cn::milliseconds>(end - start);
  m_stats.imguiDrawTime = elapsed.count() / 1000.0f;
}

void Engine::cull_objects(VkCommandBuffer cmd, const std::uint32_t objectCount,
                          const std::uint32_t objectOffset,
                          const glm::mat4& viewProj) {
  auto& currentFrame = get_current_frame();
  utils::BarrierBuilder barrierBuilder;
  barrierBuilder.add_buffer_barrier({
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .srcAccessMask = 0,
      .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = currentFrame.countBuffer.buffer,
      .offset = 0,
      .size = VK_WHOLE_SIZE,
  });
  barrierBuilder.barrier(cmd);

  vkCmdFillBuffer(cmd, currentFrame.countBuffer.buffer, 0, VK_WHOLE_SIZE, 0);

  barrierBuilder.add_buffer_barrier({
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstAccessMask =
          VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = currentFrame.countBuffer.buffer,
      .offset = 0,
      .size = VK_WHOLE_SIZE,
  });
  barrierBuilder.add_buffer_barrier({
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .srcAccessMask = 0,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstAccessMask =
          VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = currentFrame.drawCommandsBuffer.buffer,
      .offset = 0,
      .size = VK_WHOLE_SIZE,
  });

  barrierBuilder.barrier(cmd);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_CullPassPipeline);

  const CullPassPushConstants cullPassConstants{
      .meshBufferAddr = currentFrame.meshBufferAddr,
      .instanceBufferDeviceAddr = currentFrame.instanceBufferAddr,
      .commandsBufferAddr = currentFrame.drawCommandsBufferAddr,
      .countBufferAddr = currentFrame.countBufferAddr,
      .viewProj = viewProj,
      .objectsCount = objectCount,
      .objectsOffset = objectOffset,
  };
  vkCmdPushConstants(cmd, m_CullPassPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                     0, sizeof(CullPassPushConstants), &cullPassConstants);

  vkCmdDispatch(cmd, std::ceil(objectCount / 64.0f), 1, 1);

  barrierBuilder.add_buffer_barrier({
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .srcAccessMask =
          VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
      .dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = currentFrame.countBuffer.buffer,
      .offset = 0,
      .size = VK_WHOLE_SIZE,
  });
  barrierBuilder.add_buffer_barrier({
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .srcAccessMask =
          VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
      .dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = currentFrame.drawCommandsBuffer.buffer,
      .offset = 0,
      .size = VK_WHOLE_SIZE,
  });

  barrierBuilder.barrier(cmd);
}

void Engine::draw_meshes(VkCommandBuffer cmd,
                         const VkPipelineLayout drawPassPipelineLayout,
                         const VkPipeline drawPipeline,
                         const std::uint32_t objectCount, auto& pushConstants,
                         const VkShaderStageFlags pushConstantsShaderStage,
                         bool isMetalRoughness) {
  auto& currentFrame = get_current_frame();
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, drawPipeline);

  vkCmdPushConstants(cmd, drawPassPipelineLayout, pushConstantsShaderStage, 0,
                     sizeof(pushConstants), &pushConstants);
  if (isMetalRoughness) {
    // Bind textures
    const VkDescriptorBufferBindingInfoEXT bindingInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
        .address = m_metalRoughness.descriptors.get_device_address(),
        .usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                 VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT};
    vkCmdBindDescriptorBuffersEXT(cmd, 1, &bindingInfo);

    const std::uint32_t bufferIndices[]{0};
    const VkDeviceSize offsets[]{0};
    vkCmdSetDescriptorBufferOffsetsEXT(
        cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, drawPassPipelineLayout, 0,
        std::size(bufferIndices), bufferIndices, offsets);
  }

  vkCmdDrawIndexedIndirectCount(cmd, currentFrame.drawCommandsBuffer.buffer, 0,
                                currentFrame.countBuffer.buffer, 0, objectCount,
                                sizeof(VkDrawIndexedIndirectCommand));
  m_stats.drawCallCount++;
}

void Engine::draw_post(VkCommandBuffer cmd) {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    m_PostProcessPassPipeline);

  const VkDescriptorBufferBindingInfoEXT buffersInfo[]{
      {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
       .pNext = nullptr,
       .address =
           get_current_frame().drawImageDescriptorBuffer.get_device_address(),
       .usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT},
  };
  vkCmdBindDescriptorBuffersEXT(cmd, std::size(buffersInfo), buffersInfo);

  const std::uint32_t indices[]{0};
  const VkDeviceSize offsets[]{0};
  vkCmdSetDescriptorBufferOffsetsEXT(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                     m_PostProcessPassPipelineLayout, 0,
                                     std::size(offsets), indices, offsets);

  vkCmdDispatch(cmd, std::ceil(m_CommonImageExtent2D.width / 16.0f),
                std::ceil(m_CommonImageExtent2D.height / 16.0f), 1.0f);
}

void Engine::copy_frame_buffers() {
  m_OpaqueSize =
      static_cast<std::uint32_t>(m_mainDrawContext.opaqueInstances.size());
  const std::uint32_t opaqueByteSize = m_OpaqueSize * sizeof(Instance);
  std::memcpy(m_CurrentFrameInstanceBuffer,
              m_mainDrawContext.opaqueInstances.data(), opaqueByteSize);

  m_TransparentSize =
      static_cast<std::uint32_t>(m_mainDrawContext.transparentInstances.size());
  const std::uint32_t transparentByteSize =
      m_TransparentSize * sizeof(Instance);
  std::memcpy(m_CurrentFrameInstanceBuffer + m_OpaqueSize,
              m_mainDrawContext.transparentInstances.data(),
              transparentByteSize);

  std::memcpy(m_CurrentMeshBuffer, m_mainDrawContext.renderObjects.data(),
              m_mainDrawContext.renderObjects.size() * sizeof(RenderObject));
}

}  // namespace mp
