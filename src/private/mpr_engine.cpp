// clang-format off
#define VMA_IMPLEMENTATION
#define GLM_ENABLE_EXPERIMENTAL
#define VOLK_IMPLEMENTATION
#include "mpr_engine.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <VkBootstrap.h>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>
#include <ImGuizmo.h>
#include <vulkan/utility/vk_format_utils.h>
#include <vulkan/vk_enum_string_helper.h>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <chrono>
#include <format>
#include <numbers>
#include <numeric>
#include <print>
#include <ranges>
#include <thread>

#include "mpr_error_check.hpp"
#include "mpr_image.hpp"
#include "mpr_init_vk_stucts.hpp"
#include "mpr_loader.hpp"
#include "mpr_pipelines.hpp"
// clang-format on

using namespace std::chrono_literals;
namespace rn = std::ranges;
namespace vi = std::views;
namespace cn = std::chrono;

constexpr bool bUseValidationLayers = true;
constexpr auto kBaseWindowTitle = "Hello Vulkan";

#define GPU_USAGE_DISCRETE

namespace {
mp::Engine* gLoadedEngine = nullptr;

std::pair<std::uint32_t, char const* const*>
get_required_instance_extensions_for_window() {
  std::uint32_t count;
  const auto requiredExtensions = SDL_Vulkan_GetInstanceExtensions(&count);
  return {count, requiredExtensions};
}

}  // namespace

namespace mp {

Engine::~Engine() {
  if (m_isInitialized) {
    vkDeviceWaitIdle(m_device);
    m_scene.clear_all(*this);
    for (auto& frame : m_frameData) {
      frame.frameDeletionQueue.flush();
    }
    m_mainDeletionQueue.flush();
    destroy_sync();
    destroy_commands();

    destroy_swapchain();

    vkDestroyDevice(m_device, nullptr);
    vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
    vkb::destroy_debug_utils_messenger(m_instance, m_debugMessenger);
    vkDestroyInstance(m_instance, nullptr);
  }
  gLoadedEngine = nullptr;
}

Engine::Engine() {
  assert(!gLoadedEngine);
  gLoadedEngine = this;
  init_window();
  init_vulkan();
  init_swapchain();
  init_commands();
  init_sync();
  init_descriptors();
  init_pipelines();
  init_imgui();
  init_frames_data();
  init_default_data();
  init_mesh_data();

  m_camera.velocity = glm::vec3(0.0f);
  m_camera.position = glm::vec3(0.0f, 0.0f, 5.0f);

  m_camera.pitch = 0;
  m_camera.yaw = 0;
  m_isInitialized = true;
}

Engine& Engine::get() { return *gLoadedEngine; }

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

  cull_objects(cmd, m_OpaqueSize, 0);
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

  vkCmdBindIndexBuffer(cmd, m_globalIndexBuffer.buffer, 0,
                       VK_INDEX_TYPE_UINT32);
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

  cull_objects(cmd, m_TransparentSize, m_OpaqueSize);
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
                          const std::uint32_t objectOffset) {
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

  vkCmdFillBuffer(cmd, currentFrame.countBuffer.buffer, 0,
                  VK_WHOLE_SIZE, 0);

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
      .instanceBufferDeviceAddr =
          currentFrame.instanceBufferAddr,
      .commandsBufferAddr = currentFrame.drawCommandsBufferAddr,
      .countBufferAddr = currentFrame.countBufferAddr,
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
                         const VkShaderStageFlags pushConstantsShaderStage) {
  auto& currentFrame = get_current_frame();
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, drawPipeline);

  vkCmdPushConstants(cmd, drawPassPipelineLayout, pushConstantsShaderStage, 0,
                     sizeof(pushConstants), &pushConstants);
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
  m_OpaqueSize = static_cast<std::uint32_t>(m_mainDrawContext.opaqueInstances.size());
  const std::uint32_t opaqueByteSize = m_OpaqueSize * sizeof(Instance);
  std::memcpy(m_CurrentFrameInstanceBuffer,
              m_mainDrawContext.opaqueInstances.data(), opaqueByteSize);

  m_TransparentSize = static_cast<std::uint32_t>(m_mainDrawContext.transparentInstances.size());
  const std::uint32_t transparentByteSize = m_TransparentSize * sizeof(Instance);
  std::memcpy(m_CurrentFrameInstanceBuffer + m_OpaqueSize,
              m_mainDrawContext.transparentInstances.data(), transparentByteSize);

  std::memcpy(m_CurrentMeshBuffer, m_mainDrawContext.renderObjects.data(),
              m_mainDrawContext.renderObjects.size() * sizeof(RenderObject));
}

void Engine::init_frames_data() {
  constexpr auto kMaxInstances = 100'000;
  constexpr auto kMaxLights = 10'000;
  constexpr auto kMaxMeshes = 50'000;
  for (auto& frame : m_frameData) {
    frame.sceneDataBuffer =
        create_buffer(sizeof(GpuSceneData),
                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                      VMA_MEMORY_USAGE_CPU_TO_GPU);
    frame.sceneDataBufferAddr =
        frame.sceneDataBuffer.get_buffer_device_address(m_device);

    frame.lightDataBuffer =
        create_buffer(sizeof(LightData) * kMaxLights,
                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                      VMA_MEMORY_USAGE_CPU_TO_GPU);
    frame.lightDataBufferAddr =
        frame.lightDataBuffer.get_buffer_device_address(m_device);

    frame.instanceBuffer =
        create_buffer(sizeof(Instance) * kMaxInstances,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                      VMA_MEMORY_USAGE_CPU_TO_GPU);
    frame.instanceBufferAddr =
        frame.instanceBuffer.get_buffer_device_address(m_device);

    frame.meshesBuffer =
        create_buffer(sizeof(RenderObject) * kMaxMeshes,
                      VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT,
                      VMA_MEMORY_USAGE_CPU_TO_GPU);

    frame.meshBufferAddr =
        frame.meshesBuffer.get_buffer_device_address(m_device);
    frame.drawCommandsBuffer =
        create_buffer(sizeof(VkDrawIndexedIndirectCommand) * kMaxInstances,
                      VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT |
                          VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT,
                      VMA_MEMORY_USAGE_GPU_ONLY);
    frame.drawCommandsBufferAddr =
        frame.drawCommandsBuffer.get_buffer_device_address(m_device);

    frame.countBuffer =
        create_buffer(sizeof(std::uint32_t),
                      VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT |
                          VK_BUFFER_USAGE_2_TRANSFER_DST_BIT |
                              VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT,
                      VMA_MEMORY_USAGE_GPU_ONLY);
    frame.countBufferAddr =
        frame.countBuffer.get_buffer_device_address(m_device);
  }

  m_mainDeletionQueue.push_function([this] {
    for (auto& frame : m_frameData) {
      destroy_buffer(frame.instanceBuffer);
      destroy_buffer(frame.drawCommandsBuffer);
      destroy_buffer(frame.meshesBuffer);
      destroy_buffer(frame.countBuffer);
      destroy_buffer(frame.sceneDataBuffer);
      destroy_buffer(frame.lightDataBuffer);
    }
  });
}

std::uint64_t Engine::render_scene_tree_ui(Scene& scene,
                                           std::uint64_t nodeIndex,
                                           std::uint64_t selectedNode) {
  const bool isLeaf = scene.nodes.at(nodeIndex)->children.empty();
  ImGuiTreeNodeFlags flags =
      isLeaf ? ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_Bullet : 0;
  if (nodeIndex == selectedNode) {
    flags |= ImGuiTreeNodeFlags_Selected;
  }

  const ImVec4 color =
      isLeaf ? ImVec4(0.7f, 0.7f, 0.2f, 1.0f) : ImVec4(0.2f, 0.7f, 0.7f, 1.0f);

  ImGui::PushStyleColor(ImGuiCol_Text, color);
  const bool isOpened =
      ImGui::TreeNodeEx(&scene.nodes.at(nodeIndex), flags, "%s",
                        scene.nodes[nodeIndex]->name.c_str());
  ImGui::PopStyleColor();

  ImGui::PushID(nodeIndex);
  if (ImGui::IsItemClicked() && isLeaf) {
    std::println("Selected Node: {}", nodeIndex);
    selectedNode = nodeIndex;
  }

  if (isOpened) {
    for (const auto& child : scene.nodes.at(nodeIndex)->children) {
      if (const auto subNode =
              render_scene_tree_ui(scene, child->nodeIndex, selectedNode);
          subNode != UINT64_MAX) {
        selectedNode = subNode;
      }
    }
    ImGui::TreePop();
  }
  ImGui::PopID();

  return selectedNode;
}

bool Engine::edit_transform_ui(const glm::mat4& view,
                               const glm::mat4& projection,
                               glm::mat4& globalTransform) {
  static ImGuizmo::OPERATION gizmoOperation(ImGuizmo::TRANSLATE);

  ImGui::Text("Transforms:");

  if (ImGui::IsKeyPressed(ImGuiKey_W)) gizmoOperation = ImGuizmo::TRANSLATE;
  if (ImGui::IsKeyPressed(ImGuiKey_E)) gizmoOperation = ImGuizmo::ROTATE;
  if (ImGui::IsKeyPressed(ImGuiKey_R)) gizmoOperation = ImGuizmo::SCALE;

  if (ImGui::RadioButton("Translate", gizmoOperation == ImGuizmo::TRANSLATE))
    gizmoOperation = ImGuizmo::TRANSLATE;

  if (ImGui::RadioButton("Rotate", gizmoOperation == ImGuizmo::ROTATE))
    gizmoOperation = ImGuizmo::ROTATE;

  if (ImGui::RadioButton("Scale", gizmoOperation == ImGuizmo::SCALE))
    gizmoOperation = ImGuizmo::SCALE;

  float matrixTranslation[3], matrixRotation[3], matrixScale[3];
  ImGuizmo::DecomposeMatrixToComponents(glm::value_ptr(globalTransform),
                                        matrixTranslation, matrixRotation,
                                        matrixScale);
  ImGui::InputFloat3("Tr", matrixTranslation);
  ImGui::InputFloat3("Rt", matrixRotation);
  ImGui::InputFloat3("Sc", matrixScale);
  ImGuizmo::RecomposeMatrixFromComponents(matrixTranslation, matrixRotation,
                                          matrixScale,
                                          glm::value_ptr(globalTransform));

  const ImGuiIO& io = ImGui::GetIO();
  ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
  return ImGuizmo::Manipulate(glm::value_ptr(view), glm::value_ptr(projection),
                              gizmoOperation, ImGuizmo::WORLD,
                              glm::value_ptr(globalTransform));
}

void Engine::edit_node(Scene& scene, const std::uint64_t nodeIndex) {
  ImGuizmo::SetOrthographic(false);
  ImGuizmo::BeginFrame();

  auto& node = scene.nodes[nodeIndex];

  const auto& name = node->name;
  std::string label =
      name.empty() ? (std::string("Node") + std::to_string(nodeIndex)) : name;
  label = "Node: " + label;

  if (const ImGuiViewport* v = ImGui::GetMainViewport()) {
    ImGui::SetNextWindowPos(ImVec2(v->WorkSize.x * 0.83f, 200));
    ImGui::SetNextWindowSize(ImVec2(v->WorkSize.x / 6, v->WorkSize.y - 210));
  }
  ImGui::Begin("Editor", nullptr,
               ImGuiWindowFlags_NoFocusOnAppearing |
                   ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize);
  if (!name.empty()) ImGui::Text("%s", label.c_str());

  ImGui::Separator();
  ImGuizmo::PushID(1);

  auto& globalTransform = node->worldTransform;
  if (edit_transform_ui(m_sceneData.view, m_sceneData.proj, globalTransform)) {
    if (const auto parent = node->parent.lock(); parent) {
      const glm::mat4 parentWorldTransform = parent->worldTransform;
      node->localTransform =
          glm::inverse(parentWorldTransform) * globalTransform;
    } else {
      node->localTransform = globalTransform;
    }
  }

  node->edit();

  ImGui::Separator();
#if 0
    ImGui::Text("%s", "Material");

    editMaterialUI(scene, meshData, node, outUpdateMaterialIndex, textureCache);
#endif
  ImGuizmo::PopID();
  ImGui::End();
}

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

  const VkImageAspectFlags aspectFlags = format == VK_FORMAT_D32_SFLOAT
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
      static_cast<char*>(stagingBuffer.allocation->GetMappedData());
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

void Engine::run() {
  SDL_Event e;
  bool bIsRunning = true;

  while (bIsRunning) {
    auto start = cn::steady_clock::now();
    while (SDL_PollEvent(&e)) {
      if (e.type == SDL_EVENT_QUIT) {
        bIsRunning = false;
      }

      if (e.type == SDL_EVENT_WINDOW_MINIMIZED) {
        m_isRenderStopped = true;
      }

      if (e.type == SDL_EVENT_WINDOW_MAXIMIZED) {
        m_isRenderStopped = false;
      }
      m_camera.process_sdl_event(e, m_window.get());
      ImGui_ImplSDL3_ProcessEvent(&e);

      if (ImGui::GetIO().WantCaptureMouse || ImGui::GetIO().WantCaptureKeyboard)
        continue;
    }

    if (m_bSwapchainResizeRequest) {
      resize_swapchain();
    }

    if (m_isRenderStopped) {
      std::this_thread::sleep_for(100ms);
      continue;
    }

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();

    // ImGui UI
    if (ImGui::Begin("Other")) {
      ImGui::DragFloat("Render scale", &m_renderScale, 0.01f, 0.01f, 1.0f);
      ImGui::DragFloat("Camera speed", &m_camera.cameraSpeed, 0.01f, 0.01f,
                       100.0f);
      // TODO: Add debug light visualization
#if 0
      ImGui::Checkbox("Draw debug light positions", &m_IsLightsRendered);
#endif
    }
    ImGui::End();

    ImGui::Begin("Stats");
    ImGui::Text("Frame time: %f ms", m_stats.frameTime);
    ImGui::Text("GBuffer Pass time: %f ms", m_stats.gBufferPassTime);
    ImGui::Text("Deferred light pass time: %f ms",
                m_stats.gBufferLightPassTime);
    ImGui::Text("WBOIT forward pass time: %f ms",
                m_stats.transparentForwardLightPassTime);
    ImGui::Text("Post process pass time: %f ms", m_stats.postProcessPassTime);
    ImGui::Text("ImGui draw time: %f ms", m_stats.imguiDrawTime);
    ImGui::Text("Scene update tim: %f ms", m_stats.sceneUpdateTime);
    ImGui::Text("Amount of draw calls: %i", m_stats.drawCallCount);
    ImGui::Text("Amount of triangles: %i", m_stats.triangleCount);
    ImGui::End();

    {
      const ImGuiViewport* v = ImGui::GetMainViewport();
      ImGui::SetNextWindowPos(ImVec2(10, 200));
      ImGui::SetNextWindowSize(ImVec2(v->WorkSize.x / 6, v->WorkSize.y - 210));
      ImGui::Begin("Scene graph", nullptr,
                   ImGuiWindowFlags_NoFocusOnAppearing |
                       ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize);
      ImGui::Separator();
      for (const auto& topNode : m_scene.topNodes) {
        m_selectedNode =
            render_scene_tree_ui(m_scene, topNode->nodeIndex, m_selectedNode);
      }
      ImGui::Separator();
      if (ImGui::Button("Add light")) {
        const auto nodeIndex = m_scene.add_node(std::make_shared<LightNode>(
            LightData{.lightType = 0,
                      .padding0 = 0,
                      .data0 = {0.0f, 0.0f, 0.0f, 1.0f},
                      .data1 = {1.0f, 1.0f, 1.0f, 1.0f}}));

        auto& node = m_scene.nodes.find(nodeIndex)->second;
        node->worldTransform = glm::mat4(1.0f);
        node->localTransform = glm::mat4(1.0f);
        node->name = "Light";
        node->nodeIndex = nodeIndex;
        m_scene.topNodes.push_back(node);
      }
      ImGui::End();
    }

    if (m_selectedNode != UINT64_MAX) {
      edit_node(m_scene, m_selectedNode);
    }
    // ImGui UI end

    ImGui::Render();

    draw();

    auto end = cn::steady_clock::now();
    auto elapsed = cn::duration_cast<cn::milliseconds>(end - start);
    m_stats.frameTime = elapsed.count() / 1000.0f;
  }
}

FrameData& Engine::get_current_frame() {
  return m_frameData[m_frameNumber % kNumberOfFrames];
}

void Engine::init_window() {
  if (!SDL_Init(SDL_INIT_VIDEO)) {
    std::println("Failed to init SDL: {}", SDL_GetError());
  }
  atexit(SDL_Quit);

  constexpr SDL_WindowFlags windowFlags =
      SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE;

  m_window = {SDL_CreateWindow(kBaseWindowTitle, m_windowExtent.width,
                               m_windowExtent.height, windowFlags),
              WindowCleaner{}};
  if (!m_window) {
    std::println("Failed to create window: {}", SDL_GetError());
  }
}

void Engine::init_vulkan() {
  volkInitialize() >> chk;
  const auto [numberOfRequiredExtensions, requiredExtensions] =
      get_required_instance_extensions_for_window();
  const auto result =
      vkb::InstanceBuilder()
          .request_validation_layers(bUseValidationLayers)
          .use_default_debug_messenger()
          .require_api_version(1, 3, 0)
          .enable_extensions(numberOfRequiredExtensions, requiredExtensions)
          .build();

  if (!result.has_value()) {
    throw std::runtime_error("Failed to create instance");
  }
  m_instance = result.value().instance;
  m_debugMessenger = result.value().debug_messenger;

  volkLoadInstance(m_instance);

  if (!SDL_Vulkan_CreateSurface(m_window.get(), m_instance, nullptr,
                                &m_surface)) {
    throw std::runtime_error(
        std::format("Failed to create surface: {}", SDL_GetError()));
  }

  VkPhysicalDeviceDescriptorBufferFeaturesEXT descriptorBufferFeatures{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT,
      .pNext = nullptr,
      .descriptorBuffer = true,
#if 0
    .descriptorBufferImageLayoutIgnored = true,
#endif
  };

  const VkPhysicalDeviceVulkan13Features features13{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .synchronization2 = true,
      .dynamicRendering = true,
  };

  const VkPhysicalDeviceVulkan12Features features12{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &descriptorBufferFeatures,
      .drawIndirectCount = true,
      .descriptorIndexing = true,
      .shaderSampledImageArrayNonUniformIndexing = true,
      .descriptorBindingPartiallyBound = true,
      .descriptorBindingVariableDescriptorCount = true,
      .runtimeDescriptorArray = true,
      .scalarBlockLayout = true,
      .bufferDeviceAddress = true,
  };

  VkPhysicalDeviceFeatures features10{
      .independentBlend = true,
      .samplerAnisotropy = true,
  };

  vkb::PhysicalDeviceSelector selector{result.value()};

  const auto physicalDevice =
      selector.set_minimum_version(1, 3)
          .add_required_extension(VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME)
          .set_required_features_13(features13)
          .set_required_features_12(features12)
          .set_required_features(features10)
          .set_surface(m_surface)
#ifdef GPU_USAGE_DISCRETE
          .allow_any_gpu_device_type(false)
          .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
#endif
          .select()
          .value();

  vkb::DeviceBuilder deviceBuilder{physicalDevice};

  vkb::Device vkbDevice = deviceBuilder.build().value();

  m_device = vkbDevice.device;
  m_chosenGpu = vkbDevice.physical_device;
  std::println("Physical GPU: {}", vkbDevice.physical_device.name);

  m_queue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
  m_queueFamilyIndex =
      vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

  volkLoadDevice(m_device);

  VmaVulkanFunctions vulkanFunc[]{vkGetInstanceProcAddr, vkGetDeviceProcAddr};
  VmaAllocatorCreateInfo allocatorCreateInfo{
      .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
      .physicalDevice = m_chosenGpu,
      .device = m_device,
      .pVulkanFunctions = vulkanFunc,
      .instance = m_instance,
  };

  vmaCreateAllocator(&allocatorCreateInfo, &m_allocator) >> chk;

  m_mainDeletionQueue.push_function(
      [&]() { vmaDestroyAllocator(m_allocator); });
}

void Engine::create_draw_image(AllocatedImage& drawImage,
                               const VkExtent3D extent) {
  drawImage.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
  drawImage.imageExtent = extent;

  VkImageUsageFlags drawImageUsages{};
  drawImageUsages |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  drawImageUsages |= VK_IMAGE_USAGE_STORAGE_BIT;
  const VkImageCreateInfo imageCreateInfo =
      utils::image_create_info(drawImage.imageFormat, drawImageUsages, extent);
  constexpr VmaAllocationCreateInfo allocationCreateInfo{
      .usage = VMA_MEMORY_USAGE_GPU_ONLY,
      .requiredFlags = static_cast<VkMemoryPropertyFlags>(
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
  vmaCreateImage(m_allocator, &imageCreateInfo, &allocationCreateInfo,
                 &drawImage.image, &drawImage.allocation, nullptr) >>
      chk;

  const VkImageViewCreateInfo imageViewCreateInfo =
      utils::image_view_create_info(drawImage.imageFormat, drawImage.image,
                                    VK_IMAGE_ASPECT_COLOR_BIT);
  vkCreateImageView(m_device, &imageViewCreateInfo, nullptr,
                    &drawImage.imageView) >>
      chk;

  m_mainDeletionQueue.push_function([&] {
    vkDestroyImageView(m_device, drawImage.imageView, nullptr);
    vmaDestroyImage(m_allocator, drawImage.image, drawImage.allocation);
  });
}

void Engine::create_depth_image(AllocatedImage& depthImage,
                                const VkExtent3D extent) {
  depthImage.imageFormat = VK_FORMAT_D32_SFLOAT;
  depthImage.imageExtent = extent;

  constexpr VkImageUsageFlags imageUsages =
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  const VkImageCreateInfo imageCreateInfo =
      utils::image_create_info(depthImage.imageFormat, imageUsages, extent);
  constexpr VmaAllocationCreateInfo allocationCreateInfo{
      .usage = VMA_MEMORY_USAGE_GPU_ONLY,
      .requiredFlags = static_cast<VkMemoryPropertyFlags>(
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
  vmaCreateImage(m_allocator, &imageCreateInfo, &allocationCreateInfo,
                 &depthImage.image, &depthImage.allocation, nullptr) >>
      chk;

  const VkImageViewCreateInfo imageViewCreateInfo =
      utils::image_view_create_info(depthImage.imageFormat, depthImage.image,
                                    VK_IMAGE_ASPECT_DEPTH_BIT);
  vkCreateImageView(m_device, &imageViewCreateInfo, nullptr,
                    &depthImage.imageView) >>
      chk;

  m_mainDeletionQueue.push_function([&] {
    vkDestroyImageView(m_device, depthImage.imageView, nullptr);
    vmaDestroyImage(m_allocator, depthImage.image, depthImage.allocation);
  });
}

void Engine::init_swapchain() {
  create_swapchain(m_windowExtent.width, m_windowExtent.height);
  m_CommonImageExtent3D = {
      .width = m_windowExtent.width,
      .height = m_windowExtent.height,
      .depth = 1,
  };

  m_CommonImageExtent2D = {
      .width = m_windowExtent.width,
      .height = m_windowExtent.height,
  };
  for (auto& frame : m_frameData) {
    create_draw_image(frame.drawImage, m_CommonImageExtent3D);
    create_depth_image(frame.depthImage, m_CommonImageExtent3D);

    frame.gBuffer.position = create_image(
        m_CommonImageExtent3D, VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    frame.gBuffer.normal = create_image(
        m_CommonImageExtent3D, VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    frame.gBuffer.diffuse = create_image(
        m_CommonImageExtent3D, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    frame.gBuffer.specular = create_image(
        m_CommonImageExtent3D, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

    frame.oitAccImage = create_image(
        m_CommonImageExtent3D, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    frame.oitRevealImage = create_image(
        m_CommonImageExtent3D, VK_FORMAT_R16_SFLOAT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
  }

  m_mainDeletionQueue.push_function([this]() {
    for (auto& frame : m_frameData) {
      destroy_image(frame.gBuffer.position);
      destroy_image(frame.gBuffer.normal);
      destroy_image(frame.gBuffer.diffuse);
      destroy_image(frame.gBuffer.specular);

      destroy_image(frame.oitAccImage);
      destroy_image(frame.oitRevealImage);
    }
  });
}

void Engine::init_commands() {
  const VkCommandPoolCreateInfo commandPoolCreateInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
               VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = m_queueFamilyIndex};
  vkCreateCommandPool(m_device, &commandPoolCreateInfo, nullptr,
                      &m_commandPool) >>
      chk;
  vkCreateCommandPool(m_device, &commandPoolCreateInfo, nullptr,
                      &m_immCommandPool);
  for (auto& frame : m_frameData) {
    const VkCommandBufferAllocateInfo allocateInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = m_commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    vkAllocateCommandBuffers(m_device, &allocateInfo, &frame.commandBuffer) >>
        chk;
  }
  const VkCommandBufferAllocateInfo allocateInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .pNext = nullptr,
      .commandPool = m_immCommandPool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };
  vkAllocateCommandBuffers(m_device, &allocateInfo, &m_immCommandBuffer) >> chk;

  m_mainDeletionQueue.push_function(
      [&] { vkDestroyCommandPool(m_device, m_immCommandPool, nullptr); });
}

void Engine::init_sync() {
  constexpr VkFenceCreateInfo fenceCreateInfo{
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      .flags = VK_FENCE_CREATE_SIGNALED_BIT};
  constexpr VkSemaphoreCreateInfo semaphoreCreateInfo{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  for (auto& frame : m_frameData) {
    vkCreateFence(m_device, &fenceCreateInfo, nullptr, &frame.fence) >> chk;

    vkCreateSemaphore(m_device, &semaphoreCreateInfo, nullptr,
                      &frame.swapchainSemaphore) >>
        chk;
  }

  m_swapchainSemaphores.resize(m_swapchainImages.size());
  for (auto& renderSemaphore : m_swapchainSemaphores)
    vkCreateSemaphore(m_device, &semaphoreCreateInfo, nullptr,
                      &renderSemaphore) >>
        chk;

  vkCreateFence(m_device, &fenceCreateInfo, nullptr, &m_immFence) >> chk;

  m_mainDeletionQueue.push_function(
      [&] { vkDestroyFence(m_device, m_immFence, nullptr); });
}

void Engine::init_descriptors() {
  {
    m_DrawImageDescriptorSetLayout =
        DescriptorSetLayoutBuilder()
            .add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                         VK_SHADER_STAGE_COMPUTE_BIT)
            .build(m_device,
                   VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT);
    m_LightPassDescriptorSetLayout =
        DescriptorSetLayoutBuilder()
            .add_binding(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1,
                         VK_SHADER_STAGE_COMPUTE_BIT)
            .add_binding(1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1,
                         VK_SHADER_STAGE_COMPUTE_BIT)
            .add_binding(2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1,
                         VK_SHADER_STAGE_COMPUTE_BIT)
            .add_binding(3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1,
                         VK_SHADER_STAGE_COMPUTE_BIT)
            .build(m_device,
                   VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT);
  }
  for (auto& frame : m_frameData) {
    frame.lightPassDescriptorBuffer =
        DescriptorBuffer(m_device, m_LightPassDescriptorSetLayout,
                         DescriptorBufferProperties::query(m_chosenGpu));

    frame.lightPassDescriptorBuffer.create_buffer(
        [&](const std::size_t allocSize, const VkBufferUsageFlags bufferUsage) {
          return create_buffer(allocSize, bufferUsage,
                               VMA_MEMORY_USAGE_CPU_ONLY);
        });

    frame.drawImageDescriptorBuffer =
        DescriptorBuffer(m_device, m_DrawImageDescriptorSetLayout,
                         DescriptorBufferProperties::query(m_chosenGpu));

    frame.drawImageDescriptorBuffer.create_buffer(
        [&](const std::size_t allocSize, const VkBufferUsageFlags bufferUsage) {
          return create_buffer(allocSize, bufferUsage,
                               VMA_MEMORY_USAGE_CPU_ONLY);
        });

    frame.drawImageDescriptorBuffer.write_storage_image(
        0, 0, frame.drawImage.imageView, VK_IMAGE_LAYOUT_GENERAL);
    frame.lightPassDescriptorBuffer.write_sampled_image(
        0, 0, frame.gBuffer.position.imageView,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    frame.lightPassDescriptorBuffer.write_sampled_image(
        1, 0, frame.gBuffer.normal.imageView,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    frame.lightPassDescriptorBuffer.write_sampled_image(
        2, 0, frame.gBuffer.diffuse.imageView,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    frame.lightPassDescriptorBuffer.write_sampled_image(
        3, 0, frame.gBuffer.specular.imageView,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }

  m_mainDeletionQueue.push_function([&]() mutable {
    vkDestroyDescriptorSetLayout(m_device, m_LightPassDescriptorSetLayout,
                                 nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_DrawImageDescriptorSetLayout,
                                 nullptr);
    for (auto& frame : m_frameData) {
      destroy_buffer(frame.lightPassDescriptorBuffer.get_buffer());
      destroy_buffer(frame.drawImageDescriptorBuffer.get_buffer());
    }
  });
}

void Engine::init_pipelines() {
  init_light_pass_pipeline();
  init_cull_pipeline();
  init_wboit_composite_pass_pipeline();
  init_post_pipeline();
  m_metalRoughness.build_pipelines(*this);
}

void Engine::init_light_pass_pipeline() {
  constexpr VkPushConstantRange pushConstantRange{
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = static_cast<std::uint32_t>(sizeof(LightPassConstantRange)),
  };

  VkDescriptorSetLayout setLayouts[]{m_DrawImageDescriptorSetLayout,
                                     m_LightPassDescriptorSetLayout};
  const VkPipelineLayoutCreateInfo layoutCreateInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .setLayoutCount = std::size(setLayouts),
      .pSetLayouts = setLayouts,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pushConstantRange,
  };
  vkCreatePipelineLayout(m_device, &layoutCreateInfo, nullptr,
                         &m_LightPassPipelineLayout) >>
      chk;

  VkShaderModule lightPassShader;
  if (!load_shader_module("../../src/compiled_shaders/light_pass.compute.spv",
                          m_device, &lightPassShader)) {
    throw std::runtime_error("Failed to load a light pass shader");
  }

  const VkPipelineShaderStageCreateInfo shaderStage{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .pNext = nullptr,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = lightPassShader,
      .pName = "main",
  };

  const VkComputePipelineCreateInfo createInfo{
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .pNext = nullptr,
      .flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT,
      .stage = shaderStage,
      .layout = m_LightPassPipelineLayout,
  };

  vkCreateComputePipelines(m_device, nullptr, 1, &createInfo, nullptr,
                           &m_LightPassPipeline) >>
      chk;

  vkDestroyShaderModule(m_device, lightPassShader, nullptr);
  m_mainDeletionQueue.push_function([this] {
    vkDestroyPipeline(m_device, m_LightPassPipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_LightPassPipelineLayout, nullptr);
  });
}

void Engine::init_wboit_composite_pass_pipeline() {
  VkShaderModule compositeVertexShader;
  if (!load_shader_module(
          "../../src/compiled_shaders/wboit_composite.vertex.spv", m_device,
          &compositeVertexShader)) {
    throw std::runtime_error("Failed to load wboit_composite.vertex.spv");
  }
  VkShaderModule compositeFragmentShader;
  if (!load_shader_module(
          "../../src/compiled_shaders/wboit_composite.pixel.spv", m_device,
          &compositeFragmentShader)) {
    throw std::runtime_error("Failed to load wboit_composite.pixel.spv");
  }

  {
    m_WboitCompositePassDescriptorSetLayout =
        DescriptorSetLayoutBuilder()
            .add_binding(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1,
                         VK_SHADER_STAGE_FRAGMENT_BIT)
            .add_binding(1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1,
                         VK_SHADER_STAGE_FRAGMENT_BIT)
            .build(m_device,
                   VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT);
  }
  for (auto& frame : m_frameData) {
    frame.wboitCompositePassDescBuffer =
        DescriptorBuffer(m_device, m_WboitCompositePassDescriptorSetLayout,
                         DescriptorBufferProperties::query(m_chosenGpu));
    frame.wboitCompositePassDescBuffer.create_buffer(
        [this](const std::size_t allocSize,
               const VkBufferUsageFlags bufferUsage) {
          return create_buffer(allocSize, bufferUsage,
                               VMA_MEMORY_USAGE_CPU_ONLY);
        });

    frame.wboitCompositePassDescBuffer.write_sampled_image(
        0, 0, frame.oitAccImage.imageView,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    frame.wboitCompositePassDescBuffer.write_sampled_image(
        1, 0, frame.oitRevealImage.imageView,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }
  const VkDescriptorSetLayout layouts[]{
      m_WboitCompositePassDescriptorSetLayout};

  {
    const VkPipelineLayoutCreateInfo layoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = std::size(layouts),
        .pSetLayouts = layouts,

    };
    vkCreatePipelineLayout(m_device, &layoutCreateInfo, nullptr,
                           &m_WBOITCompositePassPipelineLayout) >>
        chk;
  }

  {
    PipelineBuilder pipelineBuilder;
    pipelineBuilder.pipelineLayout = m_WBOITCompositePassPipelineLayout;
    pipelineBuilder.add_shader(compositeVertexShader,
                               VK_SHADER_STAGE_VERTEX_BIT);
    pipelineBuilder.add_shader(compositeFragmentShader,
                               VK_SHADER_STAGE_FRAGMENT_BIT);
    pipelineBuilder.disable_depth_test();
    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.add_color_attachment_format(
        m_frameData.at(0).drawImage.imageFormat);
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_BACK_BIT,
                                  VK_FRONT_FACE_COUNTER_CLOCKWISE);
    pipelineBuilder.set_multisampling_none();
    pipelineBuilder.colorBlends.push_back(
        {.blendEnable = VK_TRUE,
         .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
         .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
         .colorBlendOp = VK_BLEND_OP_ADD,
         .srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
         .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
         .alphaBlendOp = VK_BLEND_OP_ADD,
         .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT |
                           VK_COLOR_COMPONENT_A_BIT});
    m_WBOITCompositePassPipeline = pipelineBuilder.build_pipeline(
        m_device, VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT);
  }

  vkDestroyShaderModule(m_device, compositeVertexShader, nullptr);
  vkDestroyShaderModule(m_device, compositeFragmentShader, nullptr);

  m_mainDeletionQueue.push_function([this] {
    vkDestroyPipeline(m_device, m_WBOITCompositePassPipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_WBOITCompositePassPipelineLayout,
                            nullptr);

    vkDestroyDescriptorSetLayout(
        m_device, m_WboitCompositePassDescriptorSetLayout, nullptr);

    for (auto& frame : m_frameData) {
      destroy_buffer(frame.wboitCompositePassDescBuffer.get_buffer());
    }
  });
}

void Engine::init_post_pipeline() {
  const VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pNext = nullptr,
      .setLayoutCount = 1,
      .pSetLayouts = &m_DrawImageDescriptorSetLayout,
      .pushConstantRangeCount = 0,
      .pPushConstantRanges = nullptr,
  };
  vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr,
                         &m_PostProcessPassPipelineLayout) >>
      chk;

  VkShaderModule postprocessShader;
  if (!load_shader_module("../../src/compiled_shaders/postprocess.compute.spv",
                          m_device, &postprocessShader)) {
    throw std::runtime_error("Failed to load postprocess pass shader");
  }

  const VkPipelineShaderStageCreateInfo shaderStage{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .pNext = nullptr,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = postprocessShader,
      .pName = "main",
  };
  const VkComputePipelineCreateInfo pipelineCreateInfo{
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .pNext = nullptr,
      .flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT,
      .stage = shaderStage,
      .layout = m_PostProcessPassPipelineLayout,
  };
  vkCreateComputePipelines(m_device, nullptr, 1, &pipelineCreateInfo, nullptr,
                           &m_PostProcessPassPipeline);

  vkDestroyShaderModule(m_device, postprocessShader, nullptr);

  m_mainDeletionQueue.push_function([this] {
    vkDestroyPipeline(m_device, m_PostProcessPassPipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_PostProcessPassPipelineLayout, nullptr);
  });
}

void Engine::init_cull_pipeline() {
#if 0
  m_CullPassDescriptorSetLayout =
      DescriptorSetLayoutBuilder()
          .build(m_device,
                 VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT);
#endif

  const VkPushConstantRange constantRange{
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = sizeof(CullPassPushConstants),
  };

  const VkPipelineLayoutCreateInfo layoutCreateInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pNext = nullptr,
      .setLayoutCount = 0,
      .pSetLayouts = nullptr,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &constantRange,
  };

  vkCreatePipelineLayout(m_device, &layoutCreateInfo, nullptr,
                         &m_CullPassPipelineLayout) >>
      chk;

  VkShaderModule cullShader;
  if (!load_shader_module("../../src/compiled_shaders/cull.compute.spv",
                          m_device, &cullShader)) {
    throw std::runtime_error("Failed to load cull shader");
  }

  const VkPipelineShaderStageCreateInfo shaderStage{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .pNext = nullptr,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = cullShader,
      .pName = "main",
  };

  const VkComputePipelineCreateInfo pipelineCreateInfo{
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .pNext = nullptr,
      .stage = shaderStage,
      .layout = m_CullPassPipelineLayout,
  };

  vkCreateComputePipelines(m_device, nullptr, 1, &pipelineCreateInfo, nullptr,
                           &m_CullPassPipeline) >>
      chk;

  vkDestroyShaderModule(m_device, cullShader, nullptr);

  m_mainDeletionQueue.push_function([this] {
    vkDestroyPipeline(m_device, m_CullPassPipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_CullPassPipelineLayout, nullptr);
  });
}

void Engine::init_imgui() {
  // 1: create descriptor pool for IMGUI
  //  the size of the pool is very oversize, but it's copied from imgui demo
  //  itself.
  const VkDescriptorPoolSize poolSizes[] = {
      {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
      {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}};

  VkDescriptorPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  poolInfo.maxSets = 1000;
  poolInfo.poolSizeCount = static_cast<uint32_t>(std::size(poolSizes));
  poolInfo.pPoolSizes = poolSizes;

  VkDescriptorPool imguiPool;
  vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &imguiPool) >> chk;

  // this initializes the core structures of imgui
  ImGui::CreateContext();

  // this initializes imgui for SDL
  ImGui_ImplSDL3_InitForVulkan(m_window.get());

  ImGui_ImplVulkan_InitInfo initInfo = {};
  initInfo.Instance = m_instance;
  initInfo.PhysicalDevice = m_chosenGpu;
  initInfo.Device = m_device;
  initInfo.Queue = m_queue;
  initInfo.DescriptorPool = imguiPool;
  initInfo.MinImageCount = m_swapchainImages.size();
  initInfo.ImageCount = m_swapchainImages.size();
  initInfo.UseDynamicRendering = true;
  initInfo.ApiVersion = VK_API_VERSION_1_4;
  initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
  initInfo.PipelineInfoMain.PipelineRenderingCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &m_swapchainImageFormat,
  };

  ImGui_ImplVulkan_Init(&initInfo);

  m_mainDeletionQueue.push_function([&, imguiPool] {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    vkDestroyDescriptorPool(m_device, imguiPool, nullptr);
  });
}

void Engine::ensure_vertex_capacity(std::size_t additionalCount) {
  if (m_globalVertexCount + additionalCount <= m_globalVertexCapacity) {
    return;
  }

  std::size_t newCapacity =
      m_globalVertexCapacity == 0 ? 1024 : m_globalVertexCapacity * 2;
  while (m_globalVertexCount + additionalCount > newCapacity) {
    newCapacity *= 2;
  }

  const std::size_t newSize = newCapacity * sizeof(Vertex);
  const AllocatedBuffer newBuffer = create_buffer(
      newSize,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      VMA_MEMORY_USAGE_GPU_ONLY);

  if (m_globalVertexCount > 0) {
    immediate_submit([&](VkCommandBuffer cmd) {
      const VkBufferCopy copyRegion{
          .srcOffset = 0,
          .dstOffset = 0,
          .size = m_globalVertexCount * sizeof(Vertex),
      };
      vkCmdCopyBuffer(cmd, m_globalVertexBuffer.buffer, newBuffer.buffer, 1,
                      &copyRegion);
    });
    destroy_buffer(m_globalVertexBuffer);
  } else if (m_globalVertexCapacity > 0) {
    destroy_buffer(m_globalVertexBuffer);
  }

  m_globalVertexBuffer = newBuffer;
  m_globalVertexCapacity = newCapacity;

  const VkBufferDeviceAddressInfo addrInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
      .buffer = m_globalVertexBuffer.buffer,
  };
  m_globalVertexBufferAddress = vkGetBufferDeviceAddress(m_device, &addrInfo);
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
  m_globalIndexCapacity = newCapacity;
}

void Engine::init_mesh_data() {
#if 0
   const std::string sponzaPath =
      "../../assets/gltf-samples/Models/AlphaBlendModeTest/glTF/AlphaBlendModeTest.gltf";
#else
  const std::string sponzaPath =
      "../../assets/gltf-samples/Models/Sponza/glTF/sponza.gltf";
#endif

  ensure_vertex_capacity(1024);  // Initial capacity
  ensure_index_capacity(1024);

  if (!load_gltf(*this, sponzaPath)) {
    throw std::runtime_error("Failed to load glTF file: " + sponzaPath);
  }

  const std::string alphaBlendMode =
      "../../assets/gltf-samples/Models/AlphaBlendModeTest/glTF/"
      "AlphaBlendModeTest.gltf";
  if (!load_gltf(*this, alphaBlendMode)) {
    throw std::runtime_error("Failed to load glTF file: " + alphaBlendMode);
  }

  m_mainDeletionQueue.push_function([this] {
    destroy_buffer(m_globalVertexBuffer);
    destroy_buffer(m_globalIndexBuffer);
  });
}

void Engine::init_default_data() {
  std::uint32_t whiteColor =
      glm::packUnorm4x8(glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
  m_whiteImage =
      create_image(&whiteColor, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM,
                   VK_IMAGE_USAGE_SAMPLED_BIT);
  std::uint32_t blackColor =
      glm::packUnorm4x8(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
  m_blackImage =
      create_image(&blackColor, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM,
                   VK_IMAGE_USAGE_SAMPLED_BIT);
  std::uint32_t greyColor =
      glm::packUnorm4x8(glm::vec4(0.5f, 0.5f, 0.5f, 1.0f));
  m_greyImage =
      create_image(&greyColor, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM,
                   VK_IMAGE_USAGE_SAMPLED_BIT);
  std::uint32_t normalFallback =
      glm::packUnorm4x8(glm::vec4(0.5f, 0.5f, 1.0f, 1.0f));
  m_normalFallback =
      create_image(&normalFallback, VkExtent3D{1, 1, 1},
                   VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

  const std::uint32_t magentaColor =
      glm::packUnorm4x8(glm::vec4(1.0f, 0.0f, 1.0f, 1.0f));
  std::array<std::uint32_t, 16 * 16> errorPixels;
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      errorPixels[i * 16 + j] = ((i % 2) ^ (j % 2)) ? magentaColor : blackColor;
    }
  }
  m_errorImage =
      create_image(errorPixels.data(), VkExtent3D{16, 16, 1},
                   VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

  VkSamplerCreateInfo samplerCreateInfo{
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_LINEAR,
      .minFilter = VK_FILTER_LINEAR,
  };
  vkCreateSampler(m_device, &samplerCreateInfo, nullptr,
                  &m_defaultSamplerLinear);
  samplerCreateInfo.minFilter = VK_FILTER_NEAREST;
  samplerCreateInfo.magFilter = VK_FILTER_NEAREST;
  vkCreateSampler(m_device, &samplerCreateInfo, nullptr,
                  &m_defaultSamplerNearest);

  assert(0 == m_metalRoughness.write_sampler(m_defaultSamplerLinear));
  assert(0 == m_metalRoughness.write_texture(m_whiteImage.imageView));
  assert(1 == m_metalRoughness.write_texture(m_blackImage.imageView));
  assert(2 == m_metalRoughness.write_texture(m_normalFallback.imageView));
  m_mainDeletionQueue.push_function([&] {
    m_metalRoughness.clear_resources(*this);

    destroy_image(m_whiteImage);
    destroy_image(m_blackImage);
    destroy_image(m_greyImage);
    destroy_image(m_errorImage);
    destroy_image(m_normalFallback);

    vkDestroySampler(m_device, m_defaultSamplerLinear, nullptr);
    vkDestroySampler(m_device, m_defaultSamplerNearest, nullptr);
  });
}

void Engine::destroy_sync() {
  for (const auto& frame : m_frameData) {
    vkDestroyFence(m_device, frame.fence, nullptr);
    vkDestroySemaphore(m_device, frame.swapchainSemaphore, nullptr);
  }

  for (const auto& renderSemaphore : m_swapchainSemaphores) {
    vkDestroySemaphore(m_device, renderSemaphore, nullptr);
  }
}

void Engine::destroy_commands() {
  vkDestroyCommandPool(m_device, m_commandPool, nullptr);
}

void Engine::create_swapchain(const std::uint32_t width,
                              const std::uint32_t height) {
  m_swapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;

  auto vkbSwapchainResult =
      vkb::SwapchainBuilder(m_chosenGpu, m_device, m_surface)
          //.use_default_format_selection()
          .set_desired_format({.format = m_swapchainImageFormat,
                               .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
          .set_desired_present_mode(VK_PRESENT_MODE_IMMEDIATE_KHR)
          .set_desired_extent(width, height)
          .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
          .set_required_min_image_count(kNumberOfFrames)
          .build();

  if (!vkbSwapchainResult.has_value()) {
    throw std::runtime_error("Failed to create swapchain");
  }

  m_swapchainExtent = vkbSwapchainResult.value().extent;
  m_swapchain = vkbSwapchainResult.value().swapchain;
  m_swapchainImages = vkbSwapchainResult.value().get_images().value();
  m_swapchainImageViews = vkbSwapchainResult.value().get_image_views().value();
}

void Engine::destroy_swapchain() {
  vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);

  for (const auto& imageView : m_swapchainImageViews) {
    vkDestroyImageView(m_device, imageView, nullptr);
  }
}

void Engine::resize_swapchain() {
  vkDeviceWaitIdle(m_device) >> chk;

  destroy_swapchain();

  int w{0}, h{0};
  SDL_GetWindowSize(m_window.get(), &w, &h);
  m_windowExtent.width = w;
  m_windowExtent.height = h;
  create_swapchain(m_windowExtent.width, m_windowExtent.height);

  m_bSwapchainResizeRequest = false;
}

void Engine::WindowCleaner::operator()(SDL_Window* window) const {
  SDL_DestroyWindow(window);
}

void Engine::update_scene() {
  const auto start = cn::steady_clock::now();
  m_camera.update(m_stats.frameTime);
  m_mainDrawContext.clear();
  m_CurrentFrameInstanceBuffer = static_cast<Instance*>(
      get_current_frame().instanceBuffer.allocationInfo.pMappedData);
  m_CurrentMeshBuffer = static_cast<RenderObject*>(
      get_current_frame().meshesBuffer.allocationInfo.pMappedData);

  m_scene.draw(glm::mat4(1.0f), m_mainDrawContext);

  copy_frame_buffers();

  const glm::mat4 proj = glm::perspective(
      glm::radians(90.0f),
      static_cast<float>(m_drawExtent.width) / m_drawExtent.height, 0.001f,
      1000.0f);

  m_sceneData.view = m_camera.get_view_matrix();
  m_sceneData.proj = proj;
  m_sceneData.projView = proj * m_sceneData.view;
  m_sceneData.cameraPos = m_camera.position;
  m_sceneData.padding0 = 0.0f;

  auto* sceneData = static_cast<GpuSceneData*>(
      get_current_frame().sceneDataBuffer.allocation->GetMappedData());
  *sceneData = m_sceneData;
  m_LightPassConstants.sceneDataBufferDeviceAddr =
      get_current_frame().sceneDataBufferAddr;
  m_LightPassConstants.lightDataBufferDeviceAddr =
      get_current_frame().lightDataBufferAddr;
  m_LightPassConstants.lightCount = m_mainDrawContext.lights.size();
  std::memcpy(get_current_frame().lightDataBuffer.allocationInfo.pMappedData,
              m_mainDrawContext.lights.data(),
              m_mainDrawContext.lights.size() * sizeof(LightData));

  m_GBufferMeshPushConstants = {
      .globalVertexBufferAddr = m_globalVertexBufferAddress,
      .instanceBufferDeviceAddr = get_current_frame().instanceBufferAddr,
      .sceneDataBufferDeviceAddr = get_current_frame().sceneDataBufferAddr,
  };

  m_WBOITForwardPassPushConstants = {
      .globalVertexBufferAddr = m_globalVertexBufferAddress,
      .instanceBufferDeviceAddr =
          m_GBufferMeshPushConstants.instanceBufferDeviceAddr,
      .sceneDataBufferDeviceAddr =
          m_GBufferMeshPushConstants.sceneDataBufferDeviceAddr,
      .lightDataBufferDeviceAddr =
          m_LightPassConstants.lightDataBufferDeviceAddr,
      .lightCount = m_LightPassConstants.lightCount};
  const auto end = cn::steady_clock::now();
  const auto elapsed = cn::duration_cast<cn::milliseconds>(end - start);

  m_stats.sceneUpdateTime = elapsed.count() / 1000.0f;
}

}  // namespace mp