#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "mpr_engine.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <VkBootstrap.h>

#include <algorithm>
#include <chrono>
#include <format>
#include <print>
#include <ranges>
#include <thread>

#define VMA_IMPLEMENTATION
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>
#include <vk_mem_alloc.h>

#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "mpr_error_check.hpp"
#include "mpr_image.hpp"
#include "mpr_init_vk_stucts.hpp"
#include "mpr_loader.hpp"
#include "mpr_pipelines.hpp"
#include "vulkan/vk_enum_string_helper.h"

using namespace std::chrono_literals;
namespace rn = std::ranges;
namespace vi = std::views;

constexpr bool bUseValidationLayers = true;
constexpr auto kBaseWindowTitle = "Hello Vulkan";

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
  init_mesh_data();

  m_isInitialized = true;
}

Engine& Engine::get() { return *gLoadedEngine; }

void Engine::draw() {
  FrameData& currentFrame = get_current_frame();
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
  VkCommandBuffer& cmd = currentFrame.commandBuffer;
  VkImage& swapchainImage = m_swapchainImages[swapchainImageIndex];

  constexpr VkCommandBufferBeginInfo beginInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .pNext = nullptr,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
      .pInheritanceInfo = nullptr,
  };

  AllocatedImage& currentDrawingImage =
      m_drawImages[m_frameNumber % kNumberOfFrames];
  AllocatedImage& currentDepthImage =
      m_depthImages[m_frameNumber % kNumberOfFrames];
  m_drawExtent.width =
      std::min(currentDrawingImage.imageExtent.width, m_swapchainExtent.width) *
      m_renderScale;
  m_drawExtent.height = std::min(currentDrawingImage.imageExtent.height,
                                 m_swapchainExtent.height) *
                        m_renderScale;

  vkBeginCommandBuffer(cmd, &beginInfo) >> chk;

  utils::transition_image(cmd, currentDrawingImage.image,
                          VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

  draw_background(cmd);

  utils::transition_image(cmd, currentDrawingImage.image,
                          VK_IMAGE_LAYOUT_GENERAL,
                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  utils::transition_image(cmd, currentDepthImage.image,
                          VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

  draw_geometry(cmd, currentDrawingImage.imageView, currentDepthImage.imageView,
                m_drawExtent);
  utils::transition_image(cmd, currentDrawingImage.image,
                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
  utils::transition_image(cmd, swapchainImage, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

  utils::copy_to_image(cmd, currentDrawingImage.image, swapchainImage,
                       m_drawExtent, m_swapchainExtent);

  utils::transition_image(cmd, swapchainImage,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

  draw_imgui(cmd, m_swapchainImageViews[swapchainImageIndex]);

  utils::transition_image(cmd, swapchainImage,
                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                          VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

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

void Engine::draw_background(VkCommandBuffer cmd) {
  VkDescriptorSet& descSet =
      m_drawImagesDescriptors[m_frameNumber % kNumberOfFrames];
  auto& currentComputeEffect = m_computeEffects[m_currentComputeEffect];

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    currentComputeEffect.pipeline);

  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout,
                          0, 1, &descSet, 0, nullptr);

  vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     sizeof(ConstantPushRange), &currentComputeEffect.data);

  vkCmdDispatch(cmd, std::ceil(m_drawExtent.width / 16.0f),
                std::ceil(m_drawExtent.height / 16.0f), 1);
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

void Engine::destroy_buffer(AllocatedBuffer& buffer) {
  vmaDestroyBuffer(m_allocator, buffer.buffer, buffer.allocation);
}

GpuMeshBuffers Engine::create_mesh_buffers(std::span<std::uint32_t> indices,
                                           std::span<Vertex> vertices) {
  const auto vertexBufferSize = vertices.size() * sizeof(Vertex);
  const auto indexBufferSize = indices.size() * sizeof(std::uint32_t);
  GpuMeshBuffers buffers;
  buffers.vertexBuffer = create_buffer(
      vertexBufferSize,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      VMA_MEMORY_USAGE_GPU_ONLY);

  const VkBufferDeviceAddressInfo bufferAddressInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
      .pNext = nullptr,
      .buffer = buffers.vertexBuffer.buffer,
  };
  buffers.vertexBufferDeviceAddr =
      vkGetBufferDeviceAddress(m_device, &bufferAddressInfo);
  assert(buffers.vertexBufferDeviceAddr);

  buffers.indexBuffer = create_buffer(
      indexBufferSize,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
      VMA_MEMORY_USAGE_GPU_ONLY);

  AllocatedBuffer stagingBuffer = create_buffer(
      vertexBufferSize + indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VMA_MEMORY_USAGE_CPU_ONLY);

  auto* data = static_cast<char*>(stagingBuffer.allocation->GetMappedData());
  std::memcpy(data, vertices.data(), vertexBufferSize);
  std::memcpy(data + vertexBufferSize, indices.data(), indexBufferSize);

  immediate_submit([&](VkCommandBuffer cmd) {
    const VkBufferCopy vRegions{
        .srcOffset = 0,
        .dstOffset = 0,
        .size = vertexBufferSize,
    };
    vkCmdCopyBuffer(cmd, stagingBuffer.buffer, buffers.vertexBuffer.buffer, 1,
                    &vRegions);

    const VkBufferCopy iRegions{
        .srcOffset = vertexBufferSize,
        .dstOffset = 0,
        .size = indexBufferSize,
    };
    vkCmdCopyBuffer(cmd, stagingBuffer.buffer, buffers.indexBuffer.buffer, 1,
                    &iRegions);
  });

  destroy_buffer(stagingBuffer);
  return buffers;
}

void Engine::run() {
  SDL_Event e;
  bool bIsRunning = true;

  const auto effectsNames =
      m_computeEffects |
      vi::transform([](const auto& effect) { return effect.name; }) |
      rn::to<std::vector>();

  const auto assetsNames =
      m_testAssets |
      vi::transform([](const auto& asset) { return asset->name.c_str(); }) |
      rn::to<std::vector>();
  while (bIsRunning) {
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
      ImGui_ImplSDL3_ProcessEvent(&e);

      if (ImGui::GetIO().WantCaptureMouse || ImGui::GetIO().WantCaptureKeyboard)
        continue;

      if (e.type == SDL_EVENT_MOUSE_WHEEL) {
        m_centerRadius += e.wheel.y;
      }

      if (e.type == SDL_EVENT_MOUSE_MOTION) {
        static float theta = glm::pi<float>() / 4;
        static float phi = glm::pi<float>() / 2;
        theta += glm::radians(e.motion.xrel * 0.25f);
        phi += glm::radians(e.motion.yrel * 0.5f);

        phi = glm::clamp(phi, 0.1f, glm::pi<float>() - 0.1f);

        const float x = {m_centerRadius * std::sinf(phi) * std::cosf(theta)};
        const float z = {m_centerRadius * std::sinf(phi) * std::sinf(theta)};
        const float y = {m_centerRadius * std::cos(phi)};
        m_eyePos = {x, y, z};
      }
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
    if (ImGui::Begin("##Other")) {
      ImGui::DragFloat("##Render scale", &m_renderScale, 0.01f, 0.01f, 1.0f);
      ImGui::End();
    }

    if (ImGui::Begin("##Assets")) {
      ImGui::ListBox("##Select asset", &m_assetIndex, assetsNames.data(),
                     assetsNames.size());
      ImGui::End();
    }

    if (ImGui::Begin("##Effects")) {
      ImGui::ListBox("##Select compute effect", &m_currentComputeEffect,
                     effectsNames.data(), effectsNames.size());

      ComputeEffect& currentComputeEffect =
          m_computeEffects[m_currentComputeEffect];
      ImGui::ColorPicker4("##Data 1", reinterpret_cast<float*>(
                                          &currentComputeEffect.data.data1));
      ImGui::Spacing();
      ImGui::ColorPicker4("##Data 2", reinterpret_cast<float*>(
                                          &currentComputeEffect.data.data2));
      ImGui::Spacing();
      ImGui::ColorPicker4("##Data 3", reinterpret_cast<float*>(
                                          &currentComputeEffect.data.data3));
      ImGui::Spacing();
      ImGui::ColorPicker4("##Data 4", reinterpret_cast<float*>(
                                          &currentComputeEffect.data.data4));
      ImGui::End();
    }
    // ImGui UI end

    ImGui::Render();

    draw();
  }
}

void Engine::draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView) {
  const auto colorAttachment = utils::attachment_info(
      targetImageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  const auto renderingInfo =
      utils::rendering_info(m_swapchainExtent, &colorAttachment, nullptr);

  vkCmdBeginRendering(cmd, &renderingInfo);

  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

  vkCmdEndRendering(cmd);
}

void Engine::draw_geometry(VkCommandBuffer cmd, VkImageView colorImageView,
                           VkImageView depthImageView,
                           const VkExtent2D imageExtent) {
  const auto colorAttachment = utils::attachment_info(
      colorImageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  const auto depthAttachment = utils::depth_attachment(
      depthImageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
  const auto renderInfo =
      utils::rendering_info(imageExtent, &colorAttachment, &depthAttachment);
  vkCmdBeginRendering(cmd, &renderInfo);
  const VkViewport viewport{
      .x = 0,
      .y = static_cast<float>(imageExtent.height),
      .width = static_cast<float>(imageExtent.width),
      .height = -static_cast<float>(imageExtent.height),
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
  };
  vkCmdSetViewport(cmd, 0, 1, &viewport);

  const VkRect2D scissor{
      .extent = imageExtent,
  };
  vkCmdSetScissor(cmd, 0, 1, &scissor);

  auto* currentAsset = m_testAssets[m_assetIndex].get();

  static glm::mat4 proj =
      glm::perspective(glm::radians(90.0f),
                       static_cast<float>(m_drawExtent.width) /
                           static_cast<float>(m_drawExtent.height),
                       1000.0f, 0.001f);
  const glm::mat4 view = glm::lookAt(m_eyePos, glm::vec3{0.0f, 0.0f, 0.0f},
                                     glm::vec3{0.0f, 1.0f, 0.0f});

  m_pushConstants.vertexBufferDeviceAddr =
      currentAsset->meshBuffers.vertexBufferDeviceAddr;
  m_pushConstants.mvpMatrix = proj * view;
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_meshPipeline);

  vkCmdPushConstants(cmd, m_meshPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                     sizeof(GpuPushConstants), &m_pushConstants);
  vkCmdBindIndexBuffer(cmd, currentAsset->meshBuffers.indexBuffer.buffer, 0,
                       VK_INDEX_TYPE_UINT32);
  for (const auto& [startIndex, count] : currentAsset->geoSurfaces) {
    vkCmdDrawIndexed(cmd, count, 1, startIndex, 0, 0);
  }

  vkCmdEndRendering(cmd);
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
      static_cast<SDL_WindowFlags>(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

  m_window = {SDL_CreateWindow(kBaseWindowTitle, m_windowExtent.width,
                               m_windowExtent.height, windowFlags),
              WindowCleaner{}};
  if (!m_window) {
    std::println("Failed to create window: {}", SDL_GetError());
  }
}

void Engine::init_vulkan() {
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

  if (!SDL_Vulkan_CreateSurface(m_window.get(), m_instance, nullptr,
                                &m_surface)) {
    throw std::runtime_error(
        std::format("Failed to create surface: {}", SDL_GetError()));
  }

  constexpr VkPhysicalDeviceVulkan13Features features13{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .synchronization2 = true,
      .dynamicRendering = true,
  };

  constexpr VkPhysicalDeviceVulkan12Features features12{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .descriptorIndexing = true,
      .bufferDeviceAddress = true,
  };

  vkb::PhysicalDeviceSelector selector{result.value()};

  const auto physicalDevice =
      selector.set_minimum_version(1, 3)
          .set_required_features_13(features13)
          .set_required_features_12(features12)
          .set_surface(m_surface)
          //.allow_any_gpu_device_type(false)
          //.prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
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

  VmaAllocatorCreateInfo allocatorCreateInfo{
      .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
      .physicalDevice = m_chosenGpu,
      .device = m_device,
      .instance = m_instance,
  };

  vmaCreateAllocator(&allocatorCreateInfo, &m_allocator) >> chk;

  m_mainDeletionQueue.push_function(
      [&]() { vmaDestroyAllocator(m_allocator); });
}

void Engine::create_draw_images(VkExtent3D extent) {
  for (auto& drawImage : m_drawImages) {
    drawImage.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    drawImage.imageExtent = extent;

    VkImageUsageFlags drawImageUsages{};
    drawImageUsages |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_STORAGE_BIT;
    const VkImageCreateInfo imageCreateInfo = utils::image_create_info(
        drawImage.imageFormat, drawImageUsages, extent);
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
}

void Engine::create_depth_images(VkExtent3D extent) {
  for (auto& depthImage : m_depthImages) {
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
}

void Engine::init_swapchain() {
  create_swapchain(m_windowExtent.width, m_windowExtent.height);
  const VkExtent3D drawImageExtent{
      .width = m_windowExtent.width,
      .height = m_windowExtent.height,
      .depth = 1,
  };
  create_draw_images(drawImageExtent);
  create_depth_images(drawImageExtent);
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
  std::vector<DescriptorAllocator::PoolSizeRatio> ratios{
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}};

  constexpr auto kMaxSets = 10;
  m_descriptorAllocator.init_pool(m_device, kMaxSets, ratios);

  {
    DescriptorLayoutBuilder builder;
    builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

    m_drawImageDescriptorSetLayout =
        builder.build(m_device, VK_SHADER_STAGE_COMPUTE_BIT);
  }

  for (auto i = 0; i < kNumberOfFrames; ++i) {
    auto& drawImageDescriptors = m_drawImagesDescriptors[i];
    auto& drawImage = m_drawImages[i];
    drawImageDescriptors = m_descriptorAllocator.allocate(
        m_device, m_drawImageDescriptorSetLayout);

    const VkDescriptorImageInfo imageInfo{
        .sampler = nullptr,
        .imageView = drawImage.imageView,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

    const VkWriteDescriptorSet drawImageWrite{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr,
        .dstSet = drawImageDescriptors,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &imageInfo,
    };

    vkUpdateDescriptorSets(m_device, 1, &drawImageWrite, 0, nullptr);
  }
  m_mainDeletionQueue.push_function([&] {
    m_descriptorAllocator.destroy_pool(m_device);
    vkDestroyDescriptorSetLayout(m_device, m_drawImageDescriptorSetLayout,
                                 nullptr);
  });
}

void Engine::init_pipelines() {
  init_background_pipelines();
  init_mesh_pipelines();
}

void Engine::init_background_pipelines() {
  m_computeEffects.reserve(2);
  constexpr VkPushConstantRange pushConstantRange{
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = sizeof(ConstantPushRange),
  };

  const VkPipelineLayoutCreateInfo layoutCreateInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .setLayoutCount = 1,
      .pSetLayouts = &m_drawImageDescriptorSetLayout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pushConstantRange,
  };
  vkCreatePipelineLayout(m_device, &layoutCreateInfo, nullptr,
                         &m_pipelineLayout) >>
      chk;

  VkShaderModule gradientShader;
  if (!mp::load_shader_module(
          "../../src/compiled_shaders/gradient_color.comp.spv", m_device,
          &gradientShader)) {
    throw std::runtime_error("Failed to load a shader");
  }

  VkShaderModule skyShader;
  if (!mp::load_shader_module("../../src/compiled_shaders/sky.comp.spv",
                              m_device, &skyShader)) {
    throw std::runtime_error("Failed to load a shader");
  }

  const VkPipelineShaderStageCreateInfo shaderStage{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .pNext = nullptr,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = gradientShader,
      .pName = "main",
  };

  VkComputePipelineCreateInfo createInfo{
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .stage = shaderStage,
      .layout = m_pipelineLayout,
  };

  ComputeEffect gradientComputeEffect{};

  gradientComputeEffect.pipelineLayout = m_pipelineLayout;
  gradientComputeEffect.name = "gradient";
  gradientComputeEffect.data = {.data1 = {1.0f, 0.0f, 0.0f, 1.0f},
                                .data2 = {0.0f, 1.0f, 0.0f, 1.0f}};
  vkCreateComputePipelines(m_device, nullptr, 1, &createInfo, nullptr,
                           &gradientComputeEffect.pipeline) >>
      chk;

  createInfo.stage.module = skyShader;

  ComputeEffect skyComputeEffect{};
  skyComputeEffect.name = "sky";
  skyComputeEffect.pipelineLayout = m_pipelineLayout;
  skyComputeEffect.data.data1 = {0.1f, 0.2f, 0.4f, 0.97f};
  vkCreateComputePipelines(m_device, nullptr, 1, &createInfo, nullptr,
                           &skyComputeEffect.pipeline) >>
      chk;

  m_computeEffects.push_back(gradientComputeEffect);
  m_computeEffects.push_back(skyComputeEffect);

  vkDestroyShaderModule(m_device, gradientShader, nullptr);
  vkDestroyShaderModule(m_device, skyShader, nullptr);
  m_mainDeletionQueue.push_function(
      [&, gradientComputeEffect, skyComputeEffect] {
        vkDestroyPipeline(m_device, gradientComputeEffect.pipeline, nullptr);
        vkDestroyPipeline(m_device, skyComputeEffect.pipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
      });
}

void Engine::init_mesh_pipelines() {
  PipelineBuilder builder;

  const VkPushConstantRange range{
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
      .offset = 0,
      .size = sizeof(GpuPushConstants),
  };
  const VkPipelineLayoutCreateInfo layoutCreateInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &range,
  };
  vkCreatePipelineLayout(m_device, &layoutCreateInfo, nullptr,
                         &m_meshPipelineLayout) >>
      chk;

  builder.pipelineLayout = m_meshPipelineLayout;

  VkShaderModule vertexShaderModule;
  if (!load_shader_module("../../src/compiled_shaders/colored_mesh.vert.spv",
                          m_device, &vertexShaderModule)) {
  }
  builder.add_shader(vertexShaderModule, VK_SHADER_STAGE_VERTEX_BIT);

  VkShaderModule fragmentShaderModule;
  if (!load_shader_module(
          "../../src/compiled_shaders/colored_triangle.frag.spv", m_device,
          &fragmentShaderModule)) {
  }
  builder.add_shader(fragmentShaderModule, VK_SHADER_STAGE_FRAGMENT_BIT);

  builder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  builder.set_polygon_mode(VK_POLYGON_MODE_FILL);
  builder.set_color_attachment_format(m_drawImages.at(0).imageFormat);
  builder.set_depth_format(m_depthImages.at(0).imageFormat);
  builder.set_cull_mode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);

  builder.set_multisampling_none();
  builder.enable_blending_additive();
  builder.enable_depth_test(true, VK_COMPARE_OP_GREATER_OR_EQUAL);

  m_meshPipeline = builder.build_pipeline(m_device);

  vkDestroyShaderModule(m_device, vertexShaderModule, nullptr);
  vkDestroyShaderModule(m_device, fragmentShaderModule, nullptr);
  m_mainDeletionQueue.push_function([&] {
    vkDestroyPipeline(m_device, m_meshPipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_meshPipelineLayout, nullptr);
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

  // 2: initialize imgui library

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

  // dynamic rendering parameters for imgui to use
  initInfo.PipelineRenderingCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  initInfo.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
  initInfo.PipelineRenderingCreateInfo.pColorAttachmentFormats =
      &m_swapchainImageFormat;

  initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

  ImGui_ImplVulkan_Init(&initInfo);

  ImGui_ImplVulkan_CreateFontsTexture();

  m_mainDeletionQueue.push_function([&, imguiPool] {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    vkDestroyDescriptorPool(m_device, imguiPool, nullptr);
  });
}

void Engine::init_mesh_data() {
  m_testAssets = load_mesh(*this, "../../assets/basicmesh.glb").value();

  m_mainDeletionQueue.push_function([&] {
    for (auto& assets : m_testAssets) {
      destroy_buffer(assets->meshBuffers.vertexBuffer);
      destroy_buffer(assets->meshBuffers.indexBuffer);
    }
  });
}

void Engine::destroy_sync() {
  for (auto& frame : m_frameData) {
    vkDestroyFence(m_device, frame.fence, nullptr);
    vkDestroySemaphore(m_device, frame.swapchainSemaphore, nullptr);
  }

  for (auto& renderSemaphore : m_swapchainSemaphores) {
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
          .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
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
}  // namespace mp