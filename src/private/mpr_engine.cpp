#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include "mpr_engine.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <VkBootstrap.h>

#include <algorithm>
#include <chrono>
#include <format>
#include <numbers>
#include <numeric>
#include <print>
#include <ranges>
#include <thread>

#define VMA_IMPLEMENTATION
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>
#include <vk_mem_alloc.h>
#include <vulkan/vk_enum_string_helper.h>

#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "mpr_error_check.hpp"
#include "mpr_image.hpp"
#include "mpr_init_vk_stucts.hpp"
#include "mpr_loader.hpp"
#include "mpr_pipelines.hpp"

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


[[nodiscard]] bool is_visible(const mp::RenderObject& renderObject,
                     const glm::mat4& projView) {
  return true;
}
}  // namespace

namespace mp {
void GLTFMetallic_Roughness::build_pipelines(Engine& engine) {
  VkShaderModule meshVertShader;
  if (!load_shader_module("../../src/compiled_shaders/mesh.vertex.spv",
                          engine.m_device, &meshVertShader)) {
    throw std::runtime_error("Failed to load mesh.vertex.spv");
  }
  VkShaderModule meshFragShader;
  if (!load_shader_module("../../src/compiled_shaders/mesh.pixel.spv",
                          engine.m_device, &meshFragShader)) {
    throw std::runtime_error("Failed to load mesh.pixel.spv");
  }

  const VkPushConstantRange pushConstantRange{
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
      .offset = 0,
      .size = static_cast<std::uint32_t>(sizeof(GpuPushConstants)),
  };

  {
    DescriptorLayoutBuilder layoutBuilder;
    layoutBuilder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    layoutBuilder.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    layoutBuilder.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    materialLayout =
        layoutBuilder.build(engine.m_device, VK_SHADER_STAGE_VERTEX_BIT |
                                                 VK_SHADER_STAGE_FRAGMENT_BIT);
  }
  const VkDescriptorSetLayout layouts[]{
      engine.m_gpuSceneDataDescriptorSetLayout, materialLayout};
  const VkPipelineLayoutCreateInfo layoutCreateInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = std::size(layouts),
      .pSetLayouts = layouts,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pushConstantRange,

  };

  VkPipelineLayout layout;
  vkCreatePipelineLayout(engine.m_device, &layoutCreateInfo, nullptr,
                         &layout) >>
      chk;
  transparentPipeline.pipelineLayout = layout;
  opaquePipeline.pipelineLayout = layout;

  PipelineBuilder pipelineBuilder;
  pipelineBuilder.pipelineLayout = opaquePipeline.pipelineLayout;
  pipelineBuilder.add_shader(meshVertShader, VK_SHADER_STAGE_VERTEX_BIT);
  pipelineBuilder.add_shader(meshFragShader, VK_SHADER_STAGE_FRAGMENT_BIT);
  pipelineBuilder.enable_depth_test(true, VK_COMPARE_OP_GREATER_OR_EQUAL);
  pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
  pipelineBuilder.set_color_attachment_format(
      engine.m_frameData.at(0).drawImage.imageFormat);
  pipelineBuilder.set_depth_format(
      engine.m_frameData.at(0).depthImage.imageFormat);
  pipelineBuilder.set_cull_mode(VK_CULL_MODE_BACK_BIT,
                                VK_FRONT_FACE_COUNTER_CLOCKWISE);

  pipelineBuilder.set_multisampling_none();
  pipelineBuilder.disable_blending();
  opaquePipeline.pipeline = pipelineBuilder.build_pipeline(engine.m_device);

  //---
  pipelineBuilder.pipelineLayout = transparentPipeline.pipelineLayout;
  pipelineBuilder.enable_depth_test(false, VK_COMPARE_OP_GREATER_OR_EQUAL);
  pipelineBuilder.enable_blending_additive();
  transparentPipeline.pipeline =
      pipelineBuilder.build_pipeline(engine.m_device);
  vkDestroyShaderModule(engine.m_device, meshVertShader, nullptr);
  vkDestroyShaderModule(engine.m_device, meshFragShader, nullptr);
}

void GLTFMetallic_Roughness::clear_resources(VkDevice device) {
  vkDestroyPipeline(device, opaquePipeline.pipeline, nullptr);
  vkDestroyPipeline(device, transparentPipeline.pipeline, nullptr);
  vkDestroyPipelineLayout(device, transparentPipeline.pipelineLayout, nullptr);

  vkDestroyDescriptorSetLayout(device, materialLayout, nullptr);
}

MaterialInstance GLTFMetallic_Roughness::write_material(
    VkDevice device, MaterialPass matPass, const MaterialResources& res,
    DescriptorAllocatorGrowable& allocator) {
  MaterialInstance materialInstance{.passType = matPass};
  if (matPass == MaterialPass::Opaque) {
    materialInstance.pipeline = &opaquePipeline;

  } else if (matPass == MaterialPass::Transparent) {
    materialInstance.pipeline = &transparentPipeline;
  } else {
    throw std::runtime_error("Unsupported pass type");
  }

  materialInstance.materialSet = allocator.allocate(device, materialLayout);
  writer.clear();
  writer.write_buffer(0, res.dataBuffer, sizeof(MaterialConstants),
                      res.dataBufferOffset, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
  writer.write_image(1, res.colorImage.imageView, res.colorSampler,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
  writer.write_image(2, res.metalRoughnessImage.imageView,
                     res.metalRoughnessSampler,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
  writer.update_set(device, materialInstance.materialSet);

  return materialInstance;
}
void MeshNode::draw(const glm::mat4& topMatrix, DrawContext& ctx) {
  glm::mat4 nodeMatrix = topMatrix * worldTransform;
  for (const auto& s : mesh->geoSurfaces) {
    ctx.opaqueSurfaces.emplace_back(s.count, s.startIndex,
                                    mesh->meshBuffers.indexBuffer.buffer,
                                    &s.material->data, s.bound, nodeMatrix,
                                    mesh->meshBuffers.vertexBufferDeviceAddr);
  }

  Node::draw(topMatrix, ctx);
}

void LoadedGLTF::draw(const glm::mat4& topMatrix, DrawContext& ctx) {
  for (const auto& node : topNodes) {
    node->draw(topMatrix, ctx);
  }
}

void LoadedGLTF::clear_all() {
  descriptorAllocator.destroy_pools(creator->m_device);
  creator->destroy_buffer(materialDataBuffer);
  for (const auto& mesh : vi::values(meshes)) {
    creator->destroy_buffer(mesh->meshBuffers.indexBuffer);
    creator->destroy_buffer(mesh->meshBuffers.vertexBuffer);
  }
  for (const auto& image : vi::values(images)) {
    if (image.image != creator->m_errorImage.image) {
      creator->destroy_image(image);
    }
  }

  for (const auto& sampler : samplers) {
    vkDestroySampler(creator->m_device, sampler, nullptr);
  }
}

Engine::~Engine() {
  if (m_isInitialized) {
    vkDeviceWaitIdle(m_device);
    m_loadedScenes.clear();
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
  currentFrame.descriptorAllocator.clear_pools(m_device);

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
  m_drawExtent.width =
      std::min(currentDrawingImage.imageExtent.width, m_swapchainExtent.width) *
      m_renderScale;
  m_drawExtent.height = std::min(currentDrawingImage.imageExtent.height,
                                 m_swapchainExtent.height) *
                        m_renderScale;

  vkBeginCommandBuffer(cmd, &beginInfo) >> chk;

  utils::BarrierBuilder barrierBuilder;
  {
    barrierBuilder.add_image_barrier(
        currentDrawingImage.image, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_WRITE_BIT_KHR, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL,
        utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));

    barrierBuilder.barrier(cmd);
  }

  draw_background(cmd);

  {
    barrierBuilder.add_image_barrier(
        currentDrawingImage.image, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR,
        VK_ACCESS_2_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
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

  draw_geometry(cmd, currentDrawingImage.imageView, currentDepthImage.imageView,
                m_drawExtent);

  {
    barrierBuilder.add_image_barrier(
        currentDrawingImage.image,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
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

void Engine::draw_background(VkCommandBuffer cmd) {
  const VkDescriptorSet& descSet = get_current_frame().drawImageDescriptorSet;
  const auto& currentComputeEffect = m_computeEffects[m_currentComputeEffect];

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    currentComputeEffect.pipeline);

  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          m_backgroundPipelineLayout, 0, 1, &descSet, 0,
                          nullptr);

  vkCmdPushConstants(cmd, m_backgroundPipelineLayout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ConstantPushRange),
                     &currentComputeEffect.data);

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
        static_cast<std::uint32_t>(
            std::floor(std::log2(std::max(extent.width, extent.height)))) +
        1;
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

AllocatedImage Engine::create_image(void* data, VkExtent3D extent,
                                    VkFormat format,
                                    VkImageUsageFlags imageUsage,
                                    bool mipMapped) {
  if (format != VK_FORMAT_R8G8B8A8_UNORM &&
      format != VK_FORMAT_B8G8R8A8_UNORM) {
    throw std::runtime_error(std::format("Unsupported image format for now: {}",
                                         string_VkFormat(format)));
  }
  constexpr auto imagePixelSize = 4;  // 8 + 8 + 8 + 8 = 32bits (4 bytes)
  const auto bufferSize =
      extent.width * extent.height * extent.depth * imagePixelSize;
  const AllocatedImage image = create_image(
      extent, format, imageUsage | VK_IMAGE_USAGE_TRANSFER_DST_BIT, mipMapped);

  const AllocatedBuffer stagingBuffer = create_buffer(
      bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

  auto* bufferData =
      static_cast<char*>(stagingBuffer.allocation->GetMappedData());
  std::memcpy(bufferData, static_cast<char*>(data), bufferSize);

  immediate_submit([&](VkCommandBuffer cmd) {
    utils::transition_image(cmd, image.image, VK_IMAGE_LAYOUT_UNDEFINED,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // TODO: Handle mip and array levels/layers
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

    utils::transition_image(cmd, image.image,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  });
  destroy_buffer(stagingBuffer);
  return image;
}

void Engine::destroy_image(const AllocatedImage& image) {
  vkDestroyImageView(m_device, image.imageView, nullptr);
  vmaDestroyImage(m_allocator, image.image, image.allocation);
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

  const AllocatedBuffer stagingBuffer = create_buffer(
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
      m_camera.process_sdl_event(e);
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
    }
    ImGui::End();

    ImGui::Begin("Stats");
    ImGui::Text("Frame time: %f ms", m_stats.frameTime);
    ImGui::Text("Draw time: %f ms", m_stats.meshDrawTime);
    ImGui::Text("Scene update tim: %f ms", m_stats.sceneUpdateTime);
    ImGui::Text("Amount of draw calls: %i", m_stats.drawCallCount);
    ImGui::Text("Amount of triangles: %i", m_stats.triangleCount);
    ImGui::End();

    if (ImGui::Begin("Effects")) {
      ImGui::ListBox("Select compute effect", &m_currentComputeEffect,
                     effectsNames.data(), effectsNames.size());

      ComputeEffect& currentComputeEffect =
          m_computeEffects[m_currentComputeEffect];
      ImGui::ColorPicker4(
          "Data 1", reinterpret_cast<float*>(&currentComputeEffect.data.data1));
      ImGui::Spacing();
      ImGui::ColorPicker4(
          "Data 2", reinterpret_cast<float*>(&currentComputeEffect.data.data2));
      ImGui::Spacing();
      ImGui::ColorPicker4(
          "Data 3", reinterpret_cast<float*>(&currentComputeEffect.data.data3));
      ImGui::Spacing();
      ImGui::ColorPicker4(
          "Data 4", reinterpret_cast<float*>(&currentComputeEffect.data.data4));
    }
    ImGui::End();
    // ImGui UI end

    ImGui::Render();

    draw();

    auto end = cn::steady_clock::now();

    auto elapsed = cn::duration_cast<cn::milliseconds>(end - start);
    m_stats.frameTime = elapsed.count() / 1000.0f;
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
  m_stats.drawCallCount = 0;
  m_stats.triangleCount = 0;
  auto start = cn::steady_clock::now();
  AllocatedBuffer gpuSceneDataBuffer =
      create_buffer(sizeof(GpuSceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    VMA_MEMORY_USAGE_CPU_TO_GPU);
  get_current_frame().frameDeletionQueue.push_function(
      [=] { destroy_buffer(gpuSceneDataBuffer); });
  auto* sceneData = static_cast<GpuSceneData*>(
      gpuSceneDataBuffer.allocation->GetMappedData());
  *sceneData = m_sceneData;

  VkDescriptorSet globalDescSet =
      get_current_frame().descriptorAllocator.allocate(
          m_device, m_gpuSceneDataDescriptorSetLayout);
  {
    DescriptorWriter globalSceneWriter;
    globalSceneWriter.write_buffer(0, gpuSceneDataBuffer.buffer,
                                   sizeof(GpuSceneData), 0,
                                   VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    globalSceneWriter.update_set(m_device, globalDescSet);
  }

  auto opaqueSurfaces =
      vi::iota(0u, m_mainDrawContext.opaqueSurfaces.size()) |
      vi::filter([&](const std::uint32_t index) {
        return is_visible(m_mainDrawContext.opaqueSurfaces[index], sceneData->projView);
      }) |
      rn::to<std::vector>();
  rn::sort(opaqueSurfaces, [&](const std::uint32_t ia, const std::uint32_t ib) {
    const auto& a = m_mainDrawContext.opaqueSurfaces[ia];
    const auto& b = m_mainDrawContext.opaqueSurfaces[ib];

    if (a.material == b.material) {
      return a.indexBuffer < b.indexBuffer;
    }
    return a.material < b.material;
  });
  // ---
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

  MaterialInstance* lastMaterial = nullptr;
  MaterialPipeline* lastPipeline = nullptr;
  VkBuffer lastIndexBuffer = nullptr;
  auto drawContext = [&](const RenderObject& ctx) {
    if (lastMaterial != ctx.material) {
      if (ctx.material->pipeline != lastPipeline) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          ctx.material->pipeline->pipeline);

        lastPipeline = ctx.material->pipeline;
      }
      const VkDescriptorSet descSets[]{globalDescSet,
                                       ctx.material->materialSet};
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              ctx.material->pipeline->pipelineLayout, 0,
                              std::size(descSets), descSets, 0, nullptr);

      lastMaterial = ctx.material;
    }

    if (lastIndexBuffer != ctx.indexBuffer) {
      vkCmdBindIndexBuffer(cmd, ctx.indexBuffer, 0, VK_INDEX_TYPE_UINT32);

      lastIndexBuffer = ctx.indexBuffer;
    }
    const GpuPushConstants ctxPushConstant{
        .transform = ctx.transform,
        .vertexBufferDeviceAddr = ctx.vertexBufferAddress};
    vkCmdPushConstants(cmd, ctx.material->pipeline->pipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GpuPushConstants),
                       &ctxPushConstant);

    vkCmdDrawIndexed(cmd, ctx.indexCount, 1, ctx.firstIndex, 0, 0);
    m_stats.drawCallCount++;
    m_stats.triangleCount += ctx.indexCount / 3;
  };
  for (const auto ctx : opaqueSurfaces) {
    drawContext(m_mainDrawContext.opaqueSurfaces[ctx]);
  }
  for (const auto& ctx : m_mainDrawContext.opaqueSurfaces) {
    drawContext(ctx);
  }

  vkCmdEndRendering(cmd);

  auto end = cn::steady_clock::now();
  auto elapsed = cn::duration_cast<cn::milliseconds>(end - start);
  m_stats.meshDrawTime = elapsed.count() / 1000.0f;
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

void Engine::create_draw_image(AllocatedImage& drawImage, VkExtent3D extent) {
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

void Engine::create_depth_image(AllocatedImage& depthImage, VkExtent3D extent) {
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
  const VkExtent3D drawImageExtent{
      .width = m_windowExtent.width,
      .height = m_windowExtent.height,
      .depth = 1,
  };
  for (auto& frame : m_frameData) {
    create_draw_image(frame.drawImage, drawImageExtent);
    create_depth_image(frame.depthImage, drawImageExtent);
  }
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
  std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> ratios{
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
  };

  m_globalDescAllocator.init(m_device, 10, ratios);
  {
    DescriptorLayoutBuilder builder;
    builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

    m_drawImageDescriptorSetLayout =
        builder.build(m_device, VK_SHADER_STAGE_COMPUTE_BIT);
  }
  {
    DescriptorLayoutBuilder builder;
    builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    m_gpuSceneDataDescriptorSetLayout = builder.build(
        m_device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
  }
  {
    DescriptorLayoutBuilder builder;
    builder.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    m_texturedSetLayout = builder.build(
        m_device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
  }

  const std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> frameSizes = {
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4},
  };
  for (auto& frame : m_frameData) {
    auto& drawImageDescriptors = frame.drawImageDescriptorSet;
    const auto& drawImage = frame.drawImage;
    drawImageDescriptors = m_globalDescAllocator.allocate(
        m_device, m_drawImageDescriptorSetLayout);

    DescriptorWriter writer;
    writer.write_image(0, drawImage.imageView, nullptr, VK_IMAGE_LAYOUT_GENERAL,
                       VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    writer.update_set(m_device, drawImageDescriptors);

    frame.descriptorAllocator.init(m_device, 1000, frameSizes);
    m_mainDeletionQueue.push_function(
        [&] { frame.descriptorAllocator.destroy_pools(m_device); });
  }

  m_mainDeletionQueue.push_function([&] mutable {
    m_globalDescAllocator.destroy_pools(m_device);
    vkDestroyDescriptorSetLayout(m_device, m_drawImageDescriptorSetLayout,
                                 nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_gpuSceneDataDescriptorSetLayout,
                                 nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_texturedSetLayout, nullptr);
  });
}

void Engine::init_pipelines() {
  init_background_pipelines();
  m_metalRoughness.build_pipelines(*this);
}

void Engine::init_background_pipelines() {
  m_computeEffects.reserve(2);
  constexpr VkPushConstantRange pushConstantRange{
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = static_cast<std::uint32_t>(sizeof(ConstantPushRange)),
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
                         &m_backgroundPipelineLayout) >>
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
      .layout = m_backgroundPipelineLayout,
  };

  ComputeEffect gradientComputeEffect{};

  gradientComputeEffect.pipelineLayout = m_backgroundPipelineLayout;
  gradientComputeEffect.name = "gradient";
  gradientComputeEffect.data =
      ConstantPushRange{.data1 = glm::vec4{1.0f, 0.0f, 0.0f, 1.0f},
                        .data2 = glm::vec4{0.0f, 1.0f, 0.0f, 1.0f}};
  vkCreateComputePipelines(m_device, nullptr, 1, &createInfo, nullptr,
                           &gradientComputeEffect.pipeline) >>
      chk;

  createInfo.stage.module = skyShader;

  ComputeEffect skyComputeEffect{};
  skyComputeEffect.name = "sky";
  skyComputeEffect.pipelineLayout = m_backgroundPipelineLayout;
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
        vkDestroyPipelineLayout(m_device, m_backgroundPipelineLayout, nullptr);
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
  const std::string structurePath = "../../assets/structure.glb";
  auto structureFile = load_gltf(*this, structurePath).value();

  m_loadedScenes[structurePath] = std::move(structureFile);
}

void Engine::init_default_data() {
  std::uint32_t whiteColor =
      glm::packUnorm4x8(glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
  m_whiteImage =
      create_image(reinterpret_cast<void*>(&whiteColor), VkExtent3D{1, 1, 1},
                   VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);
  std::uint32_t blackColor =
      glm::packUnorm4x8(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
  m_blackImage =
      create_image(reinterpret_cast<void*>(&blackColor), VkExtent3D{1, 1, 1},
                   VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);
  std::uint32_t greyColor =
      glm::packUnorm4x8(glm::vec4(0.5f, 0.5f, 0.5f, 1.0f));
  m_greyImage =
      create_image(reinterpret_cast<void*>(&greyColor), VkExtent3D{1, 1, 1},
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

  GLTFMetallic_Roughness::MaterialResources materialResources;
  materialResources.colorImage = m_whiteImage;
  materialResources.colorSampler = m_defaultSamplerLinear;
  materialResources.metalRoughnessImage = m_whiteImage;
  materialResources.metalRoughnessSampler = m_defaultSamplerLinear;

  AllocatedBuffer matConstants = create_buffer(
      sizeof(GLTFMetallic_Roughness::MaterialConstants),
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      VMA_MEMORY_USAGE_GPU_ONLY);

  const AllocatedBuffer stagingBuffer = create_buffer(
      sizeof(GLTFMetallic_Roughness::MaterialConstants),
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

  auto* data = static_cast<GLTFMetallic_Roughness::MaterialConstants*>(
      stagingBuffer.allocation->GetMappedData());
  data->colorFactors = glm::vec4(1.0f);
  data->metalRoughFactors = glm::vec4(1.0f, 0.5f, 0.0f, 0.0f);
  immediate_submit([&](VkCommandBuffer cmd) {
    const VkBufferCopy region{
        .srcOffset = 0,
        .dstOffset = 0,
        .size = sizeof(GLTFMetallic_Roughness::MaterialConstants),
    };

    vkCmdCopyBuffer(cmd, stagingBuffer.buffer, matConstants.buffer, 1, &region);
  });

  materialResources.dataBuffer = matConstants.buffer;
  materialResources.dataBufferOffset = 0;

  m_defaultData = m_metalRoughness.write_material(
      m_device, MaterialPass::Opaque, materialResources, m_globalDescAllocator);
  destroy_buffer(stagingBuffer);
  m_mainDeletionQueue.push_function([&, matConstants] {
    m_defaultData = {};
    m_metalRoughness.clear_resources(m_device);

    destroy_buffer(matConstants);

    destroy_image(m_whiteImage);
    destroy_image(m_blackImage);
    destroy_image(m_greyImage);
    destroy_image(m_errorImage);

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

void Engine::update_scene() {
  const auto start = cn::steady_clock::now();
  m_camera.update();
  m_mainDrawContext.opaqueSurfaces.clear();
  m_mainDrawContext.transparentSurfaces.clear();

  for (const auto& mesh : m_loadedScenes | std::views::values) {
    mesh->draw(glm::mat4(1.0f), m_mainDrawContext);
  }

  const glm::mat4 proj = glm::perspective(
      glm::radians(90.0f),
      static_cast<float>(m_drawExtent.width) / m_drawExtent.height, 1000.0f,
      0.001f);

  m_sceneData.view = m_camera.get_view_matrix();
  m_sceneData.proj = proj;
  m_sceneData.projView = proj * m_sceneData.view;
  m_sceneData.ambientColor = glm::vec4(0.1f, 0.1f, 0.1f, 1.0f);
  m_sceneData.sunlightColor = glm::vec4(1.0f);
  m_sceneData.sunlightDirection =
      glm::normalize(glm::vec4(.77f, 0.2f, 0.5f, 1.0f));
  const auto end = cn::steady_clock::now();
  const auto elapsed = cn::duration_cast<cn::milliseconds>(end - start);

  m_stats.sceneUpdateTime = elapsed.count() / 1000.0f;
}

}  // namespace mp