#pragma once

// clang-format off
#include <array>
#include <deque>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <cstddef>
#include <functional>
#include <cinttypes>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <glm/glm.hpp>

// clang-format on

namespace mp {

struct DeletionQueue {
  std::deque<std::function<void()> > deletors;

  template <typename T>
  void push_function(T&& function) {
    deletors.push_back(std::forward<T>(function));
  }

  void flush() {
    for (auto& deletor : std::ranges::reverse_view(deletors)) {
      deletor();
    }

    deletors.clear();
  }
};

struct FrameData {
  VkCommandBuffer commandBuffer;
  VkFence fence;
  VkSemaphore swapchainSemaphore;
  DeletionQueue frameDeletionQueue;
};

struct AllocatedImage {
  VkImage image;
  VkImageView imageView;
  VmaAllocation allocation;
  VkExtent3D imageExtent;
  VkFormat imageFormat;
};

struct ConstantPushRange {
  glm::vec4 data1;
  glm::vec4 data2;
  glm::vec4 data3;
  glm::vec4 data4;
};

struct ComputeEffect {
  const char* name;

  VkPipeline pipeline;
  VkPipelineLayout pipelineLayout;

  ConstantPushRange data;
};

struct AllocatedBuffer {
  VkBuffer buffer;
  VmaAllocation allocation;
  VmaAllocationInfo allocationInfo;
};

struct Vertex {
  glm::vec3 pos;
  float u;
  glm::vec3 normal;
  float v;
  glm::vec4 color;
};

struct GpuMeshBuffers {
  AllocatedBuffer vertexBuffer;
  AllocatedBuffer indexBuffer;
  VkDeviceAddress vertexBufferDeviceAddr;
};

struct GpuPushConstants {
  glm::mat4 mvpMatrix;
  VkDeviceAddress vertexBufferDeviceAddr;
};

}  // namespace mp
