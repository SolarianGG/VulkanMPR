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
  std::deque<std::function<void()>> deletors;

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
  glm::mat4 transform;
  VkDeviceAddress vertexBufferDeviceAddr;
};

struct GpuSceneData {
  glm::mat4 view;
  glm::mat4 proj;
  glm::mat4 projView;
  glm::vec4 ambientColor;
  glm::vec4 sunlightDirection;  // w for sun power
  glm::vec4 sunlightColor;
};

enum class MaterialPass : std::uint8_t { Opaque, Transparent, Other };

struct MaterialPipeline {
  VkPipeline pipeline;
  VkPipelineLayout pipelineLayout;
};

struct MaterialInstance {
  MaterialPipeline* pipeline;
  VkDescriptorSet materialSet;
  MaterialPass passType;
};



struct DrawContext;
class IRenderable {
 public:
  virtual void draw(const glm::mat4& topMatrix, DrawContext& ctx) = 0;

  IRenderable() = default;
  IRenderable(const IRenderable& other) = default;
  IRenderable(IRenderable&& other) noexcept = default;
  IRenderable& operator=(const IRenderable& other) = default;
  IRenderable& operator=(IRenderable&& other) noexcept = default;
  virtual ~IRenderable() = default;
};

}  // namespace mp
