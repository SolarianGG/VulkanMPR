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

#include <volk.h>
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

struct LightPassConstantRange {
  VkDeviceAddress sceneDataBufferDeviceAddr;
  VkDeviceAddress lightDataBufferDeviceAddr;
  std::uint32_t lightCount;
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
  glm::vec4 tangent;
  glm::vec4 color;
};

struct MaterialInstanceIndices {
  std::uint32_t materialID;
  std::uint32_t colorTextureID;
  std::uint32_t colorSamplerID;
  std::uint32_t metalRoughnessTextureID;
  std::uint32_t metalRoughnessSamplerID;
  std::uint32_t normalTextureID;
  std::uint32_t normalSamplerID;
};

struct Instance {
  glm::mat4 world;
  MaterialInstanceIndices materialIndices;
};

struct GpuMeshBuffers {
  AllocatedBuffer vertexBuffer;
  AllocatedBuffer indexBuffer;
  VkDeviceAddress vertexBufferDeviceAddr;
};

struct GpuPushConstants {
  VkDeviceAddress vertexBufferDeviceAddr;
  VkDeviceAddress instanceBufferDeviceAddr;
  VkDeviceAddress sceneDataBufferDeviceAddr;
};


struct GpuSceneData {
  glm::mat4 view;
  glm::mat4 proj;
  glm::mat4 projView;
  glm::vec3 cameraPos;
  float padding0;
};

struct LightData {
  std::int32_t lightType; // directional - 0, point - 1
  std::int32_t padding0[3];

  glm::vec4 data0;
  glm::vec4 data1;
};

enum class MaterialPass : std::uint8_t { Opaque, Transparent, Other };

struct MaterialPipeline {
  VkPipeline pipeline;
  VkPipelineLayout pipelineLayout;
};

struct MaterialInstance {
  MaterialPipeline* pipeline;
  MaterialPass passType;
  MaterialInstanceIndices indices;
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

struct GLTFMaterial {
  MaterialInstance data;
};
struct GeoSurface {
  std::uint32_t startIndex;
  std::uint32_t count;
  std::shared_ptr<GLTFMaterial> material;
};

struct MeshAsset {
  std::string name;

  std::vector<GeoSurface> geoSurfaces;
  GpuMeshBuffers meshBuffers;
};

}  // namespace mp
