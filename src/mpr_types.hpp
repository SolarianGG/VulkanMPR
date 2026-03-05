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
namespace mp
{

struct DeletionQueue
{
    std::deque<std::function<void()>> deletors;

    template <typename T> void push_function(T &&function)
    {
        deletors.push_back(std::forward<T>(function));
    }

    void flush()
    {
        for (auto &deletor : std::ranges::reverse_view(deletors))
        {
            deletor();
        }

        deletors.clear();
    }
};

struct AllocatedImage
{
    VkImage image;
    VkImageView imageView;
    VmaAllocation allocation;
    VkExtent3D imageExtent;
    VkFormat imageFormat;
};

struct AllocatedBuffer
{
    VkBuffer buffer;
    VmaAllocation allocation;
    VmaAllocationInfo allocationInfo;

    VkDeviceAddress get_buffer_device_address(VkDevice device) const
    {
        const VkBufferDeviceAddressInfo addrInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .buffer = buffer,
        };
        return vkGetBufferDeviceAddress(device, &addrInfo);
    }
};

struct Vertex
{
    glm::vec3 pos;
    float u;
    glm::vec3 normal;
    float v;
    glm::vec4 tangent;
    glm::vec4 color;
};

struct MaterialInstanceIndices
{
    std::uint32_t materialID;
    std::uint32_t colorTextureID;
    std::uint32_t colorSamplerID;
    std::uint32_t metalRoughnessTextureID;
    std::uint32_t metalRoughnessSamplerID;
    std::uint32_t normalTextureID;
    std::uint32_t normalSamplerID;
};

struct Instance
{
    glm::mat4 world;
    std::uint32_t meshIndex;
    MaterialInstanceIndices materialIndices;
};

struct GBufferPassPushConstants
{
    VkDeviceAddress globalVertexBufferAddr;
    VkDeviceAddress instanceBufferDeviceAddr;
    VkDeviceAddress sceneDataBufferDeviceAddr;
};

struct OITForwardPassPushConstants
{
    VkDeviceAddress globalVertexBufferAddr;     // 8 @ 0
    VkDeviceAddress instanceBufferDeviceAddr;   // 8 @ 8
    VkDeviceAddress sceneDataBufferDeviceAddr;  // 8 @ 16
    VkDeviceAddress dirLightBufferDeviceAddr;   // 8 @ 24
    std::uint32_t dirLightCount;                // 4 @ 32
    std::uint32_t _pad0;                        // 4 @ 36
    VkDeviceAddress pointLightBufferDeviceAddr; // 8 @ 40
    std::uint32_t pointLightCount;              // 4 @ 48
}; // Total: 52 bytes

struct LightPassConstantRange
{
    VkDeviceAddress sceneDataBufferDeviceAddr;  // 8 @ 0
    VkDeviceAddress dirLightBufferDeviceAddr;   // 8 @ 8
    std::uint32_t dirLightCount;                // 4 @ 16
    std::uint32_t _pad0;                        // 4 @ 20
    VkDeviceAddress pointLightBufferDeviceAddr; // 8 @ 24
    std::uint32_t pointLightCount;              // 4 @ 32
    std::uint32_t _pad1, _pad2, _pad3;          // 12 @ 36
    glm::mat4 inverseCameraViewProj;            // 64 @ 48
}; // Total: 112 bytes

struct CullPassPushConstants
{
    VkDeviceAddress meshBufferAddr;
    VkDeviceAddress instanceBufferDeviceAddr;
    VkDeviceAddress commandsBufferAddr;
    VkDeviceAddress countBufferAddr;
    glm::mat4 viewProj;
    std::uint32_t objectsCount;
    std::uint32_t objectsOffset;
};

struct GpuSceneData
{
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 projView;
    glm::vec3 cameraPos;
    float padding0;
};

struct ShadowPassPushConstants
{
    VkDeviceAddress globalVertexBufferAddr;
    VkDeviceAddress instanceBufferDeviceAddr;
    VkDeviceAddress dirLightsBufferAddr;
    std::uint32_t lightIndex;
    std::uint32_t cascadeIndex;
};

enum class MaterialPass : std::uint8_t
{
    Opaque,
    Transparent,
    Other
};

struct MaterialPipeline
{
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
};

struct MaterialInstance
{
    MaterialPipeline *pipeline;
    MaterialPass passType;
    MaterialInstanceIndices indices;
};

constexpr int kMaxCascadeCount = 4;
constexpr int MAX_CASCADES = kMaxCascadeCount;

struct DirectionalLightData
{
    glm::vec3 direction;     // 12 @ 0
    float padding;           // 4  @ 12  → matches float4 direction in shader
    glm::vec3 color;         // 12 @ 16
    float intensity;         // 4  @ 28  → matches float4 color.w in shader
    glm::ivec4 cascadeCount; // 16 @ 32 → .x = count; .yzw = padding

    std::array<float, kMaxCascadeCount> splitDistances; // 16 @ 48
    std::array<glm::mat4, kMaxCascadeCount> cascadeVPs; // 256 @ 64
}; // Total: 320 bytes
static_assert(sizeof(DirectionalLightData) == 320);

struct PointLightData
{
    glm::vec3 position;
    float range;
    glm::vec3 color;
    float intensity;
};

struct MinMax
{
    std::uint32_t min;
    std::uint32_t max;
};

struct AABB
{
    std::uint32_t minX, minY, minZ, _pad0;
    std::uint32_t maxX, maxY, maxZ, _pad1;
};

struct CascadesAABB
{
    std::array<AABB, kMaxCascadeCount> bounds;
};

struct DepthReductionPushConstants
{
    VkDeviceAddress minMaxAddr;
    float cameraNear, cameraFar;
};

struct DepthPartitionPushConstants
{
    VkDeviceAddress minMaxAddr;      // 8 @ 0
    VkDeviceAddress splitsAABBAddr;  // 8 @ 8
    VkDeviceAddress dirLightsAddr;   // 8 @ 16
    std::uint32_t dirLightCount;     // 4 @ 32
    float near, far;
    std::uint32_t _pad0;             // 4 @ 36
    std::uint32_t _pad1;             // 4 @ 36
    std::uint32_t _pad2;             // 4 @ 36
    glm::mat4 inverseCameraViewProj; // 64 @ 48
}; 

struct DirVpPushConstants
{
    VkDeviceAddress splitsAABBAddr; // 8 @ 0
    VkDeviceAddress dirLightsAddr;  // 8 @ 8
}; // Total: 16 bytes

struct DrawContext;
class IRenderable
{
  public:
    virtual void draw(const glm::mat4 &topMatrix, DrawContext &ctx) = 0;

    IRenderable() = default;
    IRenderable(const IRenderable &other) = default;
    IRenderable(IRenderable &&other) noexcept = default;
    IRenderable &operator=(const IRenderable &other) = default;
    IRenderable &operator=(IRenderable &&other) noexcept = default;
    virtual ~IRenderable() = default;
};

struct GLTFMaterial
{
    MaterialInstance data;
};
struct GeoSurface
{
    std::uint32_t startIndex;
    std::uint32_t count;
    std::int32_t vertexOffset;
    glm::vec3 min;
    glm::vec3 max;
    std::shared_ptr<GLTFMaterial> material;
};

struct MeshAsset
{
    std::string name;

    std::vector<GeoSurface> geoSurfaces;
};

} // namespace mp
