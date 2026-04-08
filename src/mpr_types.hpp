#pragma once

// clang-format off
#include <array>
#include <deque>
#include <memory>
#include <optional>
#include <ranges>
#include <string>
#include <utility>
#include <vector>
#include <functional>
#include <cinttypes>

#include <volk.h>
#include <vk_mem_alloc.h>

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/glm.hpp>

// clang-format on
namespace mp
{

constexpr auto kNumberOfFrames = 2;

constexpr float cameraNear = 0.1f;
constexpr float cameraFar = 200.0f;
constexpr int kMaxCascadeCount = 4;
constexpr int MAX_CASCADES = kMaxCascadeCount;

constexpr std::uint32_t kDirectionalShadowMapSize = 2048u;

constexpr std::uint32_t kPointLightsShadowMapSize = 8192u;
constexpr std::uint32_t kPointLightTileSize = 1024u;
constexpr std::uint32_t kMaxPointLights =
    (kPointLightsShadowMapSize * kPointLightsShadowMapSize) / (kPointLightTileSize * kPointLightTileSize);
constexpr float kPointLightNear = 0.1f;
constexpr std::uint32_t kPointLightTilesPerRow = kPointLightsShadowMapSize / kPointLightTileSize;

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

inline std::pair<float, float> compute_alpha_beta()
{
    constexpr float hFOV_orig = 143.98570868f;
    constexpr float vFOV_orig = 125.26438968f;
    constexpr float r = 0.0625f;

    const float aspect = glm::tan(glm::radians(hFOV_orig / 2.0f)) / glm::tan(glm::radians(vFOV_orig / 2.0f));
    const glm::mat4 projA = glm::perspectiveRH_ZO(glm::radians(vFOV_orig), aspect, mp::kPointLightNear, 1.0f);
    const glm::mat4 invProjA = glm::inverse(projA);

    glm::vec3 centers[4] = {
        {-1.0f, 0.0f, -1.0f},
        {1.0f, 0.0f, -1.0f},
        {0.0f, -1.0f, -1.0f},
        {0.0f, 1.0f, -1.0f},
    };
    const glm::vec3 offsets[4] = {{-r, 0, 0}, {r, 0, 0}, {0, -r, 0}, {0, r, 0}};
    glm::vec3 v[4];
    for (int i = 0; i < 4; ++i)
    {
        centers[i] += offsets[i];
        const glm::vec4 vs = invProjA * glm::vec4(centers[i], 1.0f);
        v[i] = glm::normalize(glm::vec3(vs));
    }

    const float dilatedFovX = glm::degrees(glm::acos(glm::dot(v[0], v[1])));
    const float dilatedFovY = glm::degrees(glm::acos(glm::dot(v[2], v[3])));
    return {dilatedFovX - hFOV_orig, dilatedFovY - vFOV_orig};
}

inline bool is_nearly_zero(const float value)
{
    return FP_ZERO == fpclassify(value);
}

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
    glm::u16vec2 uv;
    glm::u16vec2 normal;
    glm::u16vec2 tangent;
};

struct VertexPosition
{
    glm::vec3 pos;
};

struct VertexAttributes
{
    glm::u16vec2 uv;
    glm::u16vec2 normal;
    glm::u16vec2 tangent;
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
    std::uint32_t emissiveTextureID;
    std::uint32_t emissiveSamplerID;
};

struct Instance
{
    glm::mat4 world;
    std::uint32_t meshIndex;
    MaterialInstanceIndices materialIndices;
};

struct GBufferPassPushConstants
{
    VkDeviceAddress positionBufferAddr;
    VkDeviceAddress attributesBufferAddr;
    VkDeviceAddress instanceBufferDeviceAddr;
    VkDeviceAddress sceneDataBufferDeviceAddr;
};

struct OITForwardPassPushConstants
{
    VkDeviceAddress positionBufferAddr;
    VkDeviceAddress attributesBufferAddr;
    VkDeviceAddress instances;
    VkDeviceAddress sceneData;
    VkDeviceAddress directionalLight;
    VkDeviceAddress visiblePointLights;
    VkDeviceAddress visiblePointLightsCount;
};

struct LightPassPushConstants
{
    VkDeviceAddress sceneData;
    VkDeviceAddress directionalLight;
    VkDeviceAddress visiblePointLights;
    VkDeviceAddress visiblePointLightsCount;
    VkDeviceAddress tetrahedronDataAddr;
    float cameraNear;
    float cameraFar;

    glm::mat4 inverseCameraViewProj;
};
static_assert(sizeof(LightPassPushConstants) == 112);

struct PointLightsShadowPassPushConstants
{
    VkDeviceAddress positionBufferAddr;
    VkDeviceAddress instances;
    VkDeviceAddress visiblePointLights;
    VkDeviceAddress pointLightIndices;
    VkDeviceAddress pointLightOffsets;
    VkDeviceAddress tetrahedronDataAddr;
};

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

struct CullPointLightsPassPushConstants
{
    VkDeviceAddress pointLights;
    VkDeviceAddress pointLightsVisibleBuffer;
    VkDeviceAddress pointLightsVisibleCountBuffer;
    std::uint32_t pointLightsCount;
    std::uint32_t _padding0;
    glm::mat4 viewProj;
};

struct GpuSceneData
{
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 projView;
    glm::vec3 cameraPos;
    float padding0;
};

struct DirectionalShadowPassPushConstants
{
    VkDeviceAddress positionBufferAddr;
    VkDeviceAddress instanceBufferDeviceAddr;
    VkDeviceAddress dirLightsBufferAddr;
    std::uint32_t cascadeCount;
};

struct DirectionalShadowPassAlphaTestedPushConstants
{
    VkDeviceAddress positionBufferAddr;
    VkDeviceAddress attributesBufferAddr;
    VkDeviceAddress instanceBufferDeviceAddr;
    VkDeviceAddress dirLightsBufferAddr;
    std::uint32_t cascadeCount;
};

struct PointLightsShadowPassAlphaTestedPushConstants
{
    VkDeviceAddress positionBufferAddr;
    VkDeviceAddress attributesBufferAddr;
    VkDeviceAddress instances;
    VkDeviceAddress visiblePointLights;
    VkDeviceAddress pointLightIndices;
    VkDeviceAddress pointLightOffsets;
    VkDeviceAddress tetrahedronDataAddr;
};

enum class MaterialPass : std::uint8_t
{
    Opaque,
    AlphaTested,
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

struct TetrahedronData
{
    std::array<glm::vec4, 4> faceVectors;
    std::array<glm::vec4, 12> planeNormals;
};

struct DirectionalLightData
{
    glm::vec3 direction;
    int cascadeCount;
    glm::vec3 color;
    float intensity;
    float normalBias;
    float constantBias;
    float _pad[2];

    std::array<float, kMaxCascadeCount> splitDistances;
    std::array<glm::mat4, kMaxCascadeCount> cascadeVPs;
};

struct PointLightData
{
    glm::vec3 position;
    float range;
    glm::vec3 color;
    float intensity;
    float normalBias;
    float constantBias;
    float _pad[2];
    std::array<glm::mat4, 4> tetrahedronFacesMatrices;

#if 0
    glm::mat4 viewProject;
#endif
};

// GPU structures for SDSM, values are in uint32_t because nvidia gpus do not support atomic float min/max
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
    VkDeviceAddress minMaxAddr;
    VkDeviceAddress splitsAABBAddr;
    VkDeviceAddress dirLightsAddr;
    float near, far;
    glm::mat4 inverseCameraViewProj;
    glm::mat4 lightViewMatrix;
};

struct DirVpPushConstants
{
    VkDeviceAddress splitsAABBAddr;
    VkDeviceAddress dirLightAddr;
    glm::vec3 sceneMin;
    float _pad0{};
    glm::vec3 sceneMax;
    std::uint32_t shadowMapSize{};
    glm::mat4 lightView;
};

struct PopulateCommandsWithCascadeCountPushConstants
{
    VkDeviceAddress commands;
    VkDeviceAddress count;
    int cascadesCount;
};

struct GeneratePointLightCommandsPushConstants
{
    VkDeviceAddress meshes;
    VkDeviceAddress instances;
    VkDeviceAddress meshDrawCommands;
    VkDeviceAddress meshDrawCommandsCount;
    VkDeviceAddress visiblePointLights;
    VkDeviceAddress visiblePointLightsCount;
    std::uint32_t instanceOffset;
    std::uint32_t _padding1;
    VkDeviceAddress pointLightIndices;
    VkDeviceAddress pointLightOffsets;
    VkDeviceAddress pointLightOffsetsCounter;
};
static_assert(sizeof(GeneratePointLightCommandsPushConstants) == 80);

struct LuminanceHistogramPushConstants
{
    VkDeviceAddress histogram; // uint[256]
    float minLogLum;           // e.g. -8.0f
    float invLogLumRange;      // 1.0f / logLumRange
};

struct AverageLuminancePushConstants
{
    VkDeviceAddress histogram;
    VkDeviceAddress avgLum; // float[1], read-write EMA state
    float minLogLum;
    float logLumRange;
    float timeCoeff; // 1 - exp(-dt * adaptationSpeed), computed in C++
    std::uint32_t numPixels;
};

struct PostProcessPushConstants
{
    VkDeviceAddress avgLum;
};

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
