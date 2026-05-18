#pragma once

#include "mpr_types.hpp"

namespace mp
{

class Engine;

struct RenderObject
{
    std::uint32_t indexCount;
    std::uint32_t firstIndex;
    std::int32_t vertexOffset;
    std::uint32_t padding;
    glm::vec3 min;
    float padding1;
    glm::vec3 max;
    float padding2;

    [[nodiscard]]
    bool operator==(const RenderObject &other) const noexcept
    {
        return indexCount == other.indexCount && firstIndex == other.firstIndex && vertexOffset == other.vertexOffset;
    }
};

struct DrawContext
{
    struct RenderObjectHash
    {
        static constexpr auto kHashCombineMagicValue = 0x9e3779b9;

        std::size_t operator()(const RenderObject &ro) const
        {
            const auto h1 = std::hash<std::uint32_t>{}(ro.indexCount);
            const auto h2 = std::hash<std::uint32_t>{}(ro.firstIndex);
            const auto h3 = std::hash<std::int32_t>{}(ro.vertexOffset);

            std::size_t seed = h1;
            seed ^= h2 + kHashCombineMagicValue + (seed << 6) + (seed >> 2);
            seed ^= h3 + kHashCombineMagicValue + (seed << 6) + (seed >> 2);
            return seed;
        }
    };

    void clear();
    std::vector<RenderObject> renderObjects;

    std::vector<Instance> opaqueInstances;
    std::vector<Instance> alphaTestedInstances;
    std::vector<Instance> transparentInstances;

    std::unordered_map<RenderObject, std::uint32_t, RenderObjectHash> opaqueMeshes;
    std::unordered_map<RenderObject, std::uint32_t, RenderObjectHash> alphaTestedMeshes;
    std::unordered_map<RenderObject, std::uint32_t, RenderObjectHash> transparentMeshes;
    std::optional<DirectionalLightData> dirLight;
    std::vector<PointLightData> pointLights;
    std::vector<DDGIVolume> ddgiVolumes;
    std::vector<DDGIVolumeVisEntry> ddgiVolumesVis;

    glm::vec3 min{FLT_MAX};
    glm::vec3 max{-FLT_MAX};
};

struct Node : public IRenderable
{
    std::weak_ptr<Node> parent;
    std::string name{"Name"};
    std::uint64_t nodeIndex = 0;
    std::vector<std::shared_ptr<Node>> children;

    glm::mat4 localTransform{1.0f};
    glm::mat4 worldTransform{1.0f};

    void refresh_transform(const glm::mat4 &parentMatrix)
    {
        worldTransform = parentMatrix * localTransform;
        for (auto &c : children)
        {
            c->refresh_transform(worldTransform);
        }
    }

    void draw(const glm::mat4 &topMatrix, DrawContext &ctx) override
    {
        for (auto &c : children)
        {
            c->draw(topMatrix, ctx);
        }
    }

    virtual void edit()
    {
    }
};

struct MeshAsset;

struct MeshNode final : public Node
{
    std::shared_ptr<MeshAsset> mesh;

    MeshNode() = default;

    explicit MeshNode(std::shared_ptr<MeshAsset> mesh_) : mesh(std::move(mesh_))
    {
    }

    void draw(const glm::mat4 &topMatrix, DrawContext &ctx) override;
};

struct DirectionalLightNode : public Node
{
  public:
    DirectionalLightData m_Data{};

  public:
    DirectionalLightNode() = default;

    explicit DirectionalLightNode(DirectionalLightData data) : m_Data(std::move(data))
    {
    }

    void draw(const glm::mat4 &topMatrix, DrawContext &ctx) override;

    void edit() override;
};

struct PointLightNode : public Node
{
  public:
    PointLightData m_Data{};

  public:
    PointLightNode() = default;

    explicit PointLightNode(PointLightData data) : m_Data(std::move(data))
    {
    }

    void draw(const glm::mat4 &topMatrix, DrawContext &ctx) override;

    void edit() override;
};

struct DDGIVolumeNode : public Node
{
  public:
    DDGIVolume m_Data{};
    bool m_bVisualize{false};
    float m_ProbeRadius{0.05f};
    DDGIProbeVisMode m_visMode{DDGIProbeVisMode::GridPosition};

  public:
    DDGIVolumeNode() = default;
    explicit DDGIVolumeNode(const DDGIVolume &volume, bool bVisualize = false)
        : m_Data(volume), m_bVisualize(bVisualize)
    {
    }

    void draw(const glm::mat4 &topMatrix, DrawContext &ctx) override;

    void edit() override;
};

struct Scene final : public IRenderable
{
    std::unordered_map<std::uint64_t, std::shared_ptr<MeshAsset>> meshes;
    std::unordered_map<std::uint64_t, std::shared_ptr<Node>> nodes;
    std::unordered_map<std::uint64_t, std::pair<std::string, AllocatedImage>> images;
    std::unordered_map<std::uint64_t, std::pair<std::string, std::shared_ptr<GLTFMaterial>>> materials;

    std::vector<std::shared_ptr<Node>> topNodes;

    std::vector<VkSampler> samplers;

    std::vector<std::pair<AllocatedBuffer, VkDeviceAddress>> materialBuffers;

    Scene() = default;
    Scene(const Scene &other) = delete;
    Scene(Scene &&other) noexcept = delete;
    Scene &operator=(const Scene &other) = delete;
    Scene &operator=(Scene &&other) noexcept = delete;
    ~Scene() override = default;
    // ---
    void draw(const glm::mat4 &topMatrix, DrawContext &ctx) override;
    void add_mesh(std::shared_ptr<MeshAsset> mesh);
    void add_image(std::string imageName, const AllocatedImage &image);
    void add_material(std::string materialName, std::shared_ptr<GLTFMaterial> material);
    std::uint64_t add_node(std::shared_ptr<Node> node);
    void clear_all(Engine &engine);
};

} // namespace mp