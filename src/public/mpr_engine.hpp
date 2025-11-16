#pragma once

#include "mpr_camera.hpp"
#include "mpr_descriptors.hpp"
#include "mpr_types.hpp"

struct SDL_Window;

namespace mp {
struct GLTFMaterial {
  MaterialInstance data;
};
class Engine;

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

struct EngineStats {
  float frameTime;
  int triangleCount;
  int drawCallCount;
  float sceneUpdateTime;
  float meshDrawTime;
};

constexpr auto kNumberOfFrames = 2;

struct FrameData {
  VkCommandBuffer commandBuffer;
  VkFence fence;
  VkSemaphore swapchainSemaphore;
  DeletionQueue frameDeletionQueue;
  AllocatedImage drawImage;
  AllocatedImage depthImage;
  AllocatedBuffer sceneDataBuffer;
  VkDeviceAddress sceneDataBufferAddr;
  DescriptorBuffer sceneDataDescriptorBuffer;
  VkDescriptorSet drawImageDescriptorSet;
  AllocatedBuffer instanceBuffer;
  VkDeviceAddress instanceBufferAddr;
};

struct GltfMetallicRoughness {
  MaterialPipeline opaquePipeline;
  MaterialPipeline transparentPipeline;

  VkDescriptorSetLayout materialLayout;
  DescriptorBuffer descriptors;
  std::uint32_t currentMaterialOffset = 0;
  std::uint32_t currentSamplerOffset = 0;
  std::uint32_t currentTextureOffset = 0;

  struct alignas(256) MaterialConstants {
    glm::vec4 colorFactors;
    glm::vec4 metalRoughFactors;
  };

  void build_pipelines(Engine& engine);
  void clear_resources(Engine& engine);

  std::uint32_t write_uniform_buffer(VkDeviceAddress uniformBuffer);
  std::uint32_t write_sampler(VkSampler sampler);
  std::uint32_t write_texture(VkImageView imageView);
  MaterialPipeline* select_pipeline(const MaterialPass pass);
};
struct RenderObject {
  std::uint32_t indexCount;
  std::uint32_t firstIndex;
  VkBuffer indexBuffer;
  VkDeviceAddress vertexBufferAddress;

  [[nodiscard]]
  bool operator==(const RenderObject& other) const noexcept {
    return indexCount == other.indexCount && firstIndex == other.firstIndex &&
           indexBuffer == other.indexBuffer &&
           vertexBufferAddress == other.vertexBufferAddress;
  }
};

struct DrawContext {
  struct RenderObjectHash {
    static constexpr auto kHashCombineMagicValue = 0x9e3779b9;
    std::size_t operator()(const RenderObject& ro) const {
      const auto h1 = std::hash<VkBuffer>{}(ro.indexBuffer);
      const auto h2 = std::hash<VkDeviceAddress>{}(ro.vertexBufferAddress);
      const auto h3 = std::hash<std::uint32_t>{}(ro.indexCount);
      const auto h4 = std::hash<std::uint32_t>{}(ro.firstIndex);

      std::size_t seed = h1;
      seed ^= h2 + kHashCombineMagicValue + (seed << 6) + (seed >> 2);
      seed ^= h3 + kHashCombineMagicValue + (seed << 6) + (seed >> 2);
      seed ^= h4 + kHashCombineMagicValue + (seed << 6) + (seed >> 2);
      return seed;
    }
  };
  std::unordered_map<RenderObject, std::vector<Instance>, RenderObjectHash>
      opaqueRenderObjects;
  std::unordered_map<RenderObject, std::vector<Instance>, RenderObjectHash>
      transparentRenderObjects;
};

struct Node : public IRenderable {
  std::weak_ptr<Node> parent;
  std::string name;
  std::uint64_t nodeIndex;
  std::vector<std::shared_ptr<Node>> children;

  glm::mat4 localTransform;
  glm::mat4 worldTransform;

  void refresh_transform(const glm::mat4& parentMatrix) {
    worldTransform = parentMatrix * localTransform;
    for (auto& c : children) {
      c->refresh_transform(worldTransform);
    }
  }

  void draw(const glm::mat4& topMatrix, DrawContext& ctx) override {
    for (auto& c : children) {
      c->draw(topMatrix, ctx);
    }
  }
};
struct MeshAsset;

struct MeshNode final : public Node {
  std::shared_ptr<MeshAsset> mesh;

  MeshNode() = default;
  explicit MeshNode(std::shared_ptr<MeshAsset> mesh_)
      : mesh(std::move(mesh_)) {}
  void draw(const glm::mat4& topMatrix, DrawContext& ctx) override;
};

struct Scene final : public IRenderable {
  std::unordered_map<std::uint64_t, std::shared_ptr<MeshAsset>> meshes;
  std::unordered_map<std::uint64_t, std::shared_ptr<Node>> nodes;
  std::unordered_map<std::uint64_t, std::pair<std::string, AllocatedImage>>
      images;
  std::unordered_map<
      std::uint64_t,
      std::pair<std::string, std::shared_ptr<GLTFMaterial>>>
      materials;

  std::vector<std::shared_ptr<Node>> topNodes;

  std::vector<VkSampler> samplers;

  std::vector<std::pair<AllocatedBuffer, VkDeviceAddress>> materialBuffers;

  Scene() = default;
  Scene(const Scene& other) = delete;
  Scene(Scene&& other) noexcept = delete;
  Scene& operator=(const Scene& other) = delete;
  Scene& operator=(Scene&& other) noexcept = delete;
  ~Scene() override = default;
  // ---
  void draw(const glm::mat4& topMatrix, DrawContext& ctx) override;
  void add_mesh(std::shared_ptr<MeshAsset> mesh);
  void add_image(std::string imageName, const AllocatedImage& image);
  void add_material(std::string materialName,
                    std::shared_ptr<GLTFMaterial> material);
  std::uint64_t add_node(std::shared_ptr<Node> node);
  void clear_all(Engine& engine);
};

class Engine final {
 public:
  Engine(const Engine& other) = delete;
  Engine(Engine&& other) noexcept = delete;
  Engine& operator=(const Engine& other) = delete;
  Engine& operator=(Engine&& other) noexcept = delete;
  ~Engine();

  Engine();

  static Engine& get();

  GpuMeshBuffers create_mesh_buffers(const std::span<std::uint32_t> indices,
                                     const std::span<Vertex> vertices);
  void run();

  void immediate_submit(const std::function<void(VkCommandBuffer)>& function);
  AllocatedBuffer create_buffer(std::size_t allocSize,
                                VkBufferUsageFlags usageFlags,
                                VmaMemoryUsage memoryUsage);
  void destroy_buffer(const AllocatedBuffer& buffer);
  AllocatedImage create_image(VkExtent3D extent, VkFormat format,
                              VkImageUsageFlags imageUsage,
                              bool mipMapped = false);
  AllocatedImage create_image(void* data, VkExtent3D extent, VkFormat format,
                              VkImageUsageFlags imageUsage,
                              bool mipMapped = false);
  void destroy_image(const AllocatedImage& image);
  FrameData& get_current_frame();

  void destroy_sync();
  void destroy_commands();
  void create_draw_image(AllocatedImage& image, VkExtent3D extent);
  void create_depth_image(AllocatedImage& depthImage, VkExtent3D extent);
  void create_swapchain(const std::uint32_t width, const std::uint32_t height);
  void destroy_swapchain();
  void resize_swapchain();

  VkExtent2D m_windowExtent{1920, 1080};
  std::uint64_t m_frameNumber = 0;
  bool m_isInitialized = false;
  bool m_isRenderStopped = false;
  struct WindowCleaner {
    void operator()(SDL_Window* window) const;
  };
  std::unique_ptr<SDL_Window, WindowCleaner> m_window;

  VkInstance m_instance;
  VkDebugUtilsMessengerEXT m_debugMessenger;
  VkPhysicalDevice m_chosenGpu;
  VkDevice m_device;
  VkSurfaceKHR m_surface;

  VkSwapchainKHR m_swapchain;
  VkFormat m_swapchainImageFormat;

  std::vector<VkImage> m_swapchainImages;
  std::vector<VkImageView> m_swapchainImageViews;
  VkExtent2D m_swapchainExtent;

  VkQueue m_queue;
  std::uint32_t m_queueFamilyIndex;
  // TODO: For multithreading add 1 per thread
  VkCommandPool m_commandPool;
  std::array<FrameData, kNumberOfFrames> m_frameData;
  std::vector<VkSemaphore> m_swapchainSemaphores;

  DeletionQueue m_mainDeletionQueue;
  VmaAllocator m_allocator;
  VkExtent2D m_drawExtent;
  VkDescriptorSetLayout m_drawImageDescriptorSetLayout;
  VkDescriptorSetLayout m_gpuSceneDataDescriptorSetLayout;

  VkPipelineLayout m_backgroundPipelineLayout;

  // Simple immediate submit structures
  // For future optimizations consider adding queue
  VkCommandPool m_immCommandPool;
  VkCommandBuffer m_immCommandBuffer;
  VkFence m_immFence;

  std::vector<ComputeEffect> m_computeEffects;
  int m_currentComputeEffect{0};

  VkPipeline m_meshPipeline;

  GpuPushConstants m_pushConstants;

  bool m_bSwapchainResizeRequest = false;
  float m_renderScale{1.0f};

  AllocatedImage m_whiteImage;
  AllocatedImage m_blackImage;
  AllocatedImage m_greyImage;
  AllocatedImage m_errorImage;

  VkSampler m_defaultSamplerLinear;
  VkSampler m_defaultSamplerNearest;

  GltfMetallicRoughness m_metalRoughness;

  DrawContext m_mainDrawContext;
  Scene m_scene;

  Camera m_camera;

  EngineStats m_stats{};

  GpuSceneData m_sceneData;

  VkDescriptorPool m_drawImageDescPool;

  std::uint64_t m_selectedNode = UINT64_MAX;

 private:
  void init_window();
  void init_vulkan();
  void init_swapchain();
  void init_commands();
  void init_sync();
  void init_descriptors();
  void init_pipelines();
  void init_background_pipelines();
  void init_imgui();
  void init_mesh_data();
  void init_default_data();
  void draw();
  void draw_background(VkCommandBuffer cmd);
  void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);
  void draw_geometry(VkCommandBuffer cmd, VkImageView colorImageView,
                     VkImageView depthImageView, VkExtent2D imageExtent);
  void update_scene();


  static std::uint64_t render_scene_tree_ui(
      Scene& scene, std::uint64_t nodeIndex,
      std::uint64_t selectedNode);
  bool edit_transform_ui(const glm::mat4& view, const glm::mat4& projection, glm::mat4& globalTransform);
  void edit_node(Scene& scene, std::uint64_t nodeIndex);
};

}  // namespace mp
