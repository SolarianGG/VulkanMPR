#pragma once

#include "mpr_camera.hpp"
#include "mpr_descriptors.hpp"
#include "mpr_types.hpp"

struct SDL_Window;

namespace mp {
struct MeshAsset;
class Engine;

constexpr auto kNumberOfFrames = 2;

struct FrameData {
  VkCommandBuffer commandBuffer;
  VkFence fence;
  VkSemaphore swapchainSemaphore;
  DeletionQueue frameDeletionQueue;
  AllocatedImage drawImage;
  AllocatedImage depthImage;
  DescriptorAllocatorGrowable descriptorAllocator;
  VkDescriptorSet drawImageDescriptorSet;
};

struct GLTFMetallic_Roughness {
  MaterialPipeline opaquePipeline;
  MaterialPipeline transparentPipeline;

  VkDescriptorSetLayout materialLayout;

  struct alignas(256) MaterialConstants {
    glm::vec4 colorFactors;
    glm::vec4 metalRoughFactors;
  };

  struct MaterialResources {
    AllocatedImage colorImage;
    VkSampler colorSampler;
    AllocatedImage metalRoughnessImage;
    VkSampler metalRoughnessSampler;
    VkBuffer dataBuffer;
    std::uint32_t dataBufferOffset;
  };

  DescriptorWriter writer;

  void build_pipelines(Engine& engine);
  void clear_resources(VkDevice device);

  MaterialInstance write_material(VkDevice device, MaterialPass matPass,
                                  const MaterialResources& res,
                                  DescriptorAllocatorGrowable& allocator);
};
struct RenderObject {
  std::uint32_t indexCount;
  std::uint32_t firstIndex;
  VkBuffer indexBuffer;

  MaterialInstance* material;

  glm::mat4 transform;
  VkDeviceAddress vertexBufferAddress;
};

struct DrawContext {
  std::vector<RenderObject> opaqueSurfaces;
  std::vector<RenderObject> transparentSurfaces;
};

struct Node : public IRenderable {
  std::weak_ptr<Node> parent;
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

struct MeshNode : public Node {
  std::shared_ptr<MeshAsset> mesh;

  MeshNode() = default;
  explicit MeshNode(std::shared_ptr<MeshAsset> mesh_) : mesh(std::move(mesh_)) {}
  void draw(const glm::mat4& topMatrix, DrawContext& ctx) override;
};

struct LoadedGLTF : public IRenderable {
  std::unordered_map<std::string, std::shared_ptr<MeshAsset>> meshes;
  std::unordered_map<std::string, std::shared_ptr<Node>> nodes;
  std::unordered_map<std::string, AllocatedImage> images;
  std::unordered_map<std::string, std::shared_ptr<struct GLTFMaterial>>
      materials;

  std::vector<std::shared_ptr<Node>> topNodes;

  std::vector<VkSampler> samplers;

  DescriptorAllocatorGrowable descriptorAllocator;

  AllocatedBuffer materialDataBuffer;

  Engine* creator;

  LoadedGLTF() = default;
  LoadedGLTF(const LoadedGLTF& other) = delete;
  LoadedGLTF(LoadedGLTF&& other) noexcept = delete;
  LoadedGLTF& operator=(const LoadedGLTF& other) = delete;
  LoadedGLTF& operator=(LoadedGLTF&& other) noexcept = delete;
  ~LoadedGLTF() override { clear_all(); }
  void draw(const glm::mat4& topMatrix, DrawContext& ctx) override;

 private:
  void clear_all();
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

  GpuMeshBuffers create_mesh_buffers(std::span<std::uint32_t> indices,
                                     std::span<Vertex> vertices);
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
  GpuSceneData m_sceneData;
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
  VkPipelineLayout m_meshPipelineLayout;

  GpuPushConstants m_pushConstants;


  bool m_bSwapchainResizeRequest = false;
  float m_renderScale{1.0f};

  AllocatedImage m_whiteImage;
  AllocatedImage m_blackImage;
  AllocatedImage m_greyImage;
  AllocatedImage m_errorImage;

  VkSampler m_defaultSamplerLinear;
  VkSampler m_defaultSamplerNearest;
  VkDescriptorSetLayout m_texturedSetLayout;

  MaterialInstance m_defaultData;
  GLTFMetallic_Roughness m_metalRoughness;

  DescriptorAllocatorGrowable m_globalDescAllocator;

  DrawContext m_mainDrawContext;
  std::unordered_map<std::string, std::shared_ptr<LoadedGLTF>> m_loadedScenes;

  Camera m_camera;

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
};

}  // namespace mp
