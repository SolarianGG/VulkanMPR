#pragma once

#include "mpr_descriptors.hpp"
#include "mpr_types.hpp"
#include "mpr_camera.hpp"

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
};

class Engine final {
  friend struct GLTFMetallic_Roughness;

 public:
  Engine(const Engine& other) = delete;
  Engine(Engine&& other) noexcept = delete;
  Engine& operator=(const Engine& other) = delete;
  Engine& operator=(Engine&& other) noexcept = delete;
  ~Engine();

  Engine();

  static Engine& get();

  void draw();
  GpuMeshBuffers create_mesh_buffers(std::span<std::uint32_t> indices,
                                     std::span<Vertex> vertices);
  void run();

 private:
  void draw_background(VkCommandBuffer cmd);
  void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);
  void draw_geometry(VkCommandBuffer cmd, VkImageView colorImageView,
                     VkImageView depthImageView, VkExtent2D imageExtent);
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
  void destroy_sync();
  void destroy_commands();
  void create_draw_image(AllocatedImage& image, VkExtent3D extent);
  void create_depth_image(AllocatedImage& depthImage, VkExtent3D extent);
  void create_swapchain(const std::uint32_t width, const std::uint32_t height);
  void destroy_swapchain();
  void resize_swapchain();

 private:
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

  std::vector<std::shared_ptr<MeshAsset>> m_testAssets;
  int m_assetIndex{0};

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
  std::unordered_map<std::string, std::shared_ptr<Node>> m_loadedNodes;

  Camera m_camera;

  void update_scene();
};

}  // namespace mp
