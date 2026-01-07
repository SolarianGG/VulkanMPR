#pragma once

#include "mpr_camera.hpp"
#include "mpr_descriptors.hpp"
#include "mpr_types.hpp"
#include "mpr_materials.hpp"
#include "mpr_scene.hpp"

struct SDL_Window;

namespace mp {

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
  struct {
    AllocatedImage position; 
    AllocatedImage normal; 
    AllocatedImage diffuse; 
    AllocatedImage specular; 
  } gBuffer;
  AllocatedImage drawImage;
  AllocatedImage depthImage;
  AllocatedBuffer sceneDataBuffer;
  VkDeviceAddress sceneDataBufferAddr;
  DescriptorBuffer drawImageDescriptorBuffer;
  AllocatedBuffer instanceBuffer;
  VkDeviceAddress instanceBufferAddr;
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

  VkExtent2D m_windowExtent{1600, 900};
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

  VkPipelineLayout m_LightPassPipelineLayout;
  VkPipeline m_LightPassPipeline;
  LightPassConstantRange m_LightPassConstants;

  // Simple immediate submit structures
  // For future optimizations consider adding queue
  VkCommandPool m_immCommandPool;
  VkCommandBuffer m_immCommandBuffer;
  VkFence m_immFence;

  VkPipeline m_meshPipeline;

  GpuPushConstants m_MeshPassPushConstants;

  bool m_bSwapchainResizeRequest = false;
  float m_renderScale{1.0f};

  AllocatedImage m_whiteImage;
  AllocatedImage m_blackImage;
  AllocatedImage m_greyImage;
  AllocatedImage m_errorImage;

  VkSampler m_defaultSamplerLinear;
  VkSampler m_defaultSamplerNearest;

  GLTFMetallicRoughness m_metalRoughness;

  DrawContext m_mainDrawContext;
  Scene m_scene;

  Camera m_camera;

  EngineStats m_stats{};

  GpuSceneData m_sceneData;

  std::uint64_t m_selectedNode = UINT64_MAX;

 private:
  void init_window();
  void init_vulkan();
  void init_swapchain();
  void init_commands();
  void init_sync();
  void init_descriptors();
  void init_pipelines();
  void init_light_pass_pipeline();
  void init_imgui();
  void init_mesh_data();
  void init_default_data();
  void draw();
  void draw_background(VkCommandBuffer cmd);
  void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);
  void draw_geometry(VkCommandBuffer cmd);
  void update_scene();


  static std::uint64_t render_scene_tree_ui(
      Scene& scene, std::uint64_t nodeIndex,
      std::uint64_t selectedNode);
  bool edit_transform_ui(const glm::mat4& view, const glm::mat4& projection, glm::mat4& globalTransform);
  void edit_node(Scene& scene, std::uint64_t nodeIndex);
};

}  // namespace mp
