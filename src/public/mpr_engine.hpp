#pragma once

#include "mpr_descriptors.hpp"
#include "mpr_types.hpp"

struct SDL_Window;

namespace mp {
struct MeshAsset;

constexpr auto kNumberOfFrames = 2;

class Engine final {
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
  void destroy_buffer(AllocatedBuffer& buffer);
  FrameData& get_current_frame();

  void init_window();
  void init_vulkan();
  void init_swapchain();
  void init_commands();
  void init_sync();
  void init_descriptors();
  void init_pipelines();
  void init_background_pipelines();
  void init_mesh_pipelines();
  void init_imgui();
  void init_mesh_data();
  void destroy_sync();
  void destroy_commands();
  void create_draw_images(VkExtent3D extent);
  void create_depth_images(VkExtent3D extent);
  void create_swapchain(const std::uint32_t width, const std::uint32_t height);
  void destroy_swapchain();
  void resize_swapchain();

 private:
  VkExtent2D m_windowExtent{1700, 900};
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
  std::array<AllocatedImage, kNumberOfFrames> m_drawImages;
  std::array<AllocatedImage, kNumberOfFrames> m_depthImages;
  VkExtent2D m_drawExtent;

  DescriptorAllocator m_descriptorAllocator;
  VkDescriptorSetLayout m_drawImageDescriptorSetLayout;
  std::array<VkDescriptorSet, kNumberOfFrames> m_drawImagesDescriptors;
  VkPipelineLayout m_pipelineLayout;

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

  glm::vec3 m_eyePos;
  float m_centerRadius{5.0f};
  bool m_bSwapchainResizeRequest = false;
  float m_renderScale{1.0f};
};

}  // namespace mp
