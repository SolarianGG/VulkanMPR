#pragma once

#include "mpr_types.hpp"

#include "mpr_camera.hpp"
#include "mpr_descriptors.hpp"
#include "mpr_materials.hpp"
#include "mpr_scene.hpp"

struct SDL_Window;

namespace mp
{

struct EngineStats
{
    float frameTime;
    int triangleCount;
    int drawCallCount;
    float sceneUpdateTime;
    float gBufferPassTime;
    float gBufferLightPassTime;
    float transparentForwardLightPassTime;
    float imguiDrawTime;
    float postProcessPassTime;
    float shadowPassDrawTime;
};


struct FrameData
{
    VkCommandBuffer commandBuffer;
    VkFence fence;
    VkSemaphore swapchainSemaphore;
    DeletionQueue frameDeletionQueue;
    struct
    {
        AllocatedImage normal;
        AllocatedImage diffuse;
        AllocatedImage specular;
    } gBuffer;
    AllocatedImage drawImage;
    AllocatedImage depthImage;
    AllocatedImage oitAccImage;
    AllocatedImage oitRevealImage;
    AllocatedImage directionalShadowPassDepthArray;            // array image view (for sampling)
    AllocatedImage pointLightsShadowTileMap;

    DescriptorBuffer lightPassDescriptorBuffer;
    DescriptorBuffer wboitCompositePassDescBuffer;
    DescriptorBuffer cullPassDescBuffer;
    DescriptorBuffer drawImageDescriptorBuffer;

    AllocatedBuffer sceneDataBuffer;
    VkDeviceAddress sceneDataBufferAddr;
    AllocatedBuffer dirLightBuffer;
    VkDeviceAddress dirLightBufferAddr;
    AllocatedBuffer pointLightBuffer;
    VkDeviceAddress pointLightBufferAddr;
    AllocatedBuffer visiblePointLightsBuffer;
    VkDeviceAddress visiblePointLightsBufferAddr;
    AllocatedBuffer visiblePointLightsCountBuffer;
    VkDeviceAddress visiblePointLightsCountBufferAddr;
    AllocatedBuffer pointLightIndicesBuffer;
    VkDeviceAddress pointLightIndicesBufferAddr;
    AllocatedBuffer pointLightIndicesOffsetsBuffer;
    VkDeviceAddress pointLightIndicesOffsetsBufferAddr;
    AllocatedBuffer pointLightIndicesOffsetsCounterBuffer;
    VkDeviceAddress pointLightIndicesOffsetsCounterBufferAddr;
    AllocatedBuffer minMaxBuffer;
    VkDeviceAddress minMaxBufferAddr;
    AllocatedBuffer splitsAABBBuffer;
    VkDeviceAddress splitsAABBBufferAddr;
    DescriptorBuffer cascadeDepthDescBuffer;
    AllocatedBuffer instanceBuffer;
    VkDeviceAddress instanceBufferAddr;
    AllocatedBuffer meshesBuffer;
    VkDeviceAddress meshBufferAddr;
    AllocatedBuffer drawCommandsBuffer;
    VkDeviceAddress drawCommandsBufferAddr;
    AllocatedBuffer countBuffer;
    VkDeviceAddress countBufferAddr;
};

class Engine final
{
  public:
    Engine(const Engine &other) = delete;
    Engine(Engine &&other) noexcept = delete;
    Engine &operator=(const Engine &other) = delete;
    Engine &operator=(Engine &&other) noexcept = delete;
    ~Engine();

    Engine();

    static Engine &get();

    void run();

    void immediate_submit(const std::function<void(VkCommandBuffer)> &function);
    AllocatedBuffer create_buffer(std::size_t allocSize, VkBufferUsageFlags usageFlags, VmaMemoryUsage memoryUsage);
    void destroy_buffer(const AllocatedBuffer &buffer);
    AllocatedImage create_image(VkExtent3D extent, VkFormat format, VkImageUsageFlags imageUsage,
                                bool mipMapped = false);
    AllocatedImage create_image(void *data, VkExtent3D extent, VkFormat format, VkImageUsageFlags imageUsage,
                                bool mipMapped = false);
    void destroy_image(const AllocatedImage &image);
    FrameData &get_current_frame();

    void destroy_sync();
    void destroy_commands();
    void create_draw_image(AllocatedImage &image, VkExtent3D extent);
    void create_depth_image(AllocatedImage &depthImage, VkExtent3D extent);
    void create_swapchain(const std::uint32_t width, const std::uint32_t height);
    void destroy_swapchain();
    void resize_swapchain();

    VkExtent2D m_windowExtent{1920, 1080};
    std::uint64_t m_frameNumber = 0;
    bool m_isInitialized = false;
    bool m_isRenderStopped = false;
    struct WindowCleaner
    {
        void operator()(SDL_Window *window) const;
    };
    std::unique_ptr<SDL_Window, WindowCleaner> m_window{};

    VkInstance m_instance{};
    VkDebugUtilsMessengerEXT m_debugMessenger{};
    VkPhysicalDevice m_chosenGpu{};
    VkDevice m_device{};
    VkSurfaceKHR m_surface{};

    VkSwapchainKHR m_swapchain{};
    VkFormat m_swapchainImageFormat{};

    std::vector<VkImage> m_swapchainImages{};
    std::vector<VkImageView> m_swapchainImageViews{};
    VkExtent2D m_swapchainExtent{};

    VkQueue m_queue{};
    std::uint32_t m_queueFamilyIndex{};
    // TODO: For multithreading add 1 per thread
    VkCommandPool m_commandPool{};
    std::array<FrameData, kNumberOfFrames> m_frameData{};
    std::vector<VkSemaphore> m_swapchainSemaphores{};

    DeletionQueue m_mainDeletionQueue{};
    VmaAllocator m_allocator{};
    VkExtent2D m_drawExtent{};
    VkDescriptorSetLayout m_LightPassDescriptorSetLayout{};

    VkPipelineLayout m_LightPassPipelineLayout{};
    VkPipeline m_LightPassPipeline{};
    LightPassPushConstants m_LightPassConstants{};

    // Simple immediate submit structures
    // For future optimizations consider adding queue
    VkCommandPool m_immCommandPool{};
    VkCommandBuffer m_immCommandBuffer{};
    VkFence m_immFence{};

    VkPipeline m_meshPipeline{};

    GBufferPassPushConstants m_MeshPassPushConstants{};

    bool m_bSwapchainResizeRequest = false;

    AllocatedImage m_whiteImage{};
    AllocatedImage m_blackImage{};
    AllocatedImage m_greyImage{};
    AllocatedImage m_errorImage{};
    AllocatedImage m_normalFallback{};

    VkSampler m_defaultSamplerLinear{};
    VkSampler m_defaultSamplerNearest{};
    VkSampler m_shadowSampler{};
    VkSampler m_debugSampler{};

    GLTFMetallicRoughness m_metalRoughness{};

    DrawContext m_mainDrawContext{};
    Scene m_scene{};

    Camera m_camera{};

    EngineStats m_stats{};

    VkExtent3D m_CommonImageExtent3D{};
    VkExtent2D m_CommonImageExtent2D{};

    AllocatedBuffer m_tetrahedronBuffer{};

    AllocatedBuffer m_globalVertexBuffer{};
    VkDeviceAddress m_globalVertexBufferAddress{};
    std::size_t m_globalVertexCount = 0;
    std::size_t m_globalVertexCapacity = 0;

    AllocatedBuffer m_globalIndexBuffer{};
    std::size_t m_globalIndexCount = 0;
    std::size_t m_globalIndexCapacity = 0;

    void ensure_vertex_capacity(std::size_t additionalCount);
    void ensure_index_capacity(std::size_t additionalCount);

    GpuSceneData m_sceneData{};

    GBufferPassPushConstants m_GBufferMeshPushConstants{};
    OITForwardPassPushConstants m_WBOITForwardPassPushConstants{};

    VkPipelineLayout m_WBOITCompositePassPipelineLayout{};
    VkPipeline m_WBOITCompositePassPipeline{};

    std::uint64_t m_selectedNode = UINT64_MAX;

    VkDescriptorSetLayout m_WboitCompositePassDescriptorSetLayout{};

    VkDescriptorSetLayout m_DrawImageDescriptorSetLayout{};
    VkPipeline m_PostProcessPassPipeline{};
    VkPipelineLayout m_PostProcessPassPipelineLayout{};

#if 0
  VkDescriptorSetLayout m_CullPassDescriptorSetLayout;
#endif
    VkPipeline m_CullPassPipeline{};
    VkPipelineLayout m_CullPassPipelineLayout{};

    VkPipeline m_CullPointLightsPassPipeline{};
    VkPipelineLayout m_CullPointLightsPassPipelineLayout{};

    std::uint32_t m_OpaqueSize = 0;
    std::uint32_t m_TransparentSize = 0;

    Instance *m_CurrentFrameInstanceBuffer = nullptr;
    RenderObject *m_CurrentMeshBuffer = nullptr;

    VkPipeline m_ShadowPassPipeline{};
    VkPipelineLayout m_ShadowPassPipelineLayout{};

    VkPipeline m_PointLightShadowPassPipeline{};
    VkPipelineLayout m_PointLightShadowPassPipelineLayout{};
    VkPipeline m_GeneratePointLightCommandsPipeline{};
    VkPipelineLayout m_GeneratePointLightCommandsPipelineLayout{};

    VkPipeline m_PrepassPipeline{};
    VkPipelineLayout m_PrepassPipelineLayout{};

    DepthReductionPushConstants m_depthReductionPushConstants{};

    VkDescriptorSetLayout m_DepthPassDescSetLayout{};
    VkPipeline m_DepthReductionPipeline{};
    VkPipelineLayout m_DepthReductionPipelineLayout{};
    VkPipeline m_DepthPartitionPipeline{};
    VkPipelineLayout m_DepthPartitionPipelineLayout{};
    VkPipeline m_DirVpPipeline{};
    VkPipelineLayout m_DirVpPipelineLayout{};

    glm::mat4 m_DirLightCullMatrix{1.0f};
    glm::mat4 m_DirLightViewMatrix{1.0f};

  private:
    void init_window();
    void init_vulkan();
    void init_swapchain();
    void init_commands();
    void init_sync();
    void init_descriptors();
    void init_pipelines();
    void init_light_pass_pipeline();
    void init_wboit_composite_pass_pipeline();
    void init_depth_reduction_pass();
    void init_directional_shadow_pass();
    void init_point_shadow_pass();
    void init_prepass();
    void init_post_pipeline();
    void init_cull_meshes_pipeline();
    void init_cull_point_lights_pipeline();
    void init_generate_point_light_commands_pipeline();
    void init_imgui();
    void init_mesh_data();
    void init_default_data();
    void draw();
    void draw_directional_shadow_pass(VkCommandBuffer cmd);
    void draw_light_pass(VkCommandBuffer cmd);
    void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);
    void draw_gBuffer_pass(VkCommandBuffer cmd);
    void draw_wboit(VkCommandBuffer cmd);
    void draw_wboit_composite(VkCommandBuffer cmd);
    void update_scene();
    void draw_post(VkCommandBuffer cmd);
    void draw_prepass(VkCommandBuffer cmd);
    void compute_depth_reduction(VkCommandBuffer cmd);
    void compute_depth_partition(VkCommandBuffer cmd);
    void compute_dir_lights_vp(VkCommandBuffer cmd);
    void draw_meshes(VkCommandBuffer cmd, const VkPipelineLayout drawPassPipelineLayout, const VkPipeline drawPipeline,
                     const std::uint32_t objectCount, auto &pushConstants,
                     const VkShaderStageFlags pushConstantsShaderStage);

    void cull_objects(VkCommandBuffer cmd, std::uint32_t objectCount, std::uint32_t objectOffset,
                      const glm::mat4 &viewProj);
    void cull_point_lights(VkCommandBuffer cmd);
    void compute_point_lights_commands(VkCommandBuffer cmd);
    void draw_point_lights_shadows_pass(VkCommandBuffer cmd);
    void copy_frame_buffers();

    void init_frames_data();

    static std::uint64_t render_scene_tree_ui(Scene &scene, std::uint64_t nodeIndex, std::uint64_t selectedNode);
    bool edit_transform_ui(const glm::mat4 &view, const glm::mat4 &projection, glm::mat4 &globalTransform);
    void edit_node(Scene &scene, std::uint64_t nodeIndex);
};

} // namespace mp
