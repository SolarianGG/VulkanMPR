// clang-format off
#define GLM_ENABLE_EXPERIMENTAL
#include "mpr_engine.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <VkBootstrap.h>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>
#include <glm/gtc/type_ptr.hpp>
#include <tinyfiledialogs.h>

#include <format>
#include <print>

#include "mpr_error_check.hpp"
#include "mpr_init_vk_stucts.hpp"
#include "mpr_loader.hpp"
#include "mpr_pipelines.hpp"
#include "mpr_debug_utils.hpp"
#include "mpr_image.hpp"
// clang-format on

namespace
{

#ifdef MPR_DEBUG
constexpr bool bUseValidationLayers = true;
#else
constexpr bool bUseValidationLayers = false;
#endif
constexpr auto kBaseWindowTitle = "Hello Vulkan";

std::pair<std::uint32_t, char const *const *> get_required_instance_extensions_for_window()
{
    std::uint32_t count;
    const auto requiredExtensions = SDL_Vulkan_GetInstanceExtensions(&count);
    return {count, requiredExtensions};
}

std::vector<std::filesystem::path> parse_tiny_multiple(const wchar_t *files)
{
    std::vector<std::filesystem::path> res;
    if (!files)
        return res;

    std::wstring str = files;

    if (str.find(L'|') == std::wstring::npos)
    {
        res.emplace_back(std::move(str));
        return res;
    }

    std::vector<std::wstring> parts;
    std::wstring::size_type currentOffset = 0;
    while (true)
    {
        auto newOffset = str.find(L'|');
        if (newOffset == std::wstring::npos)
        {
            parts.emplace_back(str.substr(currentOffset));
            break;
        }
        parts.emplace_back(str.substr(currentOffset, newOffset - currentOffset));
        currentOffset = newOffset + 1;
    }

    return res;
}

} // namespace

namespace mp
{

void Engine::init_window()
{
    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        std::println("Failed to init SDL: {}", SDL_GetError());
    }
    atexit(SDL_Quit);

    constexpr SDL_WindowFlags windowFlags = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE;

    m_window = {SDL_CreateWindow(kBaseWindowTitle, m_windowExtent.width, m_windowExtent.height, windowFlags),
                WindowCleaner{}};
    if (!m_window)
    {
        std::println("Failed to create window: {}", SDL_GetError());
    }
}

void Engine::init_vulkan()
{
    volkInitialize() >> chk;
    vkb::InstanceBuilder builder;
    {
        const auto [numberOfRequiredExtensions, requiredExtensions] = get_required_instance_extensions_for_window();
        builder = vkb::InstanceBuilder()
                      .request_validation_layers(bUseValidationLayers)
                      .require_api_version(1, 3, 0)
                      .enable_extensions(numberOfRequiredExtensions, requiredExtensions);
    }

    if constexpr (bUseValidationLayers)
    {
        builder.use_default_debug_messenger();
    }

    const auto result = builder.build();

    if (!result.has_value())
    {
        throw std::runtime_error("Failed to create instance");
    }
    m_instance = result.value().instance;
    m_debugMessenger = result.value().debug_messenger;

    volkLoadInstance(m_instance);

    if (!SDL_Vulkan_CreateSurface(m_window.get(), m_instance, nullptr, &m_surface))
    {
        throw std::runtime_error(std::format("Failed to create surface: {}", SDL_GetError()));
    }

    VkPhysicalDeviceDescriptorBufferFeaturesEXT descriptorBufferFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT,
        .pNext = nullptr,
        .descriptorBuffer = true,
#if 0
    .descriptorBufferImageLayoutIgnored = true,
#endif
    };

    const VkPhysicalDeviceVulkan13Features features13{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .shaderDemoteToHelperInvocation = true,
        .synchronization2 = true,
        .dynamicRendering = true,
        .shaderIntegerDotProduct = true,
    };

    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
        .shaderBufferFloat32Atomics = true,
        .shaderBufferFloat32AtomicAdd = true,
        .shaderSharedFloat32Atomics = true,
        .shaderSharedFloat32AtomicAdd = true,
        .shaderImageFloat32Atomics = true,
        .shaderImageFloat32AtomicAdd = true,
        .sparseImageFloat32Atomics = true,
        .sparseImageFloat32AtomicAdd = true};

    const VkPhysicalDeviceVulkan12Features features12{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .drawIndirectCount = true,
        .shaderFloat16 = true,
        .descriptorIndexing = true,
        .shaderSampledImageArrayNonUniformIndexing = true,
        .descriptorBindingPartiallyBound = true,
        .descriptorBindingVariableDescriptorCount = true,
        .runtimeDescriptorArray = true,
        .scalarBlockLayout = true,
        .bufferDeviceAddress = true,
        .shaderOutputLayer = true,
    };

    const VkPhysicalDeviceVulkan11Features features11{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        .storageBuffer16BitAccess = true,
        .storagePushConstant16 = true,
        .shaderDrawParameters = true,
    };
    VkPhysicalDeviceFeatures features10{
        .independentBlend = true,
        .geometryShader = true,
        .samplerAnisotropy = true,
        .textureCompressionBC = true,
        .shaderClipDistance = true,
        .shaderInt64 = true,
        .shaderInt16 = true,
    };

    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
        .accelerationStructure = true,
    };
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
        .rayTracingPipeline = VK_TRUE,
    };

    vkb::PhysicalDeviceSelector selector{result.value()};

    std::vector<const char *> requiredExtensions{
        VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME,           VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
        VK_NV_SHADER_SUBGROUP_PARTITIONED_EXTENSION_NAME, // VK_EXT_SHADER_SUBGROUP_PARTITIONED_EXTENSION_NAME
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    };
    const auto physicalDevice = selector.set_minimum_version(1, 3)
                                    .add_required_extensions(requiredExtensions)
                                    .set_required_features_13(features13)
                                    .add_required_extension_features(descriptorBufferFeatures)
                                    .add_required_extension_features(atomicFloatFeatures)
                                    .add_required_extension_features(accelFeature)
                                    .add_required_extension_features(rtPipelineFeature)
                                    .set_required_features_12(features12)
                                    .set_required_features_11(features11)
                                    .set_required_features(features10)
                                    .set_surface(m_surface)
                                    .select();

    vkb::DeviceBuilder deviceBuilder{physicalDevice.value()};

    vkb::Device vkbDevice = deviceBuilder.build().value();

    m_device = vkbDevice.device;
    m_chosenGpu = vkbDevice.physical_device;
    std::println("Physical GPU: {}", vkbDevice.physical_device.name);

    {
        m_RTProperties.pNext = &m_ASProperties;
        VkPhysicalDeviceProperties2 rtProps{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &m_RTProperties,
        };

        vkGetPhysicalDeviceProperties2(m_chosenGpu, &rtProps);
    }

    m_queue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    m_queueFamilyIndex = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    volkLoadDevice(m_device);

    VmaVulkanFunctions vulkanFunc[]{vkGetInstanceProcAddr, vkGetDeviceProcAddr};
    VmaAllocatorCreateInfo allocatorCreateInfo{
        .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = m_chosenGpu,
        .device = m_device,
        .pVulkanFunctions = vulkanFunc,
        .instance = m_instance,
    };

    vmaCreateAllocator(&allocatorCreateInfo, &m_allocator) >> chk;

    m_mainDeletionQueue.push_function([&]() { vmaDestroyAllocator(m_allocator); });
}

void Engine::create_draw_image(AllocatedImage &drawImage, const VkExtent3D extent)
{
    drawImage.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    drawImage.imageExtent = extent;

    VkImageUsageFlags drawImageUsages{};
    drawImageUsages |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_STORAGE_BIT;
    const VkImageCreateInfo imageCreateInfo = utils::image_create_info(drawImage.imageFormat, drawImageUsages, extent);
    constexpr VmaAllocationCreateInfo allocationCreateInfo{
        .usage = VMA_MEMORY_USAGE_GPU_ONLY,
        .requiredFlags = static_cast<VkMemoryPropertyFlags>(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    vmaCreateImage(m_allocator, &imageCreateInfo, &allocationCreateInfo, &drawImage.image, &drawImage.allocation,
                   nullptr) >>
        chk;

    const VkImageViewCreateInfo imageViewCreateInfo =
        utils::image_view_create_info(drawImage.imageFormat, drawImage.image, VK_IMAGE_ASPECT_COLOR_BIT);
    vkCreateImageView(m_device, &imageViewCreateInfo, nullptr, &drawImage.imageView) >> chk;

    m_mainDeletionQueue.push_function([&] {
        vkDestroyImageView(m_device, drawImage.imageView, nullptr);
        vmaDestroyImage(m_allocator, drawImage.image, drawImage.allocation);
    });
}

void Engine::create_depth_image(AllocatedImage &depthImage, const VkExtent3D extent)
{
    depthImage.imageFormat = VK_FORMAT_D32_SFLOAT;
    depthImage.imageExtent = extent;

    constexpr VkImageUsageFlags imageUsages = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    const VkImageCreateInfo imageCreateInfo = utils::image_create_info(depthImage.imageFormat, imageUsages, extent);
    constexpr VmaAllocationCreateInfo allocationCreateInfo{
        .usage = VMA_MEMORY_USAGE_GPU_ONLY,
        .requiredFlags = static_cast<VkMemoryPropertyFlags>(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    vmaCreateImage(m_allocator, &imageCreateInfo, &allocationCreateInfo, &depthImage.image, &depthImage.allocation,
                   nullptr) >>
        chk;

    const VkImageViewCreateInfo imageViewCreateInfo =
        utils::image_view_create_info(depthImage.imageFormat, depthImage.image, VK_IMAGE_ASPECT_DEPTH_BIT);
    vkCreateImageView(m_device, &imageViewCreateInfo, nullptr, &depthImage.imageView) >> chk;

    m_mainDeletionQueue.push_function([&] {
        vkDestroyImageView(m_device, depthImage.imageView, nullptr);
        vmaDestroyImage(m_allocator, depthImage.image, depthImage.allocation);
    });
}

void Engine::init_swapchain()
{
    create_swapchain(m_windowExtent.width, m_windowExtent.height);
}

void Engine::init_commands()
{
    const VkCommandPoolCreateInfo commandPoolCreateInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                                        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                                                                 VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                                        .queueFamilyIndex = m_queueFamilyIndex};
    vkCreateCommandPool(m_device, &commandPoolCreateInfo, nullptr, &m_commandPool) >> chk;
    vkCreateCommandPool(m_device, &commandPoolCreateInfo, nullptr, &m_immCommandPool);
    for (auto &frame : m_frameData)
    {
        const VkCommandBufferAllocateInfo allocateInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr,
            .commandPool = m_commandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        vkAllocateCommandBuffers(m_device, &allocateInfo, &frame.commandBuffer) >> chk;
    }
    const VkCommandBufferAllocateInfo allocateInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = m_immCommandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    vkAllocateCommandBuffers(m_device, &allocateInfo, &m_immCommandBuffer) >> chk;

    m_mainDeletionQueue.push_function([&] { vkDestroyCommandPool(m_device, m_immCommandPool, nullptr); });
}

void Engine::init_sync()
{
    constexpr VkFenceCreateInfo fenceCreateInfo{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                                                .flags = VK_FENCE_CREATE_SIGNALED_BIT};
    constexpr VkSemaphoreCreateInfo semaphoreCreateInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (auto &frame : m_frameData)
    {
        vkCreateFence(m_device, &fenceCreateInfo, nullptr, &frame.fence) >> chk;

        vkCreateSemaphore(m_device, &semaphoreCreateInfo, nullptr, &frame.swapchainSemaphore) >> chk;
    }

    m_swapchainSemaphores.resize(m_swapchainImages.size());
    for (auto &renderSemaphore : m_swapchainSemaphores)
        vkCreateSemaphore(m_device, &semaphoreCreateInfo, nullptr, &renderSemaphore) >> chk;

    vkCreateFence(m_device, &fenceCreateInfo, nullptr, &m_immFence) >> chk;

    m_mainDeletionQueue.push_function([&] { vkDestroyFence(m_device, m_immFence, nullptr); });
}

void Engine::init_descriptors()
{

    {
        m_DrawImageDescriptorSetLayout =
            DescriptorSetLayoutBuilder()
                .add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT)
                .build(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT);
        m_LightPassDescriptorSetLayout =
            DescriptorSetLayoutBuilder()
                .add_binding(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT)
                .add_binding(1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT)
                .add_binding(2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT)
                .add_binding(3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT)
                .add_binding(4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT)
                .add_binding(5, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT)
                .add_binding(6, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT)
                .add_binding(7, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT)
                .add_binding(8, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT)
                .build(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT);
    }
    {
        m_DDGITLASDescSetLayout =
            DescriptorSetLayoutBuilder()
                .add_binding(0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR)
                .build(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT);
        m_DDGIRayDataDescSetLayout = DescriptorSetLayoutBuilder()
                                         .add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kMaxDDGIVolumes,
                                                      VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT)
                                         .build(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT);
        m_DDGIResourcesDescSetLayout = DescriptorSetLayoutBuilder()
                                           .add_binding(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kMaxDDGIVolumes,
                                                        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT |
                                                            VK_SHADER_STAGE_FRAGMENT_BIT) // Irradiance
                                           .add_binding(1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kMaxDDGIVolumes,
                                                        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT |
                                                            VK_SHADER_STAGE_FRAGMENT_BIT) // Distance
                                           .add_binding(2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kMaxDDGIVolumes,
                                                        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT |
                                                            VK_SHADER_STAGE_VERTEX_BIT) // Probe Data
                                           .add_binding(3, VK_DESCRIPTOR_TYPE_SAMPLER, 1,
                                                        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT |
                                                            VK_SHADER_STAGE_FRAGMENT_BIT) // Sampler
                                           .build(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT);
        m_DDGIProbeStorageDescSetLayout =
            DescriptorSetLayoutBuilder()
                .add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kMaxDDGIVolumes, VK_SHADER_STAGE_COMPUTE_BIT)
                .build(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT);
        m_DDGIGBufferReadDescSetLayout =
            DescriptorSetLayoutBuilder()
                .add_binding(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT)
                .add_binding(1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT)
                .add_binding(2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT)
                .build(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT);
    }
    {
        m_WboitCompositePassDescriptorSetLayout =
            DescriptorSetLayoutBuilder()
                .add_binding(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT)
                .add_binding(1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT)
                .build(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT);
        m_DepthPassDescSetLayout = DescriptorSetLayoutBuilder()
                                       .add_binding(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT)
                                       .build(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT);
#if 0
        m_CullPassDescriptorSetLayout =
            DescriptorSetLayoutBuilder()
                .build(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT);
#endif
    }

    m_mainDeletionQueue.push_function([&]() mutable {
        vkDestroyDescriptorSetLayout(m_device, m_LightPassDescriptorSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(m_device, m_DrawImageDescriptorSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(m_device, m_DDGITLASDescSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(m_device, m_DDGIRayDataDescSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(m_device, m_DDGIResourcesDescSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(m_device, m_DDGIProbeStorageDescSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(m_device, m_DDGIGBufferReadDescSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(m_device, m_WboitCompositePassDescriptorSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(m_device, m_DepthPassDescSetLayout, nullptr);
    });
}

void Engine::init_pipelines()
{
    init_light_pass_pipeline();
    init_cull_meshes_pipeline();
    init_cull_point_lights_pipeline();
    init_populate_commands_with_cascade_count();
    init_wboit_composite_pass_pipeline();
    init_post_pipeline();
    init_luminance_histogram_pipeline();
    init_average_luminance_pipeline();
    init_generate_point_light_commands_pipeline();
    init_directional_shadow_pass();
    init_point_shadow_pass();
    init_depth_reduction_pass();
    m_metalRoughness.build_pipelines(*this);
    init_prepass();
    init_alpha_tested_prepass();
    init_alpha_tested_directional_shadow_pass();
    init_alpha_tested_point_shadow_pass();
    init_ddgi_probe_pipeline();
    init_ddgi_probe_support_pipelines();
    init_ddgi_probe_reset_pipelines();
    init_ddgi_indirect_pipeline();
    init_ddgi_probe_vis_pipeline();
}

void Engine::init_light_pass_pipeline()
{
    constexpr VkPushConstantRange pushConstantRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = static_cast<std::uint32_t>(sizeof(LightPassPushConstants)),
    };

    VkDescriptorSetLayout setLayouts[]{m_DrawImageDescriptorSetLayout, m_LightPassDescriptorSetLayout};
    const VkPipelineLayoutCreateInfo layoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = std::size(setLayouts),
        .pSetLayouts = setLayouts,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstantRange,
    };
    vkCreatePipelineLayout(m_device, &layoutCreateInfo, nullptr, &m_LightPassPipelineLayout) >> chk;

    VkShaderModule lightPassShader;
    if (!load_shader_module("../../src/compiled_shaders/light_pass.compute.spv", m_device, &lightPassShader))
    {
        throw std::runtime_error("Failed to load a light pass shader");
    }

    const VkPipelineShaderStageCreateInfo shaderStage{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = lightPassShader,
        .pName = "main",
    };

    const VkComputePipelineCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT,
        .stage = shaderStage,
        .layout = m_LightPassPipelineLayout,
    };

    vkCreateComputePipelines(m_device, nullptr, 1, &createInfo, nullptr, &m_LightPassPipeline) >> chk;
    mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_PIPELINE, reinterpret_cast<uint64_t>(m_LightPassPipeline),
                               "Light Pass Pipeline");

    vkDestroyShaderModule(m_device, lightPassShader, nullptr);
    m_mainDeletionQueue.push_function([this] {
        vkDestroyPipeline(m_device, m_LightPassPipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_LightPassPipelineLayout, nullptr);
    });
}

void Engine::init_wboit_composite_pass_pipeline()
{
    VkShaderModule compositeVertexShader;
    if (!load_shader_module("../../src/compiled_shaders/wboit_composite.vertex.spv", m_device, &compositeVertexShader))
    {
        throw std::runtime_error("Failed to load wboit_composite.vertex.spv");
    }
    VkShaderModule compositeFragmentShader;
    if (!load_shader_module("../../src/compiled_shaders/wboit_composite.pixel.spv", m_device, &compositeFragmentShader))
    {
        throw std::runtime_error("Failed to load wboit_composite.pixel.spv");
    }

    const VkDescriptorSetLayout layouts[]{m_WboitCompositePassDescriptorSetLayout};

    {
        const VkPipelineLayoutCreateInfo layoutCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = std::size(layouts),
            .pSetLayouts = layouts,

        };
        vkCreatePipelineLayout(m_device, &layoutCreateInfo, nullptr, &m_WBOITCompositePassPipelineLayout) >> chk;
    }

    {
        PipelineBuilder pipelineBuilder;
        pipelineBuilder.pipelineLayout = m_WBOITCompositePassPipelineLayout;
        pipelineBuilder.add_shader(compositeVertexShader, VK_SHADER_STAGE_VERTEX_BIT);
        pipelineBuilder.add_shader(compositeFragmentShader, VK_SHADER_STAGE_FRAGMENT_BIT);
        pipelineBuilder.disable_depth_test();
        pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
        pipelineBuilder.add_color_attachment_format(VK_FORMAT_R16G16B16A16_SFLOAT);
        pipelineBuilder.set_cull_mode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);
        pipelineBuilder.set_multisampling_none();
        pipelineBuilder.colorBlends.push_back({.blendEnable = VK_TRUE,
                                               .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
                                               .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                                               .colorBlendOp = VK_BLEND_OP_ADD,
                                               .srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
                                               .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                                               .alphaBlendOp = VK_BLEND_OP_ADD,
                                               .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                                                 VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT});
        m_WBOITCompositePassPipeline =
            pipelineBuilder.build_pipeline(m_device, VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_PIPELINE,
                                   reinterpret_cast<uint64_t>(m_WBOITCompositePassPipeline),
                                   "WBOIT Composite Pipeline");
    }

    vkDestroyShaderModule(m_device, compositeVertexShader, nullptr);
    vkDestroyShaderModule(m_device, compositeFragmentShader, nullptr);

    m_mainDeletionQueue.push_function([this] {
        vkDestroyPipeline(m_device, m_WBOITCompositePassPipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_WBOITCompositePassPipelineLayout, nullptr);
    });
}

void Engine::init_depth_reduction_pass()
{
    // --- Depth Reduction pipeline ---
    {
        const VkPushConstantRange pcRange{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(DepthReductionPushConstants)};
        const VkPipelineLayoutCreateInfo layoutInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &m_DepthPassDescSetLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pcRange,
        };
        vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_DepthReductionPipelineLayout) >> chk;

        VkShaderModule shaderModule;
        if (!load_shader_module("../../src/compiled_shaders/depth_reduction.compute.spv", m_device, &shaderModule))
        {
            throw std::runtime_error("Failed to load depth_reduction.compute.spv");
        }
        const VkComputePipelineCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT,
            .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                      .module = shaderModule,
                      .pName = "main"},
            .layout = m_DepthReductionPipelineLayout,
        };
        vkCreateComputePipelines(m_device, nullptr, 1, &createInfo, nullptr, &m_DepthReductionPipeline) >> chk;
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_PIPELINE,
                                   reinterpret_cast<uint64_t>(m_DepthReductionPipeline), "Depth Reduction Pipeline");
        vkDestroyShaderModule(m_device, shaderModule, nullptr);
    }

    // --- Depth Partition pipeline ---
    {
        const VkPushConstantRange pcRange{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(DepthPartitionPushConstants)};
        const VkPipelineLayoutCreateInfo layoutInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &m_DepthPassDescSetLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pcRange,
        };
        vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_DepthPartitionPipelineLayout) >> chk;

        VkShaderModule shaderModule;
        if (!load_shader_module("../../src/compiled_shaders/depth_partition.compute.spv", m_device, &shaderModule))
        {
            throw std::runtime_error("Failed to load depth_partition.compute.spv");
        }
        const VkComputePipelineCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT,
            .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                      .module = shaderModule,
                      .pName = "main"},
            .layout = m_DepthPartitionPipelineLayout,
        };
        vkCreateComputePipelines(m_device, nullptr, 1, &createInfo, nullptr, &m_DepthPartitionPipeline) >> chk;
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_PIPELINE,
                                   reinterpret_cast<uint64_t>(m_DepthPartitionPipeline), "Depth Partition Pipeline");
        vkDestroyShaderModule(m_device, shaderModule, nullptr);
    }

    // --- Directional VP pipeline (no descriptor set) ---
    {
        const VkPushConstantRange pcRange{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(DirVpPushConstants)};
        const VkPipelineLayoutCreateInfo layoutInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 0,
            .pSetLayouts = nullptr,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pcRange,
        };
        vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_DirVpPipelineLayout) >> chk;

        VkShaderModule shaderModule;
        if (!load_shader_module("../../src/compiled_shaders/directional_vp.compute.spv", m_device, &shaderModule))
        {
            throw std::runtime_error("Failed to load directional_vp.compute.spv");
        }
        const VkComputePipelineCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                      .module = shaderModule,
                      .pName = "main"},
            .layout = m_DirVpPipelineLayout,
        };
        vkCreateComputePipelines(m_device, nullptr, 1, &createInfo, nullptr, &m_DirVpPipeline) >> chk;
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_PIPELINE, reinterpret_cast<uint64_t>(m_DirVpPipeline),
                                   "Directional VP Pipeline");
        vkDestroyShaderModule(m_device, shaderModule, nullptr);
    }

    m_mainDeletionQueue.push_function([this] {
        vkDestroyPipeline(m_device, m_DepthReductionPipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_DepthReductionPipelineLayout, nullptr);
        vkDestroyPipeline(m_device, m_DepthPartitionPipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_DepthPartitionPipelineLayout, nullptr);
        vkDestroyPipeline(m_device, m_DirVpPipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_DirVpPipelineLayout, nullptr);
    });
}

void Engine::init_directional_shadow_pass()
{
    const VkPushConstantRange pushConstantRange{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .offset = 0, .size = sizeof(DirectionalShadowPassPushConstants)};

    const VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                                              .pNext = nullptr,
                                                              .setLayoutCount = 0,
                                                              .pSetLayouts = nullptr,
                                                              .pushConstantRangeCount = 1,
                                                              .pPushConstantRanges = &pushConstantRange};
    vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &m_ShadowPassPipelineLayout) >> chk;

    VkShaderModule shadowPassVert;
    if (!mp::load_shader_module("../../src/compiled_shaders/directional_shadow_pass.vertex.spv", m_device,
                                &shadowPassVert))
    {
        throw std::runtime_error("Failed to load shadow pass vertex shader");
    }

    VkShaderModule shadowPassFrag;
    if (!mp::load_shader_module("../../src/compiled_shaders/directional_shadow_pass.pixel.spv", m_device,
                                &shadowPassFrag))
    {
        throw std::runtime_error("Failed to load shadow pass fragment shader");
    }

    mp::PipelineBuilder builder;
    builder.pipelineLayout = m_ShadowPassPipelineLayout;
    builder.enable_depth_test(true, VK_COMPARE_OP_LESS_OR_EQUAL);
    builder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    builder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    builder.add_shader(shadowPassVert, VK_SHADER_STAGE_VERTEX_BIT);
    builder.add_shader(shadowPassFrag, VK_SHADER_STAGE_FRAGMENT_BIT);
    builder.set_depth_format(VK_FORMAT_D32_SFLOAT);
    builder.set_cull_mode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);
    builder.set_multisampling_none();

    m_ShadowPassPipeline = builder.build_pipeline(m_device);
    mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_PIPELINE, reinterpret_cast<uint64_t>(m_ShadowPassPipeline),
                               "Directional Shadow Pass Pipeline");

    vkDestroyShaderModule(m_device, shadowPassVert, nullptr);
    vkDestroyShaderModule(m_device, shadowPassFrag, nullptr);

    m_mainDeletionQueue.push_function([this]() {
        vkDestroyPipeline(m_device, m_ShadowPassPipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_ShadowPassPipelineLayout, nullptr);
    });
}

void Engine::init_point_shadow_pass()
{
    const VkPushConstantRange pushConstantRange{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT,
        .offset = 0,
        .size = sizeof(PointLightsShadowPassPushConstants),
    };

    const VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .setLayoutCount = 0,
        .pSetLayouts = nullptr,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstantRange,
    };
    vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &m_PointLightShadowPassPipelineLayout) >> chk;

    VkShaderModule vert;
    if (!mp::load_shader_module("../../src/compiled_shaders/point_lights_shadow_pass.vertex.spv", m_device, &vert))
    {
        throw std::runtime_error("Failed to load point light shadow pass vertex shader");
    }

    VkShaderModule geom;
    if (!mp::load_shader_module("../../src/compiled_shaders/point_lights_shadow_pass.geometry.spv", m_device, &geom))
    {
        throw std::runtime_error("Failed to load point light shadow pass geometry shader");
    }

    VkShaderModule frag;
    if (!mp::load_shader_module("../../src/compiled_shaders/point_lights_shadow_pass.pixel.spv", m_device, &frag))
    {
        throw std::runtime_error("Failed to load point light shadow pass fragment shader");
    }

    mp::PipelineBuilder builder;
    builder.pipelineLayout = m_PointLightShadowPassPipelineLayout;
    builder.enable_depth_test(true, VK_COMPARE_OP_LESS_OR_EQUAL);
    builder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    builder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    builder.add_shader(vert, VK_SHADER_STAGE_VERTEX_BIT);
    builder.add_shader(geom, VK_SHADER_STAGE_GEOMETRY_BIT);
    builder.add_shader(frag, VK_SHADER_STAGE_FRAGMENT_BIT);
    builder.set_depth_format(VK_FORMAT_D16_UNORM);
    builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
    builder.set_multisampling_none();

    m_PointLightShadowPassPipeline = builder.build_pipeline(m_device);
    mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_PIPELINE,
                               reinterpret_cast<uint64_t>(m_PointLightShadowPassPipeline),
                               "Point Light Shadow Pass Pipeline");

    vkDestroyShaderModule(m_device, vert, nullptr);
    vkDestroyShaderModule(m_device, geom, nullptr);
    vkDestroyShaderModule(m_device, frag, nullptr);

    m_mainDeletionQueue.push_function([this]() {
        vkDestroyPipeline(m_device, m_PointLightShadowPassPipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_PointLightShadowPassPipelineLayout, nullptr);
    });
}

void Engine::init_prepass()
{
    const VkPushConstantRange pushConstantRange{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0,
        .size = static_cast<std::uint32_t>(sizeof(GBufferPassPushConstants)),
    };

    const VkDescriptorSetLayout layouts[]{m_metalRoughness.materialLayout};
    const VkPipelineLayoutCreateInfo layoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = std::size(layouts),
        .pSetLayouts = layouts,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstantRange,
    };
    vkCreatePipelineLayout(m_device, &layoutCreateInfo, nullptr, &m_PrepassPipelineLayout) >> chk;

    VkShaderModule prepassVert;
    if (!load_shader_module("../../src/compiled_shaders/prepass.vertex.spv", m_device, &prepassVert))
    {
        throw std::runtime_error("Failed to load prepass vertex shader");
    }

    VkShaderModule prepassFrag;
    if (!load_shader_module("../../src/compiled_shaders/prepass.pixel.spv", m_device, &prepassFrag))
    {
        throw std::runtime_error("Failed to load prepass pixel shader");
    }

    PipelineBuilder builder;
    builder.pipelineLayout = m_PrepassPipelineLayout;
    builder.enable_depth_test(true, VK_COMPARE_OP_LESS_OR_EQUAL);
    builder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    builder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    builder.add_shader(prepassVert, VK_SHADER_STAGE_VERTEX_BIT);
    builder.add_shader(prepassFrag, VK_SHADER_STAGE_FRAGMENT_BIT);
    builder.set_depth_format(VK_FORMAT_D32_SFLOAT);
    builder.set_cull_mode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);
    builder.set_multisampling_none();

    m_PrepassPipeline = builder.build_pipeline(m_device, VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT);
    mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_PIPELINE, reinterpret_cast<uint64_t>(m_PrepassPipeline),
                               "Prepass Pipeline");

    vkDestroyShaderModule(m_device, prepassVert, nullptr);
    vkDestroyShaderModule(m_device, prepassFrag, nullptr);

    m_mainDeletionQueue.push_function([this]() {
        vkDestroyPipeline(m_device, m_PrepassPipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_PrepassPipelineLayout, nullptr);
    });
}

void Engine::init_post_pipeline()
{
    const VkPushConstantRange postPcRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(PostProcessPushConstants),
    };
    const VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .setLayoutCount = 1,
        .pSetLayouts = &m_DrawImageDescriptorSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &postPcRange,
    };
    vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &m_PostProcessPassPipelineLayout) >> chk;

    VkShaderModule postprocessShader;
    if (!load_shader_module("../../src/compiled_shaders/postprocess.compute.spv", m_device, &postprocessShader))
    {
        throw std::runtime_error("Failed to load postprocess pass shader");
    }

    const VkPipelineShaderStageCreateInfo shaderStage{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = postprocessShader,
        .pName = "main",
    };
    const VkComputePipelineCreateInfo pipelineCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT,
        .stage = shaderStage,
        .layout = m_PostProcessPassPipelineLayout,
    };
    vkCreateComputePipelines(m_device, nullptr, 1, &pipelineCreateInfo, nullptr, &m_PostProcessPassPipeline);
    mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_PIPELINE, reinterpret_cast<uint64_t>(m_PostProcessPassPipeline),
                               "Postprocess Pipeline");

    vkDestroyShaderModule(m_device, postprocessShader, nullptr);

    m_mainDeletionQueue.push_function([this] {
        vkDestroyPipeline(m_device, m_PostProcessPassPipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_PostProcessPassPipelineLayout, nullptr);
    });
}

void Engine::init_luminance_histogram_pipeline()
{
    const VkPushConstantRange pcRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(LuminanceHistogramPushConstants),
    };
    const VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .setLayoutCount = 1,
        .pSetLayouts = &m_DrawImageDescriptorSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcRange,
    };
    vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_LuminanceHistogramPipelineLayout) >> chk;

    VkShaderModule shader;
    if (!load_shader_module("../../src/compiled_shaders/luminance_histogram.compute.spv", m_device, &shader))
    {
        throw std::runtime_error("Failed to load luminance_histogram.compute.spv");
    }

    const VkComputePipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT,
        .stage =
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader,
                .pName = "main",
            },
        .layout = m_LuminanceHistogramPipelineLayout,
    };
    vkCreateComputePipelines(m_device, nullptr, 1, &pipelineInfo, nullptr, &m_LuminanceHistogramPipeline) >> chk;
    mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_PIPELINE,
                               reinterpret_cast<uint64_t>(m_LuminanceHistogramPipeline),
                               "Luminance Histogram Pipeline");
    vkDestroyShaderModule(m_device, shader, nullptr);

    m_mainDeletionQueue.push_function([this] {
        vkDestroyPipeline(m_device, m_LuminanceHistogramPipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_LuminanceHistogramPipelineLayout, nullptr);
    });
}

void Engine::init_average_luminance_pipeline()
{
    const VkPushConstantRange pcRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(AverageLuminancePushConstants),
    };
    const VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .setLayoutCount = 0,
        .pSetLayouts = nullptr,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcRange,
    };
    vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_AverageLuminancePipelineLayout) >> chk;

    VkShaderModule shader;
    if (!load_shader_module("../../src/compiled_shaders/average_luminance.compute.spv", m_device, &shader))
    {
        throw std::runtime_error("Failed to load average_luminance.compute.spv");
    }

    const VkComputePipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT,
        .stage =
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader,
                .pName = "main",
            },
        .layout = m_AverageLuminancePipelineLayout,
    };
    vkCreateComputePipelines(m_device, nullptr, 1, &pipelineInfo, nullptr, &m_AverageLuminancePipeline) >> chk;
    mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_PIPELINE,
                               reinterpret_cast<uint64_t>(m_AverageLuminancePipeline), "Average Luminance Pipeline");
    vkDestroyShaderModule(m_device, shader, nullptr);

    m_mainDeletionQueue.push_function([this] {
        vkDestroyPipeline(m_device, m_AverageLuminancePipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_AverageLuminancePipelineLayout, nullptr);
    });
}

void Engine::init_cull_point_lights_pipeline()
{
    const VkPushConstantRange constantRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(CullPointLightsPassPushConstants),
    };

    const VkPipelineLayoutCreateInfo layoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .setLayoutCount = 0,
        .pSetLayouts = nullptr,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &constantRange,
    };

    vkCreatePipelineLayout(m_device, &layoutCreateInfo, nullptr, &m_CullPointLightsPassPipelineLayout) >> chk;

    VkShaderModule cullShader;
    if (!load_shader_module("../../src/compiled_shaders/cull_point_lights.compute.spv", m_device, &cullShader))
    {
        throw std::runtime_error("Failed to load cull shader");
    }

    const VkPipelineShaderStageCreateInfo shaderStage{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = cullShader,
        .pName = "main",
    };

    const VkComputePipelineCreateInfo pipelineCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .stage = shaderStage,
        .layout = m_CullPointLightsPassPipelineLayout,
    };

    vkCreateComputePipelines(m_device, nullptr, 1, &pipelineCreateInfo, nullptr, &m_CullPointLightsPassPipeline) >> chk;
    mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_PIPELINE,
                               reinterpret_cast<uint64_t>(m_CullPointLightsPassPipeline), "Cull Point Lights Pipeline");

    vkDestroyShaderModule(m_device, cullShader, nullptr);

    m_mainDeletionQueue.push_function([this] {
        vkDestroyPipeline(m_device, m_CullPointLightsPassPipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_CullPointLightsPassPipelineLayout, nullptr);
    });
}

void Engine::init_ddgi_probe_pipeline()
{
    const VkDescriptorSetLayout layouts[]{
        m_DDGITLASDescSetLayout,
        m_DDGIRayDataDescSetLayout,
        m_DDGIResourcesDescSetLayout,
        m_metalRoughness.materialLayout,
    };
    const VkPushConstantRange pcRange{
        .stageFlags =
            VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
        .offset = 0,
        .size = sizeof(DDGIProbePushConstants),
    };
    const VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = static_cast<std::uint32_t>(std::size(layouts)),
        .pSetLayouts = layouts,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcRange,
    };
    vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_DDGIPipelineLayout) >> chk;

    VkShaderModule rgenShader, missShader, chitShader, ahitShader;
    if (!load_shader_module("../../src/compiled_shaders/probe.raygeneration.spv", m_device, &rgenShader))
        throw std::runtime_error("Failed to load probe.raygeneration.spv");
    if (!load_shader_module("../../src/compiled_shaders/probe.miss.spv", m_device, &missShader))
        throw std::runtime_error("Failed to load probe.miss.spv");
    if (!load_shader_module("../../src/compiled_shaders/probe.closesthit.spv", m_device, &chitShader))
        throw std::runtime_error("Failed to load probe.closesthit.spv");
    if (!load_shader_module("../../src/compiled_shaders/probe.anyhit.spv", m_device, &ahitShader))
        throw std::runtime_error("Failed to load probe.anyhit.spv");

    const VkPipelineShaderStageCreateInfo stages[]{
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
         .module = rgenShader,
         .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_MISS_BIT_KHR,
         .module = missShader,
         .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
         .module = chitShader,
         .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
         .module = ahitShader,
         .pName = "main"},
    };

    const VkRayTracingShaderGroupCreateInfoKHR groups[]{
        {.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
         .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
         .generalShader = 0,
         .closestHitShader = VK_SHADER_UNUSED_KHR,
         .anyHitShader = VK_SHADER_UNUSED_KHR,
         .intersectionShader = VK_SHADER_UNUSED_KHR},
        {.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
         .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
         .generalShader = 1,
         .closestHitShader = VK_SHADER_UNUSED_KHR,
         .anyHitShader = VK_SHADER_UNUSED_KHR,
         .intersectionShader = VK_SHADER_UNUSED_KHR},
        {.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
         .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
         .generalShader = VK_SHADER_UNUSED_KHR,
         .closestHitShader = 2,
         .anyHitShader = 3,
         .intersectionShader = VK_SHADER_UNUSED_KHR},
    };

    const VkRayTracingPipelineCreateInfoKHR rtInfo{
        .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
        .flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT,
        .stageCount = static_cast<std::uint32_t>(std::size(stages)),
        .pStages = stages,
        .groupCount = static_cast<std::uint32_t>(std::size(groups)),
        .pGroups = groups,
        .maxPipelineRayRecursionDepth = 1,
        .layout = m_DDGIPipelineLayout,
    };
    vkCreateRayTracingPipelinesKHR(m_device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &rtInfo, nullptr, &m_DDGIPipeline) >>
        chk;
    mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_PIPELINE, reinterpret_cast<uint64_t>(m_DDGIPipeline),
                               "DDGI Probe Pipeline");

    vkDestroyShaderModule(m_device, rgenShader, nullptr);
    vkDestroyShaderModule(m_device, missShader, nullptr);
    vkDestroyShaderModule(m_device, chitShader, nullptr);
    vkDestroyShaderModule(m_device, ahitShader, nullptr);

    const std::uint32_t handleSize = m_RTProperties.shaderGroupHandleSize;
    const std::uint32_t handleAlign = m_RTProperties.shaderGroupHandleAlignment;
    const std::uint32_t baseAlign = m_RTProperties.shaderGroupBaseAlignment;

    const std::uint32_t entryStride = static_cast<std::uint32_t>((handleSize + handleAlign - 1u) & ~(handleAlign - 1u));
    const VkDeviceSize regionSize = static_cast<VkDeviceSize>((entryStride + baseAlign - 1u) & ~(baseAlign - 1u));

    constexpr std::uint32_t kGroupCount = 3u;
    m_ShaderHandles.resize(kGroupCount * handleSize);
    vkGetRayTracingShaderGroupHandlesKHR(m_device, m_DDGIPipeline, 0, kGroupCount, m_ShaderHandles.size(),
                                         m_ShaderHandles.data()) >>
        chk;

    m_SBTBuffer =
        create_buffer(regionSize * kGroupCount,
                      VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                      VMA_MEMORY_USAGE_CPU_TO_GPU);

    auto *sbt = static_cast<std::uint8_t *>(m_SBTBuffer.allocationInfo.pMappedData);
    std::memcpy(sbt + regionSize * 0, m_ShaderHandles.data() + handleSize * 0, handleSize);
    std::memcpy(sbt + regionSize * 1, m_ShaderHandles.data() + handleSize * 1, handleSize);
    std::memcpy(sbt + regionSize * 2, m_ShaderHandles.data() + handleSize * 2, handleSize);

    const VkDeviceAddress base = m_SBTBuffer.get_buffer_device_address(m_device);
    m_RaygenRegion = {.deviceAddress = base, .stride = regionSize, .size = regionSize};
    m_MissRegion = {.deviceAddress = base + regionSize, .stride = entryStride, .size = regionSize};
    m_HitRegion = {.deviceAddress = base + regionSize * 2, .stride = entryStride, .size = regionSize};
    m_CallableRegion = {};

    {
        m_DDGIVolumesBuffer =
            create_buffer(sizeof(DDGIVolume) * kMaxDDGIVolumes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                          VMA_MEMORY_USAGE_CPU_TO_GPU);
        m_DDGIVolumesAddr = m_DDGIVolumesBuffer.get_buffer_device_address(m_device);
    }

    m_mainDeletionQueue.push_function([this] {
        vkDestroyPipeline(m_device, m_DDGIPipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_DDGIPipelineLayout, nullptr);
        destroy_buffer(m_SBTBuffer);
        destroy_buffer(m_DDGIVolumesBuffer);
    });
}

void Engine::init_ddgi_probe_support_pipelines()
{
    const VkDescriptorSetLayout setLayouts[]{m_DDGIRayDataDescSetLayout, m_DDGIProbeStorageDescSetLayout};
    const VkPushConstantRange pcRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(DDGIProbeSupportPushConstants),
    };
    const VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = static_cast<std::uint32_t>(std::size(setLayouts)),
        .pSetLayouts = setLayouts,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcRange,
    };
    vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_DDGIProbeSupportPipelineLayout) >> chk;

    VkShaderModule irradianceShader, distanceShader, relocationShader;
    if (!load_shader_module("../../src/compiled_shaders/probe_irradiance_blending.compute.spv", m_device,
                            &irradianceShader))
        throw std::runtime_error("Failed to load probe_irradiance_blending.compute.spv");
    if (!load_shader_module("../../src/compiled_shaders/probe_distance_blending.compute.spv", m_device,
                            &distanceShader))
        throw std::runtime_error("Failed to load probe_distance_blending.compute.spv");
    if (!load_shader_module("../../src/compiled_shaders/probe_relocation.compute.spv", m_device, &relocationShader))
        throw std::runtime_error("Failed to load probe_relocation.compute.spv");

    const VkComputePipelineCreateInfo irradiancePipelineInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT,
        .stage =
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = irradianceShader,
                .pName = "main",
            },
        .layout = m_DDGIProbeSupportPipelineLayout,
    };
    const VkComputePipelineCreateInfo distancePipelineInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT,
        .stage =
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = distanceShader,
                .pName = "main",
            },
        .layout = m_DDGIProbeSupportPipelineLayout,
    };
    const VkComputePipelineCreateInfo relocationPipelineCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT,
        .stage =
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = relocationShader,
                .pName = "main",
            },
        .layout = m_DDGIProbeSupportPipelineLayout,
    };
    vkCreateComputePipelines(m_device, nullptr, 1, &irradiancePipelineInfo, nullptr,
                             &m_DDGIIrradianceBlendingPipeline) >>
        chk;
    vkCreateComputePipelines(m_device, nullptr, 1, &distancePipelineInfo, nullptr, &m_DDGIDistanceBlendingPipeline) >>
        chk;
    vkCreateComputePipelines(m_device, nullptr, 1, &relocationPipelineCreateInfo, nullptr,
                             &m_DDGIProbeRelocationPipeline) >>
        chk;
    mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_PIPELINE,
                               reinterpret_cast<uint64_t>(m_DDGIIrradianceBlendingPipeline),
                               "DDGI Irradiance Blending Pipeline");
    mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_PIPELINE,
                               reinterpret_cast<uint64_t>(m_DDGIDistanceBlendingPipeline),
                               "DDGI Distance Blending Pipeline");
    mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_PIPELINE,
                               reinterpret_cast<uint64_t>(m_DDGIProbeRelocationPipeline),
                               "DDGI Probe Relocation Pipeline");

    vkDestroyShaderModule(m_device, irradianceShader, nullptr);
    vkDestroyShaderModule(m_device, distanceShader, nullptr);
    vkDestroyShaderModule(m_device, relocationShader, nullptr);

    m_mainDeletionQueue.push_function([this] {
        vkDestroyPipeline(m_device, m_DDGIIrradianceBlendingPipeline, nullptr);
        vkDestroyPipeline(m_device, m_DDGIDistanceBlendingPipeline, nullptr);
        vkDestroyPipeline(m_device, m_DDGIProbeRelocationPipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_DDGIProbeSupportPipelineLayout, nullptr);
    });
}

void Engine::init_ddgi_probe_reset_pipelines()
{
    const VkDescriptorSetLayout setLayouts[]{m_DDGIProbeStorageDescSetLayout};
    const VkPushConstantRange pcRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(DDGIProbeSupportPushConstants),
    };
    const VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = static_cast<std::uint32_t>(std::size(setLayouts)),
        .pSetLayouts = setLayouts,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcRange,
    };
    vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_DDGIProbeResetPipelineLayout) >> chk;

    VkShaderModule relocationResetShader;
    if (!load_shader_module("../../src/compiled_shaders/probe_relocation_reset.compute.spv", m_device,
                            &relocationResetShader))
        throw std::runtime_error("Failed to load probe_relocation_reset.compute.spv");

    const VkComputePipelineCreateInfo relocationPipelineCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT,
        .stage =
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = relocationResetShader,
                .pName = "main",
            },
        .layout = m_DDGIProbeSupportPipelineLayout,
    };
    vkCreateComputePipelines(m_device, nullptr, 1, &relocationPipelineCreateInfo, nullptr,
                             &m_DDGIProbeRelocationResetPipeline) >>
        chk;
    mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_PIPELINE,
                               reinterpret_cast<uint64_t>(m_DDGIProbeRelocationResetPipeline),
                               "DDGI Relocation Reset Pipeline");

    vkDestroyShaderModule(m_device, relocationResetShader, nullptr);

    m_mainDeletionQueue.push_function([this] {
        vkDestroyPipeline(m_device, m_DDGIProbeRelocationResetPipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_DDGIProbeResetPipelineLayout, nullptr);
    });
}

void Engine::init_ddgi_indirect_pipeline()
{
    const VkDescriptorSetLayout setLayouts[]{m_DrawImageDescriptorSetLayout, m_DDGIGBufferReadDescSetLayout,
                                             m_DDGIResourcesDescSetLayout};
    const VkPushConstantRange pcRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(DDGIIndirectPushConstants),
    };
    const VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = static_cast<std::uint32_t>(std::size(setLayouts)),
        .pSetLayouts = setLayouts,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcRange,
    };
    vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_DDGIIndirectPipelineLayout) >> chk;

    VkShaderModule shader;
    if (!load_shader_module("../../src/compiled_shaders/ddgi_indirect.compute.spv", m_device, &shader))
        throw std::runtime_error("Failed to load ddgi_indirect.compute.spv");

    const VkComputePipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT,
        .stage =
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader,
                .pName = "main",
            },
        .layout = m_DDGIIndirectPipelineLayout,
    };
    vkCreateComputePipelines(m_device, nullptr, 1, &pipelineInfo, nullptr, &m_DDGIIndirectPipeline) >> chk;
    mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_PIPELINE, reinterpret_cast<uint64_t>(m_DDGIIndirectPipeline),
                               "DDGI Indirect Pipeline");

    vkDestroyShaderModule(m_device, shader, nullptr);

    m_mainDeletionQueue.push_function([this] {
        vkDestroyPipeline(m_device, m_DDGIIndirectPipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_DDGIIndirectPipelineLayout, nullptr);
    });
}

void Engine::init_populate_commands_with_cascade_count()
{
    const VkPushConstantRange constantRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(PopulateCommandsWithCascadeCountPushConstants),
    };

    const VkPipelineLayoutCreateInfo layoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .setLayoutCount = 0,
        .pSetLayouts = nullptr,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &constantRange,
    };

    vkCreatePipelineLayout(m_device, &layoutCreateInfo, nullptr, &m_PopulateCommandsWithCascadeCountPipelineLayout) >>
        chk;

    VkShaderModule populateShader;
    if (!load_shader_module("../../src/compiled_shaders/populate_commands_with_cascade_count.compute.spv", m_device,
                            &populateShader))
    {
        throw std::runtime_error("Failed to load populate shader");
    }

    const VkPipelineShaderStageCreateInfo shaderStage{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = populateShader,
        .pName = "main",
    };

    const VkComputePipelineCreateInfo pipelineCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .stage = shaderStage,
        .layout = m_PopulateCommandsWithCascadeCountPipelineLayout,
    };

    vkCreateComputePipelines(m_device, nullptr, 1, &pipelineCreateInfo, nullptr,
                             &m_PopulateCommandsWithCascadeCountPipeline) >>
        chk;
    mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_PIPELINE,
                               reinterpret_cast<uint64_t>(m_PopulateCommandsWithCascadeCountPipeline),
                               "Cull Point Lights Pipeline");

    vkDestroyShaderModule(m_device, populateShader, nullptr);

    m_mainDeletionQueue.push_function([this] {
        vkDestroyPipeline(m_device, m_PopulateCommandsWithCascadeCountPipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_PopulateCommandsWithCascadeCountPipelineLayout, nullptr);
    });
}

void Engine::init_generate_point_light_commands_pipeline()
{
    const VkPushConstantRange constantRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(GeneratePointLightCommandsPushConstants),
    };

    const VkPipelineLayoutCreateInfo layoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .setLayoutCount = 0,
        .pSetLayouts = nullptr,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &constantRange,
    };

    vkCreatePipelineLayout(m_device, &layoutCreateInfo, nullptr, &m_GeneratePointLightCommandsPipelineLayout) >> chk;

    VkShaderModule shader;
    if (!load_shader_module("../../src/compiled_shaders/generate_point_light_shadows_commands.compute.spv", m_device,
                            &shader))
    {
        throw std::runtime_error("Failed to load generate point light commands shader");
    }

    const VkPipelineShaderStageCreateInfo shaderStage{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader,
        .pName = "main",
    };

    const VkComputePipelineCreateInfo pipelineCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .stage = shaderStage,
        .layout = m_GeneratePointLightCommandsPipelineLayout,
    };

    vkCreateComputePipelines(m_device, nullptr, 1, &pipelineCreateInfo, nullptr,
                             &m_GeneratePointLightCommandsPipeline) >>
        chk;
    mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_PIPELINE,
                               reinterpret_cast<uint64_t>(m_GeneratePointLightCommandsPipeline),
                               "Generate Point Light Commands Pipeline");

    vkDestroyShaderModule(m_device, shader, nullptr);

    m_mainDeletionQueue.push_function([this] {
        vkDestroyPipeline(m_device, m_GeneratePointLightCommandsPipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_GeneratePointLightCommandsPipelineLayout, nullptr);
    });
}

void Engine::init_cull_meshes_pipeline()
{
    const VkPushConstantRange constantRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(CullPassPushConstants),
    };

    const VkPipelineLayoutCreateInfo layoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .setLayoutCount = 0,
        .pSetLayouts = nullptr,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &constantRange,
    };

    vkCreatePipelineLayout(m_device, &layoutCreateInfo, nullptr, &m_CullPassPipelineLayout) >> chk;

    VkShaderModule cullShader;
    if (!load_shader_module("../../src/compiled_shaders/cull.compute.spv", m_device, &cullShader))
    {
        throw std::runtime_error("Failed to load cull shader");
    }

    const VkPipelineShaderStageCreateInfo shaderStage{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = cullShader,
        .pName = "main",
    };

    const VkComputePipelineCreateInfo pipelineCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .stage = shaderStage,
        .layout = m_CullPassPipelineLayout,
    };

    vkCreateComputePipelines(m_device, nullptr, 1, &pipelineCreateInfo, nullptr, &m_CullPassPipeline) >> chk;
    mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_PIPELINE, reinterpret_cast<uint64_t>(m_CullPassPipeline),
                               "Cull Meshes Pipeline");

    vkDestroyShaderModule(m_device, cullShader, nullptr);

    m_mainDeletionQueue.push_function([this] {
        vkDestroyPipeline(m_device, m_CullPassPipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_CullPassPipelineLayout, nullptr);
    });
}

void Engine::init_imgui()
{
    // 1: create descriptor pool for IMGUI
    //  the size of the pool is very oversize, but it's copied from imgui demo
    //  itself.
    const VkDescriptorPoolSize poolSizes[] = {{VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
                                              {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
                                              {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
                                              {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
                                              {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
                                              {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
                                              {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
                                              {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
                                              {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
                                              {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
                                              {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}};

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = 1000;
    poolInfo.poolSizeCount = static_cast<uint32_t>(std::size(poolSizes));
    poolInfo.pPoolSizes = poolSizes;

    VkDescriptorPool imguiPool;
    vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &imguiPool) >> chk;

    // this initializes the core structures of imgui
    ImGui::CreateContext();

    // this initializes imgui for SDL
    ImGui_ImplSDL3_InitForVulkan(m_window.get());

    ImGui_ImplVulkan_InitInfo initInfo = {};
    initInfo.Instance = m_instance;
    initInfo.PhysicalDevice = m_chosenGpu;
    initInfo.Device = m_device;
    initInfo.Queue = m_queue;
    initInfo.DescriptorPool = imguiPool;
    initInfo.MinImageCount = static_cast<std::uint32_t>(m_swapchainImages.size());
    initInfo.ImageCount = static_cast<std::uint32_t>(m_swapchainImages.size());
    initInfo.UseDynamicRendering = true;
    initInfo.ApiVersion = VK_API_VERSION_1_4;
    initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &m_swapchainImageFormat,
    };

    ImGui_ImplVulkan_Init(&initInfo);

    m_mainDeletionQueue.push_function([&, imguiPool] {
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplSDL3_Shutdown();
        vkDestroyDescriptorPool(m_device, imguiPool, nullptr);
    });
}

void Engine::init_frames_data()
{
    constexpr auto kMaxInstances = 100'000;
    constexpr auto kMaxMeshes = 50'000;
    m_CommonImageExtent3D = {
        .width = m_windowExtent.width,
        .height = m_windowExtent.height,
        .depth = 1,
    };

    m_CommonImageExtent2D = {
        .width = m_windowExtent.width,
        .height = m_windowExtent.height,
    };

    {
        const VkSamplerCreateInfo shadowSamplerInfo{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_LINEAR,
            .minFilter = VK_FILTER_LINEAR,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
            .compareEnable = VK_TRUE,
            .compareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
            .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
        };
        vkCreateSampler(m_device, &shadowSamplerInfo, nullptr, &m_shadowSampler);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_SAMPLER, reinterpret_cast<uint64_t>(m_shadowSampler),
                                   "Shadow Sampler");
    }
    ddgiRayDataDescBuffer =
        DescriptorBuffer(m_device, m_DDGIRayDataDescSetLayout, DescriptorBufferProperties::query(m_chosenGpu));
    ddgiRayDataDescBuffer.create_buffer([this](const std::size_t allocSize, const VkBufferUsageFlags bufferUsage) {
        return create_buffer(allocSize, bufferUsage, VMA_MEMORY_USAGE_CPU_ONLY);
    });

    ddgiResourcesDescBuffer =
        DescriptorBuffer(m_device, m_DDGIResourcesDescSetLayout, DescriptorBufferProperties::query(m_chosenGpu));
    ddgiResourcesDescBuffer.create_buffer([this](const std::size_t allocSize, const VkBufferUsageFlags bufferUsage) {
        return create_buffer(allocSize, bufferUsage, VMA_MEMORY_USAGE_CPU_ONLY);
    });
    ddgiIrradianceStorageDescBuffer =
        DescriptorBuffer(m_device, m_DDGIProbeStorageDescSetLayout, DescriptorBufferProperties::query(m_chosenGpu));
    ddgiIrradianceStorageDescBuffer.create_buffer(
        [this](const std::size_t allocSize, const VkBufferUsageFlags bufferUsage) {
            return create_buffer(allocSize, bufferUsage, VMA_MEMORY_USAGE_CPU_ONLY);
        });
    ddgiDistanceStorageDescBuffer =
        DescriptorBuffer(m_device, m_DDGIProbeStorageDescSetLayout, DescriptorBufferProperties::query(m_chosenGpu));
    ddgiDistanceStorageDescBuffer.create_buffer(
        [this](const std::size_t allocSize, const VkBufferUsageFlags bufferUsage) {
            return create_buffer(allocSize, bufferUsage, VMA_MEMORY_USAGE_CPU_ONLY);
        });
    ddgiProbeDataStorageDescBuffer =
        DescriptorBuffer(m_device, m_DDGIProbeStorageDescSetLayout, DescriptorBufferProperties::query(m_chosenGpu));
    ddgiProbeDataStorageDescBuffer.create_buffer(
        [this](const std::size_t allocSize, const VkBufferUsageFlags bufferUsage) {
            return create_buffer(allocSize, bufferUsage, VMA_MEMORY_USAGE_CPU_ONLY);
        });
    for (std::uint32_t j = 0; j < kMaxDDGIVolumes; ++j)
    {
        // RayData
        rayDatas[j] =
            create_image_array({kMaxDDGIRays, kMaxDDGIProbesX * kMaxDDGIProbesZ, 1}, VK_FORMAT_R32G32B32A32_SFLOAT,
                               VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, kMaxDDGIProbesY);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_IMAGE, reinterpret_cast<uint64_t>(rayDatas[j].image),
                                   std::format("DDGI RayData [{}]", j).c_str());
        // Irradiance
        irradianceDatas[j] = create_image_array(
            {.width = (kMaxIrradianceTexels)*kMaxDDGIProbesX,
             .height = (kMaxIrradianceTexels)*kMaxDDGIProbesZ,
             .depth = 1},
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, kMaxDDGIProbesY);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_IMAGE, reinterpret_cast<uint64_t>(irradianceDatas[j].image),
                                   std::format("DDGI Irradiance Data [{}]", j).c_str());
        // Distance
        distanceDatas[j] = create_image_array(
            {.width = (kMaxDistanceTexels)*kMaxDDGIProbesX, .height = (kMaxDistanceTexels)*kMaxDDGIProbesZ, .depth = 1},
            VK_FORMAT_R32G32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, kMaxDDGIProbesY);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_IMAGE, reinterpret_cast<uint64_t>(distanceDatas[j].image),
                                   std::format("DDGI Distance Data [{}]", j).c_str());

        // Probe Data
        probeDatas[j] = create_image_array(
            {.width = kMaxDDGIProbesX, .height = kMaxDDGIProbesZ, .depth = 1}, VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, kMaxDDGIProbesY);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_IMAGE, reinterpret_cast<uint64_t>(probeDatas[j].image),
                                   std::format("DDGI Probe Data [{}]", j).c_str());

        immediate_submit([this, j](VkCommandBuffer cmd) {
            mp::utils::BarrierBuilder barrierBuilder;
            barrierBuilder.add_image_barrier(irradianceDatas[j].transition(
                {.stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .accessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                 .layout = VK_IMAGE_LAYOUT_GENERAL,
                 .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                 .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
            barrierBuilder.add_image_barrier(distanceDatas[j].transition(
                {.stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .accessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                 .layout = VK_IMAGE_LAYOUT_GENERAL,
                 .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                 .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
            barrierBuilder.add_image_barrier(probeDatas[j].transition(
                {.stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .accessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                 .layout = VK_IMAGE_LAYOUT_GENERAL,
                 .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                 .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
            barrierBuilder.barrier(cmd);

            const VkClearColorValue color{{0.f, 0.f, 0.f, 1.f}};
            VkImageSubresourceRange range;
            range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            range.baseMipLevel = 0;
            range.levelCount = 1;
            range.baseArrayLayer = 0;
            range.layerCount = kMaxDDGIProbesY;

            vkCmdClearColorImage(cmd, irradianceDatas[j].image, VK_IMAGE_LAYOUT_GENERAL, &color, 1, &range);
            vkCmdClearColorImage(cmd, distanceDatas[j].image, VK_IMAGE_LAYOUT_GENERAL, &color, 1, &range);
            vkCmdClearColorImage(cmd, probeDatas[j].image, VK_IMAGE_LAYOUT_GENERAL, &color, 1, &range);
        });
    }
    for (std::size_t i = 0; i < m_frameData.size(); ++i)
    {
        auto &frame = m_frameData[i];
        m_drawExtent = m_windowExtent;
        create_draw_image(frame.drawImage, m_CommonImageExtent3D);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_IMAGE, reinterpret_cast<uint64_t>(frame.drawImage.image),
                                   std::format("Draw Image [{}]", i).c_str());
        create_depth_image(frame.depthImage, m_CommonImageExtent3D);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_IMAGE, reinterpret_cast<uint64_t>(frame.depthImage.image),
                                   std::format("Depth Image [{}]", i).c_str());

        frame.gBuffer.normal = create_image(m_CommonImageExtent3D, VK_FORMAT_R32_UINT,
                                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_IMAGE,
                                   reinterpret_cast<uint64_t>(frame.gBuffer.normal.image),
                                   std::format("GBuffer Normal [{}]", i).c_str());
        frame.gBuffer.diffuse = create_image(m_CommonImageExtent3D, VK_FORMAT_R8G8B8A8_UNORM,
                                             VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                                 VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_IMAGE,
                                   reinterpret_cast<uint64_t>(frame.gBuffer.diffuse.image),
                                   std::format("GBuffer Diffuse [{}]", i).c_str());
        frame.gBuffer.specular = create_image(m_CommonImageExtent3D, VK_FORMAT_R8G8B8A8_UNORM,
                                              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_IMAGE,
                                   reinterpret_cast<uint64_t>(frame.gBuffer.specular.image),
                                   std::format("GBuffer Specular [{}]", i).c_str());
        frame.gBuffer.emissive = create_image(m_CommonImageExtent3D, VK_FORMAT_R8G8B8A8_UNORM,
                                              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_IMAGE,
                                   reinterpret_cast<uint64_t>(frame.gBuffer.emissive.image),
                                   std::format("GBuffer Emissive [{}]", i).c_str());

        frame.oitAccImage = create_image(m_CommonImageExtent3D, VK_FORMAT_R16G16B16A16_SFLOAT,
                                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_IMAGE, reinterpret_cast<uint64_t>(frame.oitAccImage.image),
                                   std::format("OIT Accumulation [{}]", i).c_str());
        frame.oitRevealImage = create_image(m_CommonImageExtent3D, VK_FORMAT_R16_SFLOAT,
                                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_IMAGE,
                                   reinterpret_cast<uint64_t>(frame.oitRevealImage.image),
                                   std::format("OIT Reveal [{}]", i).c_str());
        frame.directionalShadowPassDepthArray =
            create_image_array({kDirectionalShadowMapSize, kDirectionalShadowMapSize, 1}, VK_FORMAT_D32_SFLOAT,
                               VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                               static_cast<std::uint32_t>(MAX_CASCADES));
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_IMAGE,
                                   reinterpret_cast<uint64_t>(frame.directionalShadowPassDepthArray.image),
                                   std::format("Directional Shadow Array [{}]", i).c_str());
        frame.pointLightsShadowTileMap =
            create_image({kPointLightsShadowMapSize, kPointLightsShadowMapSize, 1}, VK_FORMAT_D16_UNORM,
                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_IMAGE,
                                   reinterpret_cast<uint64_t>(frame.pointLightsShadowTileMap.image),
                                   std::format("Point Light Shadow Map [{}]", i).c_str());

        frame.sceneDataBuffer = create_buffer(
            sizeof(GpuSceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU);
        frame.sceneDataBufferAddr = frame.sceneDataBuffer.get_buffer_device_address(m_device);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_BUFFER,
                                   reinterpret_cast<uint64_t>(frame.sceneDataBuffer.buffer),
                                   std::format("Scene Data Buffer [{}]", i).c_str());

        frame.dirLightBuffer =
            create_buffer(sizeof(DirectionalLightData),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VMA_MEMORY_USAGE_GPU_ONLY);
        frame.dirLightStagingBuffer =
            create_buffer(sizeof(DirectionalLightData), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        frame.dirLightBufferAddr = frame.dirLightBuffer.get_buffer_device_address(m_device);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_BUFFER,
                                   reinterpret_cast<uint64_t>(frame.dirLightBuffer.buffer),
                                   std::format("Directional Light Buffer [{}]", i).c_str());

        frame.pointLightBuffer =
            create_buffer(sizeof(PointLightData) * kMaxPointLights,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                          VMA_MEMORY_USAGE_CPU_TO_GPU);
        frame.pointLightBufferAddr = frame.pointLightBuffer.get_buffer_device_address(m_device);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_BUFFER,
                                   reinterpret_cast<uint64_t>(frame.pointLightBuffer.buffer),
                                   std::format("Point Light Buffer [{}]", i).c_str());

        frame.visiblePointLightsBuffer = create_buffer(
            sizeof(PointLightData) * kMaxPointLights,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        frame.visiblePointLightsBufferAddr = frame.visiblePointLightsBuffer.get_buffer_device_address(m_device);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_BUFFER,
                                   reinterpret_cast<uint64_t>(frame.visiblePointLightsBuffer.buffer),
                                   std::format("Visible Point Lights Buffer [{}]", i).c_str());

        frame.visiblePointLightsCountBuffer =
            create_buffer(sizeof(std::uint32_t),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VMA_MEMORY_USAGE_GPU_ONLY);
        frame.visiblePointLightsCountBufferAddr =
            frame.visiblePointLightsCountBuffer.get_buffer_device_address(m_device);

        frame.pointLightIndicesBuffer = create_buffer(
            sizeof(std::uint32_t) * kMaxPointLights * kMaxPointLights,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        frame.pointLightIndicesBufferAddr = frame.pointLightIndicesBuffer.get_buffer_device_address(m_device);

        frame.pointLightIndicesOffsetsBuffer =
            create_buffer(sizeof(std::uint32_t) * kMaxInstances,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VMA_MEMORY_USAGE_GPU_ONLY);
        frame.pointLightIndicesOffsetsBufferAddr =
            frame.pointLightIndicesOffsetsBuffer.get_buffer_device_address(m_device);

        frame.pointLightIndicesOffsetsCounterBuffer =
            create_buffer(sizeof(std::uint32_t),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VMA_MEMORY_USAGE_GPU_ONLY);
        frame.pointLightIndicesOffsetsCounterBufferAddr =
            frame.pointLightIndicesOffsetsCounterBuffer.get_buffer_device_address(m_device);

        frame.minMaxBuffer = create_buffer(sizeof(MinMax),
                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                           VMA_MEMORY_USAGE_GPU_ONLY);
        frame.minMaxBufferAddr = frame.minMaxBuffer.get_buffer_device_address(m_device);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_BUFFER,
                                   reinterpret_cast<uint64_t>(frame.minMaxBuffer.buffer),
                                   std::format("MinMax Buffer [{}]", i).c_str());

        frame.splitsAABBBuffer = create_buffer(sizeof(CascadesAABB),
                                               VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                               VMA_MEMORY_USAGE_GPU_ONLY);
        frame.splitsAABBBufferAddr = frame.splitsAABBBuffer.get_buffer_device_address(m_device);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_BUFFER,
                                   reinterpret_cast<uint64_t>(frame.splitsAABBBuffer.buffer),
                                   std::format("Splits AABB Buffer [{}]", i).c_str());

        frame.histogramBuffer = create_buffer(sizeof(std::uint32_t) * 256,
                                              VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                              VMA_MEMORY_USAGE_GPU_ONLY);
        frame.histogramBufferAddr = frame.histogramBuffer.get_buffer_device_address(m_device);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_BUFFER,
                                   reinterpret_cast<uint64_t>(frame.histogramBuffer.buffer),
                                   std::format("Luminance Histogram Buffer [{}]", i).c_str());

        frame.cascadeDepthDescBuffer =
            DescriptorBuffer(m_device, m_DepthPassDescSetLayout, DescriptorBufferProperties::query(m_chosenGpu));
        frame.cascadeDepthDescBuffer.create_buffer(
            [this](const std::size_t allocSize, const VkBufferUsageFlags bufferUsage) {
                return create_buffer(allocSize, bufferUsage, VMA_MEMORY_USAGE_CPU_ONLY);
            });
        frame.instanceBuffer =
            create_buffer(sizeof(Instance) * kMaxInstances,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VMA_MEMORY_USAGE_GPU_ONLY);
        frame.instanceStagingBuffer = create_buffer(sizeof(Instance) * kMaxInstances, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                    VMA_MEMORY_USAGE_CPU_TO_GPU);
        frame.instanceBufferAddr = frame.instanceBuffer.get_buffer_device_address(m_device);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_BUFFER,
                                   reinterpret_cast<uint64_t>(frame.instanceBuffer.buffer),
                                   std::format("Instance Buffer [{}]", i).c_str());

        frame.meshesBuffer =
            create_buffer(sizeof(RenderObject) * kMaxMeshes,
                          VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT,
                          VMA_MEMORY_USAGE_CPU_TO_GPU);

        frame.meshBufferAddr = frame.meshesBuffer.get_buffer_device_address(m_device);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_BUFFER,
                                   reinterpret_cast<uint64_t>(frame.meshesBuffer.buffer),
                                   std::format("Meshes Buffer [{}]", i).c_str());
        frame.drawCommandsBuffer =
            create_buffer(sizeof(VkDrawIndexedIndirectCommand) * kMaxInstances,
                          VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT,
                          VMA_MEMORY_USAGE_GPU_ONLY);
        frame.drawCommandsBufferAddr = frame.drawCommandsBuffer.get_buffer_device_address(m_device);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_BUFFER,
                                   reinterpret_cast<uint64_t>(frame.drawCommandsBuffer.buffer),
                                   std::format("Draw Commands Buffer [{}]", i).c_str());

        frame.countBuffer = create_buffer(sizeof(std::uint32_t),
                                          VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT |
                                              VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT,
                                          VMA_MEMORY_USAGE_GPU_ONLY);
        frame.countBufferAddr = frame.countBuffer.get_buffer_device_address(m_device);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_BUFFER,
                                   reinterpret_cast<uint64_t>(frame.countBuffer.buffer),
                                   std::format("Count Buffer [{}]", i).c_str());

        frame.lightPassDescriptorBuffer =
            DescriptorBuffer(m_device, m_LightPassDescriptorSetLayout, DescriptorBufferProperties::query(m_chosenGpu));

        frame.lightPassDescriptorBuffer.create_buffer(
            [&](const std::size_t allocSize, const VkBufferUsageFlags bufferUsage) {
                return create_buffer(allocSize, bufferUsage, VMA_MEMORY_USAGE_CPU_ONLY);
            });

        frame.wboitCompositePassDescBuffer = DescriptorBuffer(m_device, m_WboitCompositePassDescriptorSetLayout,
                                                              DescriptorBufferProperties::query(m_chosenGpu));
        frame.wboitCompositePassDescBuffer.create_buffer(
            [this](const std::size_t allocSize, const VkBufferUsageFlags bufferUsage) {
                return create_buffer(allocSize, bufferUsage, VMA_MEMORY_USAGE_CPU_ONLY);
            });

        frame.drawImageDescriptorBuffer =
            DescriptorBuffer(m_device, m_DrawImageDescriptorSetLayout, DescriptorBufferProperties::query(m_chosenGpu));

        frame.drawImageDescriptorBuffer.create_buffer(
            [&](const std::size_t allocSize, const VkBufferUsageFlags bufferUsage) {
                return create_buffer(allocSize, bufferUsage, VMA_MEMORY_USAGE_CPU_ONLY);
            });

        frame.ddgiTLASDescBuffer =
            DescriptorBuffer(m_device, m_DDGITLASDescSetLayout, DescriptorBufferProperties::query(m_chosenGpu));
        frame.ddgiTLASDescBuffer.create_buffer(
            [this](const std::size_t allocSize, const VkBufferUsageFlags bufferUsage) {
                return create_buffer(allocSize, bufferUsage, VMA_MEMORY_USAGE_CPU_ONLY);
            });
        frame.ddgiOutputStorageDescBuffer =
            DescriptorBuffer(m_device, m_DrawImageDescriptorSetLayout, DescriptorBufferProperties::query(m_chosenGpu));
        frame.ddgiOutputStorageDescBuffer.create_buffer(
            [this](const std::size_t allocSize, const VkBufferUsageFlags bufferUsage) {
                return create_buffer(allocSize, bufferUsage, VMA_MEMORY_USAGE_CPU_ONLY);
            });
        frame.ddgiGBufferReadDescBuffer =
            DescriptorBuffer(m_device, m_DDGIGBufferReadDescSetLayout, DescriptorBufferProperties::query(m_chosenGpu));
        frame.ddgiGBufferReadDescBuffer.create_buffer(
            [this](const std::size_t allocSize, const VkBufferUsageFlags bufferUsage) {
                return create_buffer(allocSize, bufferUsage, VMA_MEMORY_USAGE_CPU_ONLY);
            });
        frame.ddgiOutput = create_image(m_CommonImageExtent3D, VK_FORMAT_R16G16B16A16_SFLOAT,
                                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_IMAGE, reinterpret_cast<uint64_t>(frame.ddgiOutput.image),
                                   std::format("DDGI Output [{}]", i).c_str());
    }

    m_mainDeletionQueue.push_function([this] {
        vkDestroySampler(m_device, m_shadowSampler, nullptr);

        for (std::uint32_t j = 0; j < kMaxDDGIVolumes; ++j)
        {
            auto &rayData = rayDatas[j];
            auto &distanceData = distanceDatas[j];
            auto &irradianceData = irradianceDatas[j];
            auto &probeData = probeDatas[j];
            vkDestroyImageView(m_device, rayData.imageView, nullptr);
            vmaDestroyImage(m_allocator, rayData.image, rayData.allocation);
            vkDestroyImageView(m_device, distanceData.imageView, nullptr);
            vmaDestroyImage(m_allocator, distanceData.image, distanceData.allocation);
            vkDestroyImageView(m_device, irradianceData.imageView, nullptr);
            vmaDestroyImage(m_allocator, irradianceData.image, irradianceData.allocation);
            vkDestroyImageView(m_device, probeData.imageView, nullptr);
            vmaDestroyImage(m_allocator, probeData.image, probeData.allocation);
        }
        destroy_buffer(ddgiRayDataDescBuffer.get_buffer());
        destroy_buffer(ddgiResourcesDescBuffer.get_buffer());
        destroy_buffer(ddgiIrradianceStorageDescBuffer.get_buffer());
        destroy_buffer(ddgiDistanceStorageDescBuffer.get_buffer());
        destroy_buffer(ddgiProbeDataStorageDescBuffer.get_buffer());
        for (auto &frame : m_frameData)
        {

            destroy_image(frame.gBuffer.normal);
            destroy_image(frame.gBuffer.diffuse);
            destroy_image(frame.gBuffer.specular);
            destroy_image(frame.gBuffer.emissive);

            destroy_image(frame.oitAccImage);
            destroy_image(frame.oitRevealImage);
            destroy_image(frame.pointLightsShadowTileMap);

            vkDestroyImageView(m_device, frame.directionalShadowPassDepthArray.imageView, nullptr);
            vmaDestroyImage(m_allocator, frame.directionalShadowPassDepthArray.image,
                            frame.directionalShadowPassDepthArray.allocation);
            destroy_buffer(frame.instanceBuffer);
            destroy_buffer(frame.instanceStagingBuffer);
            destroy_buffer(frame.drawCommandsBuffer);
            destroy_buffer(frame.meshesBuffer);
            destroy_buffer(frame.countBuffer);
            destroy_buffer(frame.sceneDataBuffer);
            destroy_buffer(frame.dirLightBuffer);
            destroy_buffer(frame.dirLightStagingBuffer);
            destroy_buffer(frame.pointLightIndicesBuffer);
            destroy_buffer(frame.pointLightIndicesOffsetsBuffer);
            destroy_buffer(frame.pointLightIndicesOffsetsCounterBuffer);
            destroy_buffer(frame.visiblePointLightsBuffer);
            destroy_buffer(frame.visiblePointLightsCountBuffer);
            destroy_buffer(frame.pointLightBuffer);
            destroy_buffer(frame.minMaxBuffer);
            destroy_buffer(frame.splitsAABBBuffer);
            destroy_buffer(frame.histogramBuffer);
            destroy_buffer(frame.wboitCompositePassDescBuffer.get_buffer());
            destroy_buffer(frame.lightPassDescriptorBuffer.get_buffer());
            destroy_buffer(frame.drawImageDescriptorBuffer.get_buffer());
            destroy_buffer(frame.cascadeDepthDescBuffer.get_buffer());
            destroy_image(frame.ddgiOutput);
            destroy_buffer(frame.ddgiTLASDescBuffer.get_buffer());
            destroy_buffer(frame.ddgiOutputStorageDescBuffer.get_buffer());
            destroy_buffer(frame.ddgiGBufferReadDescBuffer.get_buffer());
        }
    });

    m_avgLuminanceBuffer =
        create_buffer(sizeof(float),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VMA_MEMORY_USAGE_GPU_ONLY);
    m_avgLuminanceBufferAddr = m_avgLuminanceBuffer.get_buffer_device_address(m_device);
    mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_BUFFER, reinterpret_cast<uint64_t>(m_avgLuminanceBuffer.buffer),
                               "Average Luminance Buffer");
    // Bootstrap with 1.0f so the first frame doesn't divide by zero
    immediate_submit(
        [&](VkCommandBuffer cmd) { vkCmdFillBuffer(cmd, m_avgLuminanceBuffer.buffer, 0, VK_WHOLE_SIZE, 0x3F800000u); });
    m_mainDeletionQueue.push_function([this] { destroy_buffer(m_avgLuminanceBuffer); });
}

AccelerationStructure Engine::allocate_acceleration_structure(VkAccelerationStructureCreateInfoKHR &createInfo)
{
    AccelerationStructure resultAccel{};

    resultAccel.buffer = create_buffer(createInfo.size,
                                       VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                           VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT,
                                       VMA_MEMORY_USAGE_GPU_ONLY);

    createInfo.buffer = resultAccel.buffer.buffer;
    vkCreateAccelerationStructureKHR(m_device, &createInfo, nullptr, &resultAccel.accel) >> chk;

    {
        VkAccelerationStructureDeviceAddressInfoKHR info{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
            .accelerationStructure = resultAccel.accel};
        resultAccel.address = vkGetAccelerationStructureDeviceAddressKHR(m_device, &info);
    }

    return resultAccel;
}

void Engine::create_acceleration_structure(VkAccelerationStructureTypeKHR asType,
                                           VkBuildAccelerationStructureFlagsKHR flags,
                                           VkAccelerationStructureGeometryKHR &asGeometry,
                                           VkAccelerationStructureBuildRangeInfoKHR &asBuildRangeInfo,
                                           AccelerationStructure &accelerationStructure)
{
    auto alignUp = [](auto value, size_t alignment) noexcept { return ((value + alignment - 1) & ~(alignment - 1)); };

    VkAccelerationStructureBuildGeometryInfoKHR asBuildInfo{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .type = asType,                                         // The type of acceleration structure (BLAS or TLAS)
        .flags = flags,                                         // Build flags (e.g. prefer fast trace)
        .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR, // Build mode vs update
        .geometryCount = 1,                                     // Deal with one geometry at a time
        .pGeometries = &asGeometry,                             // The geometry to build the acceleration structure from
    };

    std::vector<std::uint32_t> maxPrimCount{asBuildRangeInfo.primitiveCount};
    VkAccelerationStructureBuildSizesInfoKHR asBuildSize{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR(m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &asBuildInfo,
                                            maxPrimCount.data(), &asBuildSize);

    VkDeviceSize scratchSize =
        alignUp(asBuildSize.buildScratchSize, m_ASProperties.minAccelerationStructureScratchOffsetAlignment);

    AllocatedBuffer scratchBuffer =
        create_buffer(scratchSize,
                      VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT |
                          VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                      VMA_MEMORY_USAGE_GPU_ONLY);

    VkAccelerationStructureCreateInfoKHR createInfo{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
        .size = asBuildSize.accelerationStructureSize,
        .type = asType,
    };
    accelerationStructure = allocate_acceleration_structure(createInfo);

    immediate_submit([&](VkCommandBuffer cmd) {
        asBuildInfo.dstAccelerationStructure = accelerationStructure.accel;
        asBuildInfo.scratchData.deviceAddress = scratchBuffer.get_buffer_device_address(m_device);

        VkAccelerationStructureBuildRangeInfoKHR *pBuildRangeInfo = &asBuildRangeInfo;
        vkCmdBuildAccelerationStructuresKHR(cmd, 1, &asBuildInfo, &pBuildRangeInfo);
    });

    destroy_buffer(scratchBuffer);
}

void Engine::create_BLAS()
{
    const auto meshesCount = m_mainDrawContext.renderObjects.size();
    m_BlasAccels.resize(meshesCount);

    for (std::size_t i = 0; i < meshesCount; ++i)
    {
        VkAccelerationStructureGeometryKHR asGeometry{};
        VkAccelerationStructureBuildRangeInfoKHR asBuildRangeInfo{};

        primitive_to_geometry(m_mainDrawContext.renderObjects[i], asGeometry, asBuildRangeInfo);

        create_acceleration_structure(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                                      VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR, asGeometry,
                                      asBuildRangeInfo, m_BlasAccels[i]);
        debug::set_object_name(m_device, VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR,
                               reinterpret_cast<std::uint64_t>(m_BlasAccels[i].accel),
                               std::format("Bottom-level acceleration structure: {}", i).c_str());
    }

    std::println("Bottom-level acceleration structures built successfully");
}

void Engine::create_TLAS()
{
    auto toTransformMatrixKHR = [](const glm::mat4 &m) {
        VkTransformMatrixKHR t;
        memcpy(&t, glm::value_ptr(glm::transpose(m)), sizeof(t));
        return t;
    };
    std::vector<VkAccelerationStructureInstanceKHR> tlasInstances;
    auto appendInstances = [&tlasInstances, &toTransformMatrixKHR, this](const std::vector<Instance> &instances) {
        tlasInstances.reserve(tlasInstances.size() + instances.size());
        for (const auto &instance : instances)
        {
            VkAccelerationStructureInstanceKHR asInstance{};
            asInstance.transform = toTransformMatrixKHR(instance.world); // Position of the instance
            asInstance.instanceCustomIndex = instance.meshIndex;         // gl_InstanceCustomIndexEXT
            asInstance.accelerationStructureReference = m_BlasAccels[instance.meshIndex].address; // Will be set in
            asInstance.instanceShaderBindingTableRecordOffset = 0; // We will use the same hit group for all objects
            asInstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV; // No culling - double sided
            asInstance.mask = 0xFF;
            tlasInstances.emplace_back(asInstance);
        }
    };

    appendInstances(m_mainDrawContext.opaqueInstances);
    appendInstances(m_mainDrawContext.alphaTestedInstances);
    appendInstances(m_mainDrawContext.transparentInstances);

    const auto instancesSize = std::span<VkAccelerationStructureInstanceKHR const>(tlasInstances).size_bytes();
    AllocatedBuffer tlasInstanceBuffer =
        create_buffer(instancesSize,
                      VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                          VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
                      VMA_MEMORY_USAGE_GPU_ONLY);

    AllocatedBuffer stagingBuffer =
        create_buffer(instancesSize, VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

    std::memcpy(stagingBuffer.allocationInfo.pMappedData, tlasInstances.data(), instancesSize);
    immediate_submit([&stagingBuffer, &tlasInstanceBuffer, instancesSize](VkCommandBuffer cmd) {
        VkBufferCopy region{.srcOffset = 0, .dstOffset = 0, .size = instancesSize};
        vkCmdCopyBuffer(cmd, stagingBuffer.buffer, tlasInstanceBuffer.buffer, 1, &region);
    });
    destroy_buffer(stagingBuffer);

    {
        VkAccelerationStructureGeometryKHR asGeometry{};
        VkAccelerationStructureBuildRangeInfoKHR asBuildRangeInfo{};

        // Convert the instance information to acceleration structure geometry, similar to primitiveToGeometry()
        VkAccelerationStructureGeometryInstancesDataKHR geometryInstances{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
            .data = {.deviceAddress = tlasInstanceBuffer.get_buffer_device_address(m_device)}};
        asGeometry = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                      .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
                      .geometry = {.instances = geometryInstances}};
        asBuildRangeInfo = {.primitiveCount = static_cast<std::uint32_t>(tlasInstances.size())};

        create_acceleration_structure(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
                                      VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR, asGeometry,
                                      asBuildRangeInfo, m_TlasAccel);
        debug::set_object_name(m_device, VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR,
                               reinterpret_cast<std::uint64_t>(m_TlasAccel.accel), "TLAS");
    }

    std::println("Top-level acceleration structure has been built successfully");
    destroy_buffer(tlasInstanceBuffer);
}

void Engine::primitive_to_geometry(RenderObject &mesh, VkAccelerationStructureGeometryKHR &asGeometry,
                                   VkAccelerationStructureBuildRangeInfoKHR &asBuildRangeInfo)
{
    // Describe buffer as array of VertexObj.
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
        .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT, // vec3 vertex position data
        .vertexData = {.deviceAddress = m_globalPositionBufferAddress},
        .vertexStride = sizeof(glm::vec3),
        .maxVertex = static_cast<std::uint32_t>(m_globalPositionCount - 1),
        .indexType = VK_INDEX_TYPE_UINT32, // Index type (VK_INDEX_TYPE_UINT16 or VK_INDEX_TYPE_UINT32)
        .indexData = {.deviceAddress = m_globalIndexBufferDeviceAddress},
    };

    // Identify the above data as containing opaque triangles.
    asGeometry = VkAccelerationStructureGeometryKHR{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
        .geometry = {.triangles = triangles},
        .flags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR | VK_GEOMETRY_OPAQUE_BIT_KHR,
    };

    asBuildRangeInfo = VkAccelerationStructureBuildRangeInfoKHR{
        .primitiveCount = mesh.indexCount / 3,
        .primitiveOffset = mesh.firstIndex * static_cast<std::uint32_t>(sizeof(std::uint32_t)),
        .firstVertex = static_cast<std::uint32_t>(mesh.vertexOffset),
    };
}

void Engine::init_default_data()
{
    std::uint32_t whiteColor = glm::packUnorm4x8(glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
    m_whiteImage = create_image(&whiteColor, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);
    mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_IMAGE, reinterpret_cast<uint64_t>(m_whiteImage.image),
                               "Default White Image");
    std::uint32_t blackColor = glm::packUnorm4x8(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
    m_blackImage = create_image(&blackColor, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);
    mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_IMAGE, reinterpret_cast<uint64_t>(m_blackImage.image),
                               "Default Black Image");
    std::uint32_t normalFallback = glm::packUnorm4x8(glm::vec4(0.5f, 0.5f, 1.0f, 1.0f));
    m_normalFallback =
        create_image(&normalFallback, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);
    mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_IMAGE, reinterpret_cast<uint64_t>(m_normalFallback.image),
                               "Default Normal Fallback Image");

    VkSamplerCreateInfo samplerCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .anisotropyEnable = true,
        .maxAnisotropy = 16.0f,
        .minLod = 0,
        .maxLod = VK_LOD_CLAMP_NONE,
    };
    vkCreateSampler(m_device, &samplerCreateInfo, nullptr, &m_defaultSamplerLinear);
    mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_SAMPLER, reinterpret_cast<uint64_t>(m_defaultSamplerLinear),
                               "Default Linear Sampler");
    samplerCreateInfo.minFilter = VK_FILTER_NEAREST;
    samplerCreateInfo.magFilter = VK_FILTER_NEAREST;
    vkCreateSampler(m_device, &samplerCreateInfo, nullptr, &m_defaultSamplerNearest);
    mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_SAMPLER, reinterpret_cast<uint64_t>(m_defaultSamplerNearest),
                               "Default Nearest Sampler");

    [[maybe_unused]] auto samplerIdx = m_metalRoughness.write_sampler(m_defaultSamplerLinear);
    assert(0 == samplerIdx);
    [[maybe_unused]] auto whiteTexIdx = m_metalRoughness.write_texture(m_whiteImage.imageView);
    assert(0 == whiteTexIdx);
    [[maybe_unused]] auto blackTexIdx = m_metalRoughness.write_texture(m_blackImage.imageView);
    assert(1 == blackTexIdx);
    [[maybe_unused]] auto normalTexIdx = m_metalRoughness.write_texture(m_normalFallback.imageView);
    assert(2 == normalTexIdx);
    m_mainDeletionQueue.push_function([&] {
        m_metalRoughness.clear_resources(*this);

        destroy_image(m_whiteImage);
        destroy_image(m_blackImage);
        destroy_image(m_normalFallback);

        vkDestroySampler(m_device, m_defaultSamplerLinear, nullptr);
        vkDestroySampler(m_device, m_defaultSamplerNearest, nullptr);
    });
}

void Engine::init_mesh_data()
{
    ensure_position_capacity(1024); // Initial capacity
    ensure_attributes_capacity(1024);
    ensure_index_capacity(1024);

    const wchar_t *filters[]{L"*.gltf", L"*.glb"};
    const auto *fileNames =
        tinyfd_openFileDialogW(L"Load gltf", L"../../assets/gltf-samples/Models/Sponza/glTF/sponza.gltf",
                               std::size(filters), filters, L"Gltf files", 1);
    const auto paths = parse_tiny_multiple(fileNames);
    if (paths.empty())
    {
        const auto path = L"../../assets/gltf-samples/Models/Sponza/glTF/sponza.gltf";
        if (!load_gltf(*this, path))
        {
            throw std::runtime_error(std::format("Failed to load gltf"));
        }
    }
    else
    {
        for (const auto &path : paths)
        {
            if (!load_gltf(*this, path))
            {
                throw std::runtime_error(std::format("Failed to load gltf"));
            }
        }
    }

    m_scene.draw(glm::mat4(1.0f), m_mainDrawContext);
    create_BLAS();
    create_TLAS();

    m_mainDeletionQueue.push_function([this] {
        destroy_buffer(m_globalPositionBuffer);
        destroy_buffer(m_globalAttributesBuffer);
        destroy_buffer(m_globalIndexBuffer);

        for (auto &blas : m_BlasAccels)
        {
            vkDestroyAccelerationStructureKHR(m_device, blas.accel, nullptr);
            destroy_buffer(blas.buffer);
        }
        vkDestroyAccelerationStructureKHR(m_device, m_TlasAccel.accel, nullptr);
        destroy_buffer(m_TlasAccel.buffer);
    });
}

void Engine::init_alpha_tested_prepass()
{
    const VkPushConstantRange pushConstantRange{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0,
        .size = static_cast<std::uint32_t>(sizeof(GBufferPassPushConstants)),
    };

    const VkDescriptorSetLayout layouts[]{m_metalRoughness.materialLayout};
    const VkPipelineLayoutCreateInfo layoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = std::size(layouts),
        .pSetLayouts = layouts,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstantRange,
    };
    vkCreatePipelineLayout(m_device, &layoutCreateInfo, nullptr, &m_AlphaTestedPrepassPipelineLayout) >> chk;

    VkShaderModule prepassVert;
    if (!load_shader_module("../../src/compiled_shaders/prepass.vertex.spv", m_device, &prepassVert))
    {
        throw std::runtime_error("Failed to load prepass vertex shader");
    }

    VkShaderModule prepassFrag;
    if (!load_shader_module("../../src/compiled_shaders/prepass.pixel.spv", m_device, &prepassFrag))
    {
        throw std::runtime_error("Failed to load prepass pixel shader");
    }

    PipelineBuilder builder;
    builder.pipelineLayout = m_AlphaTestedPrepassPipelineLayout;
    builder.enable_depth_test(true, VK_COMPARE_OP_LESS_OR_EQUAL);
    builder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    builder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    builder.add_shader(prepassVert, VK_SHADER_STAGE_VERTEX_BIT);
    builder.add_shader(prepassFrag, VK_SHADER_STAGE_FRAGMENT_BIT);
    builder.set_depth_format(VK_FORMAT_D32_SFLOAT);
    builder.set_cull_mode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);
    builder.set_multisampling_none();

    m_AlphaTestedPrepassPipeline = builder.build_pipeline(m_device, VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT);
    mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_PIPELINE,
                               reinterpret_cast<uint64_t>(m_AlphaTestedPrepassPipeline),
                               "Alpha-Tested Prepass Pipeline");

    vkDestroyShaderModule(m_device, prepassVert, nullptr);
    vkDestroyShaderModule(m_device, prepassFrag, nullptr);

    m_mainDeletionQueue.push_function([this]() {
        vkDestroyPipeline(m_device, m_AlphaTestedPrepassPipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_AlphaTestedPrepassPipelineLayout, nullptr);
    });
}

void Engine::init_alpha_tested_directional_shadow_pass()
{
    const VkDescriptorSetLayout layouts[]{m_metalRoughness.materialLayout};

    const VkPushConstantRange pushConstantRange{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = 0,
        .size = sizeof(DirectionalShadowPassAlphaTestedPushConstants),
    };

    const VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .setLayoutCount = std::size(layouts),
        .pSetLayouts = layouts,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstantRange,
    };
    vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &m_AlphaTestedShadowPassPipelineLayout) >> chk;

    VkShaderModule shadowPassVert;
    if (!mp::load_shader_module("../../src/compiled_shaders/directional_shadow_pass_alpha_tested.vertex.spv", m_device,
                                &shadowPassVert))
    {
        throw std::runtime_error("Failed to load alpha-tested directional shadow pass vertex shader");
    }

    VkShaderModule shadowPassFrag;
    if (!mp::load_shader_module("../../src/compiled_shaders/directional_shadow_pass_alpha_tested.pixel.spv", m_device,
                                &shadowPassFrag))
    {
        throw std::runtime_error("Failed to load alpha-tested directional shadow pass pixel shader");
    }

    mp::PipelineBuilder builder;
    builder.pipelineLayout = m_AlphaTestedShadowPassPipelineLayout;
    builder.enable_depth_test(true, VK_COMPARE_OP_LESS_OR_EQUAL);
    builder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    builder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    builder.add_shader(shadowPassVert, VK_SHADER_STAGE_VERTEX_BIT);
    builder.add_shader(shadowPassFrag, VK_SHADER_STAGE_FRAGMENT_BIT);
    builder.set_depth_format(VK_FORMAT_D32_SFLOAT);
    builder.set_cull_mode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);
    builder.set_multisampling_none();

    m_AlphaTestedShadowPassPipeline = builder.build_pipeline(m_device, VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT);
    mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_PIPELINE,
                               reinterpret_cast<uint64_t>(m_AlphaTestedShadowPassPipeline),
                               "Alpha-Tested Directional Shadow Pass Pipeline");

    vkDestroyShaderModule(m_device, shadowPassVert, nullptr);
    vkDestroyShaderModule(m_device, shadowPassFrag, nullptr);

    m_mainDeletionQueue.push_function([this]() {
        vkDestroyPipeline(m_device, m_AlphaTestedShadowPassPipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_AlphaTestedShadowPassPipelineLayout, nullptr);
    });
}

void Engine::init_alpha_tested_point_shadow_pass()
{
    const VkDescriptorSetLayout layouts[]{m_metalRoughness.materialLayout};

    const VkPushConstantRange pushConstantRange{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = 0,
        .size = sizeof(PointLightsShadowPassAlphaTestedPushConstants),
    };

    const VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .setLayoutCount = std::size(layouts),
        .pSetLayouts = layouts,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstantRange,
    };
    vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr,
                           &m_AlphaTestedPointLightShadowPassPipelineLayout) >>
        chk;

    VkShaderModule vert;
    if (!mp::load_shader_module("../../src/compiled_shaders/point_lights_shadow_pass_alpha_tested.vertex.spv", m_device,
                                &vert))
    {
        throw std::runtime_error("Failed to load alpha-tested point light shadow pass vertex shader");
    }

    VkShaderModule geom;
    if (!mp::load_shader_module("../../src/compiled_shaders/point_lights_shadow_pass_alpha_tested.geometry.spv",
                                m_device, &geom))
    {
        throw std::runtime_error("Failed to load alpha-tested point light shadow pass geometry shader");
    }

    VkShaderModule frag;
    if (!mp::load_shader_module("../../src/compiled_shaders/point_lights_shadow_pass_alpha_tested.pixel.spv", m_device,
                                &frag))
    {
        throw std::runtime_error("Failed to load alpha-tested point light shadow pass pixel shader");
    }

    mp::PipelineBuilder builder;
    builder.pipelineLayout = m_AlphaTestedPointLightShadowPassPipelineLayout;
    builder.enable_depth_test(true, VK_COMPARE_OP_LESS_OR_EQUAL);
    builder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    builder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    builder.add_shader(vert, VK_SHADER_STAGE_VERTEX_BIT);
    builder.add_shader(geom, VK_SHADER_STAGE_GEOMETRY_BIT);
    builder.add_shader(frag, VK_SHADER_STAGE_FRAGMENT_BIT);
    builder.set_depth_format(VK_FORMAT_D16_UNORM);
    builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
    builder.set_multisampling_none();

    m_AlphaTestedPointLightShadowPassPipeline =
        builder.build_pipeline(m_device, VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT);
    mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_PIPELINE,
                               reinterpret_cast<uint64_t>(m_AlphaTestedPointLightShadowPassPipeline),
                               "Alpha-Tested Point Light Shadow Pass Pipeline");

    vkDestroyShaderModule(m_device, vert, nullptr);
    vkDestroyShaderModule(m_device, geom, nullptr);
    vkDestroyShaderModule(m_device, frag, nullptr);

    m_mainDeletionQueue.push_function([this]() {
        vkDestroyPipeline(m_device, m_AlphaTestedPointLightShadowPassPipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_AlphaTestedPointLightShadowPassPipelineLayout, nullptr);
    });
}

void Engine::init_ddgi_probe_vis_pipeline()
{
    {
        constexpr int rings = 8;
        constexpr int sectors = 8;

        std::vector<glm::vec3> vertices;
        std::vector<std::uint32_t> indices;
        vertices.reserve(rings * sectors);
        indices.reserve((rings - 1) * (sectors - 1) * 6);

        for (int r = 0; r < rings; ++r)
        {
            for (int s = 0; s < sectors; ++s)
            {
                const float y = std::sin(-glm::half_pi<float>() + glm::pi<float>() * r / (rings - 1));
                const float x =
                    std::cos(2.f * glm::pi<float>() * s / (sectors - 1)) * std::sin(glm::pi<float>() * r / (rings - 1));
                const float z =
                    std::sin(2.f * glm::pi<float>() * s / (sectors - 1)) * std::sin(glm::pi<float>() * r / (rings - 1));
                vertices.emplace_back(x, y, z);
            }
        }
        for (int r = 0; r < rings - 1; ++r)
        {
            for (int s = 0; s < sectors - 1; ++s)
            {
                indices.push_back(r * sectors + s);
                indices.push_back(r * sectors + (s + 1));
                indices.push_back((r + 1) * sectors + (s + 1));
                indices.push_back(r * sectors + s);
                indices.push_back((r + 1) * sectors + (s + 1));
                indices.push_back((r + 1) * sectors + s);
            }
        }
        m_probeSphereIndexCount = static_cast<std::uint32_t>(indices.size());

        const auto vbSize = vertices.size() * sizeof(glm::vec3);
        const auto ibSize = indices.size() * sizeof(std::uint32_t);

        m_probeSphereVertexBuffer =
            create_buffer(vbSize,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VMA_MEMORY_USAGE_GPU_ONLY);
        m_probeSphereIndexBuffer = create_buffer(
            ibSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

        AllocatedBuffer stagingVB = create_buffer(vbSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
        AllocatedBuffer stagingIB = create_buffer(ibSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

        std::memcpy(stagingVB.allocationInfo.pMappedData, vertices.data(), vbSize);
        std::memcpy(stagingIB.allocationInfo.pMappedData, indices.data(), ibSize);

        immediate_submit([&](VkCommandBuffer cmd) {
            VkBufferCopy vCopy{.srcOffset = 0, .dstOffset = 0, .size = vbSize};
            vkCmdCopyBuffer(cmd, stagingVB.buffer, m_probeSphereVertexBuffer.buffer, 1, &vCopy);
            VkBufferCopy iCopy{.srcOffset = 0, .dstOffset = 0, .size = ibSize};
            vkCmdCopyBuffer(cmd, stagingIB.buffer, m_probeSphereIndexBuffer.buffer, 1, &iCopy);
        });

        destroy_buffer(stagingVB);
        destroy_buffer(stagingIB);

        m_probeSphereVerticesAddr = m_probeSphereVertexBuffer.get_buffer_device_address(m_device);

        m_mainDeletionQueue.push_function([this] {
            destroy_buffer(m_probeSphereVertexBuffer);
            destroy_buffer(m_probeSphereIndexBuffer);
        });
    }

    // --- Pipeline layout ---
    {
        const VkDescriptorSetLayout setLayouts[]{m_DDGIResourcesDescSetLayout};
        const VkPushConstantRange pcRange{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(DDGIProbeVisPushConstants),
        };
        const VkPipelineLayoutCreateInfo layoutInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = setLayouts,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pcRange,
        };
        vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_DDGIProbeVisPipelineLayout) >> chk;
    }

    VkShaderModule vertShader;
    if (!load_shader_module("../../src/compiled_shaders/probe_vis.vertex.spv", m_device, &vertShader))
        throw std::runtime_error("Failed to load probe_vis.vertex.spv");

    VkShaderModule fragShader;
    if (!load_shader_module("../../src/compiled_shaders/probe_vis.fragment.spv", m_device, &fragShader))
        throw std::runtime_error("Failed to load probe_vis.fragment.spv");

    {
        PipelineBuilder builder;
        builder.pipelineLayout = m_DDGIProbeVisPipelineLayout;
        builder.add_shader(vertShader, VK_SHADER_STAGE_VERTEX_BIT);
        builder.add_shader(fragShader, VK_SHADER_STAGE_FRAGMENT_BIT);
        builder.enable_depth_test(true, VK_COMPARE_OP_LESS_OR_EQUAL);
        builder.set_depth_format(m_frameData.at(0).depthImage.imageFormat);
        builder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        builder.set_polygon_mode(VK_POLYGON_MODE_FILL);
        builder.set_cull_mode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);
        builder.set_multisampling_none();
        builder.add_color_attachment_format(m_frameData.at(0).drawImage.imageFormat);
        builder.colorBlends.push_back({.blendEnable = VK_FALSE,
                                       .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT});
        m_DDGIProbeVisPipeline = builder.build_pipeline(m_device, VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_PIPELINE,
                                   reinterpret_cast<uint64_t>(m_DDGIProbeVisPipeline), "DDGI Probe Vis Pipeline");
    }

    vkDestroyShaderModule(m_device, vertShader, nullptr);
    vkDestroyShaderModule(m_device, fragShader, nullptr);

    m_mainDeletionQueue.push_function([this] {
        vkDestroyPipeline(m_device, m_DDGIProbeVisPipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_DDGIProbeVisPipelineLayout, nullptr);
    });
}

void Engine::destroy_sync()
{
    for (const auto &frame : m_frameData)
    {
        vkDestroyFence(m_device, frame.fence, nullptr);
        vkDestroySemaphore(m_device, frame.swapchainSemaphore, nullptr);
    }

    for (const auto &renderSemaphore : m_swapchainSemaphores)
    {
        vkDestroySemaphore(m_device, renderSemaphore, nullptr);
    }
}

void Engine::destroy_commands()
{
    vkDestroyCommandPool(m_device, m_commandPool, nullptr);
}

void Engine::create_swapchain(const std::uint32_t width, const std::uint32_t height)
{
    m_swapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;

    auto vkbSwapchainResult =
        vkb::SwapchainBuilder(m_chosenGpu, m_device, m_surface)
            //.use_default_format_selection()
            .set_desired_format({.format = m_swapchainImageFormat, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
            .set_desired_present_mode(VK_PRESENT_MODE_IMMEDIATE_KHR)
            .set_desired_extent(width, height)
            .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
            .set_required_min_image_count(kNumberOfFrames)
            .build();

    if (!vkbSwapchainResult.has_value())
    {
        throw std::runtime_error("Failed to create swapchain");
    }

    m_swapchainExtent = vkbSwapchainResult.value().extent;
    m_swapchain = vkbSwapchainResult.value().swapchain;
    m_swapchainImages = vkbSwapchainResult.value().get_images().value();
    m_swapchainImageViews = vkbSwapchainResult.value().get_image_views().value();
    for (std::size_t i = 0; i < m_swapchainImages.size(); ++i)
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_IMAGE, reinterpret_cast<uint64_t>(m_swapchainImages[i]),
                                   std::format("Swapchain Image [{}]", i).c_str());
}

void Engine::destroy_swapchain()
{
    vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);

    for (const auto &imageView : m_swapchainImageViews)
    {
        vkDestroyImageView(m_device, imageView, nullptr);
    }
}

void Engine::resize_swapchain()
{
    vkDeviceWaitIdle(m_device) >> chk;

    destroy_swapchain();

    int w{0}, h{0};
    SDL_GetWindowSize(m_window.get(), &w, &h);
    m_windowExtent.width = w;
    m_windowExtent.height = h;
    create_swapchain(m_windowExtent.width, m_windowExtent.height);

    m_bSwapchainResizeRequest = false;
}

void Engine::write_frame_descriptors()
{
    for (std::uint32_t j = 0; j < kMaxDDGIVolumes; ++j)
    {
        ddgiRayDataDescBuffer.write_storage_image(0, j, rayDatas[j].imageView, VK_IMAGE_LAYOUT_GENERAL);
        ddgiResourcesDescBuffer.write_sampled_image(0, j, irradianceDatas[j].imageView,
                                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        ddgiResourcesDescBuffer.write_sampled_image(1, j, distanceDatas[j].imageView,
                                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        ddgiResourcesDescBuffer.write_sampled_image(2, j, probeDatas[j].imageView,
                                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        ddgiIrradianceStorageDescBuffer.write_storage_image(0, j, irradianceDatas[j].imageView,
                                                            VK_IMAGE_LAYOUT_GENERAL);
        ddgiDistanceStorageDescBuffer.write_storage_image(0, j, distanceDatas[j].imageView, VK_IMAGE_LAYOUT_GENERAL);
        ddgiProbeDataStorageDescBuffer.write_storage_image(0, j, probeDatas[j].imageView, VK_IMAGE_LAYOUT_GENERAL);
    }
    ddgiResourcesDescBuffer.write_sampler(3, 0, m_defaultSamplerLinear);
    for (auto &frame : m_frameData)
    {
        frame.drawImageDescriptorBuffer.write_storage_image(0, 0, frame.drawImage.imageView, VK_IMAGE_LAYOUT_GENERAL);

        frame.lightPassDescriptorBuffer.write_sampled_image(0, 0, frame.depthImage.imageView,
                                                            VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL);
        frame.lightPassDescriptorBuffer.write_sampled_image(1, 0, frame.gBuffer.normal.imageView,
                                                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        frame.lightPassDescriptorBuffer.write_sampled_image(2, 0, frame.gBuffer.diffuse.imageView,
                                                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        frame.lightPassDescriptorBuffer.write_sampled_image(3, 0, frame.gBuffer.specular.imageView,
                                                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        frame.lightPassDescriptorBuffer.write_sampled_image(4, 0, frame.directionalShadowPassDepthArray.imageView,
                                                            VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL);
        frame.lightPassDescriptorBuffer.write_sampler(5, 0, m_shadowSampler);
        frame.lightPassDescriptorBuffer.write_sampled_image(6, 0, frame.pointLightsShadowTileMap.imageView,
                                                            VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL);
        frame.lightPassDescriptorBuffer.write_sampled_image(7, 0, frame.gBuffer.emissive.imageView,
                                                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        frame.lightPassDescriptorBuffer.write_sampled_image(8, 0, frame.ddgiOutput.imageView,
                                                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        frame.wboitCompositePassDescBuffer.write_sampled_image(0, 0, frame.oitAccImage.imageView,
                                                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        frame.wboitCompositePassDescBuffer.write_sampled_image(1, 0, frame.oitRevealImage.imageView,
                                                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        frame.cascadeDepthDescBuffer.write_sampled_image(0, 0, frame.depthImage.imageView,
                                                         VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL);

        frame.ddgiTLASDescBuffer.write_acceleration_structure(0, 0, m_TlasAccel.address);
        frame.ddgiOutputStorageDescBuffer.write_storage_image(0, 0, frame.ddgiOutput.imageView,
                                                              VK_IMAGE_LAYOUT_GENERAL);
        frame.ddgiGBufferReadDescBuffer.write_sampled_image(0, 0, frame.depthImage.imageView,
                                                            VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL);
        frame.ddgiGBufferReadDescBuffer.write_sampled_image(1, 0, frame.gBuffer.normal.imageView,
                                                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        frame.ddgiGBufferReadDescBuffer.write_sampled_image(2, 0, frame.gBuffer.diffuse.imageView,
                                                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
}

} // namespace mp
