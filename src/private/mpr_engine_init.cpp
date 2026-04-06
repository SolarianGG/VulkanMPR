// clang-format off
#define GLM_ENABLE_EXPERIMENTAL
#include "mpr_engine.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <VkBootstrap.h>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>

#include <format>
#include <print>

#include "mpr_error_check.hpp"
#include "mpr_init_vk_stucts.hpp"
#include "mpr_loader.hpp"
#include "mpr_pipelines.hpp"
#include "mpr_debug_utils.hpp"
// clang-format on

namespace
{

#ifdef MPR_DEBUG
constexpr bool bUseValidationLayers = true;
#else
constexpr bool bUseValidationLayers = false;
#endif
constexpr auto kBaseWindowTitle = "Hello Vulkan";

#define GPU_USAGE_DISCRETE

std::pair<std::uint32_t, char const *const *> get_required_instance_extensions_for_window()
{
    std::uint32_t count;
    const auto requiredExtensions = SDL_Vulkan_GetInstanceExtensions(&count);
    return {count, requiredExtensions};
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
        .shaderClipDistance = true,
        .shaderInt64 = true,
        .shaderInt16 = true,
    };

    vkb::PhysicalDeviceSelector selector{result.value()};

    std::vector<const char *> requiredExtensions{
        VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
        VK_NV_SHADER_SUBGROUP_PARTITIONED_EXTENSION_NAME, // VK_EXT_SHADER_SUBGROUP_PARTITIONED_EXTENSION_NAME
    };
    const auto physicalDevice = selector.set_minimum_version(1, 3)
                                    .add_required_extensions(requiredExtensions)
                                    .set_required_features_13(features13)
                                    .add_required_extension_features(descriptorBufferFeatures)
                                    .add_required_extension_features(atomicFloatFeatures)
                                    .set_required_features_12(features12)
                                    .set_required_features_11(features11)
                                    .set_required_features(features10)
                                    .set_surface(m_surface)
#ifdef GPU_USAGE_DISCRETE
                                    .allow_any_gpu_device_type(false)
                                    .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
#endif
                                    .select();

    vkb::DeviceBuilder deviceBuilder{physicalDevice.value()};

    vkb::Device vkbDevice = deviceBuilder.build().value();

    m_device = vkbDevice.device;
    m_chosenGpu = vkbDevice.physical_device;
    std::println("Physical GPU: {}", vkbDevice.physical_device.name);

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
    m_CommonImageExtent3D = {
        .width = m_windowExtent.width,
        .height = m_windowExtent.height,
        .depth = 1,
    };

    m_CommonImageExtent2D = {
        .width = m_windowExtent.width,
        .height = m_windowExtent.height,
    };
    m_drawExtent = m_windowExtent;
    for (std::size_t i = 0; i < m_frameData.size(); ++i)
    {
        auto &frame = m_frameData[i];
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

        frame.oitAccImage = create_image(m_CommonImageExtent3D, VK_FORMAT_R16G16B16A16_SFLOAT,
                                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_IMAGE, reinterpret_cast<uint64_t>(frame.oitAccImage.image),
                                   std::format("OIT Accumulation [{}]", i).c_str());
        frame.oitRevealImage = create_image(m_CommonImageExtent3D, VK_FORMAT_R16_SFLOAT,
                                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_IMAGE,
                                   reinterpret_cast<uint64_t>(frame.oitRevealImage.image),
                                   std::format("OIT Reveal [{}]", i).c_str());
        // Create layered shadow depth image for CSM
        {
            const VkImageCreateInfo shadowImgInfo{
                .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .imageType = VK_IMAGE_TYPE_2D,
                .format = VK_FORMAT_D32_SFLOAT,
                .extent = {kDirectionalShadowMapSize, kDirectionalShadowMapSize, 1},
                .mipLevels = 1,
                .arrayLayers = static_cast<std::uint32_t>(MAX_CASCADES),
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            };
            constexpr VmaAllocationCreateInfo shadowAllocInfo{
                .usage = VMA_MEMORY_USAGE_GPU_ONLY,
                .requiredFlags = static_cast<VkMemoryPropertyFlags>(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
            };
            vmaCreateImage(m_allocator, &shadowImgInfo, &shadowAllocInfo, &frame.directionalShadowPassDepthArray.image,
                           &frame.directionalShadowPassDepthArray.allocation, nullptr) >>
                chk;
            frame.directionalShadowPassDepthArray.imageFormat = VK_FORMAT_D32_SFLOAT;
            frame.directionalShadowPassDepthArray.imageExtent = {kDirectionalShadowMapSize, kDirectionalShadowMapSize,
                                                                 1};

            // Array image view for sampling in light pass
            const VkImageViewCreateInfo arrayViewInfo{
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = frame.directionalShadowPassDepthArray.image,
                .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
                .format = VK_FORMAT_D32_SFLOAT,
                .subresourceRange =
                    {
                        .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = static_cast<std::uint32_t>(MAX_CASCADES),
                    },
            };
            vkCreateImageView(m_device, &arrayViewInfo, nullptr, &frame.directionalShadowPassDepthArray.imageView) >>
                chk;
            mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_IMAGE,
                                       reinterpret_cast<uint64_t>(frame.directionalShadowPassDepthArray.image),
                                       std::format("Directional Shadow Array [{}]", i).c_str());
        }
        frame.pointLightsShadowTileMap =
            create_image({kPointLightsShadowMapSize, kPointLightsShadowMapSize, 1}, VK_FORMAT_D16_UNORM,
                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_IMAGE,
                                   reinterpret_cast<uint64_t>(frame.pointLightsShadowTileMap.image),
                                   std::format("Point Light Shadow Map [{}]", i).c_str());
    }

    m_mainDeletionQueue.push_function([this]() {
        for (auto &frame : m_frameData)
        {
            destroy_image(frame.gBuffer.normal);
            destroy_image(frame.gBuffer.diffuse);
            destroy_image(frame.gBuffer.specular);

            destroy_image(frame.oitAccImage);
            destroy_image(frame.oitRevealImage);
            destroy_image(frame.pointLightsShadowTileMap);

            vkDestroyImageView(m_device, frame.directionalShadowPassDepthArray.imageView, nullptr);
            vmaDestroyImage(m_allocator, frame.directionalShadowPassDepthArray.image,
                            frame.directionalShadowPassDepthArray.allocation);
        }
    });
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
        m_mainDeletionQueue.push_function([&] { vkDestroySampler(m_device, m_shadowSampler, nullptr); });
    }
    {
        const VkSamplerCreateInfo debugSamplerInfo{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_LINEAR,
            .minFilter = VK_FILTER_LINEAR,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        };
        vkCreateSampler(m_device, &debugSamplerInfo, nullptr, &m_debugSampler);
        mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_SAMPLER, reinterpret_cast<uint64_t>(m_debugSampler),
                                   "Debug Sampler");
        m_mainDeletionQueue.push_function([&] { vkDestroySampler(m_device, m_debugSampler, nullptr); });
    }
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
                .build(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT);
    }
    for (auto &frame : m_frameData)
    {
        frame.lightPassDescriptorBuffer =
            DescriptorBuffer(m_device, m_LightPassDescriptorSetLayout, DescriptorBufferProperties::query(m_chosenGpu));

        frame.lightPassDescriptorBuffer.create_buffer(
            [&](const std::size_t allocSize, const VkBufferUsageFlags bufferUsage) {
                return create_buffer(allocSize, bufferUsage, VMA_MEMORY_USAGE_CPU_ONLY);
            });

        frame.drawImageDescriptorBuffer =
            DescriptorBuffer(m_device, m_DrawImageDescriptorSetLayout, DescriptorBufferProperties::query(m_chosenGpu));

        frame.drawImageDescriptorBuffer.create_buffer(
            [&](const std::size_t allocSize, const VkBufferUsageFlags bufferUsage) {
                return create_buffer(allocSize, bufferUsage, VMA_MEMORY_USAGE_CPU_ONLY);
            });

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
    }

    m_mainDeletionQueue.push_function([&]() mutable {
        vkDestroyDescriptorSetLayout(m_device, m_LightPassDescriptorSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(m_device, m_DrawImageDescriptorSetLayout, nullptr);
        for (auto &frame : m_frameData)
        {
            destroy_buffer(frame.lightPassDescriptorBuffer.get_buffer());
            destroy_buffer(frame.drawImageDescriptorBuffer.get_buffer());
        }
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
    init_generate_point_light_commands_pipeline();
    init_directional_shadow_pass();
    init_point_shadow_pass();
    init_depth_reduction_pass();
    m_metalRoughness.build_pipelines(*this);
    init_prepass();
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

    {
        m_WboitCompositePassDescriptorSetLayout =
            DescriptorSetLayoutBuilder()
                .add_binding(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT)
                .add_binding(1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT)
                .build(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT);
    }
    for (auto &frame : m_frameData)
    {
        frame.wboitCompositePassDescBuffer = DescriptorBuffer(m_device, m_WboitCompositePassDescriptorSetLayout,
                                                              DescriptorBufferProperties::query(m_chosenGpu));
        frame.wboitCompositePassDescBuffer.create_buffer(
            [this](const std::size_t allocSize, const VkBufferUsageFlags bufferUsage) {
                return create_buffer(allocSize, bufferUsage, VMA_MEMORY_USAGE_CPU_ONLY);
            });

        frame.wboitCompositePassDescBuffer.write_sampled_image(0, 0, frame.oitAccImage.imageView,
                                                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        frame.wboitCompositePassDescBuffer.write_sampled_image(1, 0, frame.oitRevealImage.imageView,
                                                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
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
        pipelineBuilder.add_color_attachment_format(m_frameData.at(0).drawImage.imageFormat);
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

        vkDestroyDescriptorSetLayout(m_device, m_WboitCompositePassDescriptorSetLayout, nullptr);

        for (auto &frame : m_frameData)
        {
            destroy_buffer(frame.wboitCompositePassDescBuffer.get_buffer());
        }
    });
}

void Engine::init_depth_reduction_pass()
{
    // Shared descriptor set layout: binding 0 = depth sampled image
    m_DepthPassDescSetLayout = DescriptorSetLayoutBuilder()
                                   .add_binding(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT)
                                   .build(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT);

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
        vkDestroyDescriptorSetLayout(m_device, m_DepthPassDescSetLayout, nullptr);
    });
}

void Engine::init_directional_shadow_pass()
{
    const VkPushConstantRange pushConstantRange{.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
                                                .offset = 0,
                                                .size = sizeof(DirectionalShadowPassPushConstants)};

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
    builder.set_depth_format(m_frameData.at(0).directionalShadowPassDepthArray.imageFormat);
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
    builder.set_depth_format(m_frameData.at(0).pointLightsShadowTileMap.imageFormat);
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
    builder.set_depth_format(m_frameData.at(0).depthImage.imageFormat);
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
    const VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .setLayoutCount = 1,
        .pSetLayouts = &m_DrawImageDescriptorSetLayout,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = nullptr,
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
#if 0
  m_CullPassDescriptorSetLayout =
      DescriptorSetLayoutBuilder()
          .build(m_device,
                 VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT);
#endif

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
    for (std::size_t i = 0; i < m_frameData.size(); ++i)
    {
        auto &frame = m_frameData[i];
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
            create_buffer(sizeof(std::uint32_t) * kMaxPointLights,
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

        frame.cascadeDepthDescBuffer =
            DescriptorBuffer(m_device, m_DepthPassDescSetLayout, DescriptorBufferProperties::query(m_chosenGpu));
        frame.cascadeDepthDescBuffer.create_buffer(
            [this](const std::size_t allocSize, const VkBufferUsageFlags bufferUsage) {
                return create_buffer(allocSize, bufferUsage, VMA_MEMORY_USAGE_CPU_ONLY);
            });
        frame.cascadeDepthDescBuffer.write_sampled_image(0, 0, frame.depthImage.imageView,
                                                         VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL);

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
    }

    m_mainDeletionQueue.push_function([this] {
        for (auto &frame : m_frameData)
        {
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
            destroy_buffer(frame.cascadeDepthDescBuffer.get_buffer());
        }
    });
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
    std::uint32_t greyColor = glm::packUnorm4x8(glm::vec4(0.5f, 0.5f, 0.5f, 1.0f));
    m_greyImage = create_image(&greyColor, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);
    mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_IMAGE, reinterpret_cast<uint64_t>(m_greyImage.image),
                               "Default Grey Image");
    std::uint32_t normalFallback = glm::packUnorm4x8(glm::vec4(0.5f, 0.5f, 1.0f, 1.0f));
    m_normalFallback =
        create_image(&normalFallback, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);
    mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_IMAGE, reinterpret_cast<uint64_t>(m_normalFallback.image),
                               "Default Normal Fallback Image");

    const std::uint32_t magentaColor = glm::packUnorm4x8(glm::vec4(1.0f, 0.0f, 1.0f, 1.0f));
    std::array<std::uint32_t, 16 * 16> errorPixels;
    for (int i = 0; i < 16; ++i)
    {
        for (int j = 0; j < 16; ++j)
        {
            errorPixels[i * 16 + j] = ((i % 2) ^ (j % 2)) ? magentaColor : blackColor;
        }
    }
    m_errorImage =
        create_image(errorPixels.data(), VkExtent3D{16, 16, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);
    mp::debug::set_object_name(m_device, VK_OBJECT_TYPE_IMAGE, reinterpret_cast<uint64_t>(m_errorImage.image),
                               "Default Error Image");

    VkSamplerCreateInfo samplerCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
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
        destroy_image(m_greyImage);
        destroy_image(m_errorImage);
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

#if 1
    const std::string sponzaPath = "../../assets/gltf-samples/Models/Sponza/glTF/sponza.gltf";
    if (!load_gltf(*this, sponzaPath))
    {
        throw std::runtime_error("Failed to load glTF file: " + sponzaPath);
    }
#endif

#if 0
    const std::string bistroPath = "../../assets/bistro_exterior.glb";
    if (!load_gltf(*this, bistroPath))
    {
        throw std::runtime_error("Failed to load glTF file: " + bistroPath);
    }
#endif
#if 0
  const std::string alphaBlendMode =
      "../../assets/gltf-samples/Models/AlphaBlendModeTest/glTF/"
      "AlphaBlendModeTest.gltf";
  if (!load_gltf(*this, alphaBlendMode)) {
    throw std::runtime_error("Failed to load glTF file: " + alphaBlendMode);
  }
#endif

    m_mainDeletionQueue.push_function([this] {
        destroy_buffer(m_globalPositionBuffer);
        destroy_buffer(m_globalAttributesBuffer);
        destroy_buffer(m_globalIndexBuffer);
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

} // namespace mp
