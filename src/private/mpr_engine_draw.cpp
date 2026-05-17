// clang-format off
#include "mpr_engine.hpp"

#include <imgui.h>
#include <imgui_impl_vulkan.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <ranges>

#include "mpr_error_check.hpp"
#include "mpr_image.hpp"
#include "mpr_init_vk_stucts.hpp"
#include "mpr_debug_utils.hpp"
// clang-format on

namespace cn = std::chrono;
namespace rn = std::ranges;

namespace mp
{

void Engine::draw()
{
    FrameData &currentFrame = get_current_frame();
    // Wait if command buffer is in execution on the gpu
    vkWaitForFences(m_device, 1, &currentFrame.fence, true, std::numeric_limits<std::uint64_t>::max()) >> chk;
    currentFrame.frameDeletionQueue.flush();
    update_scene();

    // Get the current image from the swapchain

    std::uint32_t swapchainImageIndex;
    const VkResult swapchainAcquireRes =
        vkAcquireNextImageKHR(m_device, m_swapchain, std::numeric_limits<std::uint64_t>::max(),
                              currentFrame.swapchainSemaphore, nullptr, &swapchainImageIndex);
    if (swapchainAcquireRes == VK_ERROR_OUT_OF_DATE_KHR)
    {
        m_bSwapchainResizeRequest = true;
        return;
    }
    if (swapchainAcquireRes == VK_SUBOPTIMAL_KHR)
    {
        m_bSwapchainResizeRequest = true;
    }
    else
    {
        swapchainAcquireRes >> chk;
    }
    VkSemaphore &signalSemaphore = m_swapchainSemaphores[swapchainImageIndex];

    // Reset command buffer
    VkCommandBuffer &cmd = currentFrame.commandBuffer;
    VkImage &swapchainImage = m_swapchainImages[swapchainImageIndex];
    VkImageView &swapchainImageView = m_swapchainImageViews[swapchainImageIndex];

    constexpr VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };

    AllocatedImage &currentDrawingImage = currentFrame.drawImage;
    AllocatedImage &currentDepthImage = currentFrame.depthImage;
    auto &gBuffer = currentFrame.gBuffer;
    m_drawExtent.width = std::min(currentDrawingImage.imageExtent.width, m_swapchainExtent.width);
    m_drawExtent.height = std::min(currentDrawingImage.imageExtent.height, m_swapchainExtent.height);

    vkBeginCommandBuffer(cmd, &beginInfo) >> chk;

    copy_staging_buffers(cmd);

    utils::BarrierBuilder barrierBuilder;
    m_stats.drawCallCount = 0;
    m_stats.triangleCount = 0;

    vkCmdBindIndexBuffer(cmd, m_globalIndexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

    compute_cull_point_lights(cmd);
    // Draw prepass
    {

        barrierBuilder.add_image_barrier(currentDepthImage.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
             .accessMask =
                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
             .layout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_DEPTH_BIT)}));
        barrierBuilder.barrier(cmd);
    }
    draw_prepass(cmd);

    draw_point_lights_shadows_pass(cmd);

    // Compute minZ/maxZ from prepass depth
    {
        barrierBuilder.add_image_barrier(currentDepthImage.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
             .layout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_DEPTH_BIT)}));
        barrierBuilder.barrier(cmd);
    }
    compute_depth_reduction(cmd);

    // Compute tight partitions
    {
    }
    compute_depth_partition(cmd);

    // Compute every light's vp
    {
    }
    compute_dir_lights_vp(cmd);

    // Shadow pass

    {
        barrierBuilder.add_image_barrier(currentFrame.directionalShadowPassDepthArray.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
             .accessMask =
                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
             .layout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_DEPTH_BIT)}));
        barrierBuilder.add_buffer_barrier(currentFrame.dirLightBuffer.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT,
             .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
        barrierBuilder.barrier(cmd);
    }
    draw_directional_shadow_pass(cmd);

    // Base pass (GPass)
    {
        barrierBuilder.add_image_barrier(gBuffer.normal.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
             .accessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
             .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
        barrierBuilder.add_image_barrier(gBuffer.diffuse.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
             .accessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
             .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
        barrierBuilder.add_image_barrier(gBuffer.specular.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
             .accessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
             .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
        barrierBuilder.add_image_barrier(gBuffer.emissive.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
             .accessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
             .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
        // Transition depth back from DEPTH_READ_ONLY (compute) to DEPTH_ATTACHMENT (GBuffer)
        barrierBuilder.add_image_barrier(currentDepthImage.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
             .accessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
             .layout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_DEPTH_BIT)}));
        barrierBuilder.barrier(cmd);
    }

    draw_gBuffer_pass(cmd);

    if (m_DDGIVolumeCount > 0)
    {

        // DDGI Ray Generation
        {
            for (std::uint32_t i = 0; i < m_DDGIVolumeCount; ++i)
            {
                barrierBuilder.add_image_barrier(rayDatas[i].transition(
                    {.stageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                     .accessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                     .layout = VK_IMAGE_LAYOUT_GENERAL,
                     .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                     .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
                barrierBuilder.add_image_barrier(irradianceDatas[i].transition(
                    {.stageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                     .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
                     .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                     .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
                barrierBuilder.add_image_barrier(distanceDatas[i].transition(
                    {.stageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                     .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
                     .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                     .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
                barrierBuilder.add_image_barrier(probeDatas[i].transition(
                    {.stageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                     .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
                     .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                     .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
            }
            if (m_mainDrawContext.dirLight.has_value())
                barrierBuilder.add_buffer_barrier(
                    currentFrame.dirLightBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                                                            .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
                                                            .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
            if (!m_mainDrawContext.pointLights.empty())
                barrierBuilder.add_buffer_barrier(currentFrame.pointLightBuffer.transition(
                    {.stageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                     .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
                     .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
            barrierBuilder.barrier(cmd);
        }
        trace_ddgi_probe_pass(cmd);

        // DDGI Probe Blending
        {
            for (std::uint32_t i = 0; i < m_DDGIVolumeCount; ++i)
            {
                barrierBuilder.add_image_barrier(rayDatas[i].transition(
                    {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
                     .layout = VK_IMAGE_LAYOUT_GENERAL,
                     .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                     .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
                barrierBuilder.add_image_barrier(irradianceDatas[i].transition(
                    {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     .accessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                     .layout = VK_IMAGE_LAYOUT_GENERAL,
                     .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                     .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
                barrierBuilder.add_image_barrier(distanceDatas[i].transition(
                    {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     .accessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                     .layout = VK_IMAGE_LAYOUT_GENERAL,
                     .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                     .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
            }
            barrierBuilder.barrier(cmd);
        }
        compute_ddgi_irradiance_blending(cmd);
        compute_ddgi_distance_blending(cmd);

        // DDGI Relocation

        {
            for (std::uint32_t i = 0; i < m_DDGIVolumeCount; ++i)
            {
                if (m_mainDrawContext.ddgiVolumes[i].probeRelocationEnabled == 1)
                {
                    barrierBuilder.add_image_barrier(probeDatas[i].transition(
                        {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         .accessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                         .layout = VK_IMAGE_LAYOUT_GENERAL,
                         .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                         .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
                }
                else
                {
                    barrierBuilder.add_image_barrier(probeDatas[i].transition(
                        {.stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         .accessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         .layout = VK_IMAGE_LAYOUT_GENERAL,
                         .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                         .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
                }
            }
            barrierBuilder.barrier(cmd);
        }
        compute_ddgi_relocation(cmd);

        // DDGI Indirect
        {
            for (std::uint32_t i = 0; i < m_DDGIVolumeCount; ++i)
            {
                barrierBuilder.add_image_barrier(irradianceDatas[i].transition(
                    {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
                     .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                     .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
                barrierBuilder.add_image_barrier(distanceDatas[i].transition(
                    {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
                     .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                     .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
                barrierBuilder.add_image_barrier(probeDatas[i].transition(
                    {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
                     .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                     .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
            }
            barrierBuilder.add_image_barrier(currentFrame.ddgiOutput.transition(
                {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 .accessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                 .layout = VK_IMAGE_LAYOUT_GENERAL,
                 .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                 .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
            barrierBuilder.add_image_barrier(gBuffer.normal.transition(
                {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
                 .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                 .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
            barrierBuilder.add_image_barrier(gBuffer.diffuse.transition(
                {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
                 .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                 .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
            barrierBuilder.add_image_barrier(currentDepthImage.transition(
                {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
                 .layout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
                 .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                 .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_DEPTH_BIT)}));
            barrierBuilder.barrier(cmd);
        }
        compute_ddgi_indirect(cmd);
    }

    // Light pass
    {
        barrierBuilder.add_image_barrier(currentFrame.ddgiOutput.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
             .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
        barrierBuilder.add_image_barrier(gBuffer.specular.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
             .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
        barrierBuilder.add_image_barrier(gBuffer.emissive.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
             .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
        barrierBuilder.add_image_barrier(currentDrawingImage.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             .accessMask = VK_ACCESS_2_SHADER_WRITE_BIT_KHR,
             .layout = VK_IMAGE_LAYOUT_GENERAL,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
        barrierBuilder.add_image_barrier(currentFrame.directionalShadowPassDepthArray.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
             .layout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_DEPTH_BIT)}));
        barrierBuilder.add_image_barrier(currentFrame.pointLightsShadowTileMap.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
             .layout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_DEPTH_BIT)}));
        barrierBuilder.add_buffer_barrier(
            currentFrame.dirLightBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                    .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
                                                    .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));

        barrierBuilder.barrier(cmd);
    }

    draw_light_pass(cmd);

    // forward WBOIT pass

    {
        barrierBuilder.add_image_barrier(currentFrame.oitAccImage.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
             .accessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
             .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
        barrierBuilder.add_image_barrier(currentFrame.oitRevealImage.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
             .accessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
             .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
        barrierBuilder.add_image_barrier(currentDepthImage.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
             .accessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
             .layout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_DEPTH_BIT)}));
        barrierBuilder.barrier(cmd);
    }

    draw_wboit(cmd);

    // composite weight blended OIT
    {
        barrierBuilder.add_image_barrier(currentFrame.oitAccImage.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
             .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
             .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
        barrierBuilder.add_image_barrier(currentFrame.oitRevealImage.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
             .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
             .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
        barrierBuilder.add_image_barrier(currentDrawingImage.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
             .accessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
             .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
        barrierBuilder.barrier(cmd);
    }

    draw_wboit_composite(cmd);

    // DDGI probe visualization
    if (m_mainDrawContext.ddgiVolumesVis.size() > 0)
    {
        barrierBuilder.add_image_barrier(currentDrawingImage.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
             .accessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
             .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
        barrierBuilder.add_image_barrier(currentDepthImage.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
             .accessMask =
                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
             .layout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_DEPTH_BIT)}));
        barrierBuilder.barrier(cmd);

        draw_ddgi_probe_vis(cmd);
    }

    // Postprocess (gamma + tone)
    {
        barrierBuilder.add_image_barrier(currentDrawingImage.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             .accessMask = VK_ACCESS_2_SHADER_WRITE_BIT_KHR | VK_ACCESS_2_SHADER_READ_BIT,
             .layout = VK_IMAGE_LAYOUT_GENERAL,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
        barrierBuilder.barrier(cmd);
    }

    // Auto-exposure: reset histogram
    {
        barrierBuilder.add_buffer_barrier(
            currentFrame.histogramBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                     .accessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                                     .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
        barrierBuilder.barrier(cmd);
    }
    vkCmdFillBuffer(cmd, currentFrame.histogramBuffer.buffer, 0, VK_WHOLE_SIZE, 0);

    {
        barrierBuilder.add_buffer_barrier(currentFrame.histogramBuffer.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             .accessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
        barrierBuilder.barrier(cmd);
    }

    compute_luminance_histogram(cmd);

    {
        barrierBuilder.add_buffer_barrier(
            currentFrame.histogramBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                     .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
                                                     .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
        barrierBuilder.add_buffer_barrier(
            m_avgLuminanceBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                             .accessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                                             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
        barrierBuilder.barrier(cmd);
    }

    compute_average_luminance(cmd);

    {
        barrierBuilder.add_buffer_barrier(
            m_avgLuminanceBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                             .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
                                             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
        barrierBuilder.barrier(cmd);
    }

    compute_post(cmd);

    // copy to swapchain
    {
        barrierBuilder.add_image_barrier(currentDrawingImage.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
             .accessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
             .layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));

        barrierBuilder.add_image_barrier(
            VkImageMemoryBarrier2{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                  .srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                                  .srcAccessMask = 0,
                                  .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                  .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                  .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                  .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                  .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                  .image = swapchainImage,
                                  .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)});
        barrierBuilder.barrier(cmd);
    }

    utils::copy_to_image(cmd, currentDrawingImage.image, swapchainImage, m_drawExtent, m_swapchainExtent);

    // Imgui
    {
        barrierBuilder.add_image_barrier(VkImageMemoryBarrier2{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = swapchainImage,
            .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)});
        barrierBuilder.barrier(cmd);
    }

    draw_imgui(cmd, swapchainImageView);

    {
        barrierBuilder.add_image_barrier(VkImageMemoryBarrier2{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
            .dstAccessMask = 0,
            .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = swapchainImage,
            .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)});
        barrierBuilder.barrier(cmd);
    }

    vkEndCommandBuffer(currentFrame.commandBuffer) >> chk;

    const auto waitSemaphoreInfo =
        utils::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, currentFrame.swapchainSemaphore);

    const auto signalSemaphoreInfo =
        utils::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, signalSemaphore);

    const auto cmdInfo = utils::command_buffer_submit_info(cmd);

    const auto renderSubmitInfo = utils::submit_info(&cmdInfo, &waitSemaphoreInfo, &signalSemaphoreInfo);
    vkResetFences(m_device, 1, &currentFrame.fence) >> chk;
    vkQueueSubmit2(m_queue, 1, &renderSubmitInfo, currentFrame.fence) >> chk;

    const VkPresentInfoKHR presentInfo{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &signalSemaphore,
        .swapchainCount = 1,
        .pSwapchains = &m_swapchain,
        .pImageIndices = &swapchainImageIndex,
        .pResults = nullptr,
    };
    const VkResult swapchainPresentResult = vkQueuePresentKHR(m_queue, &presentInfo);
    if (swapchainPresentResult == VK_ERROR_OUT_OF_DATE_KHR || swapchainPresentResult == VK_SUBOPTIMAL_KHR)
    {
        m_bSwapchainResizeRequest = true;
    }
    else
    {
        swapchainPresentResult >> chk;
    }
    ++m_frameNumber;
}

void Engine::draw_directional_shadow_pass(VkCommandBuffer cmd)
{
    if (!m_mainDrawContext.dirLight.has_value())
        return;
    mp::debug::cmd_begin_label(cmd, "Directional Shadow Pass", {0.9f, 0.2f, 0.2f, 1.f});
    const auto &light = m_mainDrawContext.dirLight.value();
    const auto start = cn::steady_clock::now();
    const VkExtent2D shadowPassExtent{kDirectionalShadowMapSize, kDirectionalShadowMapSize};
    auto &currentFrame = get_current_frame();

    const std::uint32_t cascadeCount = static_cast<std::uint32_t>(std::clamp(light.cascadeCount, 1, MAX_CASCADES));

    compute_cull_objects(cmd, m_OpaqueSize, 0, m_DirLightCullMatrix);

    {
        utils::BarrierBuilder barrierBuilder;
        barrierBuilder.add_buffer_barrier(currentFrame.countBuffer.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             .accessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
        barrierBuilder.add_buffer_barrier(currentFrame.drawCommandsBuffer.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             .accessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
        barrierBuilder.barrier(cmd);
    }

    compute_populate_commands_with_cascade_count(cmd, m_OpaqueSize);

    {
        utils::BarrierBuilder barrierBuilder;
        barrierBuilder.add_buffer_barrier(
            currentFrame.countBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                                                 .accessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                                                 .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
        barrierBuilder.add_buffer_barrier(
            currentFrame.drawCommandsBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                                                        .accessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                                                        .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
        barrierBuilder.barrier(cmd);
    }

    const auto depthAttachment = utils::depth_attachment(currentFrame.directionalShadowPassDepthArray.imageView,
                                                         VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    const VkRenderingInfo renderInfo{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .pNext = nullptr,
        .renderArea = {.extent = shadowPassExtent},
        .layerCount = cascadeCount,
        .colorAttachmentCount = 0,
        .pColorAttachments = nullptr,
        .pDepthAttachment = &depthAttachment,
        .pStencilAttachment = nullptr,
    };

    vkCmdBeginRendering(cmd, &renderInfo);

    const VkViewport viewport{
        .x = 0,
        .y = static_cast<float>(shadowPassExtent.height),
        .width = static_cast<float>(shadowPassExtent.width),
        .height = -static_cast<float>(shadowPassExtent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    const VkRect2D scissor{.extent = shadowPassExtent};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    const DirectionalShadowPassPushConstants shadowPassPushConstants{
        .positionBufferAddr = m_globalPositionBufferAddress,
        .instanceBufferDeviceAddr = currentFrame.instanceBufferAddr,
        .dirLightsBufferAddr = currentFrame.dirLightBufferAddr,
        .cascadeCount = cascadeCount,
    };
    draw_meshes(cmd, m_ShadowPassPipelineLayout, m_ShadowPassPipeline, m_OpaqueSize, shadowPassPushConstants,
                VK_SHADER_STAGE_VERTEX_BIT);

    vkCmdEndRendering(cmd);

    if (m_AlphaTestedSize > 0)
    {
        compute_cull_objects(cmd, m_AlphaTestedSize, m_OpaqueSize, m_DirLightCullMatrix);

        {
            utils::BarrierBuilder barrierBuilder;
            barrierBuilder.add_buffer_barrier(currentFrame.countBuffer.transition(
                {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 .accessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
                 .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
            barrierBuilder.add_buffer_barrier(currentFrame.drawCommandsBuffer.transition(
                {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 .accessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
                 .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
            barrierBuilder.barrier(cmd);
        }

        compute_populate_commands_with_cascade_count(cmd, m_AlphaTestedSize);

        {
            utils::BarrierBuilder barrierBuilder;
            barrierBuilder.add_buffer_barrier(
                currentFrame.countBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                                                     .accessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                                                     .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
            barrierBuilder.add_buffer_barrier(
                currentFrame.drawCommandsBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                                                            .accessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                                                            .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
            barrierBuilder.barrier(cmd);
        }

        // LOAD depth — opaque shadow already wrote to shadow array
        const auto atDepthAttachment = utils::depth_attachment(currentFrame.directionalShadowPassDepthArray.imageView,
                                                               VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, false);
        const VkRenderingInfo atRenderInfo{
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .pNext = nullptr,
            .renderArea = {.extent = shadowPassExtent},
            .layerCount = cascadeCount,
            .colorAttachmentCount = 0,
            .pColorAttachments = nullptr,
            .pDepthAttachment = &atDepthAttachment,
            .pStencilAttachment = nullptr,
        };
        vkCmdBeginRendering(cmd, &atRenderInfo);
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        const VkDescriptorBufferBindingInfoEXT atBindingInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
            .address = m_metalRoughness.descriptors.get_device_address(),
            .usage =
                VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT};
        vkCmdBindDescriptorBuffersEXT(cmd, 1, &atBindingInfo);

        const std::uint32_t atBufferIndices[]{0};
        const VkDeviceSize atOffsets[]{0};
        vkCmdSetDescriptorBufferOffsetsEXT(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_AlphaTestedShadowPassPipelineLayout,
                                           0, 1, atBufferIndices, atOffsets);

        const DirectionalShadowPassAlphaTestedPushConstants atPushConstants{
            .positionBufferAddr = m_globalPositionBufferAddress,
            .attributesBufferAddr = m_GBufferMeshPushConstants.attributesBufferAddr,
            .instanceBufferDeviceAddr = currentFrame.instanceBufferAddr,
            .dirLightsBufferAddr = currentFrame.dirLightBufferAddr,
            .cascadeCount = cascadeCount,
        };
        draw_meshes(cmd, m_AlphaTestedShadowPassPipelineLayout, m_AlphaTestedShadowPassPipeline, m_AlphaTestedSize,
                    atPushConstants, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

        vkCmdEndRendering(cmd);
    }

    mp::debug::cmd_end_label(cmd);
    const auto end = cn::steady_clock::now();
    const auto elapsed = cn::duration_cast<cn::milliseconds>(end - start);
    m_stats.shadowPassDrawTime = elapsed.count() / 1000.0f;
}

void Engine::draw_gBuffer_pass(VkCommandBuffer cmd)
{
    mp::debug::cmd_begin_label(cmd, "GBuffer Pass", {0.6f, 0.3f, 0.9f, 1.f});
    const auto start = cn::steady_clock::now();
    auto &currentFrame = get_current_frame();
    auto &gBuffer = currentFrame.gBuffer;
    auto &depthImage = currentFrame.depthImage;

    // ---
    VkClearValue val{.color = {0.0f, 0.0f, 0.0f, 1.0f}};
    const auto normalAttachment =
        utils::attachment_info(gBuffer.normal.imageView, &val, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    const auto diffuseAttachment =
        utils::attachment_info(gBuffer.diffuse.imageView, &val, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    const auto specularAttachment =
        utils::attachment_info(gBuffer.specular.imageView, &val, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    const auto emissiveAttachment =
        utils::attachment_info(gBuffer.emissive.imageView, &val, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    const auto depthAttachment =
        utils::depth_attachment(depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, false);

    VkRenderingAttachmentInfo attachments[]{normalAttachment, diffuseAttachment, specularAttachment,
                                            emissiveAttachment};
    const auto renderInfo =
        utils::rendering_info(m_CommonImageExtent2D, std::size(attachments), attachments, &depthAttachment);

    compute_cull_objects(cmd, m_OpaqueSize, 0, m_sceneData.projView);
    {
        utils::BarrierBuilder barrierBuilder;
        barrierBuilder.add_buffer_barrier(
            currentFrame.countBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                                                 .accessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                                                 .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
        barrierBuilder.add_buffer_barrier(
            currentFrame.drawCommandsBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                                                        .accessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                                                        .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
        barrierBuilder.barrier(cmd);
    }
    vkCmdBeginRendering(cmd, &renderInfo);
    const VkViewport viewport{
        .x = 0,
        .y = static_cast<float>(m_CommonImageExtent2D.height),
        .width = static_cast<float>(m_CommonImageExtent2D.width),
        .height = -static_cast<float>(m_CommonImageExtent2D.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    const VkRect2D scissor{
        .extent = m_CommonImageExtent2D,
    };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    const VkDescriptorBufferBindingInfoEXT gBufferBindingInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
        .address = m_metalRoughness.descriptors.get_device_address(),
        .usage =
            VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT};
    vkCmdBindDescriptorBuffersEXT(cmd, 1, &gBufferBindingInfo);

    const std::uint32_t gBufferBufferIndices[]{0};
    const VkDeviceSize gBufferOffsets[]{0};
    vkCmdSetDescriptorBufferOffsetsEXT(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                       m_metalRoughness.opaquePipeline.pipelineLayout, 0, 1, gBufferBufferIndices,
                                       gBufferOffsets);

    draw_meshes(cmd, m_metalRoughness.opaquePipeline.pipelineLayout, m_metalRoughness.opaquePipeline.pipeline,
                m_OpaqueSize, m_GBufferMeshPushConstants, VK_SHADER_STAGE_VERTEX_BIT);

    vkCmdEndRendering(cmd);

    if (m_AlphaTestedSize > 0)
    {
        compute_cull_objects(cmd, m_AlphaTestedSize, m_OpaqueSize, m_sceneData.projView);

        {
            utils::BarrierBuilder barrierBuilder;
            barrierBuilder.add_buffer_barrier(
                currentFrame.countBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                                                     .accessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                                                     .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
            barrierBuilder.add_buffer_barrier(
                currentFrame.drawCommandsBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                                                            .accessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                                                            .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
            barrierBuilder.barrier(cmd);
        }

        // LOAD all attachments — opaque GBuffer already wrote them
        const auto normalAT =
            utils::attachment_info(gBuffer.normal.imageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        const auto diffuseAT =
            utils::attachment_info(gBuffer.diffuse.imageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        const auto specularAT =
            utils::attachment_info(gBuffer.specular.imageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        const auto emissiveAT =
            utils::attachment_info(gBuffer.emissive.imageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        const auto depthAT =
            utils::depth_attachment(depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, false);
        VkRenderingAttachmentInfo atColorAttachments[]{normalAT, diffuseAT, specularAT, emissiveAT};
        const auto atRenderInfo =
            utils::rendering_info(m_CommonImageExtent2D, std::size(atColorAttachments), atColorAttachments, &depthAT);

        vkCmdBeginRendering(cmd, &atRenderInfo);
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        vkCmdBindDescriptorBuffersEXT(cmd, 1, &gBufferBindingInfo);
        vkCmdSetDescriptorBufferOffsetsEXT(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                           m_metalRoughness.alphaTestedPipeline.pipelineLayout, 0, 1,
                                           gBufferBufferIndices, gBufferOffsets);

        draw_meshes(cmd, m_metalRoughness.alphaTestedPipeline.pipelineLayout,
                    m_metalRoughness.alphaTestedPipeline.pipeline, m_AlphaTestedSize, m_GBufferMeshPushConstants,
                    VK_SHADER_STAGE_VERTEX_BIT);

        vkCmdEndRendering(cmd);
    }

    mp::debug::cmd_end_label(cmd);
    const auto end = cn::steady_clock::now();
    const auto elapsed = cn::duration_cast<cn::milliseconds>(end - start);
    m_stats.gBufferPassTime = elapsed.count() / 1000.0f;
}

void Engine::draw_light_pass(const VkCommandBuffer cmd)
{
    mp::debug::cmd_begin_label(cmd, "Light Pass", {1.0f, 0.8f, 0.2f, 1.f});
    const auto start = cn::steady_clock::now();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_LightPassPipeline);

    const VkDescriptorBufferBindingInfoEXT buffersInfo[]{
        {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
         .pNext = nullptr,
         .address = get_current_frame().drawImageDescriptorBuffer.get_device_address(),
         .usage =
             VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT},
        {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
         .pNext = nullptr,
         .address = get_current_frame().lightPassDescriptorBuffer.get_device_address(),
         .usage =
             VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT}};
    vkCmdBindDescriptorBuffersEXT(cmd, std::size(buffersInfo), buffersInfo);

    const std::uint32_t indices[]{0, 1};
    const VkDeviceSize offsets[]{0, 0};
    vkCmdSetDescriptorBufferOffsetsEXT(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_LightPassPipelineLayout, 0,
                                       std::size(offsets), indices, offsets);

    vkCmdPushConstants(cmd, m_LightPassPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(LightPassPushConstants),
                       &m_LightPassConstants);

    vkCmdDispatch(cmd, std::ceil(m_CommonImageExtent2D.width / 16.0f), std::ceil(m_CommonImageExtent2D.height / 16.0f),
                  1);

    mp::debug::cmd_end_label(cmd);
    const auto end = cn::steady_clock::now();
    const auto elapsed = cn::duration_cast<cn::milliseconds>(end - start);
    m_stats.gBufferLightPassTime = elapsed.count() / 1000.0f;
}

void Engine::draw_wboit(VkCommandBuffer cmd)
{
    mp::debug::cmd_begin_label(cmd, "WBOIT Forward Pass", {0.2f, 0.8f, 0.9f, 1.f});
    const auto start = cn::steady_clock::now();
    auto &currentFrame = get_current_frame();

    const VkClearValue clearAccum{.color = {0.0f, 0.0f, 0.0f, 0.0f}};
    const VkClearValue clearReveal{.color = {1.0f, 1.0f, 1.0f, 1.0f}};
    const VkRenderingAttachmentInfo colorAttachments[]{
        utils::attachment_info(currentFrame.oitAccImage.imageView, &clearAccum,
                               VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL),
        utils::attachment_info(currentFrame.oitRevealImage.imageView, &clearReveal,
                               VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)};
    const auto depthAttachment =
        utils::depth_attachment(currentFrame.depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, false);
    const auto renderInfo =
        utils::rendering_info(m_CommonImageExtent2D, std::size(colorAttachments), colorAttachments, &depthAttachment);

    compute_cull_objects(cmd, m_TransparentSize, m_OpaqueSize + m_AlphaTestedSize, m_sceneData.projView);
    {
        utils::BarrierBuilder barrierBuilder;
        barrierBuilder.add_buffer_barrier(
            currentFrame.countBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                                                 .accessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                                                 .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
        barrierBuilder.add_buffer_barrier(
            currentFrame.drawCommandsBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                                                        .accessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                                                        .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
        barrierBuilder.barrier(cmd);
    }
    vkCmdBeginRendering(cmd, &renderInfo);

    const VkDescriptorBufferBindingInfoEXT wboitBindingInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
        .address = m_metalRoughness.descriptors.get_device_address(),
        .usage =
            VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT};
    vkCmdBindDescriptorBuffersEXT(cmd, 1, &wboitBindingInfo);

    const std::uint32_t wboitBufferIndices[]{0};
    const VkDeviceSize wboitOffsets[]{0};
    vkCmdSetDescriptorBufferOffsetsEXT(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                       m_metalRoughness.transparentPipeline.pipelineLayout, 0, 1, wboitBufferIndices,
                                       wboitOffsets);

    draw_meshes(cmd, m_metalRoughness.transparentPipeline.pipelineLayout, m_metalRoughness.transparentPipeline.pipeline,
                m_TransparentSize, m_WBOITForwardPassPushConstants,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    vkCmdEndRendering(cmd);
    mp::debug::cmd_end_label(cmd);
    const auto end = cn::steady_clock::now();
    const auto elapsed = cn::duration_cast<cn::milliseconds>(end - start);
    m_stats.transparentForwardLightPassTime = elapsed.count() / 1000.0f;
}

void Engine::draw_wboit_composite(VkCommandBuffer cmd)
{
    mp::debug::cmd_begin_label(cmd, "WBOIT Composite Pass", {0.1f, 0.7f, 0.9f, 1.f});
    const auto start = cn::steady_clock::now();

    const auto attachment = utils::attachment_info(get_current_frame().drawImage.imageView, nullptr,
                                                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    const VkRenderingInfo renderingInfo = utils::rendering_info(m_CommonImageExtent2D, 1, &attachment, nullptr);
    vkCmdBeginRendering(cmd, &renderingInfo);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_WBOITCompositePassPipeline);

    const VkDescriptorBufferBindingInfoEXT buffersInfo[]{
        {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
         .pNext = nullptr,
         .address = get_current_frame().wboitCompositePassDescBuffer.get_device_address(),
         .usage =
             VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT}};
    vkCmdBindDescriptorBuffersEXT(cmd, std::size(buffersInfo), buffersInfo);

    const std::uint32_t indices[]{0};
    const VkDeviceSize offsets[]{0};
    vkCmdSetDescriptorBufferOffsetsEXT(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_WBOITCompositePassPipelineLayout, 0,
                                       std::size(offsets), indices, offsets);

    vkCmdDraw(cmd, 6, 1, 0, 0);

    vkCmdEndRendering(cmd);
    mp::debug::cmd_end_label(cmd);
    const auto end = cn::steady_clock::now();
    const auto elapsed = cn::duration_cast<cn::milliseconds>(end - start);
    m_stats.postProcessPassTime = elapsed.count() / 1000.0f;
}

void Engine::draw_imgui(const VkCommandBuffer cmd, const VkImageView targetImageView)
{
    mp::debug::cmd_begin_label(cmd, "ImGui", {0.7f, 0.7f, 0.7f, 1.f});
    const auto start = cn::steady_clock::now();
    const auto colorAttachment =
        utils::attachment_info(targetImageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    const auto renderingInfo = utils::rendering_info(m_swapchainExtent, 1, &colorAttachment, nullptr);

    vkCmdBeginRendering(cmd, &renderingInfo);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

    vkCmdEndRendering(cmd);
    mp::debug::cmd_end_label(cmd);
    const auto end = cn::steady_clock::now();
    const auto elapsed = cn::duration_cast<cn::milliseconds>(end - start);
    m_stats.imguiDrawTime = elapsed.count() / 1000.0f;
}

void Engine::compute_cull_objects(VkCommandBuffer cmd, const std::uint32_t objectCount,
                                  const std::uint32_t objectOffset, const glm::mat4 &viewProj)
{
    mp::debug::cmd_begin_label(cmd, "Cull Objects", {0.8f, 0.8f, 0.2f, 1.f});
    auto &currentFrame = get_current_frame();
    utils::BarrierBuilder barrierBuilder;
    barrierBuilder.add_buffer_barrier(
        currentFrame.countBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                             .accessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
    barrierBuilder.barrier(cmd);

    vkCmdFillBuffer(cmd, currentFrame.countBuffer.buffer, 0, VK_WHOLE_SIZE, 0);

    barrierBuilder.add_buffer_barrier(
        currentFrame.countBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                             .accessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
                                             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
    barrierBuilder.add_buffer_barrier(currentFrame.drawCommandsBuffer.transition(
        {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
         .accessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
         .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
    barrierBuilder.barrier(cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_CullPassPipeline);

    const CullPassPushConstants cullPassConstants{
        .meshBufferAddr = currentFrame.meshBufferAddr,
        .instanceBufferDeviceAddr = currentFrame.instanceBufferAddr,
        .commandsBufferAddr = currentFrame.drawCommandsBufferAddr,
        .countBufferAddr = currentFrame.countBufferAddr,
        .viewProj = viewProj,
        .objectsCount = objectCount,
        .objectsOffset = objectOffset,
    };
    vkCmdPushConstants(cmd, m_CullPassPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CullPassPushConstants),
                       &cullPassConstants);

    vkCmdDispatch(cmd, std::ceil(objectCount / 64.0f), 1, 1);

    mp::debug::cmd_end_label(cmd);
}

void Engine::compute_cull_point_lights(VkCommandBuffer cmd)
{
    mp::debug::cmd_begin_label(cmd, "Cull Point Lights", {0.8f, 0.7f, 0.2f, 1.f});
    auto &currentFrame = get_current_frame();
    utils::BarrierBuilder barrierBuilder;
    barrierBuilder.add_buffer_barrier(
        currentFrame.pointLightIndicesOffsetsBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                                .accessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                                                .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
    barrierBuilder.add_buffer_barrier(
        currentFrame.visiblePointLightsCountBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                               .accessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                                               .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
    barrierBuilder.add_buffer_barrier(
        currentFrame.pointLightIndicesOffsetsCounterBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                                       .accessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                                                       .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
    barrierBuilder.barrier(cmd);

    vkCmdFillBuffer(cmd, currentFrame.pointLightIndicesOffsetsBuffer.buffer, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(cmd, currentFrame.visiblePointLightsCountBuffer.buffer, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(cmd, currentFrame.pointLightIndicesOffsetsCounterBuffer.buffer, 0, VK_WHOLE_SIZE, 0);

    barrierBuilder.add_buffer_barrier(currentFrame.pointLightIndicesOffsetsBuffer.transition(
        {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
         .accessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
         .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
    barrierBuilder.add_buffer_barrier(currentFrame.visiblePointLightsCountBuffer.transition(
        {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
         .accessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
         .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
    barrierBuilder.add_buffer_barrier(currentFrame.pointLightIndicesOffsetsCounterBuffer.transition(
        {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
         .accessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
         .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
    barrierBuilder.add_buffer_barrier(currentFrame.pointLightIndicesBuffer.transition(
        {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
         .accessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
         .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
    barrierBuilder.barrier(cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_CullPointLightsPassPipeline);

    const std::uint32_t pointLightsCount = m_mainDrawContext.pointLights.size();
    const CullPointLightsPassPushConstants cullPassConstants{
        .pointLights = currentFrame.pointLightBufferAddr,
        .pointLightsVisibleBuffer = currentFrame.visiblePointLightsBufferAddr,
        .pointLightsVisibleCountBuffer = currentFrame.visiblePointLightsCountBufferAddr,
        .pointLightsCount = pointLightsCount,
        .viewProj = m_sceneData.projView,
    };
    vkCmdPushConstants(cmd, m_CullPointLightsPassPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(CullPointLightsPassPushConstants), &cullPassConstants);

    vkCmdDispatch(cmd, std::ceil(pointLightsCount / 64.0f), 1, 1);

    const VkPipelineStageFlags2 dstStages =
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT;
    barrierBuilder.add_buffer_barrier(
        currentFrame.visiblePointLightsBuffer.transition({.stageMask = dstStages,
                                                          .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
                                                          .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
    barrierBuilder.add_buffer_barrier(
        currentFrame.visiblePointLightsCountBuffer.transition({.stageMask = dstStages,
                                                               .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
                                                               .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
    barrierBuilder.barrier(cmd);
    mp::debug::cmd_end_label(cmd);
}

void Engine::compute_point_lights_commands(VkCommandBuffer cmd, const std::uint32_t instanceCount,
                                           const std::uint32_t instanceOffset)
{
    if (m_mainDrawContext.pointLights.empty() || instanceCount == 0)
        return;
    mp::debug::cmd_begin_label(cmd, "Generate Point Light Commands", {0.7f, 0.5f, 0.2f, 1.f});

    auto &currentFrame = get_current_frame();

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_GeneratePointLightCommandsPipeline);

    const GeneratePointLightCommandsPushConstants pc{
        .meshes = currentFrame.meshBufferAddr,
        .instances = currentFrame.instanceBufferAddr,
        .meshDrawCommands = currentFrame.drawCommandsBufferAddr,
        .meshDrawCommandsCount = currentFrame.countBufferAddr,
        .visiblePointLights = currentFrame.visiblePointLightsBufferAddr,
        .visiblePointLightsCount = currentFrame.visiblePointLightsCountBufferAddr,
        .instanceOffset = instanceOffset,
        ._padding1 = 0,
        .pointLightIndices = currentFrame.pointLightIndicesBufferAddr,
        .pointLightOffsets = currentFrame.pointLightIndicesOffsetsBufferAddr,
        .pointLightOffsetsCounter = currentFrame.pointLightIndicesOffsetsCounterBufferAddr,
    };
    vkCmdPushConstants(cmd, m_GeneratePointLightCommandsPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(GeneratePointLightCommandsPushConstants), &pc);

    vkCmdDispatch(cmd, instanceCount, 1, 1);
    mp::debug::cmd_end_label(cmd);
}

void Engine::draw_point_lights_shadows_pass(VkCommandBuffer cmd)
{
    auto &currentFrame = get_current_frame();

    // Always transition tile map to DEPTH_ATTACHMENT_OPTIMAL each frame so that
    // the pre-light-pass barrier (DEPTH_ATTACHMENT -> DEPTH_READ_ONLY) always
    // sees the correct oldLayout, even when there are no point lights.
    {
        utils::BarrierBuilder barrierBuilder;
        barrierBuilder.add_image_barrier(currentFrame.pointLightsShadowTileMap.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
             .accessMask =
                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
             .layout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_DEPTH_BIT)}));
        barrierBuilder.barrier(cmd);
    }

    if (m_mainDrawContext.pointLights.empty())
        return;
    mp::debug::cmd_begin_label(cmd, "Point Light Shadow Pass", {0.9f, 0.4f, 0.1f, 1.f});

    // Generate point light commands
    {
        utils::BarrierBuilder barrierBuilder;
        barrierBuilder.add_buffer_barrier(
            currentFrame.countBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                 .accessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                                 .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
        barrierBuilder.barrier(cmd);

        vkCmdFillBuffer(cmd, currentFrame.countBuffer.buffer, 0, VK_WHOLE_SIZE, 0);

        barrierBuilder.add_buffer_barrier(currentFrame.countBuffer.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             .accessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
        barrierBuilder.add_buffer_barrier(currentFrame.drawCommandsBuffer.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             .accessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
        barrierBuilder.barrier(cmd);
    }
    compute_point_lights_commands(cmd, m_OpaqueSize, 0);

    {
        utils::BarrierBuilder barrierBuilder;
        barrierBuilder.add_buffer_barrier(
            currentFrame.drawCommandsBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                                                        .accessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                                                        .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
        barrierBuilder.add_buffer_barrier(
            currentFrame.countBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                                                 .accessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                                                 .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
        barrierBuilder.add_buffer_barrier(currentFrame.pointLightIndicesBuffer.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT,
             .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
        barrierBuilder.add_buffer_barrier(currentFrame.pointLightIndicesOffsetsBuffer.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT,
             .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
        barrierBuilder.barrier(cmd);
    }

    const VkExtent2D shadowExtent{kPointLightsShadowMapSize, kPointLightsShadowMapSize};

    const auto depthAttachment = utils::depth_attachment(currentFrame.pointLightsShadowTileMap.imageView,
                                                         VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    const VkRenderingInfo renderInfo{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .pNext = nullptr,
        .renderArea = {.extent = shadowExtent},
        .layerCount = 1,
        .colorAttachmentCount = 0,
        .pColorAttachments = nullptr,
        .pDepthAttachment = &depthAttachment,
        .pStencilAttachment = nullptr,
    };

    vkCmdBeginRendering(cmd, &renderInfo);

    const VkViewport viewport{
        .x = 0,
        .y = static_cast<float>(shadowExtent.height),
        .width = static_cast<float>(shadowExtent.width),
        .height = -static_cast<float>(shadowExtent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    const VkRect2D scissor{.extent = shadowExtent};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    const PointLightsShadowPassPushConstants pc{
        .positionBufferAddr = m_globalPositionBufferAddress,
        .instances = currentFrame.instanceBufferAddr,
        .visiblePointLights = currentFrame.visiblePointLightsBufferAddr,
        .pointLightIndices = currentFrame.pointLightIndicesBufferAddr,
        .pointLightOffsets = currentFrame.pointLightIndicesOffsetsBufferAddr,
        .tetrahedronDataAddr = m_tetrahedronBuffer.get_buffer_device_address(m_device),
    };

    draw_meshes(cmd, m_PointLightShadowPassPipelineLayout, m_PointLightShadowPassPipeline, m_OpaqueSize, pc,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT);

    vkCmdEndRendering(cmd);

    if (m_AlphaTestedSize > 0)
    {
        utils::BarrierBuilder barrierBuilder;
        barrierBuilder.add_buffer_barrier(
            currentFrame.countBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                 .accessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                                 .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
        barrierBuilder.barrier(cmd);
        vkCmdFillBuffer(cmd, currentFrame.countBuffer.buffer, 0, VK_WHOLE_SIZE, 0);
        barrierBuilder.add_buffer_barrier(currentFrame.countBuffer.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             .accessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
        barrierBuilder.add_buffer_barrier(currentFrame.drawCommandsBuffer.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             .accessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
        barrierBuilder.barrier(cmd);

        compute_point_lights_commands(cmd, m_AlphaTestedSize, m_OpaqueSize);

        barrierBuilder.add_buffer_barrier(
            currentFrame.drawCommandsBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                                                        .accessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                                                        .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
        barrierBuilder.add_buffer_barrier(
            currentFrame.countBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                                                 .accessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                                                 .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
        barrierBuilder.barrier(cmd);

        // LOAD depth — opaque shadow already wrote to tile map
        const auto atDepthAttachment = utils::depth_attachment(currentFrame.pointLightsShadowTileMap.imageView,
                                                               VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, false);
        const VkRenderingInfo atRenderInfo{
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .pNext = nullptr,
            .renderArea = {.extent = shadowExtent},
            .layerCount = 1,
            .colorAttachmentCount = 0,
            .pColorAttachments = nullptr,
            .pDepthAttachment = &atDepthAttachment,
            .pStencilAttachment = nullptr,
        };
        vkCmdBeginRendering(cmd, &atRenderInfo);
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        const VkDescriptorBufferBindingInfoEXT atBindingInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
            .address = m_metalRoughness.descriptors.get_device_address(),
            .usage =
                VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT};
        vkCmdBindDescriptorBuffersEXT(cmd, 1, &atBindingInfo);

        const std::uint32_t atBufferIndices[]{0};
        const VkDeviceSize atOffsets[]{0};
        vkCmdSetDescriptorBufferOffsetsEXT(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                           m_AlphaTestedPointLightShadowPassPipelineLayout, 0, 1, atBufferIndices,
                                           atOffsets);

        const PointLightsShadowPassAlphaTestedPushConstants atPc{
            .positionBufferAddr = m_globalPositionBufferAddress,
            .attributesBufferAddr = m_GBufferMeshPushConstants.attributesBufferAddr,
            .instances = currentFrame.instanceBufferAddr,
            .visiblePointLights = currentFrame.visiblePointLightsBufferAddr,
            .pointLightIndices = currentFrame.pointLightIndicesBufferAddr,
            .pointLightOffsets = currentFrame.pointLightIndicesOffsetsBufferAddr,
            .tetrahedronDataAddr = m_tetrahedronBuffer.get_buffer_device_address(m_device),
        };
        draw_meshes(cmd, m_AlphaTestedPointLightShadowPassPipelineLayout, m_AlphaTestedPointLightShadowPassPipeline,
                    m_AlphaTestedSize, atPc,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

        vkCmdEndRendering(cmd);
    }

    mp::debug::cmd_end_label(cmd);
}

void Engine::draw_meshes(VkCommandBuffer cmd, const VkPipelineLayout drawPassPipelineLayout,
                         const VkPipeline drawPipeline, const std::uint32_t objectCount, auto &pushConstants,
                         const VkShaderStageFlags pushConstantsShaderStage)
{
    auto &currentFrame = get_current_frame();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, drawPipeline);

    vkCmdPushConstants(cmd, drawPassPipelineLayout, pushConstantsShaderStage, 0, sizeof(pushConstants), &pushConstants);

    vkCmdDrawIndexedIndirectCount(cmd, currentFrame.drawCommandsBuffer.buffer, 0, currentFrame.countBuffer.buffer, 0,
                                  objectCount, sizeof(VkDrawIndexedIndirectCommand));
    m_stats.drawCallCount++;
}

void Engine::compute_post(VkCommandBuffer cmd)
{
    mp::debug::cmd_begin_label(cmd, "Post-Process", {0.9f, 0.5f, 0.9f, 1.f});
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PostProcessPassPipeline);

    const VkDescriptorBufferBindingInfoEXT buffersInfo[]{
        {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
         .pNext = nullptr,
         .address = get_current_frame().drawImageDescriptorBuffer.get_device_address(),
         .usage =
             VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT},
    };
    vkCmdBindDescriptorBuffersEXT(cmd, std::size(buffersInfo), buffersInfo);

    const std::uint32_t indices[]{0};
    const VkDeviceSize offsets[]{0};
    vkCmdSetDescriptorBufferOffsetsEXT(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PostProcessPassPipelineLayout, 0,
                                       std::size(offsets), indices, offsets);

    const PostProcessPushConstants postPc{.avgLum = m_avgLuminanceBufferAddr};
    vkCmdPushConstants(cmd, m_PostProcessPassPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(postPc), &postPc);

    vkCmdDispatch(cmd, std::ceil(m_CommonImageExtent2D.width / 16.0f), std::ceil(m_CommonImageExtent2D.height / 16.0f),
                  1.0f);
    mp::debug::cmd_end_label(cmd);
}

void Engine::compute_luminance_histogram(VkCommandBuffer cmd)
{
    mp::debug::cmd_begin_label(cmd, "Luminance Histogram", {0.8f, 0.6f, 0.2f, 1.f});
    auto &currentFrame = get_current_frame();

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_LuminanceHistogramPipeline);

    const VkDescriptorBufferBindingInfoEXT buffersInfo[]{
        {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
         .pNext = nullptr,
         .address = currentFrame.drawImageDescriptorBuffer.get_device_address(),
         .usage =
             VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT},
    };
    vkCmdBindDescriptorBuffersEXT(cmd, std::size(buffersInfo), buffersInfo);

    const std::uint32_t indices[]{0};
    const VkDeviceSize offsets[]{0};
    vkCmdSetDescriptorBufferOffsetsEXT(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_LuminanceHistogramPipelineLayout, 0,
                                       std::size(offsets), indices, offsets);

    const LuminanceHistogramPushConstants pc{
        .histogram = currentFrame.histogramBufferAddr,
        .minLogLum = m_autoExposureMinLogLum,
        .invLogLumRange = 1.0f / m_autoExposureLogLumRange,
    };
    vkCmdPushConstants(cmd, m_LuminanceHistogramPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    vkCmdDispatch(cmd, static_cast<std::uint32_t>(std::ceil(m_CommonImageExtent2D.width / 16.0f)),
                  static_cast<std::uint32_t>(std::ceil(m_CommonImageExtent2D.height / 16.0f)), 1);
    mp::debug::cmd_end_label(cmd);
}

void Engine::compute_average_luminance(VkCommandBuffer cmd)
{
    mp::debug::cmd_begin_label(cmd, "Average Luminance", {0.8f, 0.4f, 0.1f, 1.f});
    auto &currentFrame = get_current_frame();

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_AverageLuminancePipeline);

    const float timeCoeff = 1.0f - std::exp(-m_stats.frameTime * m_autoExposureAdaptationSpeed);
    const AverageLuminancePushConstants pc{
        .histogram = currentFrame.histogramBufferAddr,
        .avgLum = m_avgLuminanceBufferAddr,
        .minLogLum = m_autoExposureMinLogLum,
        .logLumRange = m_autoExposureLogLumRange,
        .timeCoeff = timeCoeff,
        .numPixels = m_CommonImageExtent2D.width * m_CommonImageExtent2D.height,
    };
    vkCmdPushConstants(cmd, m_AverageLuminancePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    vkCmdDispatch(cmd, 1, 1, 1);
    mp::debug::cmd_end_label(cmd);
}

void Engine::draw_prepass(VkCommandBuffer cmd)
{
    mp::debug::cmd_begin_label(cmd, "Prepass", {0.4f, 0.6f, 1.0f, 1.f});
    auto &currentFrame = get_current_frame();

    compute_cull_objects(cmd, m_OpaqueSize, 0, m_sceneData.projView);

    {
        utils::BarrierBuilder barrierBuilder;
        barrierBuilder.add_buffer_barrier(
            currentFrame.countBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                                                 .accessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                                                 .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
        barrierBuilder.add_buffer_barrier(
            currentFrame.drawCommandsBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                                                        .accessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                                                        .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
        barrierBuilder.barrier(cmd);
    }

    const auto depthAttachment =
        utils::depth_attachment(currentFrame.depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    const auto renderInfo = utils::rendering_info(m_CommonImageExtent2D, 0, nullptr, &depthAttachment);

    vkCmdBeginRendering(cmd, &renderInfo);

    const VkViewport viewport{
        .x = 0,
        .y = static_cast<float>(m_CommonImageExtent2D.height),
        .width = static_cast<float>(m_CommonImageExtent2D.width),
        .height = -static_cast<float>(m_CommonImageExtent2D.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    const VkRect2D scissor{.extent = m_CommonImageExtent2D};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    const VkDescriptorBufferBindingInfoEXT prepassBindingInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
        .address = m_metalRoughness.descriptors.get_device_address(),
        .usage =
            VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT};
    vkCmdBindDescriptorBuffersEXT(cmd, 1, &prepassBindingInfo);

    const std::uint32_t prepassBufferIndices[]{0};
    const VkDeviceSize prepassOffsets[]{0};
    vkCmdSetDescriptorBufferOffsetsEXT(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_PrepassPipelineLayout, 0, 1,
                                       prepassBufferIndices, prepassOffsets);

    draw_meshes(cmd, m_PrepassPipelineLayout, m_PrepassPipeline, m_OpaqueSize, m_GBufferMeshPushConstants,
                VK_SHADER_STAGE_VERTEX_BIT);

    vkCmdEndRendering(cmd);

    if (m_AlphaTestedSize > 0)
    {
        compute_cull_objects(cmd, m_AlphaTestedSize, m_OpaqueSize, m_sceneData.projView);

        {
            utils::BarrierBuilder barrierBuilder;
            barrierBuilder.add_buffer_barrier(
                currentFrame.countBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                                                     .accessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                                                     .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
            barrierBuilder.add_buffer_barrier(
                currentFrame.drawCommandsBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                                                            .accessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                                                            .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
            barrierBuilder.barrier(cmd);
        }

        // LOAD depth — opaque prepass already wrote depth
        const auto atDepthAttachment =
            utils::depth_attachment(currentFrame.depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, false);
        const auto atRenderInfo = utils::rendering_info(m_CommonImageExtent2D, 0, nullptr, &atDepthAttachment);

        vkCmdBeginRendering(cmd, &atRenderInfo);
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        const VkDescriptorBufferBindingInfoEXT atPrepassBindingInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
            .address = m_metalRoughness.descriptors.get_device_address(),
            .usage =
                VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT};
        vkCmdBindDescriptorBuffersEXT(cmd, 1, &atPrepassBindingInfo);

        const std::uint32_t atPrepassBufferIndices[]{0};
        const VkDeviceSize atPrepassOffsets[]{0};
        vkCmdSetDescriptorBufferOffsetsEXT(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_AlphaTestedPrepassPipelineLayout, 0,
                                           1, atPrepassBufferIndices, atPrepassOffsets);

        draw_meshes(cmd, m_AlphaTestedPrepassPipelineLayout, m_AlphaTestedPrepassPipeline, m_AlphaTestedSize,
                    m_GBufferMeshPushConstants, VK_SHADER_STAGE_VERTEX_BIT);

        vkCmdEndRendering(cmd);
    }

    mp::debug::cmd_end_label(cmd);
}

void Engine::compute_populate_commands_with_cascade_count(VkCommandBuffer cmd, std::uint32_t objectCount)
{
    mp::debug::cmd_begin_label(cmd, "Populate commands with cascade count", {0.33f, 0.66f, 0.99f, 1.f});

    auto &currentFrame = get_current_frame();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PopulateCommandsWithCascadeCountPipeline);
    const auto cascadeCount = m_mainDrawContext.dirLight.value().cascadeCount;
    const PopulateCommandsWithCascadeCountPushConstants pushConstants{.commands = currentFrame.drawCommandsBufferAddr,
                                                                      .count = currentFrame.countBufferAddr,
                                                                      .cascadesCount = cascadeCount};
    vkCmdPushConstants(cmd, m_PopulateCommandsWithCascadeCountPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PopulateCommandsWithCascadeCountPushConstants), &pushConstants);

    vkCmdDispatch(cmd, std::ceil(objectCount / 64.0f), 1, 1);
    mp::debug::cmd_end_label(cmd);
}

void Engine::compute_depth_reduction(VkCommandBuffer cmd)
{
    mp::debug::cmd_begin_label(cmd, "Depth Reduction", {0.5f, 0.9f, 0.5f, 1.f});
    auto &frame = get_current_frame();

    MinMax initMinMax;
    initMinMax.min = std::bit_cast<std::uint32_t>(std::numeric_limits<float>::max());
    initMinMax.max = 0u; // sentinel for unsigned InterlockedMax: all positive linearized depths > 0
    vkCmdUpdateBuffer(cmd, frame.minMaxBuffer.buffer, 0, sizeof(MinMax), &initMinMax);

    utils::BarrierBuilder barrierBuilder;
    barrierBuilder.add_buffer_barrier(
        frame.minMaxBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                       .accessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                                       .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
    barrierBuilder.barrier(cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_DepthReductionPipeline);

    const VkDescriptorBufferBindingInfoEXT bindingInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
        .address = frame.cascadeDepthDescBuffer.get_device_address(),
        .usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT,
    };
    vkCmdBindDescriptorBuffersEXT(cmd, 1, &bindingInfo);

    const std::uint32_t bufferIndex{0};
    const VkDeviceSize offset{0};
    vkCmdSetDescriptorBufferOffsetsEXT(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_DepthReductionPipelineLayout, 0, 1,
                                       &bufferIndex, &offset);

    const DepthReductionPushConstants pushConstants{
        .minMaxAddr = frame.minMaxBufferAddr,
        .cameraNear = cameraNear,
        .cameraFar = cameraFar,
    };
    vkCmdPushConstants(cmd, m_DepthReductionPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(DepthReductionPushConstants), &pushConstants);

    vkCmdDispatch(cmd, static_cast<std::uint32_t>(std::ceil(m_CommonImageExtent2D.width / 16.0f)),
                  static_cast<std::uint32_t>(std::ceil(m_CommonImageExtent2D.height / 16.0f)), 1);
    mp::debug::cmd_end_label(cmd);
}

void Engine::compute_depth_partition(VkCommandBuffer cmd)
{
    if (!m_mainDrawContext.dirLight.has_value())
        return;
    mp::debug::cmd_begin_label(cmd, "Depth Partition", {0.3f, 0.8f, 0.3f, 1.f});
    auto &frame = get_current_frame();
    // Reset splitsAABB: min* = FLT_MAX, max* = -FLT_MAX
    CascadesAABB initAABB{};
    for (auto &aabb : initAABB.bounds)
    {
        aabb.minX = aabb.minY = aabb.minZ = std::bit_cast<std::uint32_t>(std::numeric_limits<float>::max());
        aabb.maxX = aabb.maxY = aabb.maxZ = std::bit_cast<std::uint32_t>(-std::numeric_limits<float>::max());
        aabb._pad0 = aabb._pad1 = 0;
    }
    vkCmdUpdateBuffer(cmd, frame.splitsAABBBuffer.buffer, 0, sizeof(CascadesAABB), &initAABB);

    utils::BarrierBuilder barrierBuilder;
    // minMax: COMPUTE WRITE -> COMPUTE READ
    barrierBuilder.add_buffer_barrier(
        frame.minMaxBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                       .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
                                       .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
    // splitsAABB: TRANSFER WRITE -> COMPUTE READ|WRITE
    barrierBuilder.add_buffer_barrier(
        frame.splitsAABBBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                           .accessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                                           .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
    barrierBuilder.barrier(cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_DepthPartitionPipeline);

    const VkDescriptorBufferBindingInfoEXT bindingInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
        .address = frame.cascadeDepthDescBuffer.get_device_address(),
        .usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT,
    };
    vkCmdBindDescriptorBuffersEXT(cmd, 1, &bindingInfo);

    const std::uint32_t bufferIndex{0};
    const VkDeviceSize offset{0};
    vkCmdSetDescriptorBufferOffsetsEXT(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_DepthPartitionPipelineLayout, 0, 1,
                                       &bufferIndex, &offset);

    const DepthPartitionPushConstants pushConstants{
        .minMaxAddr = frame.minMaxBufferAddr,
        .splitsAABBAddr = frame.splitsAABBBufferAddr,
        .dirLightsAddr = frame.dirLightBufferAddr,
        .near = cameraNear,
        .far = cameraFar,
        .inverseCameraViewProj = glm::inverse(m_sceneData.projView),
        .lightViewMatrix = m_DirLightViewMatrix,
    };
    vkCmdPushConstants(cmd, m_DepthPartitionPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(DepthPartitionPushConstants), &pushConstants);

    vkCmdDispatch(cmd, static_cast<std::uint32_t>(std::ceil(m_CommonImageExtent2D.width / 16.0f)),
                  static_cast<std::uint32_t>(std::ceil(m_CommonImageExtent2D.height / 16.0f)), 1);
    mp::debug::cmd_end_label(cmd);
}

void Engine::compute_dir_lights_vp(VkCommandBuffer cmd)
{
    if (!m_mainDrawContext.dirLight.has_value())
        return;
    mp::debug::cmd_begin_label(cmd, "Directional VP", {0.2f, 0.7f, 0.2f, 1.f});
    auto &frame = get_current_frame();

    // splitsAABB and dirLight (splitDistances written by partition): COMPUTE WRITE -> COMPUTE READ
    utils::BarrierBuilder barrierBuilder;
    barrierBuilder.add_buffer_barrier(
        frame.splitsAABBBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                           .accessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                                           .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
    barrierBuilder.add_buffer_barrier(
        frame.dirLightBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                         .accessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                                         .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
    barrierBuilder.barrier(cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_DirVpPipeline);

    const DirVpPushConstants pushConstants{
        .splitsAABBAddr = frame.splitsAABBBufferAddr,
        .dirLightAddr = frame.dirLightBufferAddr,
        .sceneMin = m_mainDrawContext.min,
        .sceneMax = m_mainDrawContext.max,
        .shadowMapSize = kDirectionalShadowMapSize,
        .lightView = m_DirLightViewMatrix,
    };
    vkCmdPushConstants(cmd, m_DirVpPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DirVpPushConstants),
                       &pushConstants);

    vkCmdDispatch(cmd, 1, 1, 1);
    mp::debug::cmd_end_label(cmd);
}

void Engine::copy_staging_buffers(VkCommandBuffer cmd)
{
    auto &frame = get_current_frame();
    utils::BarrierBuilder barrierBuilder;

    const std::uint32_t totalInstances = m_OpaqueSize + m_AlphaTestedSize + m_TransparentSize;
    if (totalInstances > 0)
    {
        const VkBufferCopy instanceCopy{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = totalInstances * sizeof(Instance),
        };
        vkCmdCopyBuffer(cmd, frame.instanceStagingBuffer.buffer, frame.instanceBuffer.buffer, 1, &instanceCopy);
        barrierBuilder.add_buffer_barrier(frame.instanceBuffer.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                          VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT,
             .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
    }

    if (m_mainDrawContext.dirLight.has_value())
    {
        const VkBufferCopy dirLightCopy{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = sizeof(DirectionalLightData),
        };
        vkCmdCopyBuffer(cmd, frame.dirLightStagingBuffer.buffer, frame.dirLightBuffer.buffer, 1, &dirLightCopy);
        barrierBuilder.add_buffer_barrier(
            frame.dirLightBuffer.transition({.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                             .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
                                             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED}));
    }

    barrierBuilder.barrier(cmd);
}

void Engine::copy_frame_buffers()
{
    m_OpaqueSize = static_cast<std::uint32_t>(m_mainDrawContext.opaqueInstances.size());
    std::memcpy(m_CurrentFrameInstanceBuffer, m_mainDrawContext.opaqueInstances.data(),
                m_OpaqueSize * sizeof(Instance));

    m_AlphaTestedSize = static_cast<std::uint32_t>(m_mainDrawContext.alphaTestedInstances.size());
    std::memcpy(m_CurrentFrameInstanceBuffer + m_OpaqueSize, m_mainDrawContext.alphaTestedInstances.data(),
                m_AlphaTestedSize * sizeof(Instance));

    m_TransparentSize = static_cast<std::uint32_t>(m_mainDrawContext.transparentInstances.size());
    std::memcpy(m_CurrentFrameInstanceBuffer + m_OpaqueSize + m_AlphaTestedSize,
                m_mainDrawContext.transparentInstances.data(), m_TransparentSize * sizeof(Instance));

    std::memcpy(m_CurrentMeshBuffer, m_mainDrawContext.renderObjects.data(),
                m_mainDrawContext.renderObjects.size() * sizeof(RenderObject));
}

void Engine::trace_ddgi_probe_pass(VkCommandBuffer cmd)
{
    mp::debug::cmd_begin_label(cmd, "DDGI Probe Pass", {0.2f, 0.9f, 0.4f, 1.f});
    auto &currentFrame = get_current_frame();

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_DDGIPipeline);

    const VkDescriptorBufferBindingInfoEXT bindingInfos[]{
        {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
         .address = currentFrame.ddgiTLASDescBuffer.get_device_address(),
         .usage =
             VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT},
        {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
         .address = ddgiRayDataDescBuffer.get_device_address(),
         .usage =
             VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT},
        {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
         .address = ddgiResourcesDescBuffer.get_device_address(),
         .usage =
             VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT},
        {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
         .address = m_metalRoughness.descriptors.get_device_address(),
         .usage =
             VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT},
    };
    vkCmdBindDescriptorBuffersEXT(cmd, std::size(bindingInfos), bindingInfos);

    const std::uint32_t setIndices[]{0, 1, 2, 3};
    const VkDeviceSize setOffsets[]{0, 0, 0, 0};
    vkCmdSetDescriptorBufferOffsetsEXT(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_DDGIPipelineLayout, 0,
                                       std::size(setIndices), setIndices, setOffsets);

    const VkDeviceAddress dirLightAddr =
        m_mainDrawContext.dirLight.has_value() ? currentFrame.dirLightBufferAddr : VkDeviceAddress{0};

    DDGIProbePushConstants pc{
        .volumes = m_DDGIVolumesAddr,
        .dirLight = dirLightAddr,
        .pointLights = currentFrame.pointLightBufferAddr,
        .pointLightsCount = static_cast<std::uint32_t>(m_mainDrawContext.pointLights.size()),
        .vPositions = m_globalPositionBufferAddress,
        .vAttributes = m_globalAttributesBufferAddress,
        .indices = m_globalIndexBufferDeviceAddress,
        .instances = currentFrame.instanceBufferAddr,
        .meshes = currentFrame.meshBufferAddr,
        .rayNormalBias = m_DDGIRayNormalBias,
        .rayViewBias = m_DDGIRayViewBias,
        .skyRadiance = m_SkyRadiance,
    };
    for (std::uint32_t i = 0; i < m_DDGIVolumeCount; ++i)
    {
        const auto &volData = m_mainDrawContext.ddgiVolumes[i];
        pc.currentVolumeIndex = i;
        vkCmdPushConstants(cmd, m_DDGIPipelineLayout,
                           VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                               VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                           0, sizeof(DDGIProbePushConstants), &pc);

        vkCmdTraceRaysKHR(cmd, &m_RaygenRegion, &m_MissRegion, &m_HitRegion, &m_CallableRegion, volData.probeNumRays,
                          volData.probeCounts.x * volData.probeCounts.z, volData.probeCounts.y);
    }

    mp::debug::cmd_end_label(cmd);
}

void Engine::compute_ddgi_irradiance_blending(VkCommandBuffer cmd)
{
    mp::debug::cmd_begin_label(cmd, "DDGI Irradiance Blending", {0.1f, 0.8f, 0.5f, 1.f});
    auto &currentFrame = get_current_frame();

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_DDGIIrradianceBlendingPipeline);

    const VkDescriptorBufferBindingInfoEXT bindingInfos[]{
        {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
         .address = ddgiRayDataDescBuffer.get_device_address(),
         .usage =
             VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT},
        {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
         .address = ddgiIrradianceStorageDescBuffer.get_device_address(),
         .usage =
             VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT},
    };
    vkCmdBindDescriptorBuffersEXT(cmd, std::size(bindingInfos), bindingInfos);

    const std::uint32_t setIndices[]{0, 1};
    const VkDeviceSize setOffsets[]{0, 0};
    vkCmdSetDescriptorBufferOffsetsEXT(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_DDGIProbeSupportPipelineLayout, 0,
                                       std::size(setIndices), setIndices, setOffsets);

    for (std::uint32_t i = 0; i < m_DDGIVolumeCount; ++i)
    {
        const auto &vol = m_mainDrawContext.ddgiVolumes[i];
        const DDGIProbeSupportPushConstants pc{.volumes = m_DDGIVolumesAddr, .currentVolumeIndex = i};
        vkCmdPushConstants(cmd, m_DDGIProbeSupportPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(DDGIProbeSupportPushConstants), &pc);
        vkCmdDispatch(cmd, static_cast<std::uint32_t>(vol.probeCounts.x), static_cast<std::uint32_t>(vol.probeCounts.z),
                      static_cast<std::uint32_t>(vol.probeCounts.y));
    }

    mp::debug::cmd_end_label(cmd);
}

void Engine::compute_ddgi_distance_blending(VkCommandBuffer cmd)
{
    mp::debug::cmd_begin_label(cmd, "DDGI Distance Blending", {0.1f, 0.6f, 0.8f, 1.f});
    auto &currentFrame = get_current_frame();

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_DDGIDistanceBlendingPipeline);

    const VkDescriptorBufferBindingInfoEXT bindingInfos[]{
        {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
         .address = ddgiRayDataDescBuffer.get_device_address(),
         .usage =
             VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT},
        {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
         .address = ddgiDistanceStorageDescBuffer.get_device_address(),
         .usage =
             VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT},
    };
    vkCmdBindDescriptorBuffersEXT(cmd, std::size(bindingInfos), bindingInfos);

    const std::uint32_t setIndices[]{0, 1};
    const VkDeviceSize setOffsets[]{0, 0};
    vkCmdSetDescriptorBufferOffsetsEXT(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_DDGIProbeSupportPipelineLayout, 0,
                                       std::size(setIndices), setIndices, setOffsets);

    for (std::uint32_t i = 0; i < m_DDGIVolumeCount; ++i)
    {
        const auto &vol = m_mainDrawContext.ddgiVolumes[i];
        const DDGIProbeSupportPushConstants pc{.volumes = m_DDGIVolumesAddr, .currentVolumeIndex = i};
        vkCmdPushConstants(cmd, m_DDGIProbeSupportPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(DDGIProbeSupportPushConstants), &pc);
        vkCmdDispatch(cmd, static_cast<std::uint32_t>(vol.probeCounts.x), static_cast<std::uint32_t>(vol.probeCounts.z),
                      static_cast<std::uint32_t>(vol.probeCounts.y));
    }

    mp::debug::cmd_end_label(cmd);
}

void Engine::compute_ddgi_relocation(VkCommandBuffer cmd)
{
    mp::debug::cmd_begin_label(cmd, "DDGI probe Relocation", {0.6f, 0.3f, 0.9f, 1.f});

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_DDGIProbeRelocationPipeline);

    const VkDescriptorBufferBindingInfoEXT bindingInfos[]{
        {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
         .address = ddgiRayDataDescBuffer.get_device_address(),
         .usage =
             VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT},
        {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
         .address = ddgiProbeDataStorageDescBuffer.get_device_address(),
         .usage =
             VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT},
    };
    vkCmdBindDescriptorBuffersEXT(cmd, std::size(bindingInfos), bindingInfos);

    const std::uint32_t setIndices[]{0, 1};
    const VkDeviceSize setOffsets[]{0, 0};
    vkCmdSetDescriptorBufferOffsetsEXT(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_DDGIProbeSupportPipelineLayout, 0,
                                       std::size(setIndices), setIndices, setOffsets);

    for (std::uint32_t i = 0; i < m_DDGIVolumeCount; ++i)
    {
        const auto &vol = m_mainDrawContext.ddgiVolumes[i];
        if (vol.probeRelocationEnabled == 1)
        {
            const DDGIProbeSupportPushConstants pc{.volumes = m_DDGIVolumesAddr, .currentVolumeIndex = i};
            vkCmdPushConstants(cmd, m_DDGIProbeSupportPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(DDGIProbeSupportPushConstants), &pc);
            vkCmdDispatch(cmd,
                          static_cast<std::uint32_t>(
                              std::ceil(vol.probeCounts.x * vol.probeCounts.z * vol.probeCounts.y / 32.0f)),
                          1, 1);
        }
        else
        {
            // TODO: When classification added, replace it with distinct pass
            VkClearColorValue clearColorValue{
                .float32 = {0.0f, 0.0f, 0.0f, 1.0f},
            };

            VkImageSubresourceRange range{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                          .baseMipLevel = 0,
                                          .levelCount = 1,
                                          .baseArrayLayer = 0,
                                          .layerCount = kMaxDDGIProbesY};
            vkCmdClearColorImage(cmd, probeDatas[i].image, VK_IMAGE_LAYOUT_GENERAL, &clearColorValue, 1, &range);
        };
    }

    mp::debug::cmd_end_label(cmd);
}

void Engine::compute_ddgi_indirect(VkCommandBuffer cmd)
{
    mp::debug::cmd_begin_label(cmd, "DDGI Indirect", {0.0f, 0.7f, 0.3f, 1.f});
    auto &currentFrame = get_current_frame();

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_DDGIIndirectPipeline);

    const VkDescriptorBufferBindingInfoEXT bindingInfos[]{
        {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
         .address = currentFrame.ddgiOutputStorageDescBuffer.get_device_address(),
         .usage =
             VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT},
        {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
         .address = currentFrame.ddgiGBufferReadDescBuffer.get_device_address(),
         .usage =
             VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT},
        {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
         .address = ddgiResourcesDescBuffer.get_device_address(),
         .usage =
             VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT},
    };
    vkCmdBindDescriptorBuffersEXT(cmd, std::size(bindingInfos), bindingInfos);

    const std::uint32_t setIndices[]{0, 1, 2};
    const VkDeviceSize setOffsets[]{0, 0, 0};
    vkCmdSetDescriptorBufferOffsetsEXT(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_DDGIIndirectPipelineLayout, 0,
                                       std::size(setIndices), setIndices, setOffsets);

    const DDGIIndirectPushConstants pc{
        .volumes = m_DDGIVolumesAddr,
        .volumeCount = m_DDGIVolumeCount,
        .sceneData = currentFrame.sceneDataBufferAddr,
    };
    vkCmdPushConstants(cmd, m_DDGIIndirectPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(DDGIIndirectPushConstants), &pc);

    const std::uint32_t groupsX = (m_drawExtent.width + 15) / 16;
    const std::uint32_t groupsY = (m_drawExtent.height + 15) / 16;
    vkCmdDispatch(cmd, groupsX, groupsY, 1);

    mp::debug::cmd_end_label(cmd);
}

void Engine::draw_ddgi_probe_vis(VkCommandBuffer cmd)
{
    if (m_mainDrawContext.ddgiVolumesVis.empty())
        return;

    mp::debug::cmd_begin_label(cmd, "DDGI Probe Visualization", {0.3f, 0.9f, 0.3f, 1.f});

    auto &currentFrame = get_current_frame();

    const auto colorAttachment =
        utils::attachment_info(currentFrame.drawImage.imageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    const auto depthAttachment =
        utils::depth_attachment(currentFrame.depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, false);
    const auto renderingInfo = utils::rendering_info(m_CommonImageExtent2D, 1, &colorAttachment, &depthAttachment);

    vkCmdBeginRendering(cmd, &renderingInfo);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_DDGIProbeVisPipeline);

    const VkDescriptorBufferBindingInfoEXT bindingInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
        .address = ddgiResourcesDescBuffer.get_device_address(),
        .usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT,
    };
    vkCmdBindDescriptorBuffersEXT(cmd, 1, &bindingInfo);
    const std::uint32_t setIndex = 0;
    const VkDeviceSize setOffset = 0;
    vkCmdSetDescriptorBufferOffsetsEXT(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_DDGIProbeVisPipelineLayout, 0, 1,
                                       &setIndex, &setOffset);

    const VkViewport viewport{.x = 0.f,
                              .y = static_cast<float>(m_CommonImageExtent2D.height),
                              .width = static_cast<float>(m_CommonImageExtent2D.width),
                              .height = -static_cast<float>(m_CommonImageExtent2D.height),
                              .minDepth = 0.f,
                              .maxDepth = 1.f};
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    const VkRect2D scissor{.offset = {0, 0}, .extent = m_CommonImageExtent2D};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindIndexBuffer(cmd, m_probeSphereIndexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

    for (const auto &entry : m_mainDrawContext.ddgiVolumesVis)
    {
        const auto probeCount = static_cast<std::uint32_t>(entry.volume.probeCounts.x * entry.volume.probeCounts.y *
                                                           entry.volume.probeCounts.z);

        const DDGIProbeVisPushConstants pc{
            .volumes = m_DDGIVolumesAddr,
            .sphereVertices = m_probeSphereVerticesAddr,
            .sceneData = currentFrame.sceneDataBufferAddr,
            .volumeIndex = entry.volumeIdx,
            .probeRadius = 0.05f,
            .visMode = entry.mode,
        };
        vkCmdPushConstants(cmd, m_DDGIProbeVisPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(DDGIProbeVisPushConstants), &pc);

        vkCmdDrawIndexed(cmd, m_probeSphereIndexCount, probeCount, 0, 0, 0);
    }

    vkCmdEndRendering(cmd);
    mp::debug::cmd_end_label(cmd);
}

} // namespace mp
