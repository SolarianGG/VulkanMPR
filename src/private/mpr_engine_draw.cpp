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
    const VkCommandBuffer &cmd = currentFrame.commandBuffer;
    const VkImage &swapchainImage = m_swapchainImages[swapchainImageIndex];

    constexpr VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };

    const AllocatedImage &currentDrawingImage = currentFrame.drawImage;
    const AllocatedImage &currentDepthImage = currentFrame.depthImage;
    const auto &gBuffer = currentFrame.gBuffer;
    m_drawExtent.width = std::min(currentDrawingImage.imageExtent.width, m_swapchainExtent.width) * m_renderScale;
    m_drawExtent.height = std::min(currentDrawingImage.imageExtent.height, m_swapchainExtent.height) * m_renderScale;

    vkBeginCommandBuffer(cmd, &beginInfo) >> chk;

    utils::BarrierBuilder barrierBuilder;
    m_stats.drawCallCount = 0;
    m_stats.triangleCount = 0;

    vkCmdBindIndexBuffer(cmd, m_globalIndexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

    // Draw prepass
    {

        barrierBuilder.add_image_barrier(
            currentDepthImage.image, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            utils::init_subresource_range(VK_IMAGE_ASPECT_DEPTH_BIT));
        barrierBuilder.barrier(cmd);
    }
    draw_prepass(cmd);

    // Compute minZ/maxZ from prepass depth
    {
        barrierBuilder.add_image_barrier(
            currentDepthImage.image, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_READ_BIT, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL, utils::init_subresource_range(VK_IMAGE_ASPECT_DEPTH_BIT));
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
        barrierBuilder.add_image_barrier(
            currentFrame.shadowPassDepthArray.image, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            utils::init_subresource_range(VK_IMAGE_ASPECT_DEPTH_BIT));
        barrierBuilder.add_buffer_barrier({
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = currentFrame.dirLightBuffer.buffer,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        });
        barrierBuilder.barrier(cmd);
    }
    draw_shadow_pass(cmd);

    // Base pass (GPass)
    {
        barrierBuilder.add_image_barrier(gBuffer.normal.image, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
                                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                         utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
        barrierBuilder.add_image_barrier(gBuffer.diffuse.image, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
                                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                         utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
        barrierBuilder.add_image_barrier(gBuffer.specular.image, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
                                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                         utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
        // Transition depth back from DEPTH_READ_ONLY (compute) to DEPTH_ATTACHMENT (GBuffer)
        barrierBuilder.add_image_barrier(
            currentDepthImage.image, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT, VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
            VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, utils::init_subresource_range(VK_IMAGE_ASPECT_DEPTH_BIT));
        barrierBuilder.barrier(cmd);
    }

    draw_gBuffer_pass(cmd);

    // Light pass
    {
        barrierBuilder.add_image_barrier(gBuffer.normal.image, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
                                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                         utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
        barrierBuilder.add_image_barrier(gBuffer.diffuse.image, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
                                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                         utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
        barrierBuilder.add_image_barrier(gBuffer.specular.image, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
                                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                         utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
        barrierBuilder.add_image_barrier(currentDrawingImage.image, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT_KHR,
                                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                         utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
        barrierBuilder.add_image_barrier(
            currentFrame.shadowPassDepthArray.image,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_READ_BIT, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL, utils::init_subresource_range(VK_IMAGE_ASPECT_DEPTH_BIT));
        barrierBuilder.add_image_barrier(
            currentDepthImage.image,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
            utils::init_subresource_range(VK_IMAGE_ASPECT_DEPTH_BIT));
        barrierBuilder.add_buffer_barrier({
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = currentFrame.dirLightBuffer.buffer,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        });

        barrierBuilder.barrier(cmd);
    }

    draw_light_pass(cmd);

    // forward WBOIT pass

    {
        barrierBuilder.add_image_barrier(currentFrame.oitAccImage.image, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
                                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                         utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
        barrierBuilder.add_image_barrier(currentFrame.oitRevealImage.image, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
                                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                         utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
        barrierBuilder.barrier(cmd);
    }

    draw_wboit(cmd);

    // composite weight blended OIT
    {
        barrierBuilder.add_image_barrier(
            currentFrame.oitAccImage.image, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
        barrierBuilder.add_image_barrier(
            currentFrame.oitRevealImage.image, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
        barrierBuilder.add_image_barrier(
            currentDrawingImage.image, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT_KHR,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
        barrierBuilder.barrier(cmd);
    }

    draw_wboit_composite(cmd);

    // Postprocess (gamma + tone)
    {
        barrierBuilder.add_image_barrier(currentDrawingImage.image, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
                                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                         VK_ACCESS_2_SHADER_WRITE_BIT_KHR | VK_ACCESS_2_SHADER_READ_BIT,
                                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                                         utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
        barrierBuilder.barrier(cmd);
    }

    draw_post(cmd);

    // copy to swapchain
    {
        barrierBuilder.add_image_barrier(currentDrawingImage.image, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                         VK_ACCESS_2_SHADER_WRITE_BIT_KHR | VK_ACCESS_2_SHADER_READ_BIT,
                                         VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                                         VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                         utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));

        barrierBuilder.add_image_barrier(swapchainImage, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                                         VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                         utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
        barrierBuilder.barrier(cmd);
    }

    utils::copy_to_image(cmd, currentDrawingImage.image, swapchainImage, m_drawExtent, m_swapchainExtent);

    // Imgui
    {
        barrierBuilder.add_image_barrier(swapchainImage, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                         VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                         VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,

                                         utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));

        barrierBuilder.barrier(cmd);
    }

    draw_imgui(cmd, m_swapchainImageViews[swapchainImageIndex]);

    {
        barrierBuilder.add_image_barrier(swapchainImage, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                         VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                         VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0,
                                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                         utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));

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

void Engine::draw_shadow_pass(VkCommandBuffer cmd)
{
    if (!m_mainDrawContext.dirLight.has_value())
        return;
    const auto &light = m_mainDrawContext.dirLight.value();
    const auto start = cn::steady_clock::now();
    constexpr VkExtent2D shadowPassExtent{2048, 2048};
    auto &currentFrame = get_current_frame();

    const std::uint32_t cascadeCount = static_cast<std::uint32_t>(std::clamp(light.cascadeCount.x, 1, MAX_CASCADES));

    cull_objects(cmd, m_OpaqueSize, 0, m_LightCullMatrix);

    const auto depthAttachment =
        utils::depth_attachment(currentFrame.shadowPassDepthArray.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
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

    const ShadowPassPushConstants shadowPassPushConstants{
        .globalVertexBufferAddr = m_globalVertexBufferAddress,
        .instanceBufferDeviceAddr = currentFrame.instanceBufferAddr,
        .dirLightsBufferAddr = currentFrame.dirLightBufferAddr,
        .cascadeCount = cascadeCount,
    };
    draw_meshes(cmd, m_ShadowPassPipelineLayout, m_ShadowPassPipeline, m_OpaqueSize, shadowPassPushConstants,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT, false);

    vkCmdEndRendering(cmd);

    const auto end = cn::steady_clock::now();
    const auto elapsed = cn::duration_cast<cn::milliseconds>(end - start);
    m_stats.shadowPassDrawTime = elapsed.count() / 1000.0f;
}

void Engine::draw_gBuffer_pass(VkCommandBuffer cmd)
{
    const auto start = cn::steady_clock::now();
    auto &gBuffer = get_current_frame().gBuffer;
    auto &depthImage = get_current_frame().depthImage;

    // ---
    VkClearValue val{.color = {0.0f, 0.0f, 0.0f, 1.0f}};
    const auto normalAttachment =
        utils::attachment_info(gBuffer.normal.imageView, &val, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    const auto diffuseAttachment =
        utils::attachment_info(gBuffer.diffuse.imageView, &val, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    const auto specularAttachment =
        utils::attachment_info(gBuffer.specular.imageView, &val, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    const auto depthAttachment =
        utils::depth_attachment(depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, false);

    VkRenderingAttachmentInfo attachments[]{normalAttachment, diffuseAttachment,
                                            specularAttachment};
    const auto renderInfo =
        utils::rendering_info(m_CommonImageExtent2D, std::size(attachments), attachments, &depthAttachment);

    cull_objects(cmd, m_OpaqueSize, 0, m_sceneData.projView);
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

    draw_meshes(cmd, m_metalRoughness.opaquePipeline.pipelineLayout, m_metalRoughness.opaquePipeline.pipeline,
                m_OpaqueSize, m_GBufferMeshPushConstants, VK_SHADER_STAGE_VERTEX_BIT);

    vkCmdEndRendering(cmd);
    const auto end = cn::steady_clock::now();
    const auto elapsed = cn::duration_cast<cn::milliseconds>(end - start);
    m_stats.gBufferPassTime = elapsed.count() / 1000.0f;
}

void Engine::draw_light_pass(const VkCommandBuffer cmd)
{
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

    vkCmdPushConstants(cmd, m_LightPassPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(LightPassConstantRange),
                       &m_LightPassConstants);

    vkCmdDispatch(cmd, std::ceil(m_CommonImageExtent2D.width / 16.0f), std::ceil(m_CommonImageExtent2D.height / 16.0f),
                  1);

    const auto end = cn::steady_clock::now();
    const auto elapsed = cn::duration_cast<cn::milliseconds>(end - start);
    m_stats.gBufferLightPassTime = elapsed.count() / 1000.0f;
}

void Engine::draw_wboit(VkCommandBuffer cmd)
{
    const auto start = cn::steady_clock::now();
    const auto &currentFrame = get_current_frame();

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

    cull_objects(cmd, m_TransparentSize, m_OpaqueSize, m_sceneData.projView);
    vkCmdBeginRendering(cmd, &renderInfo);
    draw_meshes(cmd, m_metalRoughness.transparentPipeline.pipelineLayout, m_metalRoughness.transparentPipeline.pipeline,
                m_TransparentSize, m_WBOITForwardPassPushConstants,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    vkCmdEndRendering(cmd);
    const auto end = cn::steady_clock::now();
    const auto elapsed = cn::duration_cast<cn::milliseconds>(end - start);
    m_stats.transparentForwardLightPassTime = elapsed.count() / 1000.0f;
}

void Engine::draw_wboit_composite(VkCommandBuffer cmd)
{
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
    const auto end = cn::steady_clock::now();
    const auto elapsed = cn::duration_cast<cn::milliseconds>(end - start);
    m_stats.postProcessPassTime = elapsed.count() / 1000.0f;
}

void Engine::draw_imgui(const VkCommandBuffer cmd, const VkImageView targetImageView)
{
    const auto start = cn::steady_clock::now();
    const auto colorAttachment =
        utils::attachment_info(targetImageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    const auto renderingInfo = utils::rendering_info(m_swapchainExtent, 1, &colorAttachment, nullptr);

    vkCmdBeginRendering(cmd, &renderingInfo);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

    vkCmdEndRendering(cmd);
    const auto end = cn::steady_clock::now();
    const auto elapsed = cn::duration_cast<cn::milliseconds>(end - start);
    m_stats.imguiDrawTime = elapsed.count() / 1000.0f;
}

void Engine::cull_objects(VkCommandBuffer cmd, const std::uint32_t objectCount, const std::uint32_t objectOffset,
                          const glm::mat4 &viewProj)
{
    auto &currentFrame = get_current_frame();
    utils::BarrierBuilder barrierBuilder;
    barrierBuilder.add_buffer_barrier({
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .srcAccessMask = 0,
        .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = currentFrame.countBuffer.buffer,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    });
    barrierBuilder.barrier(cmd);

    vkCmdFillBuffer(cmd, currentFrame.countBuffer.buffer, 0, VK_WHOLE_SIZE, 0);

    barrierBuilder.add_buffer_barrier({
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = currentFrame.countBuffer.buffer,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    });
    barrierBuilder.add_buffer_barrier({
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .srcAccessMask = 0,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = currentFrame.drawCommandsBuffer.buffer,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    });

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

    barrierBuilder.add_buffer_barrier({
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
        .dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = currentFrame.countBuffer.buffer,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    });
    barrierBuilder.add_buffer_barrier({
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
        .dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = currentFrame.drawCommandsBuffer.buffer,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    });

    barrierBuilder.barrier(cmd);
}

void Engine::draw_meshes(VkCommandBuffer cmd, const VkPipelineLayout drawPassPipelineLayout,
                         const VkPipeline drawPipeline, const std::uint32_t objectCount, auto &pushConstants,
                         const VkShaderStageFlags pushConstantsShaderStage, bool isMetalRoughness)
{
    auto &currentFrame = get_current_frame();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, drawPipeline);

    vkCmdPushConstants(cmd, drawPassPipelineLayout, pushConstantsShaderStage, 0, sizeof(pushConstants), &pushConstants);
    if (isMetalRoughness)
    {
        // Bind textures
        const VkDescriptorBufferBindingInfoEXT bindingInfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
            .address = m_metalRoughness.descriptors.get_device_address(),
            .usage =
                VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT};
        vkCmdBindDescriptorBuffersEXT(cmd, 1, &bindingInfo);

        const std::uint32_t bufferIndices[]{0};
        const VkDeviceSize offsets[]{0};
        vkCmdSetDescriptorBufferOffsetsEXT(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, drawPassPipelineLayout, 0,
                                           std::size(bufferIndices), bufferIndices, offsets);
    }

    vkCmdDrawIndexedIndirectCount(cmd, currentFrame.drawCommandsBuffer.buffer, 0, currentFrame.countBuffer.buffer, 0,
                                  objectCount, sizeof(VkDrawIndexedIndirectCommand));
    m_stats.drawCallCount++;
}

void Engine::draw_post(VkCommandBuffer cmd)
{
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

    vkCmdDispatch(cmd, std::ceil(m_CommonImageExtent2D.width / 16.0f), std::ceil(m_CommonImageExtent2D.height / 16.0f),
                  1.0f);
}

void Engine::draw_prepass(VkCommandBuffer cmd)
{
    auto &currentFrame = get_current_frame();

    cull_objects(cmd, m_OpaqueSize, 0, m_sceneData.projView);

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

    draw_meshes(cmd, m_PrepassPipelineLayout, m_PrepassPipeline, m_OpaqueSize, m_GBufferMeshPushConstants,
                VK_SHADER_STAGE_VERTEX_BIT);

    vkCmdEndRendering(cmd);
}

void Engine::compute_depth_reduction(VkCommandBuffer cmd)
{
    auto &frame = get_current_frame();

    MinMax initMinMax;
    initMinMax.min = std::bit_cast<std::uint32_t>(std::numeric_limits<float>::max());
    initMinMax.max = std::bit_cast<std::uint32_t>(-std::numeric_limits<float>::max());
    vkCmdUpdateBuffer(cmd, frame.minMaxBuffer.buffer, 0, sizeof(MinMax), &initMinMax);

    utils::BarrierBuilder barrierBuilder;
    barrierBuilder.add_buffer_barrier({
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = frame.minMaxBuffer.buffer,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    });
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

    constexpr float cameraNear = 0.1f;
    constexpr float cameraFar = 100.0f;
    const DepthReductionPushConstants pushConstants{
        .minMaxAddr = frame.minMaxBufferAddr,
        .cameraNear = cameraNear,
        .cameraFar = cameraFar,
    };
    vkCmdPushConstants(cmd, m_DepthReductionPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(DepthReductionPushConstants), &pushConstants);

    vkCmdDispatch(cmd, static_cast<std::uint32_t>(std::ceil(m_CommonImageExtent2D.width / 16.0f)),
                  static_cast<std::uint32_t>(std::ceil(m_CommonImageExtent2D.height / 16.0f)), 1);
}

void Engine::compute_depth_partition(VkCommandBuffer cmd)
{
    if (!m_mainDrawContext.dirLight.has_value())
        return;
    auto &frame = get_current_frame();

    // Reset splitsAABB: min* = FLT_MAX, max* = -FLT_MAX
    CascadesAABB initAABB{};
    for (auto &aabb : initAABB.bounds)
    {
        aabb.minX = aabb.minY = aabb.minZ = std::bit_cast<std::uint32_t>(std::numeric_limits<float>::max());
        aabb.maxX = aabb.maxY = aabb.maxZ = std::bit_cast<std::uint32_t>(-std::numeric_limits<float>::max());
        aabb._pad0 = aabb._pad1 = 0;
    }
    constexpr std::size_t dirLightCount = 1;
    vkCmdUpdateBuffer(cmd, frame.splitsAABBBuffer.buffer, 0, sizeof(CascadesAABB), &initAABB);

    utils::BarrierBuilder barrierBuilder;
    // minMax: COMPUTE WRITE -> COMPUTE READ
    barrierBuilder.add_buffer_barrier({
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = frame.minMaxBuffer.buffer,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    });
    // splitsAABB: TRANSFER WRITE -> COMPUTE READ|WRITE
    barrierBuilder.add_buffer_barrier({
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = frame.splitsAABBBuffer.buffer,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    });
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

    constexpr float cameraNear = 0.1f;
    constexpr float cameraFar = 100.0f;
    const DepthPartitionPushConstants pushConstants{
        .minMaxAddr = frame.minMaxBufferAddr,
        .splitsAABBAddr = frame.splitsAABBBufferAddr,
        .dirLightsAddr = frame.dirLightBufferAddr,
        .dirLightCount = static_cast<std::uint32_t>(dirLightCount),
        .near = cameraNear,
        .far = cameraFar,
        .inverseCameraViewProj = glm::inverse(m_sceneData.projView),
    };
    vkCmdPushConstants(cmd, m_DepthPartitionPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(DepthPartitionPushConstants), &pushConstants);

    vkCmdDispatch(cmd, static_cast<std::uint32_t>(std::ceil(m_CommonImageExtent2D.width / 16.0f)),
                  static_cast<std::uint32_t>(std::ceil(m_CommonImageExtent2D.height / 16.0f)), 1);
}

void Engine::compute_dir_lights_vp(VkCommandBuffer cmd)
{
    if (!m_mainDrawContext.dirLight.has_value())
        return;
    auto &frame = get_current_frame();

    // splitsAABB and dirLight (splitDistances written by partition): COMPUTE WRITE -> COMPUTE READ
    utils::BarrierBuilder barrierBuilder;
    barrierBuilder.add_buffer_barrier({
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = frame.splitsAABBBuffer.buffer,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    });
    barrierBuilder.add_buffer_barrier({
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = frame.dirLightBuffer.buffer,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    });
    barrierBuilder.barrier(cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_DirVpPipeline);

    const DirVpPushConstants pushConstants{
        .splitsAABBAddr = frame.splitsAABBBufferAddr,
        .dirLightsAddr = frame.dirLightBufferAddr,
        .sceneMin = m_mainDrawContext.min,
        .sceneMax = m_mainDrawContext.max,
        .shadowMapSize = 2048u,
    };
    vkCmdPushConstants(cmd, m_DirVpPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DirVpPushConstants),
                       &pushConstants);

    vkCmdDispatch(cmd, 1, 1, 1);
}

void Engine::copy_frame_buffers()
{
    m_OpaqueSize = static_cast<std::uint32_t>(m_mainDrawContext.opaqueInstances.size());
    const std::uint32_t opaqueByteSize = m_OpaqueSize * sizeof(Instance);
    std::memcpy(m_CurrentFrameInstanceBuffer, m_mainDrawContext.opaqueInstances.data(), opaqueByteSize);

    m_TransparentSize = static_cast<std::uint32_t>(m_mainDrawContext.transparentInstances.size());
    const std::uint32_t transparentByteSize = m_TransparentSize * sizeof(Instance);
    std::memcpy(m_CurrentFrameInstanceBuffer + m_OpaqueSize, m_mainDrawContext.transparentInstances.data(),
                transparentByteSize);

    std::memcpy(m_CurrentMeshBuffer, m_mainDrawContext.renderObjects.data(),
                m_mainDrawContext.renderObjects.size() * sizeof(RenderObject));
}

} // namespace mp
