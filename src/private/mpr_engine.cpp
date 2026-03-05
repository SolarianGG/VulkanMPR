// clang-format off
#define GLM_ENABLE_EXPERIMENTAL
#define VMA_IMPLEMENTATION
#define VOLK_IMPLEMENTATION
#include "mpr_engine.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <VkBootstrap.h>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <ranges>
#include <thread>

#include <vk_mem_alloc.h>

#include "mpr_error_check.hpp"
#include "mpr_image.hpp"
#include "mpr_init_vk_stucts.hpp"
#include "mpr_loader.hpp"
#include "mpr_pipelines.hpp"
// clang-format on

using namespace std::chrono_literals;
namespace cn = std::chrono;

namespace
{
mp::Engine *gLoadedEngine = nullptr;
} // namespace

namespace mp
{

Engine::~Engine()
{
    if (m_isInitialized)
    {
        vkDeviceWaitIdle(m_device);
        m_scene.clear_all(*this);
        for (auto &frame : m_frameData)
        {
            frame.frameDeletionQueue.flush();
        }
        m_mainDeletionQueue.flush();
        destroy_sync();
        destroy_commands();

        destroy_swapchain();

        vkDestroyDevice(m_device, nullptr);
        vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
        vkb::destroy_debug_utils_messenger(m_instance, m_debugMessenger);
        vkDestroyInstance(m_instance, nullptr);
    }
    gLoadedEngine = nullptr;
}

Engine::Engine()
{
    assert(!gLoadedEngine);
    gLoadedEngine = this;
    init_window();
    init_vulkan();
    init_swapchain();
    init_commands();
    init_sync();
    init_descriptors();
    init_pipelines();
    init_imgui();
    init_frames_data();
    init_default_data();
    init_mesh_data();

    m_camera.velocity = glm::vec3(0.0f);
    m_camera.position = glm::vec3(0.0f, 0.0f, 5.0f);

    m_camera.pitch = 0;
    m_camera.yaw = 0;
    m_isInitialized = true;
}

Engine &Engine::get()
{
    return *gLoadedEngine;
}

void Engine::run()
{
    SDL_Event e;
    bool bIsRunning = true;

    while (bIsRunning)
    {
        auto start = cn::steady_clock::now();
        while (SDL_PollEvent(&e))
        {
            if (e.type == SDL_EVENT_QUIT)
            {
                bIsRunning = false;
            }

            if (e.type == SDL_EVENT_WINDOW_MINIMIZED)
            {
                m_isRenderStopped = true;
            }

            if (e.type == SDL_EVENT_WINDOW_MAXIMIZED)
            {
                m_isRenderStopped = false;
            }
            m_camera.process_sdl_event(e, m_window.get());
            ImGui_ImplSDL3_ProcessEvent(&e);

            if (ImGui::GetIO().WantCaptureMouse || ImGui::GetIO().WantCaptureKeyboard)
                continue;
        }

        if (m_bSwapchainResizeRequest)
        {
            resize_swapchain();
        }

        if (m_isRenderStopped)
        {
            std::this_thread::sleep_for(100ms);
            continue;
        }

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        // ImGui UI
        if (ImGui::Begin("Other"))
        {
            ImGui::DragFloat("Render scale", &m_renderScale, 0.01f, 0.01f, 1.0f);
            ImGui::DragFloat("Camera speed", &m_camera.cameraSpeed, 0.01f, 0.01f, 100.0f);
            ImGui::DragFloat("Light distance", &m_LightDistance, 0.01f, 0.01f, 100.0f);
            // TODO: Add debug light visualization
#if 0
      ImGui::Checkbox("Draw debug light positions", &m_IsLightsRendered);
#endif
        }
        ImGui::End();

        // TODO: These stats are only showing cpu execution time of vulkan commands,
        // for gpu metrics I plan to integrate tracy
        ImGui::Begin("Stats");
        ImGui::Text("Frame time: %f ms", m_stats.frameTime);
        ImGui::Text("Shadow Pass time: %f ms", m_stats.shadowPassDrawTime);
        ImGui::Text("GBuffer Pass time: %f ms", m_stats.gBufferPassTime);
        ImGui::Text("Deferred light pass time: %f ms", m_stats.gBufferLightPassTime);
        ImGui::Text("WBOIT forward pass time: %f ms", m_stats.transparentForwardLightPassTime);
        ImGui::Text("Post process pass time: %f ms", m_stats.postProcessPassTime);
        ImGui::Text("ImGui draw time: %f ms", m_stats.imguiDrawTime);
        ImGui::Text("Scene update tim: %f ms", m_stats.sceneUpdateTime);
        ImGui::Text("Amount of draw calls: %i", m_stats.drawCallCount);
        ImGui::Text("Amount of triangles: %i", m_stats.triangleCount);
        ImGui::End();

        {
            const ImGuiViewport *v = ImGui::GetMainViewport();
            ImGui::SetNextWindowPos(ImVec2(10, 200));
            ImGui::SetNextWindowSize(ImVec2(v->WorkSize.x / 6, v->WorkSize.y - 210));
            ImGui::Begin("Scene graph", nullptr,
                         ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize);
            ImGui::Separator();
            for (const auto &topNode : m_scene.topNodes)
            {
                m_selectedNode = render_scene_tree_ui(m_scene, topNode->nodeIndex, m_selectedNode);
            }
            ImGui::Separator();
            if (ImGui::Button("Add light"))
            {
                const auto nodeIndex = m_scene.add_node(std::make_shared<DirectionalLightNode>(DirectionalLightData{
                    .direction = {0.0f, -1.0f, 0.0f},
                    .padding = 0.0f,
                    .color = {1.0f, 1.0f, 1.0f},
                    .intensity = 1.0f,
                    .cascadeCount = {4, 0, 0, 0},
                }));

                auto &node = m_scene.nodes.find(nodeIndex)->second;
                node->worldTransform = glm::mat4(1.0f);
                node->localTransform = glm::mat4(1.0f);
                node->name = "Light";
                node->nodeIndex = nodeIndex;
                m_scene.topNodes.push_back(node);
            }
            ImGui::End();
        }

        if (m_selectedNode != UINT64_MAX)
        {
            edit_node(m_scene, m_selectedNode);
        }
        // ImGui UI end

        ImGui::Render();

        draw();

        auto end = cn::steady_clock::now();
        auto elapsed = cn::duration_cast<cn::milliseconds>(end - start);
        m_stats.frameTime = elapsed.count() / 1000.0f;
    }
}

FrameData &Engine::get_current_frame()
{
    return m_frameData[m_frameNumber % kNumberOfFrames];
}

void Engine::WindowCleaner::operator()(SDL_Window *window) const
{
    SDL_DestroyWindow(window);
}

void Engine::update_scene()
{
    const auto start = cn::steady_clock::now();
    m_camera.update(m_stats.frameTime);
    m_mainDrawContext.clear();
    m_CurrentFrameInstanceBuffer =
        static_cast<Instance *>(get_current_frame().instanceBuffer.allocationInfo.pMappedData);
    m_CurrentMeshBuffer = static_cast<RenderObject *>(get_current_frame().meshesBuffer.allocationInfo.pMappedData);

    m_scene.draw(glm::mat4(1.0f), m_mainDrawContext);

    copy_frame_buffers();

    constexpr float cameraNear = 0.1f;
    constexpr float cameraFar = 100.0f;
    const glm::mat4 proj = glm::perspectiveRH_ZO(
        glm::radians(90.0f), static_cast<float>(m_drawExtent.width) / m_drawExtent.height, cameraNear, cameraFar);

    m_sceneData.view = m_camera.get_view_matrix();
    m_sceneData.proj = proj;
    m_sceneData.projView = proj * m_sceneData.view;
    m_sceneData.cameraPos = m_camera.position;
    m_sceneData.padding0 = 0.0f;

    const auto inverseViewProj = glm::inverse(m_sceneData.projView);
    std::array<glm::vec3, 8> frustumCorners{
        glm::vec3{-1.0f, -1.0f, 0.0f}, {1.0f, -1.0f, 0.0f}, {-1.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 0.0f},
        {-1.0f, -1.0f, 1.0f},          {1.0f, -1.0f, 1.0f}, {-1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f},
    };
    for (auto &corner : frustumCorners)
    {
        const glm::vec4 inversedCorner = inverseViewProj * glm::vec4(corner, 1.0f);
        corner = inversedCorner / inversedCorner.w;
    }

    const auto frustumCenter = (frustumCorners[0] + frustumCorners[1] + frustumCorners[2] + frustumCorners[3] +
                                frustumCorners[4] + frustumCorners[5] + frustumCorners[6] + frustumCorners[7]) /
                               8.0f;

    if (!m_mainDrawContext.dirLights.empty())
    {
        auto &light = m_mainDrawContext.dirLights.front();
        const auto lightDir = glm::normalize(light.direction);
        const glm::vec3 up = (std::abs(glm::dot(lightDir, glm::vec3(0.0f, 1.0f, 0.0f))) > 0.99f)
                                 ? glm::vec3(0.0f, 0.0f, 1.0f)
                                 : glm::vec3(0.0f, 1.0f, 0.0f);
        // Store light view matrix in cascadeVPs[0]; partition shader uses it
        const auto lightPos = frustumCenter - lightDir * m_LightDistance;
        light.cascadeVPs[0] = glm::lookAtRH(lightPos, lightPos + lightDir, up);
    }

    auto &frame = get_current_frame();
    auto *sceneData = static_cast<GpuSceneData *>(frame.sceneDataBuffer.allocationInfo.pMappedData);
    *sceneData = m_sceneData;

    std::memcpy(frame.dirLightBuffer.allocationInfo.pMappedData, m_mainDrawContext.dirLights.data(),
                m_mainDrawContext.dirLights.size() * sizeof(DirectionalLightData));
    std::memcpy(frame.pointLightBuffer.allocationInfo.pMappedData, m_mainDrawContext.pointLights.data(),
                m_mainDrawContext.pointLights.size() * sizeof(PointLightData));

    m_LightPassConstants = {
        .sceneDataBufferDeviceAddr = frame.sceneDataBufferAddr,
        .dirLightBufferDeviceAddr = frame.dirLightBufferAddr,
        .dirLightCount = static_cast<std::uint32_t>(m_mainDrawContext.dirLights.size()),
        .pointLightBufferDeviceAddr = frame.pointLightBufferAddr,
        .pointLightCount = static_cast<std::uint32_t>(m_mainDrawContext.pointLights.size()),
        .inverseCameraViewProj = inverseViewProj,
    };

    m_GBufferMeshPushConstants = {
        .globalVertexBufferAddr = m_globalVertexBufferAddress,
        .instanceBufferDeviceAddr = frame.instanceBufferAddr,
        .sceneDataBufferDeviceAddr = frame.sceneDataBufferAddr,
    };

    m_WBOITForwardPassPushConstants = {
        .globalVertexBufferAddr = m_globalVertexBufferAddress,
        .instanceBufferDeviceAddr = frame.instanceBufferAddr,
        .sceneDataBufferDeviceAddr = frame.sceneDataBufferAddr,
        .dirLightBufferDeviceAddr = frame.dirLightBufferAddr,
        .dirLightCount = static_cast<std::uint32_t>(m_mainDrawContext.dirLights.size()),
        .pointLightBufferDeviceAddr = frame.pointLightBufferAddr,
        .pointLightCount = static_cast<std::uint32_t>(m_mainDrawContext.pointLights.size()),
    };
    const auto end = cn::steady_clock::now();
    const auto elapsed = cn::duration_cast<cn::milliseconds>(end - start);

    m_stats.sceneUpdateTime = elapsed.count() / 1000.0f;
}

} // namespace mp
