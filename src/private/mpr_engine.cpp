// clang-format off
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
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
            // TODO: Add debug light visualization
#if 0
      ImGui::Checkbox("Draw debug light positions", &m_IsLightsRendered);
#endif
        }
        ImGui::End();

        // TODO: These stats are only showing cpu execution time of vulkan commands,
        // for gpu metrics I plan to integrate tracy
        ImGui::Begin("Stats");
        ImGui::Text("Frame time: %f s", m_stats.frameTime);
        ImGui::Text("Shadow Pass time: %f s", m_stats.shadowPassDrawTime);
        ImGui::Text("GBuffer Pass time: %f s", m_stats.gBufferPassTime);
        ImGui::Text("Deferred light pass time: %f s", m_stats.gBufferLightPassTime);
        ImGui::Text("WBOIT forward pass time: %f s", m_stats.transparentForwardLightPassTime);
        ImGui::Text("Post process pass time: %f s", m_stats.postProcessPassTime);
        ImGui::Text("ImGui draw time: %f s", m_stats.imguiDrawTime);
        ImGui::Text("Scene update tim: %f s", m_stats.sceneUpdateTime);
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
            if (!m_mainDrawContext.dirLight.has_value() && ImGui::Button("Add Directional light"))
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
                node->name = "Directional Light";
                node->nodeIndex = nodeIndex;
                m_scene.topNodes.push_back(node);
            }
            if (ImGui::Button("Add Point light"))
            {
                const auto nodeIndex = m_scene.add_node(std::make_shared<PointLightNode>(
                    PointLightData{.position = (m_mainDrawContext.max + m_mainDrawContext.min) * 0.5f,
                                   .range = 10.0f,
                                   .color = glm::vec3(1.0f, 1.0f, 1.0f),
                                   .intensity = 3.0f}));

                auto &node = m_scene.nodes.find(nodeIndex)->second;
                node->worldTransform = glm::mat4(1.0f);
                node->localTransform = glm::mat4(1.0f);
                node->name = "Point Light";
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
    const glm::mat4 proj = glm::perspective(
        glm::radians(90.0f), static_cast<float>(m_drawExtent.width) / m_drawExtent.height, cameraNear, cameraFar);

    m_sceneData.view = m_camera.get_view_matrix();
    m_sceneData.proj = proj;
    m_sceneData.projView = proj * m_sceneData.view;
    m_sceneData.cameraPos = m_camera.position;
    m_sceneData.padding0 = 0.0f;

    const auto inverseViewProj = glm::inverse(m_sceneData.projView);

    m_mainDrawContext.max += 1.0f;
    m_mainDrawContext.min -= 1.0f;
    const auto sceneDistance = (m_mainDrawContext.max) - (m_mainDrawContext.min);
    const auto sceneMaxDistance = glm::max(sceneDistance.x, glm::max(sceneDistance.y, sceneDistance.z));
    if (m_mainDrawContext.dirLight.has_value())
    {
        auto &light = m_mainDrawContext.dirLight.value();
        const auto lightDir = glm::normalize(light.direction);
        glm::vec3 up = (std::abs(glm::dot(lightDir, glm::vec3(0.0f, 1.0f, 0.0f))) > 0.99f)
                           ? glm::vec3(0.0f, 0.0f, 1.0f)
                           : glm::vec3(0.0f, 1.0f, 0.0f);

        const glm::vec3 sceneCenter = (m_mainDrawContext.max + m_mainDrawContext.min) * 0.5f;
        const auto lightPos = sceneCenter - lightDir * sceneMaxDistance;
        light.cascadeVPs[0] = m_DirLightViewMatrix =  glm::lookAt(lightPos, sceneCenter, up);

        glm::vec3 lsMin(FLT_MAX);
        glm::vec3 lsMax(-FLT_MAX);
        for (int cx = 0; cx <= 1; cx++)
            for (int cy = 0; cy <= 1; cy++)
                for (int cz = 0; cz <= 1; cz++)
                {
                    const glm::vec3 worldPt(cx ? m_mainDrawContext.max.x : m_mainDrawContext.min.x,
                                            cy ? m_mainDrawContext.max.y : m_mainDrawContext.min.y,
                                            cz ? m_mainDrawContext.max.z : m_mainDrawContext.min.z);
                    const glm::vec3 ls = glm::vec3(light.cascadeVPs[0] * glm::vec4(worldPt, 1.0f));
                    lsMin = glm::min(lsMin, ls);
                    lsMax = glm::max(lsMax, ls);
                }

        m_DirLightCullMatrix = glm::ortho(lsMin.x, lsMax.x, lsMin.y, lsMax.y, -lsMax.z, -lsMin.z) * light.cascadeVPs[0];
    }

    auto &frame = get_current_frame();
    auto *sceneData = static_cast<GpuSceneData *>(frame.sceneDataBuffer.allocationInfo.pMappedData);
    *sceneData = m_sceneData;

    if (m_mainDrawContext.dirLight.has_value())
    {
        std::memcpy(frame.dirLightBuffer.allocationInfo.pMappedData, &m_mainDrawContext.dirLight.value(),
                    sizeof(DirectionalLightData));
    }
    std::memcpy(frame.pointLightBuffer.allocationInfo.pMappedData, m_mainDrawContext.pointLights.data(),
                m_mainDrawContext.pointLights.size() * sizeof(PointLightData));

    m_LightPassConstants = {
        .sceneDataBufferDeviceAddr = frame.sceneDataBufferAddr,
        .dirLightBufferDeviceAddr = frame.dirLightBufferAddr,
        .dirLightCount = m_mainDrawContext.dirLight.has_value() ? 1u : 0u,
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
        .dirLightCount = m_mainDrawContext.dirLight.has_value() ? 1u : 0u,
        .pointLightBufferDeviceAddr = frame.pointLightBufferAddr,
        .pointLightCount = static_cast<std::uint32_t>(m_mainDrawContext.pointLights.size()),
    };
    const auto end = cn::steady_clock::now();
    const auto elapsed = cn::duration_cast<cn::milliseconds>(end - start);

    m_stats.sceneUpdateTime = elapsed.count() / 1000.0f;
}

} // namespace mp
