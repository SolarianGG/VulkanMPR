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
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <print>
#include <random>
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

std::pair<float, float> compute_alpha_beta()
{
    constexpr float hFOV_orig = 143.98570868f;
    constexpr float vFOV_orig = 125.26438968f;
    constexpr float r = 0.0625f;

    const float aspect = glm::tan(glm::radians(hFOV_orig / 2.0f)) / glm::tan(glm::radians(vFOV_orig / 2.0f));
    const glm::mat4 projA = glm::perspectiveRH_ZO(glm::radians(vFOV_orig), aspect, mp::kPointLightNear, 1.0f);
    const glm::mat4 invProjA = glm::inverse(projA);

    glm::vec3 centers[4] = {
        {-1.0f, 0.0f, -1.0f},
        {1.0f, 0.0f, -1.0f},
        {0.0f, -1.0f, -1.0f},
        {0.0f, 1.0f, -1.0f},
    };
    const glm::vec3 offsets[4] = {{-r, 0, 0}, {r, 0, 0}, {0, -r, 0}, {0, r, 0}};
    glm::vec3 v[4];
    for (int i = 0; i < 4; ++i)
    {
        centers[i] += offsets[i];
        const glm::vec4 vs = invProjA * glm::vec4(centers[i], 1.0f);
        v[i] = glm::normalize(glm::vec3(vs));
    }

    const float dilatedFovX = glm::degrees(glm::acos(glm::dot(v[0], v[1])));
    const float dilatedFovY = glm::degrees(glm::acos(glm::dot(v[2], v[3])));
    return {dilatedFovX - hFOV_orig, dilatedFovY - vFOV_orig};
}

mp::TetrahedronData compute_tetrahedron_data()
{
    const auto [alpha, beta] = compute_alpha_beta();

    const glm::vec3 faceVecs[4] = {
        {0.0f, -0.57735026f, 0.81649661f},
        {0.0f, -0.57735026f, -0.81649661f},
        {-0.81649661f, 0.57735026f, 0.0f},
        {0.81649661f, 0.57735026f, 0.0f},
    };
    const glm::vec3 corners[4] = {-faceVecs[0], -faceVecs[1], -faceVecs[2], -faceVecs[3]};

    mp::TetrahedronData data{};
    for (int i = 0; i < 4; ++i)
        data.faceVectors[i] = glm::vec4(faceVecs[i], 0.0f);

    int planeIdx = 0;
    for (int i = 0; i < 4; ++i)
    {
        int neighbors[3];
        int ni = 0;
        for (int j = 0; j < 4; ++j)
            if (j != i)
                neighbors[ni++] = j;

        for (int s = 0; s < 3; ++s)
        {
            int j = neighbors[s];
            glm::vec3 edgeCorners[2];
            int ec = 0;
            for (int k = 0; k < 4; ++k)
                if (k != i && k != j)
                    edgeCorners[ec++] = corners[k];

            glm::vec3 n = glm::normalize(glm::cross(edgeCorners[0], edgeCorners[1]));
            if (glm::dot(n, faceVecs[i]) < 0.0f)
                n = -n;
            const glm::vec3 rotAxis = glm::normalize(glm::cross(faceVecs[i], faceVecs[j]));
            n = glm::rotate(glm::angleAxis(glm::radians(alpha), rotAxis), n);

            data.planeNormals[planeIdx++] = glm::vec4(glm::normalize(n), 0.0f);
        }
    }

    return data;
}

void compute_tetrahedron_shadow_matrices(mp::PointLightData &pointLight, const std::uint32_t lightIndex)
{
    static constexpr std::array<glm::vec3, 4> kFaceVecs{
        glm::vec3{0.0f, -0.57735026f, 0.81649661f},
        glm::vec3{0.0f, -0.57735026f, -0.81649661f},
        glm::vec3{-0.81649661f, 0.57735026f, 0.0f},
        glm::vec3{0.81649661f, 0.57735026f, 0.0f},
    };
    // Up vectors perpendicular to each face direction
    static std::array<glm::vec3, 4> kFaceUps{
        glm::vec3{0.0f, 1.0f, 0.0f},
        glm::vec3{1.0f, 0.0f, 0.0f},
        glm::normalize(glm::vec3{-1.0f, 0.0f, -1.0f}),
        glm::vec3{0.0f, 0.0f, 1.0f},
    };
    static const auto [alpha, beta] = compute_alpha_beta();

    const float hFOV_AC = 143.98570868f + alpha;
    const float vFOV_AC = 125.26438968f + beta;
    const float aspect_AC = glm::tan(glm::radians(hFOV_AC / 2.0f)) / glm::tan(glm::radians(vFOV_AC / 2.0f));
    const glm::mat4 projAC =
        glm::perspectiveRH_ZO(glm::radians(vFOV_AC), aspect_AC, mp::kPointLightNear, pointLight.range);

    const float hFOV_BD = 125.26438968f + beta;
    const float vFOV_BD = 143.98570868f + alpha;
    const float aspect_BD = glm::tan(glm::radians(hFOV_BD / 2.0f)) / glm::tan(glm::radians(vFOV_BD / 2.0f));
    const glm::mat4 projBD =
        glm::perspectiveRH_ZO(glm::radians(vFOV_BD), aspect_BD, mp::kPointLightNear, pointLight.range);

    const glm::mat4 projMatrices[4] = {projAC, projBD, projAC, projBD};

    constexpr float s = static_cast<float>(mp::kPointLightTileSize) / static_cast<float>(mp::kPointLightsShadowMapSize);
    const float px = (lightIndex % mp::kPointLightTilesPerRow) * s;
    const float py = (lightIndex / mp::kPointLightTilesPerRow) * s;

    // Tile center in NDC [-1,1] (viewport uses negative height for Y-flip)
    const float cx = 2.0f * px + s - 1.0f;
    const float cy = 1.0f - 2.0f * py - s;

    // 4 faces arranged in a cross pattern within the tile:
    // A,C span full width (scale_x=s) and half height (scale_y=s/2)
    // B,D span half width (scale_x=s/2) and full height (scale_y=s)
    constexpr float firstModifier = 1.07f;
    constexpr float secondModifier = 2.93f;
    glm::mat4 genMatrices[4]{};
    {
        glm::mat4 matrix(1.0f);
        matrix[0][0] = s / firstModifier;
        matrix[1][1] = s / secondModifier;
        matrix[3][0] = cx;
        matrix[3][1] = cy - s / 2.0f;

        genMatrices[0] = matrix;
    }
    {
        glm::mat4 matrix(1.0f);
        matrix[0][0] = s / secondModifier;
        matrix[1][1] = s / firstModifier;
        matrix[3][0] = cx + s / 2.0f;
        matrix[3][1] = cy;

        genMatrices[1] = matrix;
    }
    {
        glm::mat4 matrix(1.0f);
        matrix[0][0] = s / firstModifier;
        matrix[1][1] = s / secondModifier;
        matrix[3][0] = cx;
        matrix[3][1] = cy + s / 2.0f;

        genMatrices[2] = matrix;
    }
    {
        glm::mat4 matrix(1.0f);
        matrix[0][0] = s / secondModifier;
        matrix[1][1] = s / firstModifier;
        matrix[3][0] = cx - s / 2.0f;
        matrix[3][1] = cy;

        genMatrices[3] = matrix;
    }
    for (int i = 0; i < 4; ++i)
    {
        const glm::mat4 viewMatrix =
            glm::lookAtRH(pointLight.position, pointLight.position + kFaceVecs[i], kFaceUps[i]);
        pointLight.tetrahedronFacesMatrices[i] = genMatrices[i] * projMatrices[i] * viewMatrix;
    }
}
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

    // Create tetrahedron data buffer for point light shadows
    {
        TetrahedronData tetraData = compute_tetrahedron_data();
        m_tetrahedronBuffer = create_buffer(
            sizeof(TetrahedronData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU);
        std::memcpy(m_tetrahedronBuffer.allocationInfo.pMappedData, &tetraData, sizeof(tetraData));
        m_mainDeletionQueue.push_function([this] { destroy_buffer(m_tetrahedronBuffer); });
    }

    m_camera.velocity = glm::vec3(0.0f);
    m_camera.position = glm::vec3(0.0f, 0.0f, 5.0f);

    m_camera.pitch = 0;
    m_camera.yaw = 0;
    m_isInitialized = true;

    m_LightPassConstants.dirNormalBias = 0.001f;
    m_LightPassConstants.dirConstantBias = 0.001f;
    m_LightPassConstants.pointNormalBias = 0.03f;
    m_LightPassConstants.pointConstantBias = 0.002f;
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
            ImGui::DragFloat("Camera speed", &m_camera.cameraSpeed, 0.01f, 0.01f, 100.0f);
            ImGui::DragFloat("Dir normal bias", &m_LightPassConstants.dirNormalBias, 0.001f, 0.001f, 1.0f);
            ImGui::DragFloat("Dir constant bias", &m_LightPassConstants.dirConstantBias, 0.001f, 0.001f, 1.0f);
            ImGui::DragFloat("Point normal bias", &m_LightPassConstants.pointNormalBias, 0.001f, 0.001f, 1.0f);
            ImGui::DragFloat("Point constant bias", &m_LightPassConstants.pointConstantBias, 0.001f, 0.001f, 1.0f);
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

    const glm::mat4 proj = glm::perspectiveRH_ZO(
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
        m_DirLightViewMatrix = glm::lookAtRH(lightPos, sceneCenter, up);

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

        m_DirLightCullMatrix =
            glm::orthoRH_ZO(lsMin.x, lsMax.x, lsMin.y, lsMax.y, -lsMax.z, -lsMin.z) * light.cascadeVPs[0];
    }

    auto &frame = get_current_frame();
    auto *sceneData = static_cast<GpuSceneData *>(frame.sceneDataBuffer.allocationInfo.pMappedData);
    *sceneData = m_sceneData;

    if (m_mainDrawContext.dirLight.has_value())
    {
        std::memcpy(frame.dirLightBuffer.allocationInfo.pMappedData, &m_mainDrawContext.dirLight.value(),
                    sizeof(DirectionalLightData));
    }
    for (std::uint32_t i = 0; i < m_mainDrawContext.pointLights.size(); ++i)
    {
        compute_tetrahedron_shadow_matrices(m_mainDrawContext.pointLights[i], i);
    }
    std::memcpy(frame.pointLightBuffer.allocationInfo.pMappedData, m_mainDrawContext.pointLights.data(),
                m_mainDrawContext.pointLights.size() * sizeof(PointLightData));

    m_LightPassConstants.sceneData = frame.sceneDataBufferAddr;
    m_LightPassConstants.directionalLight = frame.dirLightBufferAddr;
    m_LightPassConstants.visiblePointLights = frame.visiblePointLightsBufferAddr;
    m_LightPassConstants.visiblePointLightsCount = frame.visiblePointLightsCountBufferAddr;
    m_LightPassConstants.tetrahedronDataAddr = m_tetrahedronBuffer.get_buffer_device_address(m_device);
    m_LightPassConstants.cameraNear = cameraNear, m_LightPassConstants.cameraFar = cameraFar;
    m_LightPassConstants.inverseCameraViewProj = inverseViewProj;

    m_GBufferMeshPushConstants = {
        .globalVertexBufferAddr = m_globalVertexBufferAddress,
        .instanceBufferDeviceAddr = frame.instanceBufferAddr,
        .sceneDataBufferDeviceAddr = frame.sceneDataBufferAddr,
    };

    m_WBOITForwardPassPushConstants = {.vertices = m_globalVertexBufferAddress,
                                       .instances = frame.instanceBufferAddr,
                                       .sceneData = frame.sceneDataBufferAddr,
                                       .directionalLight = frame.dirLightBufferAddr,
                                       .visiblePointLights = frame.visiblePointLightsBufferAddr,
                                       .visiblePointLightsCount = frame.visiblePointLightsCountBufferAddr};
    const auto end = cn::steady_clock::now();
    const auto elapsed = cn::duration_cast<cn::milliseconds>(end - start);

    m_stats.sceneUpdateTime = elapsed.count() / 1000.0f;
}

} // namespace mp
