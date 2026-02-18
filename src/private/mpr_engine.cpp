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

namespace {
mp::Engine* gLoadedEngine = nullptr;
}  // namespace

namespace mp {

Engine::~Engine() {
  if (m_isInitialized) {
    vkDeviceWaitIdle(m_device);
    m_scene.clear_all(*this);
    for (auto& frame : m_frameData) {
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

Engine::Engine() {
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

Engine& Engine::get() { return *gLoadedEngine; }

void Engine::run() {
  SDL_Event e;
  bool bIsRunning = true;

  while (bIsRunning) {
    auto start = cn::steady_clock::now();
    while (SDL_PollEvent(&e)) {
      if (e.type == SDL_EVENT_QUIT) {
        bIsRunning = false;
      }

      if (e.type == SDL_EVENT_WINDOW_MINIMIZED) {
        m_isRenderStopped = true;
      }

      if (e.type == SDL_EVENT_WINDOW_MAXIMIZED) {
        m_isRenderStopped = false;
      }
      m_camera.process_sdl_event(e, m_window.get());
      ImGui_ImplSDL3_ProcessEvent(&e);

      if (ImGui::GetIO().WantCaptureMouse || ImGui::GetIO().WantCaptureKeyboard)
        continue;
    }

    if (m_bSwapchainResizeRequest) {
      resize_swapchain();
    }

    if (m_isRenderStopped) {
      std::this_thread::sleep_for(100ms);
      continue;
    }

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();

    // ImGui UI
    if (ImGui::Begin("Other")) {
      ImGui::DragFloat("Render scale", &m_renderScale, 0.01f, 0.01f, 1.0f);
      ImGui::DragFloat("Camera speed", &m_camera.cameraSpeed, 0.01f, 0.01f,
                       100.0f);
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
    ImGui::Text("Deferred light pass time: %f ms",
                m_stats.gBufferLightPassTime);
    ImGui::Text("WBOIT forward pass time: %f ms",
                m_stats.transparentForwardLightPassTime);
    ImGui::Text("Post process pass time: %f ms", m_stats.postProcessPassTime);
    ImGui::Text("ImGui draw time: %f ms", m_stats.imguiDrawTime);
    ImGui::Text("Scene update tim: %f ms", m_stats.sceneUpdateTime);
    ImGui::Text("Amount of draw calls: %i", m_stats.drawCallCount);
    ImGui::Text("Amount of triangles: %i", m_stats.triangleCount);
    ImGui::End();

    {
      const ImGuiViewport* v = ImGui::GetMainViewport();
      ImGui::SetNextWindowPos(ImVec2(10, 200));
      ImGui::SetNextWindowSize(ImVec2(v->WorkSize.x / 6, v->WorkSize.y - 210));
      ImGui::Begin("Scene graph", nullptr,
                   ImGuiWindowFlags_NoFocusOnAppearing |
                       ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize);
      ImGui::Separator();
      for (const auto& topNode : m_scene.topNodes) {
        m_selectedNode =
            render_scene_tree_ui(m_scene, topNode->nodeIndex, m_selectedNode);
      }
      ImGui::Separator();
      if (ImGui::Button("Add light")) {
        const auto nodeIndex = m_scene.add_node(std::make_shared<LightNode>(
            LightData{.lightType = 0,
                      .padding0 = 0,
                      .data0 = {0.0f, 0.0f, 0.0f, 1.0f},
                      .data1 = {1.0f, 1.0f, 1.0f, 1.0f}}));

        auto& node = m_scene.nodes.find(nodeIndex)->second;
        node->worldTransform = glm::mat4(1.0f);
        node->localTransform = glm::mat4(1.0f);
        node->name = "Light";
        node->nodeIndex = nodeIndex;
        m_scene.topNodes.push_back(node);
      }
      ImGui::End();
    }

    if (m_selectedNode != UINT64_MAX) {
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

FrameData& Engine::get_current_frame() {
  return m_frameData[m_frameNumber % kNumberOfFrames];
}

void Engine::WindowCleaner::operator()(SDL_Window* window) const {
  SDL_DestroyWindow(window);
}

void Engine::update_scene() {
  const auto start = cn::steady_clock::now();
  m_camera.update(m_stats.frameTime);
  m_mainDrawContext.clear();
  m_CurrentFrameInstanceBuffer = static_cast<Instance*>(
      get_current_frame().instanceBuffer.allocationInfo.pMappedData);
  m_CurrentMeshBuffer = static_cast<RenderObject*>(
      get_current_frame().meshesBuffer.allocationInfo.pMappedData);

  m_scene.draw(glm::mat4(1.0f), m_mainDrawContext);

  copy_frame_buffers();

  constexpr float cameraNear = 0.001f;
  constexpr float cameraFar = 100.0f;
  const glm::mat4 proj = glm::perspectiveRH_ZO(
      glm::radians(90.0f),
      static_cast<float>(m_drawExtent.width) / m_drawExtent.height, cameraNear,
      cameraFar);

  m_sceneData.view = m_camera.get_view_matrix();
  m_sceneData.proj = proj;
  m_sceneData.projView = proj * m_sceneData.view;
  m_sceneData.cameraPos = m_camera.position;
  m_sceneData.padding0 = 0.0f;

  const glm::mat4 inverseProjView = glm::inverse(m_sceneData.projView);
  std::array<glm::vec3, 8> frustum{
      glm::vec3{-1.0f, -1.0f, 0.0f},
      glm::vec3{1.0f, -1.0f, 0.0f},
      glm::vec3{
          -1.0f,
          1.0f,
          0.0f,
      },
      glm::vec3{1.0f, 1.0f, 0.0f},
      glm::vec3{-1.0f, -1.0f, 1.0f},
      glm::vec3{1.0f, -1.0f, 1.0f},
      glm::vec3{-1.0f, 1.0f, 1.0f},
      glm::vec3{1.0f, 1.0f, 1.0f},
  };

  for (auto& corner : frustum) {
    glm::vec4 pt = inverseProjView * glm::vec4(corner, 1.0f);
    corner = glm::vec3(pt) / pt.w;
  }

  for (auto& light : m_mainDrawContext.lights) {
    if (light.lightType != 0) continue;

    const glm::vec3& sceneMin = m_mainDrawContext.opaqueSceneMin;
    const glm::vec3& sceneMax = m_mainDrawContext.opaqueSceneMax;

    if (sceneMin.x > sceneMax.x) {
      light.lightVP = glm::mat4(1.0f);
      continue;
    }

    const glm::vec3 lightDir = glm::normalize(glm::vec3(light.data0));
    const glm::vec3 sceneCenter = (sceneMin + sceneMax) * 0.5f;

    const glm::vec3 up =
        (std::abs(glm::dot(lightDir, glm::vec3(0.0f, 1.0f, 0.0f))) > 0.99f)
            ? glm::vec3(0.0f, 0.0f, 1.0f)
            : glm::vec3(0.0f, 1.0f, 0.0f);
    glm::mat4 lightView =
        glm::lookAtRH(sceneCenter - lightDir, sceneCenter, up);

    // Transform scene AABB corners into light space
    glm::vec3 lsMin(FLT_MAX);
    glm::vec3 lsMax(-FLT_MAX);
    for (int cx = 0; cx <= 1; cx++)
      for (int cy = 0; cy <= 1; cy++)
        for (int cz = 0; cz <= 1; cz++) {
          const glm::vec3 worldPt(cx ? sceneMax.x : sceneMin.x,
                                  cy ? sceneMax.y : sceneMin.y,
                                  cz ? sceneMax.z : sceneMin.z);
          const glm::vec3 ls = glm::vec3(lightView * glm::vec4(worldPt, 1.0f));
          lsMin = glm::min(lsMin, ls);
          lsMax = glm::max(lsMax, ls);
        }

    glm::vec3 frustumLsMin(FLT_MAX);
    glm::vec3 frustumLsMax(-FLT_MAX);
    for (const auto& corner : frustum) {
      const auto lightSpaceCorner =
          glm::vec3(lightView * glm::vec4(corner, 1.0f));
      frustumLsMin = glm::min(frustumLsMin, lightSpaceCorner);
      frustumLsMax = glm::max(frustumLsMax, lightSpaceCorner);
    }

    lsMin.x = glm::clamp(lsMin.x, frustumLsMin.x, frustumLsMax.x);
    lsMin.y = glm::clamp(lsMin.y, frustumLsMin.y, frustumLsMax.y);
    lsMax.x = glm::clamp(lsMax.x, frustumLsMin.x, frustumLsMax.x);
    lsMax.y = glm::clamp(lsMax.y, frustumLsMin.y, frustumLsMax.y);

    constexpr float zPad = 1.0f;
    lsMin.z -= zPad;
    lsMax.z += zPad;

    lightView = glm::lookAtRH(sceneCenter - lightDir + lightDir * lsMin.z,
                              sceneCenter + lightDir * lsMin.z, up);

    const glm::mat4 lightProj = glm::orthoRH_ZO(
        lsMin.x, lsMax.x, lsMin.y, lsMax.y, 0.0f, lsMax.z - lsMin.z);

    light.lightVP = lightProj * lightView;
  }

  auto* sceneData = static_cast<GpuSceneData*>(
      get_current_frame().sceneDataBuffer.allocation->GetMappedData());
  *sceneData = m_sceneData;
  m_LightPassConstants.sceneDataBufferDeviceAddr =
      get_current_frame().sceneDataBufferAddr;
  m_LightPassConstants.lightDataBufferDeviceAddr =
      get_current_frame().lightDataBufferAddr;
  m_LightPassConstants.lightCount = m_mainDrawContext.lights.size();
  std::memcpy(get_current_frame().lightDataBuffer.allocationInfo.pMappedData,
              m_mainDrawContext.lights.data(),
              m_mainDrawContext.lights.size() * sizeof(LightData));

  m_GBufferMeshPushConstants = {
      .globalVertexBufferAddr = m_globalVertexBufferAddress,
      .instanceBufferDeviceAddr = get_current_frame().instanceBufferAddr,
      .sceneDataBufferDeviceAddr = get_current_frame().sceneDataBufferAddr,
  };

  m_WBOITForwardPassPushConstants = {
      .globalVertexBufferAddr = m_globalVertexBufferAddress,
      .instanceBufferDeviceAddr =
          m_GBufferMeshPushConstants.instanceBufferDeviceAddr,
      .sceneDataBufferDeviceAddr =
          m_GBufferMeshPushConstants.sceneDataBufferDeviceAddr,
      .lightDataBufferDeviceAddr =
          m_LightPassConstants.lightDataBufferDeviceAddr,
      .lightCount = m_LightPassConstants.lightCount};
  const auto end = cn::steady_clock::now();
  const auto elapsed = cn::duration_cast<cn::milliseconds>(end - start);

  m_stats.sceneUpdateTime = elapsed.count() / 1000.0f;
}

}  // namespace mp
