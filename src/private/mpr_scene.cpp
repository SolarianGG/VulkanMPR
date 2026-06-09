// clang-format off
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "mpr_scene.hpp"

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <imgui.h>
#include <ImGuizmo.h>

#include "mpr_engine.hpp"
#include "mpr_math.hpp"

// clang-format on

namespace
{

bool edit_transform_ui(const glm::mat4 &view, const glm::mat4 &projection, glm::mat4 &globalTransform)
{
    static ImGuizmo::OPERATION gizmoOperation(ImGuizmo::TRANSLATE);

    ImGui::Text("Transforms:");

    if (ImGui::IsKeyPressed(ImGuiKey_W))
        gizmoOperation = ImGuizmo::TRANSLATE;
    if (ImGui::IsKeyPressed(ImGuiKey_E))
        gizmoOperation = ImGuizmo::ROTATE;
    if (ImGui::IsKeyPressed(ImGuiKey_R))
        gizmoOperation = ImGuizmo::SCALE;

    if (ImGui::RadioButton("Translate", gizmoOperation == ImGuizmo::TRANSLATE))
        gizmoOperation = ImGuizmo::TRANSLATE;

    if (ImGui::RadioButton("Rotate", gizmoOperation == ImGuizmo::ROTATE))
        gizmoOperation = ImGuizmo::ROTATE;

    if (ImGui::RadioButton("Scale", gizmoOperation == ImGuizmo::SCALE))
        gizmoOperation = ImGuizmo::SCALE;

    float matrixTranslation[3], matrixRotation[3], matrixScale[3];
    ImGuizmo::DecomposeMatrixToComponents(glm::value_ptr(globalTransform), matrixTranslation, matrixRotation,
                                          matrixScale);
    bool bFieldsChanged = false;
    bFieldsChanged |= ImGui::InputFloat3("Tr", matrixTranslation);
    bFieldsChanged |= ImGui::InputFloat3("Rt", matrixRotation);
    bFieldsChanged |= ImGui::InputFloat3("Sc", matrixScale);
    if (bFieldsChanged)
    {
        ImGuizmo::RecomposeMatrixFromComponents(matrixTranslation, matrixRotation, matrixScale,
                                                glm::value_ptr(globalTransform));
    }

    const ImGuiIO &io = ImGui::GetIO();
    ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
    const bool bManipulated = ImGuizmo::Manipulate(glm::value_ptr(view), glm::value_ptr(projection), gizmoOperation,
                                                   ImGuizmo::WORLD, glm::value_ptr(globalTransform));
    return bManipulated || bFieldsChanged;
}

void compute_tetrahedron_shadow_matrices(mp::PointLightData &pointLight)
{
    static constexpr float s = 0.45954602f; // sin(27.3678)
    static constexpr float c = 0.88815527f; // cos(27.3678)
    static constexpr std::array<glm::vec3, 4> kLookDirs{
        glm::vec3{0.0f, -s, c},
        glm::vec3{0.0f, -s, -c},
        glm::vec3{-c, s, 0.0f},
        glm::vec3{c, s, 0.0f},
    };
    static std::array<glm::vec3, 4> kFaceUps{
        glm::vec3{0.0f, 1.0f, 0.0f},
        glm::vec3{1.0f, 0.0f, 0.0f},
        glm::normalize(glm::vec3{s, c, 0.0f}),
        glm::vec3{0.0f, 0.0f, 1.0f},
    };
    static const auto [alpha, beta] = mp::compute_alpha_beta();

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

    for (int i = 0; i < 4; ++i)
    {
        const glm::mat4 viewMatrix =
            glm::lookAtRH(pointLight.position, pointLight.position + kLookDirs[i], kFaceUps[i]);
        pointLight.tetrahedronFacesMatrices[i] = projMatrices[i] * viewMatrix;
    }
}

} // namespace

namespace mp
{

void DrawContext::clear()
{
    min = glm::vec3{FLT_MAX};
    max = glm::vec3{-FLT_MAX};
    bGeometryUpdated = false;
    opaqueMeshes.clear();
    alphaTestedMeshes.clear();
    transparentMeshes.clear();
    dirLight = std::nullopt;
    pointLights.clear();
    ddgiVolumes.clear();
    ddgiVolumesVis.clear();
    renderObjects.clear();
    opaqueInstances.clear();
    alphaTestedInstances.clear();
    transparentInstances.clear();
}

bool Node::edit(const GpuSceneData &sceneData)
{
    bool bIsUpdated = edit_transform_ui(sceneData.view, sceneData.proj, worldTransform);

    if (bIsUpdated)
    {
        if (const auto p = parent.lock(); p)
        {
            const glm::mat4 parentWorldTransform = p->worldTransform;
            localTransform = glm::inverse(parentWorldTransform) * worldTransform;
        }
        else
        {
            localTransform = worldTransform;
        }
    }

    return bIsUpdated;
}

bool MeshNode::edit(const GpuSceneData &sceneData)
{
    m_bIsUpdated = Node::edit(sceneData);

    return m_bIsUpdated;
}

void MeshNode::draw(const glm::mat4 &topMatrix, DrawContext &ctx)
{
    const glm::mat4 nodeMatrix = topMatrix * worldTransform;
    for (const auto &s : mesh->geoSurfaces)
    {
        const auto passType = s.material->data.passType;
        if (passType == MaterialPass::Other)
            continue;
        const RenderObject rObject{.indexCount = s.count,
                                   .firstIndex = s.startIndex,
                                   .vertexOffset = s.vertexOffset,
                                   .min = s.min,
                                   .max = s.max};

        auto &instanceVec = (passType == MaterialPass::Opaque)        ? ctx.opaqueInstances
                            : (passType == MaterialPass::AlphaTested) ? ctx.alphaTestedInstances
                                                                      : ctx.transparentInstances;
        auto &submeshMap = (passType == MaterialPass::Opaque)        ? ctx.opaqueMeshes
                           : (passType == MaterialPass::AlphaTested) ? ctx.alphaTestedMeshes
                                                                     : ctx.transparentMeshes;

        auto it = submeshMap.find(rObject);
        uint32_t submeshIndex;
        if (it != submeshMap.end())
        {
            submeshIndex = it->second;
        }
        else
        {
            submeshIndex = static_cast<uint32_t>(ctx.renderObjects.size());
            ctx.renderObjects.push_back(rObject);
            submeshMap[rObject] = submeshIndex;
        }

        instanceVec.push_back(Instance{
            .world = nodeMatrix,
            .meshIndex = submeshIndex,
            .materialIndices = s.material->data.indices,
        });

        if (passType == MaterialPass::Opaque || passType == MaterialPass::AlphaTested)
        {
            for (int cx = 0; cx <= 1; cx++)
                for (int cy = 0; cy <= 1; cy++)
                    for (int cz = 0; cz <= 1; cz++)
                    {
                        const glm::vec3 localPt(cx ? s.max.x : s.min.x, cy ? s.max.y : s.min.y, cz ? s.max.z : s.min.z);
                        const glm::vec3 worldPt = glm::vec3(nodeMatrix * glm::vec4(localPt, 1.0f));
                        ctx.min = glm::min(ctx.min, worldPt);
                        ctx.max = glm::max(ctx.max, worldPt);
                    }
        }
    }

    ctx.bGeometryUpdated |= m_bIsUpdated;
    m_bIsUpdated = false;

    Node::draw(topMatrix, ctx);
}

void DirectionalLightNode::draw(const glm::mat4 &topMatrix, DrawContext &ctx)
{
    glm::mat3 R(worldTransform);
    R[0] = normalize(R[0]);
    R[1] = normalize(R[1]);
    R[2] = normalize(R[2]);
    const glm::quat q = glm::quat_cast(R);
    m_Data.direction = glm::rotate(q, glm::vec3{0.0f, -1.0f, 0.0f});
    ctx.dirLight = m_Data;

    Node::draw(topMatrix, ctx);
}

void PointLightNode::draw(const glm::mat4 &topMatrix, DrawContext &ctx)
{
    const auto newPos = glm::vec3{worldTransform[3].x, worldTransform[3].y, worldTransform[3].z};
    if (newPos != m_Data.position)
    {
        m_Data.position = newPos;
        compute_tetrahedron_shadow_matrices(m_Data);
    }
    ctx.pointLights.push_back(m_Data);

    Node::draw(topMatrix, ctx);
}

bool DirectionalLightNode::edit(const GpuSceneData &sceneData)
{
    bool bIsUpdated = Node::edit(sceneData);

    ImGui::Text("Directional light");
    bIsUpdated |= ImGui::ColorPicker3("Color", reinterpret_cast<float *>(&m_Data.color));
    bIsUpdated |= ImGui::DragFloat("Intensity", &m_Data.intensity, 0.01f, 0.01f, 100.0f);
    bIsUpdated |= ImGui::SliderInt("Cascades", &m_Data.cascadeCount, 1, kMaxCascadeCount);
    bIsUpdated |= ImGui::DragFloat("Normal bias", &m_Data.normalBias, 0.0001f, 0.0001f, 0.1f, "%.4f");
    bIsUpdated |= ImGui::DragFloat("Constant bias", &m_Data.constantBias, 0.0001f, 0.0001f, 0.1f, "%.4f");
    return bIsUpdated;
}

bool PointLightNode::edit(const GpuSceneData &sceneData)
{
    bool bIsUpdated = Node::edit(sceneData);

    ImGui::Text("Point light");
    bIsUpdated |= ImGui::ColorPicker3("Color", reinterpret_cast<float *>(&m_Data.color));
    bIsUpdated |= ImGui::DragFloat("Range", &m_Data.range, 0.01f, 0.01f, 100.0f);
    bIsUpdated |= ImGui::DragFloat("Intensity", &m_Data.intensity, 0.01f, 0.01f, 100.0f);
    bIsUpdated |= ImGui::DragFloat("Normal bias", &m_Data.normalBias, 0.0001f, 0.0001f, 0.1f, "%.4f");
    bIsUpdated |= ImGui::DragFloat("Constant bias", &m_Data.constantBias, 0.0001f, 0.0001f, 0.1f, "%.4f");
    return bIsUpdated;
}

void DDGIVolumeNode::draw(const glm::mat4 &topMatrix, DrawContext &ctx)
{
    m_Data.origin = glm::vec3{worldTransform[3]};
    {
        glm::mat3 R(worldTransform);
        R[0] = normalize(R[0]);
        R[1] = normalize(R[1]);
        R[2] = normalize(R[2]);

        const glm::quat q = glm::quat_cast(R);
        m_Data.rotation = glm::vec4{q.x, q.y, q.z, q.w};
    }
    m_Data.probeRayRotation = mp::math::random_rotation_quaternion();
    m_Data.probeSpacing = glm::max(m_Data.probeSpacing, glm::vec3(0.01f, 0.01f, 0.01f));
    if (ctx.ddgiVolumes.size() < kMaxDDGIVolumes)
    {
        const auto volumeIdx = static_cast<std::uint32_t>(ctx.ddgiVolumes.size());
        ctx.ddgiVolumes.push_back(m_Data);
        if (m_bVisualize)
            ctx.ddgiVolumesVis.emplace_back(DDGIVolumeVisEntry{
                .volume = m_Data, .volumeIdx = volumeIdx, .probeRadius = m_ProbeRadius, .mode = m_visMode});
    }

    Node::draw(topMatrix, ctx);
}

bool DDGIVolumeNode::edit(const GpuSceneData &sceneData)
{
    bool bIsUpdated = Node::edit(sceneData);

    ImGui::Text("DDGI Volume");
    ImGui::Checkbox("Visualize Probes", &m_bVisualize);
    if (m_bVisualize)
    {
        static const char *kModes[] = {"Grid Position", "Irradiance", "Distance"};
        int idx = static_cast<int>(m_visMode);
        if (ImGui::Combo("Vis Mode", &idx, kModes, IM_ARRAYSIZE(kModes)))
            m_visMode = static_cast<DDGIProbeVisMode>(idx);
        ImGui::DragFloat("Probe radius", &m_ProbeRadius, 0.001f, 0.001f, 0.5f);
    }
    bIsUpdated |= ImGui::DragFloat3("Probe Spacing", glm::value_ptr(m_Data.probeSpacing), 0.05f, 0.01f, 10.0f);
    bIsUpdated |= ImGui::DragFloat("Max Ray Dist", &m_Data.probeMaxRayDistance, 0.1f, 0.1f, 100000.0f);
    bIsUpdated |= ImGui::DragInt("Rays per Probe", &m_Data.probeNumRays, 1, 1, static_cast<int>(kMaxDDGIRays));
    int counts[3] = {m_Data.probeCounts.x, m_Data.probeCounts.y, m_Data.probeCounts.z};
    bool probeCountUpdated = ImGui::DragInt3("Probe Counts", counts, 1, 1, static_cast<int>(kMaxDDGIProbesX));
    if (probeCountUpdated)
        m_Data.probeCounts = {counts[0], counts[1], counts[2]};
    bIsUpdated |= probeCountUpdated;
    bIsUpdated |= ImGui::DragFloat("Probe Normal Bias", &m_Data.probeNormalBias, 0.0001f, 0.0001f, 0.5f, "%.4f");
    bIsUpdated |= ImGui::DragFloat("Probe View Bias", &m_Data.probeViewBias, 0.0001f, 0.0001f, 0.5f, "%.4f");
    bIsUpdated |= ImGui::DragFloat("Irradiance gamma", &m_Data.probeIrradianceEncodingGamma, 0.1f, 0.1f, 10.0f);
    bIsUpdated |= ImGui::DragFloat("Irradiance Threshold", &m_Data.probeIrradianceThreshold, 0.1f, 0.1f, 10.0f);
    bIsUpdated |= ImGui::DragFloat("Brightness Threshold", &m_Data.probeBrightnessThreshold, 0.1f, 0.1f, 10.0f);
    bIsUpdated |= ImGui::Checkbox("Relocation enabled", reinterpret_cast<bool *>(&m_Data.probeRelocationEnabled));
    if (m_Data.probeRelocationEnabled == 1)
    {
        bIsUpdated |=
            ImGui::DragFloat("Min front face distance", &m_Data.probeMinFrontfaceDistance, 0.01f, 0.f, 100.0f);
    }
    bIsUpdated |=
        ImGui::Checkbox("Classification enabled", reinterpret_cast<bool *>(&m_Data.probeClassificationEnabled));
    if (m_Data.probeRelocationEnabled == 1 || m_Data.probeClassificationEnabled == 1)
    {
        bIsUpdated |=
            ImGui::DragFloat("Fixed ray backface threshold", &m_Data.probeFixedRayBackfaceThreshold, 0.01f, 0.f, 1.0f);
    }
    return bIsUpdated;
}

void Scene::draw(const glm::mat4 &topMatrix, DrawContext &ctx)
{
    for (const auto &node : topNodes)
    {
        node->draw(topMatrix, ctx);
    }
}

void Scene::add_mesh(std::shared_ptr<MeshAsset> mesh)
{
    static std::atomic_uint64_t counter = 0;
    meshes[counter++] = std::move(mesh);
}

void Scene::add_image(std::string imageName, const AllocatedImage &image)
{
    static std::atomic_uint64_t counter = 0;
    images[counter++] = {std::move(imageName), image};
}

void Scene::add_material(std::string materialName, std::shared_ptr<GLTFMaterial> material)
{
    static std::atomic_uint64_t counter = 0;
    materials[counter++] = {std::move(materialName), std::move(material)};
}

std::uint64_t Scene::add_node(std::shared_ptr<Node> node)
{
    static std::atomic_uint64_t counter = 0;
    nodes[counter] = std::move(node);
    return counter++;
}

void Scene::clear_all(Engine &engine)
{
    for (auto &buffer : materialBuffers | std::views::keys)
    {
        engine.destroy_buffer(buffer);
    }
    for (const auto &image : std::views::values(std::views::values(images)))
    {
        engine.destroy_image(image);
    }

    for (const auto &sampler : samplers)
    {
        vkDestroySampler(engine.m_device, sampler, nullptr);
    }
}

} // namespace mp