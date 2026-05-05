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

#include "mpr_engine.hpp"
#include "mpr_math.hpp"

// clang-format on

namespace
{

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

    Node::draw(topMatrix, ctx);
}

void DirectionalLightNode::draw(const glm::mat4 &topMatrix, DrawContext &ctx)
{
    m_Data.direction = glm::normalize(glm::vec3(worldTransform[2]));
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

void DirectionalLightNode::edit()
{
    Node::edit();

    ImGui::Text("Directional light");
    ImGui::ColorPicker3("Color", reinterpret_cast<float *>(&m_Data.color));
    ImGui::DragFloat("Intensity", &m_Data.intensity, 0.01f, 0.01f, 100.0f);
    ImGui::SliderInt("Cascades", &m_Data.cascadeCount, 1, kMaxCascadeCount);
    ImGui::DragFloat("Normal bias", &m_Data.normalBias, 0.0001f, 0.0001f, 0.1f, "%.4f");
    ImGui::DragFloat("Constant bias", &m_Data.constantBias, 0.0001f, 0.0001f, 0.1f, "%.4f");
}

void PointLightNode::edit()
{
    Node::edit();

    ImGui::Text("Point light");
    ImGui::ColorPicker3("Color", reinterpret_cast<float *>(&m_Data.color));
    ImGui::DragFloat("Range", &m_Data.range, 0.01f, 0.01f, 100.0f);
    ImGui::DragFloat("Intensity", &m_Data.intensity, 0.01f, 0.01f, 100.0f);
    ImGui::DragFloat("Normal bias", &m_Data.normalBias, 0.0001f, 0.0001f, 0.1f, "%.4f");
    ImGui::DragFloat("Constant bias", &m_Data.constantBias, 0.0001f, 0.0001f, 0.1f, "%.4f");
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
    if (ctx.ddgiVolumes.size() < kMaxDDGIVolumes)
    {
        const auto volumeIdx = static_cast<std::uint32_t>(ctx.ddgiVolumes.size());
        ctx.ddgiVolumes.push_back(m_Data);
        if (m_bVisualize)
            ctx.ddgiVolumesVis.emplace_back(m_Data, volumeIdx);
    }

    Node::draw(topMatrix, ctx);
}

void DDGIVolumeNode::edit()
{
    Node::edit();

    ImGui::Text("DDGI Volume");
    ImGui::Checkbox("Visualize Probes", &m_bVisualize);
    ImGui::DragFloat3("Probe Spacing", glm::value_ptr(m_Data.probeSpacing), 0.05f, 0.01f, 10.0f);
    ImGui::DragFloat("Max Ray Dist", &m_Data.probeMaxRayDistance, 0.1f, 0.1f, 1000.0f);
    ImGui::DragInt("Rays per Probe", &m_Data.probeNumRays, 1, 1, static_cast<int>(kMaxDDGIRays));
    int counts[3] = {m_Data.probeCounts.x, m_Data.probeCounts.y, m_Data.probeCounts.z};
    if (ImGui::DragInt3("Probe Counts", counts, 1, 1, static_cast<int>(kMaxDDGIProbesX)))
        m_Data.probeCounts = {counts[0], counts[1], counts[2]};
    ImGui::DragFloat("Ray Normal Bias", &m_Data.rayNormalBias, 0.0001f, 0.0001f, 0.5f, "%.4f");
    ImGui::DragFloat("Ray View Bias",   &m_Data.rayViewBias,   0.0001f, 0.0001f, 0.5f, "%.4f");
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