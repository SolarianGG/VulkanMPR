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

// clang-format on

namespace
{

void compute_tetrahedron_shadow_matrices(mp::PointLightData &pointLight)
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
            glm::lookAtRH(pointLight.position, pointLight.position + kFaceVecs[i], kFaceUps[i]);
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
    transparentMeshes.clear();
    dirLight = std::nullopt;
    pointLights.clear();
    renderObjects.clear();
    opaqueInstances.clear();
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

        auto &instanceVec = (passType == MaterialPass::Opaque) ? ctx.opaqueInstances : ctx.transparentInstances;
        auto &submeshMap = (passType == MaterialPass::Opaque) ? ctx.opaqueMeshes : ctx.transparentMeshes;

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

        if (passType == MaterialPass::Opaque)
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
    ImGui::SliderInt("Cascades", &m_Data.cascadeCount.x, 1, kMaxCascadeCount);
}

void PointLightNode::edit()
{
    Node::edit();

    ImGui::Text("Point light");
    ImGui::ColorPicker3("Color", reinterpret_cast<float *>(&m_Data.color));
    ImGui::DragFloat("Range", &m_Data.range, 0.01f, 0.01f, 100.0f);
    ImGui::DragFloat("Intensity", &m_Data.intensity, 0.01f, 0.01f, 100.0f);
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