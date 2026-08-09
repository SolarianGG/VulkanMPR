// clang-format off
#define GLM_ENABLE_EXPERIMENTAL
#include "mpr_engine.hpp"

#include <imgui.h>
#include <ImGuizmo.h>

#include <glm/gtc/type_ptr.hpp>

#include <print>
#include <string>
// clang-format on

namespace mp
{

std::uint64_t Engine::render_scene_tree_ui(Scene &scene, std::uint64_t nodeIndex, std::uint64_t selectedNode)
{
    ZoneScoped;
    const bool isLeaf = scene.nodes.at(nodeIndex)->children.empty();
    ImGuiTreeNodeFlags flags = isLeaf ? ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_Bullet : 0;

    if (nodeIndex == selectedNode)
    {
        flags |= ImGuiTreeNodeFlags_Selected;
    }

    const ImVec4 color = isLeaf ? ImVec4(0.7f, 0.7f, 0.2f, 1.0f) : ImVec4(0.2f, 0.7f, 0.7f, 1.0f);

    ImGui::PushStyleColor(ImGuiCol_Text, color);
    const bool isOpened =
        ImGui::TreeNodeEx(&scene.nodes.at(nodeIndex), flags, "%s", scene.nodes[nodeIndex]->name.c_str());
    ImGui::PopStyleColor();

    ImGui::PushID(nodeIndex);
    if (ImGui::IsItemClicked() && isLeaf)
    {
        std::println("Selected Node: {}", nodeIndex);
        selectedNode = nodeIndex;
    }

    if (isOpened)
    {
        for (const auto &child : scene.nodes.at(nodeIndex)->children)
        {
            if (const auto subNode = render_scene_tree_ui(scene, child->nodeIndex, selectedNode); subNode != UINT64_MAX)
            {
                selectedNode = subNode;
            }
        }
        ImGui::TreePop();
    }
    ImGui::PopID();

    return selectedNode;
}

void Engine::edit_node(Scene &scene, const std::uint64_t nodeIndex)
{
    ZoneScoped;
    ImGuizmo::SetOrthographic(false);
    ImGuizmo::BeginFrame();

    auto &node = scene.nodes[nodeIndex];

    const auto &name = node->name;
    std::string label = name.empty() ? (std::string("Node") + std::to_string(nodeIndex)) : name;
    label = "Node: " + label;

    if (const ImGuiViewport *v = ImGui::GetMainViewport())
    {
        ImGui::SetNextWindowPos(ImVec2(v->WorkSize.x * 0.83f, 200));
        ImGui::SetNextWindowSize(ImVec2(v->WorkSize.x / 6, v->WorkSize.y - 210));
    }
    ImGui::Begin("Editor", nullptr,
                 ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize);
    if (!name.empty())
        ImGui::Text("%s", label.c_str());

    ImGui::Separator();
    ImGuizmo::PushID(1);

    node->edit(m_sceneData);

    ImGui::Separator();
#if 0
    ImGui::Text("%s", "Material");

    editMaterialUI(scene, meshData, node, outUpdateMaterialIndex, textureCache);
#endif
    ImGuizmo::PopID();
    ImGui::End();
}

} // namespace mp
