// clang-format off
#include "mpr_scene.hpp"

#include "mpr_engine.hpp"

// clang-format on


namespace mp {

void MeshNode::draw(const glm::mat4& topMatrix, DrawContext& ctx) {
  glm::mat4 nodeMatrix = topMatrix * worldTransform;
  for (const auto& s : mesh->geoSurfaces) {
    const RenderObject rObject{
        .indexCount = s.count,
        .firstIndex = s.startIndex,
        .indexBuffer = mesh->meshBuffers.indexBuffer.buffer,
        .vertexBufferAddress = mesh->meshBuffers.vertexBufferDeviceAddr};
    switch (s.material->data.passType) {
      case MaterialPass::Transparent: {
        ctx.transparentRenderObjects[rObject].emplace_back(
            nodeMatrix, s.material->data.indices);
      } break;
      case MaterialPass::Opaque: {
        ctx.opaqueRenderObjects[rObject].emplace_back(nodeMatrix,
                                                      s.material->data.indices);
      } break;
      case MaterialPass::Other: {
        assert(false);
      } break;
    }
  }

  Node::draw(topMatrix, ctx);
}

void Scene::draw(const glm::mat4& topMatrix, DrawContext& ctx) {
  for (const auto& node : topNodes) {
    node->draw(topMatrix, ctx);
  }
}

void Scene::add_mesh(std::shared_ptr<MeshAsset> mesh) {
  static std::atomic_uint64_t counter = 0;
  meshes[counter++] = std::move(mesh);
}
void Scene::add_image(std::string imageName, const AllocatedImage& image) {
  static std::atomic_uint64_t counter = 0;
  images[counter++] = {std::move(imageName), image};
}

void Scene::add_material(std::string materialName,
                         std::shared_ptr<GLTFMaterial> material) {
  static std::atomic_uint64_t counter = 0;
  materials[counter++] = {std::move(materialName), std::move(material)};
}

std::uint64_t Scene::add_node(std::shared_ptr<Node> node) {
  static std::atomic_uint64_t counter = 0;
  nodes[counter] = std::move(node);
  return counter++;
}

void Scene::clear_all(Engine& engine) {
  for (auto& buffer : materialBuffers | std::views::keys) {
    engine.destroy_buffer(buffer);
  }
  for (const auto& mesh : std::views::values(meshes)) {
    engine.destroy_buffer(mesh->meshBuffers.indexBuffer);
    engine.destroy_buffer(mesh->meshBuffers.vertexBuffer);
  }
  for (const auto& image : std::views::values(std::views::values(images))) {
    if (image.image != engine.m_errorImage.image) {
      engine.destroy_image(image);
    }
  }

  for (const auto& sampler : samplers) {
    vkDestroySampler(engine.m_device, sampler, nullptr);
  }
}





}