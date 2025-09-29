#define GLM_ENABLE_EXPERIMENTAL
#include "mpr_loader.hpp"

#include <stb_image.h>

#include <fastgltf/base64.hpp>
#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/math.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>
#include <fastgltf/util.hpp>
#include <glm/gtx/quaternion.hpp>
#include <print>

#include "mpr_engine.hpp"
#include "mpr_init_vk_stucts.hpp"

namespace mp {
std::optional<std::vector<std::shared_ptr<MeshAsset>>> load_mesh(
    mp::Engine& engine, const std::filesystem::path& filePath) {
  if (!std::filesystem::is_regular_file(filePath)) {
    std::println(
        "Provided file path: {} is not a regular file or does not exists",
        filePath.string());
    return std::nullopt;
  }
  std::println("Loading GLTF: {}", filePath.string());

  auto gltfFile = fastgltf::MappedGltfFile::FromPath(filePath);
  if (!gltfFile) {
    std::println("Failed to open glTF file: {}",
                 fastgltf::getErrorMessage(gltfFile.error()));
    return std::nullopt;
  }

  constexpr auto supportedExtensions =
      fastgltf::Extensions::KHR_mesh_quantization |
      fastgltf::Extensions::KHR_texture_transform |
      fastgltf::Extensions::KHR_materials_variants;
  fastgltf::Parser parser(supportedExtensions);

  constexpr auto gltfOptions = fastgltf::Options::DontRequireValidAssetMember |
                               fastgltf::Options::AllowDouble |
                               fastgltf::Options::LoadExternalBuffers |
                               fastgltf::Options::LoadExternalImages |
                               fastgltf::Options::GenerateMeshIndices;

  // TODO: Add non-binary gltf format loading
  auto load = parser.loadGltfBinary(gltfFile.get(), filePath.parent_path(),
                                    gltfOptions);
  if (load.error() != fastgltf::Error::None) {
    std::println("Failed to load glTF: {}",
                 fastgltf::getErrorMessage(load.error()));
    return std::nullopt;
  }
  auto asset = std::move(load.get());
  std::vector<std ::shared_ptr<MeshAsset>> meshes;
  std::vector<Vertex> vertices;
  std::vector<std::uint32_t> indices;
  for (auto& mesh : asset.meshes) {
    MeshAsset newMesh;

    vertices.clear();
    indices.clear();

    newMesh.name = mesh.name;
    for (auto& p : mesh.primitives) {
      GeoSurface newSurface;
      newSurface.startIndex = static_cast<uint32_t>(indices.size());
      newSurface.count = static_cast<uint32_t>(
          asset.accessors[p.indicesAccessor.value()].count);

      size_t initial_vtx = vertices.size();

      // load indexes
      {
        fastgltf::Accessor& indexaccessor =
            asset.accessors[p.indicesAccessor.value()];
        indices.reserve(indices.size() + indexaccessor.count);

        fastgltf::iterateAccessor<std::uint32_t>(
            asset, indexaccessor, [&](const std::uint32_t idx) {
              indices.push_back(idx + initial_vtx);
            });
      }

      // load vertex positions
      {
        fastgltf::Accessor& posAccessor =
            asset.accessors[p.findAttribute("POSITION")->accessorIndex];
        vertices.resize(vertices.size() + posAccessor.count);

        fastgltf::iterateAccessorWithIndex<glm::vec3>(
            asset, posAccessor, [&](const glm::vec3 v, const size_t index) {
              Vertex newvtx;
              newvtx.pos = v;
              newvtx.normal = {1, 0, 0};
              newvtx.color = glm::vec4{1.f};
              newvtx.u = 0;
              newvtx.v = 0;
              vertices[initial_vtx + index] = newvtx;
            });
      }

      // load vertex normals
      auto normals = p.findAttribute("NORMAL");
      if (normals != p.attributes.end()) {
        fastgltf::iterateAccessorWithIndex<glm::vec3>(
            asset, asset.accessors[normals->accessorIndex],
            [&](const glm::vec3 v, const size_t index) {
              vertices[initial_vtx + index].normal = v;
            });
      }

      // load UVs
      auto uv = p.findAttribute("TEXCOORD_0");
      if (uv != p.attributes.end()) {
        fastgltf::iterateAccessorWithIndex<glm::vec2>(
            asset, asset.accessors[uv->accessorIndex],
            [&](const glm::vec2 v, const size_t index) {
              vertices[initial_vtx + index].u = v.x;
              vertices[initial_vtx + index].v = v.y;
            });
      }

      // load vertex colors
      auto colors = p.findAttribute("COLOR_0");
      if (colors != p.attributes.end()) {
        fastgltf::iterateAccessorWithIndex<glm::vec4>(
            asset, asset.accessors[colors->accessorIndex],
            [&](const glm::vec4 v, const size_t index) {
              vertices[initial_vtx + index].color = v;
            });
      }
      newMesh.geoSurfaces.push_back(newSurface);
    }

    constexpr bool kOverrideColors = true;
    if (kOverrideColors) {
      for (Vertex& vtx : vertices) {
        vtx.color = glm::vec4(glm::abs(vtx.normal), 1.f);
      }
    }
    newMesh.meshBuffers = engine.create_mesh_buffers(indices, vertices);

    meshes.emplace_back(std::make_shared<MeshAsset>(std::move(newMesh)));
  }

  return meshes;
}
}  // namespace mp