#define GLM_ENABLE_EXPERIMENTAL
#include "mpr_loader.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <algorithm>
#include <fastgltf/base64.hpp>
#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/math.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>
#include <fastgltf/util.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <print>
#include <ranges>

namespace {
VkFilter extract_filter(const fastgltf::Filter filter) {
  switch (filter) {
    // nearest samplers
    case fastgltf::Filter::Nearest:
    case fastgltf::Filter::NearestMipMapNearest:
    case fastgltf::Filter::NearestMipMapLinear:
      return VK_FILTER_NEAREST;

    // linear samplers
    case fastgltf::Filter::Linear:
    case fastgltf::Filter::LinearMipMapNearest:
    case fastgltf::Filter::LinearMipMapLinear:
      return VK_FILTER_LINEAR;
  }
}

VkSamplerMipmapMode extract_mip_map_mode(const fastgltf::Filter filter) {
  switch (filter) {
    case fastgltf::Filter::Nearest:
    case fastgltf::Filter::NearestMipMapNearest:
    case fastgltf::Filter::LinearMipMapNearest:
      return VK_SAMPLER_MIPMAP_MODE_NEAREST;

    case fastgltf::Filter::Linear:
    case fastgltf::Filter::NearestMipMapLinear:
    case fastgltf::Filter::LinearMipMapLinear:
      return VK_SAMPLER_MIPMAP_MODE_LINEAR;
  }
}

std::optional<mp::AllocatedImage> load_image(mp::Engine& engine,
                                             fastgltf::Asset& asset,
                                             fastgltf::Image& image) {
  mp::AllocatedImage newImage;
  int width, height, nChannels;
  auto* data = std::visit(
      fastgltf::visitor{
          [](auto& arg) -> void* { return nullptr; },
          [&](fastgltf::sources::URI& filePath) -> void* {
            assert(filePath.fileByteOffset == 0);
            assert(filePath.uri.isLocalPath());
            const std::string path(filePath.uri.path().begin(),
                                   filePath.uri.path().end());
            return stbi_load(path.c_str(), &width, &height, &nChannels, 4);
          },
          [&](fastgltf::sources::Array& vector) -> void* {
            return stbi_load_from_memory(
                reinterpret_cast<const stbi_uc*>(vector.bytes.data()),
                static_cast<int>(vector.bytes.size()), &width, &height,
                &nChannels, 4);
          },
          [&](fastgltf::sources::BufferView& view) -> void* {
            auto& bufferView = asset.bufferViews[view.bufferViewIndex];
            auto& buffer = asset.buffers[bufferView.bufferIndex];
            return std::visit(
                fastgltf::visitor{
                    [](auto& arg) -> void* { return nullptr; },
                    [&](fastgltf::sources::Array& vector) -> void* {
                      return stbi_load_from_memory(
                          reinterpret_cast<const stbi_uc*>(
                              vector.bytes.data() + bufferView.byteOffset),
                          static_cast<int>(bufferView.byteLength), &width,
                          &height, &nChannels, 4);
                    }},
                buffer.data);
          },
      },
      image.data);
  if (data) {
    const VkExtent3D extent{static_cast<std::uint32_t>(width),
                            static_cast<std::uint32_t>(height), 1u};

    constexpr VkFormat imageFormat = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkImageUsageFlags imageUsage = VK_IMAGE_USAGE_SAMPLED_BIT;
    newImage = engine.create_image(data, extent, imageFormat, imageUsage, true);
  } else {
    return std::nullopt;
  }
  stbi_image_free(data);

  if (newImage.image == nullptr) {
    return std::nullopt;
  }
  return newImage;
}
}  // namespace

namespace mp {
bool load_gltf(mp::Engine& engine, const std::filesystem::path& filePath) {
  if (!std::filesystem::exists(filePath)) {
    std::println(
        "Provided file path: {} is not a regular file or does not exists",
        filePath.string());
    return false;
  }
  std::println("Loading GLTF: {}", filePath.string());

  auto gltfFile = fastgltf::MappedGltfFile::FromPath(filePath);
  if (!gltfFile) {
    std::println("Failed to open glTF file: {}",
                 fastgltf::getErrorMessage(gltfFile.error()));
    return false;
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
                               fastgltf::Options::DecomposeNodeMatrices |
                               fastgltf::Options::GenerateMeshIndices;

  auto load =
      parser.loadGltf(gltfFile.get(), filePath.parent_path(), gltfOptions);
  if (load.error() != fastgltf::Error::None) {
    std::println("Failed to load glTF: {}",
                 fastgltf::getErrorMessage(load.error()));
    return false;
  }
  fastgltf::Asset asset = std::move(load.get());

  assert(asset.lights.empty());
  Scene& file = engine.m_scene;

  for (auto& sampler : asset.samplers) {
    const VkSamplerCreateInfo samplerCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .magFilter = extract_filter(
            sampler.magFilter.value_or(fastgltf::Filter::Nearest)),
        .minFilter = extract_filter(
            sampler.minFilter.value_or(fastgltf::Filter::Nearest)),
        .mipmapMode = extract_mip_map_mode(
            sampler.minFilter.value_or(fastgltf::Filter::Nearest)),
        .minLod = 0,
        .maxLod = VK_LOD_CLAMP_NONE,
    };

    VkSampler vkSampler;
    vkCreateSampler(engine.m_device, &samplerCreateInfo, nullptr, &vkSampler);
    file.samplers.push_back(vkSampler);
  }

  std::vector<AllocatedImage> images;
  std::vector<std::shared_ptr<GLTFMaterial>> materials;
  std::vector<std::shared_ptr<MeshAsset>> meshes;
  std::vector<std::shared_ptr<Node>> nodes;
  images.reserve(asset.images.size());
  for (auto& image : asset.images) {
    if (auto img = load_image(engine, asset, image); img.has_value()) {
      images.push_back(img.value());
      file.add_image(image.name.c_str(), img.value());
    } else {
      images.push_back(engine.m_errorImage);
      std::println("Failed to load texture: {}", image.name);
    }
  }

  auto& [currentBuff, currentBuffAddr] = file.materialBuffers.emplace_back(
      engine.create_buffer(sizeof(GltfMetallicRoughness::MaterialConstants) *
                               asset.materials.size(),
                           VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                           VMA_MEMORY_USAGE_CPU_TO_GPU),
      VkDeviceAddress{0});
  const VkBufferDeviceAddressInfo addrInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
      .pNext = nullptr,
      .buffer = currentBuff.buffer,
  };
  currentBuffAddr = vkGetBufferDeviceAddress(engine.m_device, &addrInfo);
  int dataIndex = 0;
  auto* sceneMaterialsConstants =
      static_cast<GltfMetallicRoughness::MaterialConstants*>(
          currentBuff.allocationInfo.pMappedData);

  std::unordered_map<std::size_t, std::uint32_t> texturesCache;
  std::unordered_map<std::size_t, std::uint32_t> samplersCache;
  for (auto& material : asset.materials) {
    auto newMat = std::make_shared<GLTFMaterial>();
    file.add_material(material.name.c_str(), newMat);

    GltfMetallicRoughness::MaterialConstants materialConstants;
    materialConstants.colorFactors = {material.pbrData.baseColorFactor[0],
                                      material.pbrData.baseColorFactor[1],
                                      material.pbrData.baseColorFactor[2],
                                      material.pbrData.baseColorFactor[3]};
    materialConstants.metalRoughFactors = {material.pbrData.metallicFactor,
                                           material.pbrData.roughnessFactor,
                                           0.0f, 0.0f};
    sceneMaterialsConstants[dataIndex] = materialConstants;

    auto passType = MaterialPass::Opaque;
    if (material.alphaMode == fastgltf::AlphaMode::Blend) {
      passType = MaterialPass::Transparent;
    }
    newMat->data.passType = passType;
    newMat->data.pipeline = engine.m_metalRoughness.select_pipeline(passType);
    newMat->data.indices = {
        .materialID = engine.m_metalRoughness.write_uniform_buffer(
            currentBuffAddr +
            dataIndex * sizeof(GltfMetallicRoughness::MaterialConstants)),
        .colorTextureID = 0,
        .colorSamplerID = 0,
        .metalRoughnessTextureID = 0,
        .metalRoughnessSamplerID = 0};

    auto getCached =
        [&](const std::size_t resourceIndex,
            std::unordered_map<std::size_t, std::uint32_t>& cache,
            const std::function<std::uint32_t(std::size_t)>& getIndexFunc) {
          if (const auto it = texturesCache.find(resourceIndex);
              it != texturesCache.end()) {
            return it->second;
          }
          const auto index = getIndexFunc(resourceIndex);
          cache.try_emplace(resourceIndex, index);
          return index;
        };
    auto getTextureIndex = [&engine, &images](const std::size_t index) {
      return engine.m_metalRoughness.write_texture(images[index].imageView);
    };

    auto getSamplerIndex = [&engine, &file](const std::size_t index) {
      return engine.m_metalRoughness.write_sampler(file.samplers[index]);
    };
    if (material.pbrData.baseColorTexture.has_value()) {
      const auto textureIndex =
          material.pbrData.baseColorTexture.value().textureIndex;
      const auto imgIndex = asset.textures[textureIndex].imageIndex.value();
      const auto samplerIndex =
          asset.textures[textureIndex].samplerIndex.value();
      newMat->data.indices.colorTextureID =
          getCached(imgIndex, texturesCache, getTextureIndex);
      newMat->data.indices.colorSamplerID =
          getCached(samplerIndex, samplersCache, getSamplerIndex);
    }
    if (material.pbrData.metallicRoughnessTexture.has_value()) {
      const auto textureIndex =
          material.pbrData.metallicRoughnessTexture.value().textureIndex;
      const auto imgIndex = asset.textures[textureIndex].imageIndex.value();
      const auto samplerIndex =
          asset.textures[textureIndex].samplerIndex.value();
      newMat->data.indices.metalRoughnessTextureID =
          getCached(imgIndex, texturesCache, getTextureIndex);
      newMat->data.indices.metalRoughnessSamplerID =
          getCached(samplerIndex, samplersCache, getSamplerIndex);
    }

    materials.push_back(std::move(newMat));
    dataIndex++;
  }

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

      size_t initialVtx = vertices.size();

      // load indexes
      {
        fastgltf::Accessor& indexAccessor =
            asset.accessors[p.indicesAccessor.value()];
        indices.reserve(indices.size() + indexAccessor.count);

        fastgltf::iterateAccessor<std::uint32_t>(
            asset, indexAccessor, [&](const std::uint32_t idx) {
              indices.push_back(idx + initialVtx);
            });
      }

      // load vertex positions
      {
        fastgltf::Accessor& posAccessor =
            asset.accessors[p.findAttribute("POSITION")->accessorIndex];
        vertices.resize(vertices.size() + posAccessor.count);

        fastgltf::iterateAccessorWithIndex<glm::vec3>(
            asset, posAccessor, [&](const glm::vec3 v, const size_t index) {
              Vertex newVtx;
              newVtx.pos = v;
              newVtx.normal = {1, 0, 0};
              newVtx.color = glm::vec4{1.f};
              newVtx.u = 0;
              newVtx.v = 0;
              vertices[initialVtx + index] = newVtx;
            });
      }

      // load vertex normals
      auto normals = p.findAttribute("NORMAL");
      if (normals != p.attributes.end()) {
        fastgltf::iterateAccessorWithIndex<glm::vec3>(
            asset, asset.accessors[normals->accessorIndex],
            [&](const glm::vec3 v, const size_t index) {
              vertices[initialVtx + index].normal = v;
            });
      }

      // load UVs
      auto uv = p.findAttribute("TEXCOORD_0");
      if (uv != p.attributes.end()) {
        fastgltf::iterateAccessorWithIndex<glm::vec2>(
            asset, asset.accessors[uv->accessorIndex],
            [&](const glm::vec2 v, const size_t index) {
              vertices[initialVtx + index].u = v.x;
              vertices[initialVtx + index].v = v.y;
            });
      }

      // load vertex colors
      auto colors = p.findAttribute("COLOR_0");
      if (colors != p.attributes.end()) {
        fastgltf::iterateAccessorWithIndex<glm::vec4>(
            asset, asset.accessors[colors->accessorIndex],
            [&](const glm::vec4 v, const size_t index) {
              vertices[initialVtx + index].color = v;
            });
      }

      if (p.materialIndex.has_value()) {
        newSurface.material = materials[p.materialIndex.value()];
      } else {
        std::println("Mesh does not contains material");
        newSurface.material = materials[0];
      }

      newMesh.geoSurfaces.push_back(newSurface);
    }

    newMesh.meshBuffers = engine.create_mesh_buffers(indices, vertices);

    meshes.emplace_back(std::make_shared<MeshAsset>(std::move(newMesh)));
    file.add_mesh(meshes.back());
  }

  // Adding nodes
  for (auto& node : asset.nodes) {
    std::shared_ptr<Node> newNode;
    if (node.meshIndex.has_value()) {
      newNode = std::make_shared<MeshNode>(meshes[node.meshIndex.value()]);
    } else {
      newNode = std::make_shared<Node>();
    }

    if (node.name.empty()) {
      node.name =
          std::format("Node_{}_{}", filePath.filename().string(), nodes.size());
    }
    newNode->name = node.name;
    nodes.push_back(newNode);
    newNode->nodeIndex = file.add_node(newNode);

    std::visit(
        fastgltf::visitor{
            [&](fastgltf::math::fmat4x4 matrix) {
              memcpy(&newNode->localTransform, matrix.data(), sizeof(matrix));
            },
            [&](fastgltf::TRS transform) {
              const glm::vec3 tl(transform.translation[0],
                                 transform.translation[1],
                                 transform.translation[2]);
              const glm::quat rot(transform.rotation[3], transform.rotation[0],
                                  transform.rotation[1], transform.rotation[2]);
              const glm::vec3 sc(transform.scale[0], transform.scale[1],
                                 transform.scale[2]);

              const glm::mat4 tm = glm::translate(glm::mat4(1.f), tl);
              const glm::mat4 rm = glm::toMat4(rot);
              const glm::mat4 sm = glm::scale(glm::mat4(1.f), sc);

              newNode->localTransform = tm * rm * sm;
            }},
        node.transform);
  }

  for (const auto [assetNode, sceneNode] :
       std::views::zip(asset.nodes, nodes)) {
    for (const auto& c : assetNode.children) {
      sceneNode->children.push_back(nodes[c]);
      nodes[c]->parent = sceneNode;
    }
  }

  for (auto& node : nodes) {
    if (node->parent.lock() == nullptr) {
      file.topNodes.push_back(node);
      node->refresh_transform(glm::mat4(1.0f));
    }
  }

  return true;
}
}  // namespace mp