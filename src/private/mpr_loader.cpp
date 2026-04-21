#define GLM_ENABLE_EXPERIMENTAL
#include "mpr_loader.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "dds.hpp"
#include "mpr_image.hpp"
#include "mpr_init_vk_stucts.hpp"

#include <algorithm>
#include <fastgltf/base64.hpp>
#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/math.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>
#include <fastgltf/util.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/packing.hpp>
#include <glm/gtx/quaternion.hpp>
#include <meshoptimizer.h>
#include <print>
#include <ranges>

namespace
{

glm::vec2 OctahedronWrap(const glm::vec2 v)
{
    glm::vec2 w = 1.0f - glm::abs(glm::vec2(v.y, v.x));
    if (v.x < 0.0f)
        w.x = -w.x;
    if (v.y < 0.0f)
        w.y = -w.y;
    return w;
}

glm::u16vec2 QuantizeNormal(glm::vec3 n)
{
    n /= (glm::abs(n.x) + glm::abs(n.y) + glm::abs(n.z));
    n = glm::vec3(n.z > 0.0f ? glm::vec2(n.x, n.y) : OctahedronWrap(glm::vec2(n.x, n.y)), n.z);
    n = glm::vec3(glm::vec2(n.x, n.y) * 0.5f + 0.5f, n.z);

    return glm::packHalf(glm::vec2(n.x, n.y));
}

glm::u16vec2 QuantizeTangent(const glm::vec4 &t)
{
    glm::u16vec2 q = QuantizeNormal(glm::vec3(t.x, t.y, t.z));

    // If bitangent sign is positive, we set first bit of Y component to 1, otherwise (if it is negative) we set 0
    q.y = t.w == 1.0f ? q.y | 1u : q.y & ~1u;

    return q;
}

VkFilter extract_filter(const fastgltf::Filter filter)
{
    switch (filter)
    {
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

VkSamplerMipmapMode extract_mip_map_mode(const fastgltf::Filter filter)
{
    switch (filter)
    {
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

std::optional<mp::AllocatedImage> load_image(mp::Engine &engine, fastgltf::Asset &asset, fastgltf::Image &image,
                                             const VkFormat imageFormat, const VkImageUsageFlags imageUsage)
{
    constexpr std::uint32_t kDdsMagic = 0x20534444u; // 'DDS '

    auto tryLoadDDS = [&](const std::uint8_t *ptr,
                          std::size_t size) -> std::optional<mp::AllocatedImage> {
        if (size < 4 || *reinterpret_cast<const std::uint32_t *>(ptr) != kDdsMagic)
            return std::nullopt;
        dds::Image ddsImage;
        if (dds::readImage(ptr, size, &ddsImage) != dds::ReadResult::Success)
            return std::nullopt;
        return engine.create_image(&ddsImage, imageUsage);
    };

    auto loadStbi = [&](const std::uint8_t *ptr, int size) -> std::optional<mp::AllocatedImage> {
        int width, height, nChannels;
        void *data = stbi_load_from_memory(reinterpret_cast<const stbi_uc *>(ptr), size, &width, &height, &nChannels, 4);
        if (!data)
            return std::nullopt;
        const VkExtent3D extent{static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), 1u};
        mp::AllocatedImage result = engine.create_image(data, extent, imageFormat, imageUsage, true);
        stbi_image_free(data);
        return result;
    };

    return std::visit(
        fastgltf::visitor{
            [](auto &) -> std::optional<mp::AllocatedImage> { return std::nullopt; },
            [&](fastgltf::sources::URI &filePath) -> std::optional<mp::AllocatedImage> {
                assert(filePath.fileByteOffset == 0);
                assert(filePath.uri.isLocalPath());
                const std::string path(filePath.uri.path().begin(), filePath.uri.path().end());
                const std::string ext = path.size() >= 4 ? path.substr(path.size() - 4) : std::string{};
                if (ext == ".dds" || ext == ".DDS")
                {
                    dds::Image ddsImage;
                    if (dds::readFile(path, &ddsImage) != dds::ReadResult::Success)
                        return std::nullopt;
                    return engine.create_image(&ddsImage, imageUsage);
                }
                int width, height, nChannels;
                void *data = stbi_load(path.c_str(), &width, &height, &nChannels, 4);
                if (!data)
                    return std::nullopt;
                const VkExtent3D extent{static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), 1u};
                mp::AllocatedImage result = engine.create_image(data, extent, imageFormat, imageUsage, true);
                stbi_image_free(data);
                return result;
            },
            [&](fastgltf::sources::Array &vector) -> std::optional<mp::AllocatedImage> {
                const auto *bytes = reinterpret_cast<const std::uint8_t *>(vector.bytes.data());
                if (auto dds = tryLoadDDS(bytes, vector.bytes.size()))
                    return dds;
                return loadStbi(bytes, static_cast<int>(vector.bytes.size()));
            },
            [&](fastgltf::sources::BufferView &view) -> std::optional<mp::AllocatedImage> {
                const auto &bufferView = asset.bufferViews[view.bufferViewIndex];
                auto &buffer = asset.buffers[bufferView.bufferIndex];
                return std::visit(
                    fastgltf::visitor{
                        [](auto &) -> std::optional<mp::AllocatedImage> { return std::nullopt; },
                        [&](fastgltf::sources::Array &vector) -> std::optional<mp::AllocatedImage> {
                            const auto *bytes = reinterpret_cast<const std::uint8_t *>(
                                vector.bytes.data() + bufferView.byteOffset);
                            if (auto dds = tryLoadDDS(bytes, bufferView.byteLength))
                                return dds;
                            return loadStbi(bytes, static_cast<int>(bufferView.byteLength));
                        }},
                    buffer.data);
            },
        },
        image.data);
}

VkSampler load_sampler(mp::Engine &engine, fastgltf::Sampler &sampler)
{
    const VkSamplerCreateInfo samplerCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .magFilter = extract_filter(sampler.magFilter.value_or(fastgltf::Filter::Nearest)),
        .minFilter = extract_filter(sampler.minFilter.value_or(fastgltf::Filter::Nearest)),
        .mipmapMode = extract_mip_map_mode(sampler.minFilter.value_or(fastgltf::Filter::Nearest)),
        .anisotropyEnable = true,
        .maxAnisotropy = 16.0f,
        .minLod = 0,
        .maxLod = VK_LOD_CLAMP_NONE,
    };

    VkSampler vkSampler;
    vkCreateSampler(engine.m_device, &samplerCreateInfo, nullptr, &vkSampler) >> mp::chk;
    return vkSampler;
}

} // namespace

namespace mp
{
bool load_gltf(mp::Engine &engine, const std::filesystem::path &filePath)
{
    namespace cn = std::chrono;
    using namespace std::chrono_literals;
    if (!std::filesystem::exists(filePath))
    {
        std::println("Provided file path: {} is not a regular file or does not exists", filePath.string());
        return false;
    }
    std::println("Loading GLTF: {}", filePath.string());

    auto gltfFile = fastgltf::MappedGltfFile::FromPath(filePath);
    if (!gltfFile)
    {
        std::println("Failed to open glTF file: {}", fastgltf::getErrorMessage(gltfFile.error()));
        return false;
    }

    constexpr auto supportedExtensions =
        fastgltf::Extensions::KHR_mesh_quantization | fastgltf::Extensions::KHR_texture_transform |
        fastgltf::Extensions::KHR_lights_punctual | fastgltf::Extensions::KHR_materials_variants |
        fastgltf::Extensions::KHR_texture_basisu | fastgltf::Extensions::MSFT_texture_dds;
    fastgltf::Parser parser(supportedExtensions);

    constexpr auto gltfOptions = fastgltf::Options::DontRequireValidAssetMember | fastgltf::Options::AllowDouble |
                                 fastgltf::Options::LoadExternalBuffers | fastgltf::Options::LoadExternalImages |
                                 fastgltf::Options::DecomposeNodeMatrices | fastgltf::Options::GenerateMeshIndices;

    const auto start = cn::steady_clock::now();
    auto load = parser.loadGltf(gltfFile.get(), filePath.parent_path(), gltfOptions);
    if (load.error() != fastgltf::Error::None)
    {
        std::println("Failed to load glTF: {}", fastgltf::getErrorMessage(load.error()));
        return false;
    }
    fastgltf::Asset asset = std::move(load.get());

    Scene &file = engine.m_scene;

    std::unordered_map<std::size_t, std::uint32_t> samplers;
    std::unordered_map<std::size_t, std::uint32_t> images;
    std::vector<std::shared_ptr<GLTFMaterial>> materials;
    std::vector<std::shared_ptr<MeshAsset>> meshes;
    std::vector<std::shared_ptr<Node>> nodes;

    auto &[currentBuff, currentBuffAddr] = file.materialBuffers.emplace_back(
        engine.create_buffer(sizeof(GLTFMetallicRoughness::MaterialConstants) * asset.materials.size(),
                             VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                             VMA_MEMORY_USAGE_CPU_TO_GPU),
        VkDeviceAddress{0});
    const VkBufferDeviceAddressInfo addrInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext = nullptr,
        .buffer = currentBuff.buffer,
    };
    currentBuffAddr = vkGetBufferDeviceAddress(engine.m_device, &addrInfo);
    int dataIndex = 0;
    auto *sceneMaterialsConstants =
        static_cast<GLTFMetallicRoughness::MaterialConstants *>(currentBuff.allocationInfo.pMappedData);

    auto getImage = [&engine, &images, &file, &asset](const std::size_t imgIndex, const VkFormat format,
                                                      const VkImageUsageFlags imageUsage, std::uint32_t &instanceID) {
        if (const auto imageIt = images.find(imgIndex); imageIt != images.end())
        {
            instanceID = imageIt->second;
        }
        else
        {
            auto &fastgltfImage = asset.images.at(imgIndex);
            const auto imageOpt = load_image(engine, asset, fastgltfImage, format, imageUsage);
            if (imageOpt.has_value())
            {
                instanceID = engine.m_metalRoughness.write_texture(imageOpt.value().imageView);
                file.add_image(fastgltfImage.name.c_str(), imageOpt.value());
                images.try_emplace(imgIndex, instanceID);
            }
            else
            {
                std::println("Failed to load texture");
            }
        }
    };

    auto getSampler = [&file, &engine, &samplers, &asset](const std::size_t samplerIndex, std::uint32_t &instanceID) {
        if (samplerIndex >= asset.samplers.size())
        {
            instanceID = 0; // no sampler defined — use engine default (slot 0)
            return;
        }
        if (const auto samplerIt = samplers.find(samplerIndex); samplerIt != samplers.end())
        {
            instanceID = samplerIt->second;
        }
        else
        {
            auto &fastgltfSampler = asset.samplers.at(samplerIndex);
            const VkSampler sampler = load_sampler(engine, fastgltfSampler);
            if (sampler)
            {
                instanceID = engine.m_metalRoughness.write_sampler(sampler);
                file.samplers.push_back(sampler);
                samplers.try_emplace(samplerIndex, instanceID);
            }
            else
            {
                std::println("Failed to load sampler");
            }
        }
    };
    const auto fastgltfParseTime = cn::steady_clock::now();
    for (auto &material : asset.materials)
    {
        auto newMat = std::make_shared<GLTFMaterial>();
        file.add_material(material.name.c_str(), newMat);

        GLTFMetallicRoughness::MaterialConstants materialConstants;
        materialConstants.colorFactors = {material.pbrData.baseColorFactor[0], material.pbrData.baseColorFactor[1],
                                          material.pbrData.baseColorFactor[2], material.pbrData.baseColorFactor[3]};
        materialConstants.metalRoughFactors = {material.pbrData.metallicFactor, material.pbrData.roughnessFactor, 0.0f,
                                               0.0f};
        materialConstants.alphaCutoff = material.alphaMode == fastgltf::AlphaMode::Mask ? material.alphaCutoff : 0.0f;
        materialConstants.emissiveFactors = {material.emissiveFactor[0], material.emissiveFactor[1],
                                             material.emissiveFactor[2], 1.0f};
        sceneMaterialsConstants[dataIndex] = materialConstants;

        auto passType = MaterialPass::Opaque;
        if (material.alphaMode == fastgltf::AlphaMode::Blend)
        {
            passType = MaterialPass::Transparent;
        }
        else if (material.alphaMode == fastgltf::AlphaMode::Mask)
        {
            passType = MaterialPass::AlphaTested;
        }
        newMat->data.passType = passType;
        newMat->data.pipeline = engine.m_metalRoughness.select_pipeline(passType);
        newMat->data.indices = {
            .materialID = engine.m_metalRoughness.write_uniform_buffer(
                currentBuffAddr + dataIndex * sizeof(GLTFMetallicRoughness::MaterialConstants)),
            .colorTextureID = 0,
            .colorSamplerID = 0,
            .metalRoughnessTextureID = 0,
            .metalRoughnessSamplerID = 0,
            .normalTextureID = 2, // NOTE: Normal fallback texture index
            .normalSamplerID = 0,
            .emissiveTextureID = 1, // NOTE: Black fallback texture index
            .emissiveSamplerID = 0,
        };

        auto resolveSamplerIndex = [](const auto &asset, const auto textureIndex) {
            return asset.textures[textureIndex].samplerIndex.has_value()
                       ? asset.textures[textureIndex].samplerIndex.value()
                       : 0;
        };

        if (material.normalTexture.has_value())
        {
            const auto textureIndex = material.normalTexture.value().textureIndex;
            const auto imgIndex = asset.textures[textureIndex].imageIndex.value();
            const auto samplerIndex = resolveSamplerIndex(asset, textureIndex);
            getImage(imgIndex, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT,
                     newMat->data.indices.normalTextureID);
            getSampler(samplerIndex, newMat->data.indices.normalSamplerID);
        }
        if (material.pbrData.baseColorTexture.has_value())
        {
            const auto textureIndex = material.pbrData.baseColorTexture.value().textureIndex;
            const auto imgIndex = asset.textures[textureIndex].imageIndex.value();
            const auto samplerIndex = resolveSamplerIndex(asset, textureIndex);
            getImage(imgIndex, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_USAGE_SAMPLED_BIT,
                     newMat->data.indices.colorTextureID);
            getSampler(samplerIndex, newMat->data.indices.colorSamplerID);
        }
        if (material.pbrData.metallicRoughnessTexture.has_value())
        {
            const auto textureIndex = material.pbrData.metallicRoughnessTexture.value().textureIndex;
            const auto imgIndex = asset.textures[textureIndex].imageIndex.value();
            const auto samplerIndex = resolveSamplerIndex(asset, textureIndex);
            getImage(imgIndex, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT,
                     newMat->data.indices.metalRoughnessTextureID);
            getSampler(samplerIndex, newMat->data.indices.metalRoughnessSamplerID);
        }
        if (material.emissiveTexture.has_value())
        {
            const auto textureIndex = material.emissiveTexture.value().textureIndex;
            const auto imgIndex = asset.textures[textureIndex].imageIndex.value();
            const auto samplerIndex = resolveSamplerIndex(asset, textureIndex);
            getImage(imgIndex, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_USAGE_SAMPLED_BIT,
                     newMat->data.indices.emissiveTextureID);
            getSampler(samplerIndex, newMat->data.indices.emissiveSamplerID);
        }

        materials.push_back(std::move(newMat));
        dataIndex++;
    }
    const auto materialsTextureSamplersLoadingTime = cn::steady_clock::now();

    // Accumulate all mesh data on the CPU before a single GPU upload
    std::vector<Vertex> allVertices;
    std::vector<mp::VertexPosition> allPositions;
    std::vector<mp::VertexAttributes> allAttributes;
    std::vector<std::uint32_t> allIndices;
    const std::size_t globalVertexBase = engine.m_globalPositionCount;
    const std::size_t globalIndexBase = engine.m_globalIndexCount;

    for (auto &mesh : asset.meshes)
    {
        MeshAsset newMesh;
        newMesh.name = mesh.name;

        for (auto &p : mesh.primitives)
        {
            GeoSurface newSurface;
            newSurface.count = static_cast<std::uint32_t>(asset.accessors[p.indicesAccessor.value()].count);
            newSurface.min = glm::vec3(0.0f);
            newSurface.max = glm::vec3(0.0f);

            std::vector<Vertex> primVertices;
            std::vector<std::uint32_t> primIndices;

            // load indexes (0-based, relative to primVertices)
            {
                fastgltf::Accessor &indexAccessor = asset.accessors[p.indicesAccessor.value()];
                primIndices.reserve(indexAccessor.count);
                fastgltf::iterateAccessor<std::uint32_t>(asset, indexAccessor,
                                                         [&](const std::uint32_t idx) { primIndices.push_back(idx); });
            }

            // load vertex positions
            {
                fastgltf::Accessor &posAccessor = asset.accessors[p.findAttribute("POSITION")->accessorIndex];
                primVertices.resize(posAccessor.count);

                newSurface.min =
                    std::visit(fastgltf::visitor{[](const std::monostate &e) { return glm::vec3(0.0f); },
                                                 [](const auto &min) { return glm::vec3(min[0], min[1], min[2]); }},
                               posAccessor.min);
                newSurface.max =
                    std::visit(fastgltf::visitor{[](const std::monostate &e) { return glm::vec3(0.0f); },
                                                 [](const auto &max) { return glm::vec3(max[0], max[1], max[2]); }},
                               posAccessor.max);

                fastgltf::iterateAccessorWithIndex<glm::vec3>(
                    asset, posAccessor, [&](const glm::vec3 v, const size_t index) {
                        primVertices[index].pos = v;
                        primVertices[index].normal = QuantizeNormal({1, 0, 0});
                        primVertices[index].uv = glm::packHalf(glm::vec2(0.0f, 0.0f));
                    });
            }

            // load vertex normals
            auto normals = p.findAttribute("NORMAL");
            if (normals != p.attributes.end())
            {
                fastgltf::iterateAccessorWithIndex<glm::vec3>(
                    asset, asset.accessors[normals->accessorIndex],
                    [&](const glm::vec3 v, const size_t index) { primVertices[index].normal = QuantizeNormal(v); });
            }

            // load UVs
            auto uv = p.findAttribute("TEXCOORD_0");
            if (uv != p.attributes.end())
            {
                fastgltf::iterateAccessorWithIndex<glm::vec2>(
                    asset, asset.accessors[uv->accessorIndex],
                    [&](const glm::vec2 v, const size_t index) { primVertices[index].uv = glm::packHalf(v); });
            }

            // load tangents
            auto tangent = p.findAttribute("TANGENT");
            if (tangent != p.attributes.end())
            {
                assert(uv != p.attributes.end());
                fastgltf::iterateAccessorWithIndex<glm::vec4>(asset, asset.accessors[tangent->accessorIndex],
                                                              [&](const glm::vec4 v, const std::size_t index) {
                                                                  primVertices[index].tangent = QuantizeTangent(v);
                                                              });
            }
            else
            {
                for (auto &vtx : primVertices)
                {
                    vtx.tangent = QuantizeTangent(glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
                }
            }

            {
                std::vector<std::uint32_t> remap(primVertices.size());
                const std::size_t uniqueVertexCount =
                    meshopt_generateVertexRemap(remap.data(), primIndices.data(), primIndices.size(),
                                                primVertices.data(), primVertices.size(), sizeof(Vertex));

                std::vector<std::uint32_t> remappedIndices(primIndices.size());
                std::vector<Vertex> remappedVertices(uniqueVertexCount);
                meshopt_remapIndexBuffer(remappedIndices.data(), primIndices.data(), primIndices.size(), remap.data());
                meshopt_remapVertexBuffer(remappedVertices.data(), primVertices.data(), primVertices.size(),
                                          sizeof(Vertex), remap.data());

                primIndices = std::move(remappedIndices);
                primVertices = std::move(remappedVertices);
            }

            meshopt_optimizeVertexCache(primIndices.data(), primIndices.data(), primIndices.size(),
                                        primVertices.size());

            {
                std::vector<float> positions(primVertices.size() * 3);
                for (std::size_t i = 0; i < primVertices.size(); ++i)
                {
                    positions[i * 3 + 0] = primVertices[i].pos.x;
                    positions[i * 3 + 1] = primVertices[i].pos.y;
                    positions[i * 3 + 2] = primVertices[i].pos.z;
                }
                meshopt_optimizeOverdraw(primIndices.data(), primIndices.data(), primIndices.size(), positions.data(),
                                         primVertices.size(), 3 * sizeof(float), 1.05f);
            }

            meshopt_optimizeVertexFetch(primVertices.data(), primIndices.data(), primIndices.size(),
                                        primVertices.data(), primVertices.size(), sizeof(Vertex));

            newSurface.startIndex = static_cast<std::uint32_t>(allIndices.size());
            newSurface.vertexOffset = static_cast<std::int32_t>(allVertices.size());

            allVertices.insert(allVertices.end(), primVertices.begin(), primVertices.end());
            allIndices.insert(allIndices.end(), primIndices.begin(), primIndices.end());

            for (const auto &v : primVertices)
            {
                allPositions.push_back({v.pos});
                allAttributes.push_back({v.uv, v.normal, v.tangent});
            }

            if (p.materialIndex.has_value())
            {
                newSurface.material = materials[p.materialIndex.value()];
            }
            else
            {
                std::println("Mesh does not contains material");
                newSurface.material = materials.at(0);
            }

            newMesh.geoSurfaces.push_back(newSurface);
        }

        
        meshes.emplace_back(std::make_shared<MeshAsset>(std::move(newMesh)));
        file.add_mesh(meshes.back());
    }

    engine.ensure_position_capacity(allPositions.size());
    engine.ensure_attributes_capacity(allAttributes.size());
    engine.ensure_index_capacity(allIndices.size());

    AllocatedBuffer staging{};
    engine.immediate_submit([&](VkCommandBuffer cmd) {
        const std::size_t pSize = allPositions.size() * sizeof(mp::VertexPosition);
        const std::size_t aSize = allAttributes.size() * sizeof(mp::VertexAttributes);
        const std::size_t iSize = allIndices.size() * sizeof(std::uint32_t);

        staging =
            engine.create_buffer(pSize + aSize + iSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
        char *dst = static_cast<char *>(staging.allocationInfo.pMappedData);
        std::memcpy(dst, allPositions.data(), pSize);
        std::memcpy(dst + pSize, allAttributes.data(), aSize);
        std::memcpy(dst + pSize + aSize, allIndices.data(), iSize);

        const VkBufferCopy pCopy{
            .srcOffset = 0,
            .dstOffset = globalVertexBase * sizeof(mp::VertexPosition),
            .size = pSize,
        };
        vkCmdCopyBuffer(cmd, staging.buffer, engine.m_globalPositionBuffer.buffer, 1, &pCopy);

        const VkBufferCopy aCopy{
            .srcOffset = pSize,
            .dstOffset = globalVertexBase * sizeof(mp::VertexAttributes),
            .size = aSize,
        };
        vkCmdCopyBuffer(cmd, staging.buffer, engine.m_globalAttributesBuffer.buffer, 1, &aCopy);

        const VkBufferCopy iCopy{
            .srcOffset = pSize + aSize,
            .dstOffset = globalIndexBase * sizeof(std::uint32_t),
            .size = iSize,
        };
        vkCmdCopyBuffer(cmd, staging.buffer, engine.m_globalIndexBuffer.buffer, 1, &iCopy);
    });
    engine.destroy_buffer(staging);

    engine.m_globalPositionCount += allPositions.size();
    engine.m_globalAttributesCount += allAttributes.size();
    engine.m_globalIndexCount += allIndices.size();

    // Adjust all surface offsets from accumulated-array-local to global GPU offsets
    for (auto &mesh : meshes)
    {
        for (auto &surf : mesh->geoSurfaces)
        {
            surf.startIndex += static_cast<std::uint32_t>(globalIndexBase);
            surf.vertexOffset += static_cast<std::int32_t>(globalVertexBase);
        }
    }
    const auto meshProcessing = cn::steady_clock::now();

    // Adding nodes
    for (auto &node : asset.nodes)
    {
        std::shared_ptr<Node> newNode;
        if (node.meshIndex.has_value())
        {
            newNode = std::make_shared<MeshNode>(meshes[node.meshIndex.value()]);
        }
        else if (node.lightIndex.has_value())
        {
            auto &gltfLight = asset.lights[node.lightIndex.value()];
            const glm::vec3 color{gltfLight.color.x(), gltfLight.color.y(), gltfLight.color.z()};
            if (gltfLight.type == fastgltf::LightType::Point)
            {
                newNode = std::make_shared<PointLightNode>(PointLightData{
                    .position = {},
                    .range = gltfLight.range.value_or(10.0f),
                    .color = color,
                    .intensity = gltfLight.intensity,
                    .normalBias = 0.03f,
                    .constantBias = 0.001f,
                });
            }
            else
            {
                newNode = std::make_shared<DirectionalLightNode>(DirectionalLightData{
                    .direction = {},
                    .cascadeCount = 4,
                    .color = color,
                    .intensity = gltfLight.intensity,
                    .normalBias = 0.001f,
                    .constantBias = 0.001f,
                });
            }
        }
        else
        {
            newNode = std::make_shared<Node>();
        }

        if (node.name.empty())
        {
            node.name = std::format("Node_{}_{}", filePath.filename().string(), nodes.size());
        }
        newNode->name = node.name;
        nodes.push_back(newNode);
        newNode->nodeIndex = file.add_node(newNode);

        std::visit(fastgltf::visitor{[&](fastgltf::math::fmat4x4 matrix) {
                                         memcpy(&newNode->localTransform, matrix.data(), sizeof(matrix));
                                     },
                                     [&](fastgltf::TRS transform) {
                                         const glm::vec3 tl(transform.translation[0], transform.translation[1],
                                                            transform.translation[2]);
                                         const glm::quat rot(transform.rotation[3], transform.rotation[0],
                                                             transform.rotation[1], transform.rotation[2]);
                                         const glm::vec3 sc(transform.scale[0], transform.scale[1], transform.scale[2]);

                                         const glm::mat4 tm = glm::translate(glm::mat4(1.f), tl);
                                         const glm::mat4 rm = glm::toMat4(rot);
                                         const glm::mat4 sm = glm::scale(glm::mat4(1.f), sc);

                                         newNode->localTransform = tm * rm * sm;
                                     }},
                   node.transform);
    }

    for (const auto [assetNode, sceneNode] : std::views::zip(asset.nodes, nodes))
    {
        for (const auto &c : assetNode.children)
        {
            sceneNode->children.push_back(nodes[c]);
            nodes[c]->parent = sceneNode;
        }
    }

    for (auto &node : nodes)
    {
        if (node->parent.lock() == nullptr)
        {
            file.topNodes.push_back(node);
            node->refresh_transform(glm::mat4(1.0f));
        }
    }
    const auto nodeProcessingTime = cn::steady_clock::now();

    std::println("FastglTF parsing time: {}",
                 cn::duration_cast<cn::milliseconds>(fastgltfParseTime - start).count() / 1000.0f);
    std::println("Materials + texturs + samplers parsing time: {}",
                 cn::duration_cast<cn::milliseconds>(materialsTextureSamplersLoadingTime - fastgltfParseTime).count() /
                     1000.0f);
    std::println("Meshes parsing time: {}",
                 cn::duration_cast<cn::milliseconds>(meshProcessing - materialsTextureSamplersLoadingTime).count() /
                     1000.0f);
    std::println("Node parsing time: {}",
                 cn::duration_cast<cn::milliseconds>(nodeProcessingTime - meshProcessing).count() / 1000.0f);
    return true;
}

std::optional<AllocatedImage> load_hdr(mp::Engine &engine, const std::filesystem::path &filePath)
{
    if (!std::filesystem::exists(filePath))
    {
        std::println("HDR file not found: {}", filePath.string());
        return std::nullopt;
    }

    int width, height, channels;
    float *data = stbi_loadf(filePath.string().c_str(), &width, &height, &channels, 4);
    if (!data)
    {
        std::println("Failed to load HDR '{}': {}", filePath.string(), stbi_failure_reason());
        return std::nullopt;
    }

    const VkExtent3D extent{
        static_cast<std::uint32_t>(width),
        static_cast<std::uint32_t>(height),
        1u,
    };
    constexpr std::uint32_t kBytesPerPixel = 4 * sizeof(float); // RGBA f32 = 16 bytes
    const std::size_t bufferSize = static_cast<std::size_t>(width) * height * kBytesPerPixel;

    AllocatedImage image = engine.create_image(extent, VK_FORMAT_R32G32B32A32_SFLOAT,
                                                     VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    const AllocatedBuffer stagingBuffer =
        engine.create_buffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    std::memcpy(stagingBuffer.allocationInfo.pMappedData, data, bufferSize);
    stbi_image_free(data);

    engine.immediate_submit([&](const VkCommandBuffer cmd) {
        utils::BarrierBuilder barrierBuilder;
        barrierBuilder.add_image_barrier(image.transition(
            {.stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
             .accessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
             .layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
             .queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .subresourceRange = utils::init_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT)}));
        barrierBuilder.barrier(cmd);

        const VkBufferImageCopy copyRegion{
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageOffset = {},
            .imageExtent = extent,
        };
        vkCmdCopyBufferToImage(cmd, stagingBuffer.buffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                               &copyRegion);

        utils::transition_image(cmd, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    engine.destroy_buffer(stagingBuffer);

    if (image.image == nullptr)
    {
        return std::nullopt;
    }
    return image;
}
} // namespace mp