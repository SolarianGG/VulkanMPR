#pragma once

#include <filesystem>
#include <optional>
#include <unordered_map>

#include "mpr_types.hpp"
#include "mpr_engine.hpp"

namespace mp {


struct GLTFMaterial {
  MaterialInstance data;
};

struct GeoSurface {
  std::uint32_t startIndex;
  std::uint32_t count;
  std::shared_ptr<GLTFMaterial> material;
};

struct MeshAsset {
  std::string name;

  std::vector<GeoSurface> geoSurfaces;
  GpuMeshBuffers meshBuffers;
};

std::optional<std::shared_ptr<LoadedGLTF>> load_gltf(
    mp::Engine& engine, const std::filesystem::path& filePath);

}