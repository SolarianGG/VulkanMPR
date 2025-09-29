#pragma once

#include <filesystem>
#include <optional>
#include <unordered_map>

#include "mpr_types.hpp"

namespace mp {

struct GeoSurface {
  std::uint32_t startIndex;
  std::uint32_t count;
};

struct MeshAsset {
  std::string name;

  std::vector<GeoSurface> geoSurfaces;
  GpuMeshBuffers meshBuffers;
};

class Engine;

std::optional<std::vector<std::shared_ptr<MeshAsset>>> load_mesh(
    mp::Engine& engine, const std::filesystem::path& filePath);

}