#pragma once

#include <filesystem>
#include <optional>
#include <unordered_map>

#include "mpr_types.hpp"
#include "mpr_engine.hpp"

namespace mp {
bool load_gltf(
    mp::Engine& engine, const std::filesystem::path& filePath);

}