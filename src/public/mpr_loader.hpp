#pragma once

#include <filesystem>
#include <optional>
#include <unordered_map>

#include "mpr_types.hpp"
#include "mpr_engine.hpp"

namespace mp {

std::optional<std::shared_ptr<LoadedGLTF>> load_gltf(
    mp::Engine& engine, const std::filesystem::path& filePath);

}