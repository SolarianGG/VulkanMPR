cmake_minimum_required (VERSION 3.29)

if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("VulkanMPR")


set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(VCPKG_CXX_FLAGS "-DIMGUI_IMPL_VULKAN_USE_LOADER")

set(SRC_PATH "${PROJECT_SOURCE_DIR}/src")
set(INCLUDE_PATH "${SRC_PATH}/public/")
set(COMP_PATH "${SRC_PATH}/private/")
set(IMGUI_PATH "${PROJECT_SOURCE_DIR}/external/imgui")

find_package(Vulkan REQUIRED)
find_package(volk CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(vk-bootstrap CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(SDL3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(fastgltf CONFIG REQUIRED)

# Have to build ImGui from source since vcpkg's imgui package are prebuild without Volk support
add_library(ImGui STATIC 
  "${IMGUI_PATH}/imgui.cpp"
  "${IMGUI_PATH}/imgui_draw.cpp"
  "${IMGUI_PATH}/imgui_demo.cpp"
  "${IMGUI_PATH}/imgui_tables.cpp"
  "${IMGUI_PATH}/imgui_widgets.cpp"
  "${IMGUI_PATH}/misc/cpp/imgui_stdlib.cpp"
  "${IMGUI_PATH}/backends/imgui_impl_vulkan.cpp"
  "${IMGUI_PATH}/backends/imgui_impl_sdl3.cpp"
)

target_include_directories(ImGui PRIVATE ${IMGUI_PATH} "${IMGUI_PATH}/backends")

target_compile_definitions(ImGui PRIVATE 
  IMGUI_IMPL_VULKAN_USE_VOLK
)

target_link_libraries(ImGui PRIVATE 
  Vulkan::Vulkan
  volk::volk volk::volk_headers
  SDL3::SDL3
)

add_executable(VulkanMPR "${SRC_PATH}/VulkanMPR.cpp"
                         "${COMP_PATH}/mpr_engine.cpp"
                         "${COMP_PATH}/mpr_error_check.cpp"
                         "${COMP_PATH}/mpr_image.cpp"
                         "${COMP_PATH}/mpr_descriptors.cpp"
                         "${COMP_PATH}/mpr_pipelines.cpp"
                         "${COMP_PATH}/mpr_loader.cpp" 
                          "${COMP_PATH}/mpr_camera.cpp")

target_include_directories(VulkanMPR PRIVATE ${INCLUDE_PATH} ${SRC_PATH} ${IMGUI_PATH} "${IMGUI_PATH}/backends")
target_link_libraries(VulkanMPR PRIVATE 
  Vulkan::Vulkan 
  GPUOpen::VulkanMemoryAllocator
  vk-bootstrap::vk-bootstrap vk-bootstrap::vk-bootstrap-compiler-warnings
  ${Stb_INCLUDE_DIR}
  SDL3::SDL3
  glm::glm
  fastgltf::fastgltf
  volk::volk volk::volk_headers
  ImGui
)

