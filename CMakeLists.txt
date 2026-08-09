cmake_minimum_required (VERSION 3.28)

project ("VulkanMPR")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(VCPKG_CXX_FLAGS "-DIMGUI_IMPL_VULKAN_USE_LOADER")


set(SRC_PATH "${PROJECT_SOURCE_DIR}/src")
set(INCLUDE_PATH "${SRC_PATH}/public/")
set(COMP_PATH "${SRC_PATH}/private/")
set(IMGUI_PATH "${PROJECT_SOURCE_DIR}/external/imgui")
set(IMGUIZMO_PATH "${PROJECT_SOURCE_DIR}/external/imguizmo")
set(DDSIMAGE_PATH "${PROJECT_SOURCE_DIR}/external/dds_image")


find_package(Vulkan REQUIRED)
find_package(volk CONFIG REQUIRED)
find_package(VulkanUtilityLibraries CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(vk-bootstrap CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(SDL3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(fastgltf CONFIG REQUIRED)
find_package(meshoptimizer CONFIG REQUIRED)
find_package(tinyfiledialogs CONFIG REQUIRED)
find_package(Tracy CONFIG REQUIRED)

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

add_library(ImGuizmo STATIC 
  "${IMGUIZMO_PATH}/ImGuizmo.cpp"
  "${IMGUIZMO_PATH}/ImSequencer.cpp"
  "${IMGUIZMO_PATH}/GraphEditor.cpp"
  "${IMGUIZMO_PATH}/ImCurveEdit.cpp"
  "${IMGUIZMO_PATH}/ImGradient.cpp"
)

target_include_directories(ImGuizmo PRIVATE ${IMGUIZMO_PATH} ${IMGUI_PATH})
target_link_libraries(ImGuizmo PRIVATE ImGui)

add_executable(VulkanMPR "${SRC_PATH}/VulkanMPR.cpp"
                         "${COMP_PATH}/mpr_engine.cpp"
                         "${COMP_PATH}/mpr_engine_draw.cpp"
                         "${COMP_PATH}/mpr_engine_init.cpp"
                         "${COMP_PATH}/mpr_engine_resources.cpp"
                         "${COMP_PATH}/mpr_engine_ui.cpp"
                         "${COMP_PATH}/mpr_error_check.cpp"
                         "${COMP_PATH}/mpr_image.cpp"
                         "${COMP_PATH}/mpr_descriptors.cpp"
                         "${COMP_PATH}/mpr_pipelines.cpp"
                         "${COMP_PATH}/mpr_loader.cpp"
                         "${COMP_PATH}/mpr_camera.cpp"
                         "${COMP_PATH}/mpr_materials.cpp"
                         "${COMP_PATH}/mpr_scene.cpp"
 )

target_include_directories(VulkanMPR PRIVATE
  ${INCLUDE_PATH}
  ${SRC_PATH} 
  ${IMGUI_PATH} 
  ${IMGUIZMO_PATH}
  ${Stb_INCLUDE_DIR}
  "${IMGUI_PATH}/backends"
  "${DDSIMAGE_PATH}"
  )


target_link_libraries(VulkanMPR PRIVATE
  Vulkan::Vulkan
  GPUOpen::VulkanMemoryAllocator
  vk-bootstrap::vk-bootstrap vk-bootstrap::vk-bootstrap-compiler-warnings
  SDL3::SDL3
  glm::glm
  fastgltf::fastgltf
  volk::volk volk::volk_headers
  Vulkan::UtilityHeaders
  meshoptimizer::meshoptimizer
  tinyfiledialogs::tinyfiledialogs
  ImGui
  ImGuizmo
  Tracy::TracyClient
)

target_compile_definitions(VulkanMPR PRIVATE
  $<$<CONFIG:Debug>:MPR_DEBUG>
  TRACY_ENABLE
)

if (WIN32)
add_custom_command(TARGET VulkanMPR POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E copy
  ${VCPKG_INSTALLED_DIR}/x64-windows/bin/TracyClient.dll
  $<TARGET_FILE_DIR:VulkanMPR>/TracyClient.dll
  COMMENT "Copying tracy client DLL to output directory"
)
endif()

