cmake_minimum_required (VERSION 3.29)

if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("VulkanMPR")


set (CMAKE_CXX_STANDARD 23)
set (CMAKE_CXX_STANDARD_REQUIRED True)

set(SRC_PATH "${PROJECT_SOURCE_DIR}/src")
set(INCLUDE_PATH "${SRC_PATH}/public/")
set(COMP_PATH "${SRC_PATH}/private/")

find_package(Vulkan REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(vk-bootstrap CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(SDL3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(fastgltf CONFIG REQUIRED)

add_executable(VulkanMPR "${SRC_PATH}/VulkanMPR.cpp"
                         "${COMP_PATH}/mpr_engine.cpp"
                         "${COMP_PATH}/mpr_error_check.cpp"
                         "${COMP_PATH}/mpr_image.cpp"
                         "${COMP_PATH}/mpr_descriptors.cpp"
                         "${COMP_PATH}/mpr_pipelines.cpp"
                         "${COMP_PATH}/mpr_loader.cpp" 
                          "${COMP_PATH}/mpr_camera.cpp")
target_include_directories(VulkanMPR PRIVATE ${INCLUDE_PATH} ${SRC_PATH})
target_link_libraries(VulkanMPR PRIVATE 
  Vulkan::Vulkan 
  GPUOpen::VulkanMemoryAllocator
  vk-bootstrap::vk-bootstrap vk-bootstrap::vk-bootstrap-compiler-warnings
  ${Stb_INCLUDE_DIR}
  SDL3::SDL3
  imgui::imgui
  glm::glm
  fastgltf::fastgltf
)

