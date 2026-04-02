### VulkanMPR
A high-performance, modern graphics engine built in C++ and Vulkan, designed for real-time rendering and experimentation with advanced rendering techniques.

### Dependencies:
- SDL3
- Vulkan
- GLM
- fastgltf 
- Volk
- Imgui
- Imguizmo
- Vulkan Memory Allocator
- Vulkan-bootstrap
- Stb

### Requirements:
- CMake 3.28 or higher
- Vcpkg
- A CMake compatible builder (Ninja or Visual Studio)
- python 3.x 
- Git
- Vulkan SDK
- GPU that supports descriptor buffers, spirv 1.6 / shader model 6.5

### Building:
- git clone --recursive https://github.com/SolarianGG/VulkanMPR.git
- python ./vcpkg_init_project.py
- cmake --preset=default

#### Debug (default):
- python ./shader_cmp.py
- cmake --build build

#### Release:
- python ./shader_cmp.py --release
- cmake --build build --config Release


### Core

- Fully **GPU-Driven** architecture.
- **Bindless rendering** for large numbers of textures and buffers via **descriptor buffers** extension.
- **Slang** shader language integration
- **glTF** model loading
- **Scene graph**
- **Metal-roughness** PBR materials

### Rendering features

- 

### Rendering features overview

### Perfomance Overview


