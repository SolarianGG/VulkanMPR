### Vulkan Learning Project

This is my personal Vulkan learning project, where I explore and implement various graphics techniques using the Vulkan API.

### Overview

The goal of this project is to build a modular and modern Vulkan-based rendering engine while deepening my understanding of real-time graphics systems.
It focuses on learning GPU resource management, descriptor systems, shader reflection, and advanced rendering techniques such as bindless materials and batching.

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

### Building:
- git clone --recursive https://github.com/SolarianGG/VulkanMPR.git
- python ./vcpkg_init_project.py
- python ./shader_cmp.py
- cmake --preset=default
- cmake --build build

### Implemented Features:
- Vulkan deferred renderer

![Deferred](./screenshots/deferred.png)
- GLTF model loading
- Bindless material model via descriptor buffers extension
- Slang shader language integration
- Batching and instancing
![Instancing](./screenshots/instancing.png)
- Scene graph (transforms, nodes, cameras, lights, materials)
![Scenegraph](./screenshots/scene-manip-tr.png)
- Normal mapping

Without normal mapping
![WithoutNM](./screenshots/nomp.png)
With normal mapping
![NM](./screenshots/mp.png)
- Metal-roughness PBR material
- Weight blended order independent transparency
![WBOIT](./screenshots/wboit.png)
- Gamma correction on albedo textures + postprocess pass


### Planned Features
- Indirect drawing
- Frustrum culling
- Render doc debug markers
- Directional shadows
- HDR + tonemap + gamma correction
- PBR
- IBL
- Bloom
- CSM
- SSAO
- Multithreaded cmd record
- Blur
- TAA

