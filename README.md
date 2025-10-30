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
- cmake --preset=default
- cmake --build build

### Screenshots:
![Screenshot1](./screenshots/structure.png)
![Screenshot2](./screenshots/sponza.png)

### Implemented Features:
- Vulkan forward renderer
- GLTF model loading
- Bindless material model via descriptor buffers extension
- Slang shader language integration
- Batching and instancing

### Planned Features
**Rendering & Lighting**
- Local illumination models (Blinn-Phong, Lambert)
- Shadow mapping & Cascaded shadow maps
- PBR (Cook-Torrance)
- Environment mapping, skybox, global illumination

**Architecture**
- Scene graph
- Frame graph
- Multithreaded rendering (TaskFlow)
- GPU-driven & mesh shader-based rendering

**Effects**
- Post-processing
- Particle system
- Hybrid rasterization and ray tracing

