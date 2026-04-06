## VulkanMPR
A high-performance, modern graphics engine built in C++ and Vulkan, designed for real-time rendering and experimentation with advanced rendering techniques.

## Dependencies:
- SDL3
- Vulkan
- GLM
- fastgltf 
- Volk
- meshoptimizer
- Imgui
- Imguizmo
- Vulkan Memory Allocator
- Vulkan-bootstrap
- Stb

## Requirements:
- CMake 3.28 or higher
- Vcpkg
- A CMake compatible builder (Ninja or Visual Studio)
- python 3.x 
- Git
- Vulkan SDK
- GPU that supports descriptor buffers, spirv 1.6 / shader model 6.5

## Building:
- git clone --recursive https://github.com/SolarianGG/VulkanMPR.git
- python ./vcpkg_init_project.py
- cmake --preset=default

### Debug (default):
- python ./shader_cmp.py
- cmake --build build

### Release:
- python ./shader_cmp.py --release
- cmake --build build --config Release


## Core

- Fully **GPU-Driven** architecture.
- **Bindless rendering** for large numbers of textures and buffers via **descriptor buffers** extension.
- **Slang** shader language integration
- **glTF** model loading
- **Scene graph**
- **Metal-roughness** PBR materials
- **Debug labels** for more convenient RenderDoc + Nvidia nsights pipeline inspection

## Rendering features

- **Deferred shading**
- **SDSM** (Sample distribution shadow maps) **CSM** (Cascaded Shadow Maps) for directional light shadows.
- **Omnidirectional tiled tetrahedron shadow mapping** for point light shadows
- GPU Visible point lights culling
- GPU Meshes **frustum culling**
- **Gamma correction**
- **WBOIT** (Weight blended order independent transparency)
- **Normal mapping**

## Demo

### Pipeline overview

<table>
  <tr>
    <td><img src="screenshots/pipeline_overview.png" width="600"/></td>
    <td><img src="screenshots/pipelines_frame_time.png" width="600"/></td>
  </tr>
</table>

### Sponza

<table>
  <tr>
    <td><img src="screenshots/Sponza_1.png" width="600"/></td>
    <td><img src="screenshots/Sponza_2.png" width="600"/></td>
  </tr>
  <tr>
    <td><img src="screenshots/Sponza_3.png" width="600"/></td>
    <td><img src="screenshots/Sponza_4.png" width="600"/></td>
  </tr>
</table>

<table>
    <tr>
    <td><img src="screenshots/Full_Scene.png" width="1100"/></td>
  </tr>
</table>

### Shadows

#### SDSM CSM

<table>
  <tr>
    <td><img src="screenshots/CSM_1_cascade.png" width="600"/></td>
    <td><img src="screenshots/CSM_1_cascade_final.png" width="600"/></td>
  </tr>
  <tr>
    <td><img src="screenshots/CSM_2_cascades.png" width="600"/></td>
    <td><img src="screenshots/CSM_2_cascades_final.png" width="600"/></td>
  </tr>
  <tr>
    <td><img src="screenshots/CSM_3_cascades.png" width="600"/></td>
    <td><img src="screenshots/CSM_3_cascades_final.png" width="600"/></td>
  </tr>
  <tr>
    <td><img src="screenshots/CSM_4_cascades.png" width="600"/></td>
    <td><img src="screenshots/CSM_4_cascades_final.png" width="600"/></td>
  </tr>
</table>

#### Difference between SDSM CSM and Simple Directional shadow mapping

<table>
  <tr>
    <td><img src="screenshots/dsm.png" width="600"/></td>
    <td><img src="screenshots/csm.png" width="600"/></td>
  </tr>
  <tr>
    <td><img src="screenshots/dsm_screen.png" width="600"/></td>
    <td><img src="screenshots/csm_cascades.png" width="600"/></td>
  </tr>
</table>

#### Omnidirectional tiled tetrahedron shadow mapping

<table>
  <tr>
    <td><img src="screenshots/tetrahedron_shadow_map.png" width="600"/></td>
    <td><img src="screenshots/Sponza_1.png" width="600"/></td>
  </tr>
</table>

#### Avoiding Shadow acne and Peter panning by using normal offset bias + constant bias

<table>
  <tr>
    <td><img src="screenshots/peter_panning.png" width="600"/></td>
    <td><img src="screenshots/PCF_Bias.png" width="600"/></td>
  </tr>
</table>


### Normal mapping

<table>
  <tr>
    <td><img src="screenshots/nonp.png" width="600"/></td>
    <td><img src="screenshots/np.png" width="600"/></td>
  </tr>
</table>


### Gamma correction 

<table>
  <tr>
    <td><img src="screenshots/no_gamma.png" width="600"/></td>
    <td><img src="screenshots/gamma.png" width="600"/></td>
  </tr>
</table>

### WBOIT

<table>
  <tr>
    <td><img src="screenshots/wboit.png" width="600"/></td>
  </tr>
</table>


### Scene manipulation

<table>
  <tr>
    <td><img src="screenshots/scene-manip-tr.png" width="400"/></td>
    <td><img src="screenshots/scene-manip-rt.png" width="400"/></td>
    <td><img src="screenshots/scene-manip-sc.png" width="400"/></td>
  </tr>
</table>


## Reference papers

[SDSM](https://www.researchgate.net/publication/220791941_Sample_distribution_Shadow_Maps)

Omdirectional tiled tetrahedron shadow mapping (GPU Pro 6)

[Vertex data quantization](https://daniilvinn.github.io/2024/05/04/omniforce-vertex-quantization.html)



## Planned rendering feautres

- GT7 Tone mapping
- Auto exposure
- Switch from Lambert to Burley diffuse BRDF
- IBL
- TAA / FXAA / SMAA
- SSAO
- SSR
- Bend studio contact shadows
- Occlusion culling
- Bloom
- Depth of field
- Motion blur
- Lens flare
- Chromatic abberation
- SSGI
- Clustered deferred
- Descriptor heaps
- Frame graph
- BC texture compression
- Async compute


