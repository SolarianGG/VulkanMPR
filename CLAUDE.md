# Project: VulkanMPR 
- This is a Vulkan renderer project using C++, Slang shaders (.slang), and CMake. The codebase implements cascaded shadow mapping (CSM/SDSM), glTF loading, and GPU-driven rendering.


## Code Style

- Use C++ Core Guidelines and Google C++ Style guide for code quality
- Use @.clang-format for code style

## Commands

- `cmake --build build`: For building of the project
- `python ./shader_cmp.py`: For shader compilation
- `vcpkg add port *LibraryName*`: For installing libraries

## Architecture

- `/src/shaders`: Directory containing all the shaders
- `/src/private`: Directory containing .cpp files
- `/src/public`: Directory containint header files
- `/assets/`: Directory containing glTF assets

## Debugging Guidelines 
- When debugging rendering issues (shadows, lighting, matrices), ask clarifying questions and request concrete data (screenshots, CSV dumps, debug output) before proposing fixes. Do not guess at root causes.

## Workflow
- When asked to analyze code or debug an issue, provide the analysis/diagnosis ONLY. Do not start writing implementation plans or making code changes unless explicitly asked to do so.
- Be sure that project compiles after making series of code changes
- Always ask for addition of any library (vcpkg)
- If you are not sure abouth something, ask right away
- After making code changes, always verify the build compiles successfully by running the CMake build command before reporting completion.
- Prefer precise, minimal fixes over broad refactors. Do not add unnecessary features or change code beyond what was requested.