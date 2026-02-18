# Project: VulkanMPR 

See @README.md for project overview


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

## Important notes

- Always ask for addition of any library (vcpkg)
- If you are not sure abouth something, ask right away