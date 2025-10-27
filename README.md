Vulkan Learning Project

That is my Vulkan API learning project, where I implement various Vulkan API + Computer graphic techniques.

Building

In order to build the project, you must have cmake, vcpkg and cmake compatible builder (i.e Ninja or Visual studio)

Run:
  python ./vcpkg_init_project.py
  cmake --preset=default
  cmake --build build
