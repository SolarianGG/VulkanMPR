#!/bin/python
import os
import argparse
import platform

DEFAULT_GENERATOR = "Visual Studio 17 2022" if platform.system() == 'Windows' else 'Ninja'


def make_json_args(generator, build_type):
    cmake_user_presets = f'''
{{
    "version": 2,
    "configurePresets": [
        {{
        "name": "default",
        "inherits": "vcpkg",
        "environment": {{
          "VCPKG_ROOT": "{os.environ["VCPKG_ROOT"]}"
        }}
      }}
    ]
}}
'''

    cmake_presets = f'''
{{
  "version": 2,
  "configurePresets": [
    {{
      "name": "vcpkg",
      "generator": "{generator}",
      "binaryDir": "${{sourceDir}}/build",
      "cacheVariables": {{
        "CMAKE_TOOLCHAIN_FILE": "$env{{VCPKG_ROOT}}/scripts/buildsystems/vcpkg.cmake",
        "CMAKE_EXPORT_COMPILE_COMMANDS" : "1",
        "CMAKE_BUILD_TYPE": "{build_type}"
      }}
    }}
  ]
}}
'''
    return cmake_user_presets, cmake_presets


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-G', '--generator',
                        help="Select project files generator")
    parser.add_argument('-B', '--build', help="Select build type")
    args = parser.parse_args()
    generator = args.generator if args.generator else DEFAULT_GENERATOR
    build_type = args.build if args.build else "Debug"
    return generator, build_type


def create_json_files(user_presets, presets):
    try:
        with open('CMakeUserPresets.json', 'wt') as file:
            file.write(user_presets)
        with open('CMakePresets.json', 'wt') as file:
            file.write(presets)
    except OSError:
        print('Error creating json files')


if __name__ == '__main__':
    g, bt = parse_args()
    up, p = make_json_args(g, bt)
    create_json_files(up, p)
    print('Files have been generated!')
