import os
import re
import subprocess

def compile_shaders(input_dir="src/shaders", output_dir="src/compiled_shaders"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)

        # GLSL
        if filename.endswith((".vert", ".frag", ".comp")):
            output_path = os.path.join(output_dir, filename + ".spv")
            command = ["glslc", input_path, "-o", output_path]
            try:
                subprocess.run(command, check=True)
                print(f"Compiled {filename} -> {output_path}")
            except subprocess.CalledProcessError:
                print(f"Failed to compile {filename}")
        
        # SLANG
        elif filename.endswith(".slang"):
            with open(input_path, "r", encoding="utf-8") as f:
                source = f.read()

            matches = re.findall(
                r'\[shader\("(\w+)"\)\]\s*[^\S\r\n]*[\w<>\s:*&]+\s+(\w+)\s*\(',
                source
            )

            if not matches:
                print(f"No shader entry points found in {filename}")
                continue

            for stage, entry_name in matches:
                output_path = os.path.join(
                    output_dir,
                    f"{os.path.splitext(filename)[0]}.{stage}.spv"
                )

                command = [
                    "slangc",
                    input_path,
                    "-target", "spirv",
                    "-entry", entry_name,
                    "-profile", _slang_profile(stage),
                    "-o", output_path
                ]

                try:
                    subprocess.run(command, check=True)
                    print(f"Compiled {filename} [{stage}] -> {output_path}")
                except subprocess.CalledProcessError:
                    print(f"Failed to compile {filename} [{stage}]")

def _slang_profile(stage: str) -> str:
    profiles = {
        "vertex": "vs_6_0",
        "hull": "hs_6_0",
        "domain": "ds_6_0",
        "geometry": "gs_6_0",
        "pixel": "ps_6_0",
        "compute": "cs_6_0",
        "amplification": "as_6_0",
        "mesh": "ms_6_0",
        "raygeneration": "rgs_6_0",
        "closesthit": "chs_6_0",
        "miss": "ms_6_0",
        "anyhit": "ahs_6_0",
        "intersection": "is_6_0",
        "callable": "callable_6_0",
    }
    return profiles.get(stage, "vs_6_0") 


if __name__ == "__main__":
    compile_shaders()
