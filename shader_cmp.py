import os
import re
import subprocess

def compile_shaders(input_dir="src/shaders", output_dir="src/compiled_shaders"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # List to store compiled SPIR-V files for validation
    compiled_files = []
    
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)

        # GLSL
        if filename.endswith((".vert", ".frag", ".comp")):
            output_path = os.path.join(output_dir, filename + ".spv")
            command = ["glslc", input_path, "-o", output_path]
            try:
                subprocess.run(command, check=True)
                print(f"Compiled {filename} -> {output_path}")
                compiled_files.append(output_path)
            except subprocess.CalledProcessError:
                print(f"Failed to compile {filename}")
        
        # SLANG
        elif filename.endswith(".slang"):
            with open(input_path, "r", encoding="utf-8") as f:
                source = f.read()

            matches = re.findall(
                r'\[shader\("(\w+)"\)\](?:\s*\[[^\]]+\]\s*)*\s*[\w<>\s:*&]+\s+(\w+)\s*\(',
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
                    "-g1",
                    "-o", output_path
                ]

                try:
                    subprocess.run(command, check=True)
                    print(f"Compiled {filename} [{stage}] -> {output_path}")
                    compiled_files.append(output_path)
                except subprocess.CalledProcessError:
                    print(f"Failed to compile {filename} [{stage}]")

    # Validate all compiled SPIR-V files
    validate_compiled_shaders(compiled_files)

def validate_compiled_shaders(compiled_files):
    """Validate all compiled SPIR-V files using spirv-val"""
    if not compiled_files:
        print("No shaders to validate")
        return

    print("\nValidating compiled SPIR-V shaders...")
    validation_errors = []
    
    for spv_file in compiled_files:
        if os.path.exists(spv_file):
            command = ["spirv-val", spv_file]
            try:
                result = subprocess.run(command, check=False, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Validation failed for {spv_file}:")
                    print(f"  Error: {result.stderr.strip()}")
                    validation_errors.append(spv_file)
                else:
                    print(f"Validated {spv_file} ✓")
            except FileNotFoundError:
                print("spirv-val not found. Please install SPIRV-Tools or add it to PATH.")
                break
        else:
            print(f"Warning: {spv_file} does not exist and cannot be validated")
    
    if validation_errors:
        print(f"\nValidation completed with {len(validation_errors)} error(s).")
    else:
        print(f"\nAll {len(compiled_files)} shaders validated successfully.")

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