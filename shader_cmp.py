import os
import subprocess

def compile_shaders(input_dir="src/shaders", output_dir="src/compiled_shaders"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".vert") or filename.endswith(".frag") or filename.endswith(".comp"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename + ".spv")
            
            command = ["glslc", input_path, "-o", output_path]
            
            try:
                subprocess.run(command, check=True)
                print(f"Compiled {filename} -> {output_path}")
            except subprocess.CalledProcessError:
                print(f"Failed to compile {filename}")

if __name__ == "__main__":
    compile_shaders()
