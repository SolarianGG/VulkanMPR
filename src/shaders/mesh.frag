#version 460

#extension GL_GOOGLE_include_directive : require

#include "input_structs.glsl"

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inColor;
layout (location = 2) in vec2 inUV;


layout (location = 0) out vec4 outColor;

void main()
{
  vec3 norm = normalize(inNormal);
  vec3 lightDir = normalize(-sceneData.sunlightDirection.xyz);
  float lightValue = max(dot(lightDir, norm), 0.0);



  vec3 color = inColor * texture(colorTexture, inUV).xyz;
  vec3 ambientColor = sceneData.ambientColor.xyz * color; 
  vec3 diffuseColor = sceneData.sunlightDirection.w * lightValue * color;
  outColor = vec4(ambientColor + diffuseColor, 1.0);
}