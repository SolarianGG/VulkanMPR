#version 460

#extension GL_EXT_buffer_reference2 : require
#extension GL_GOOGLE_include_directive : require

#include "input_structs.glsl"

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec3 outColor;
layout (location = 2) out vec2 outUV;

struct Vertex {
  vec3 pos;
  float u;
  vec3 norm;
  float v;
  vec4 col;
};

layout(buffer_reference, std430) readonly buffer VertexBuffer {
   Vertex vertices[];
};

layout(push_constant) uniform PushConstants {
  mat4 transform;
  VertexBuffer vertexBuffer;
} pushConstants;

void main() {
  Vertex vertex = pushConstants.vertexBuffer.vertices[gl_VertexIndex];

  gl_Position = sceneData.viewProj * pushConstants.transform * vec4(vertex.pos, 1.0); 

  // Make sure u do not scale the matrix or perform only uniform scaling
  outNormal = (pushConstants.transform * vec4(vertex.norm, 0.0)).xyz;
  outColor = vertex.col.xyz * materialData.colorFactors.xyz;
  outUV = vec2(vertex.u, vertex.v);

}