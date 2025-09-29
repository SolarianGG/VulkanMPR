#version 460

#extension GL_EXT_buffer_reference2 : require

layout (location = 0) out vec3 outColor;
layout (location = 1) out vec2 outUV;


struct Vertex {
  vec3 pos;
  float u;
  vec3 normal;
  float v;
  vec4 color;
};

layout(buffer_reference, std430) readonly buffer VertexBuffer {
  Vertex vertices[];
};


layout (push_constant) uniform constants {
 mat4 world;
 VertexBuffer vertexBuffer;
} PushConstants;


void main() 
{
  Vertex v = PushConstants.vertexBuffer.vertices[gl_VertexIndex];
  gl_Position = PushConstants.world * vec4(v.pos, 1.0f);
  outColor = v.color.xyz;
  outUV = vec2(v.u, v.v);
}