#version 460

layout (location = 0) out vec3 outColor;


void main()
{
  const vec3 positions[3] = {
    {-0.5f, 0.5f, 0.0f},
    {0.f, -0.5f, 0.0f},
    {0.5f, 0.5f, 0.0f},
  };

  const vec3 colors[3] = {
    {1.f, 0.f, 0.0f},
    {0.f, 1.f, 0.0f},
    {0.f, 0.f, 1.0f},
  };
  gl_Position = vec4(positions[gl_VertexIndex], 1.0f);
  outColor = colors[gl_VertexIndex];
}