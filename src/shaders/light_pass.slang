[[vk::binding(0, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> outputTexture;

[[vk::binding(1, 0)]]
[[vk::image_format("rgba32f")]]
Texture2D<float4> positionTexture;
[[vk::binding(2, 0)]]
[[vk::image_format("rgba32f")]]
Texture2D<float4> normalTexture;
[[vk::binding(3, 0)]]
[[vk::image_format("rgba8")]]
Texture2D<float4> diffuseTexture;
[[vk::binding(4, 0)]]
[[vk::image_format("rgba8")]]
Texture2D<float4> specularTexture;

struct SceneData
{
    float4x4 view;
    float4x4 proj;
    float4x4 projView;
};

struct LightData {
};

struct PushConstants {
    SceneData *sceneData;
    LightData *lights;
    uint lightCount;
};

[[vk::push_constant]]
PushConstants pushConstants;


[shader("compute")]
[numthreads(16, 16, 1)]
void computeMain(uint3 threadID: SV_DispatchThreadID) {
  uint width;
  uint height;
  outputTexture.GetDimensions(width, height);

  // NOTE: Make sure that gBuffer textures dimensions are equal to the outputTexture dimensions
  if (threadID.x >= width || threadID.y >= height) {
      return;
  }


  int3 textureCoords = int3(threadID.x, threadID.y, 0);
  float4 position = positionTexture.Load(textureCoords);
  float4 normal = normalTexture.Load(textureCoords);
  float4 diffuse = diffuseTexture.Load(textureCoords);
  float4 specular = specularTexture.Load(textureCoords);

  outputTexture[threadID.xy] = diffuse;
}