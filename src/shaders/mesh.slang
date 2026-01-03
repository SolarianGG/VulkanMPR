struct Vertex
{
    float3 pos;
    float u;
    float3 norm;
    float v;
    float4 col;
};

struct Instance
{
    float4x4 worldTransform;
    uint materialID;
    uint colorTextureID;
    uint colorSamplerID;
    uint metalRoughTextureID;
    uint metalRoughSamplerID;
};

struct SceneData 
{
    float4x4 view;
    float4x4 proj;
    float4x4 projView;
};

struct GLTFMaterialData {
    float4 colorFactors;
    float4 metalRoughnessFactors;
};

struct PushConstants {
    Vertex *vertices;
    Instance *instances;
    SceneData *sceneData;
};
[[vk::binding(0, 0)]] ConstantBuffer<GLTFMaterialData> materials[];
[[vk::binding(1, 0)]] SamplerState samplers[];
[[vk::binding(2, 0)]] Texture2D textures[];

[[vk::push_constant]]
PushConstants pushConstants;

struct VSOutput
{
    float4 pos : SV_Position;
    float4 worldPos;
    float3 norm;
    float3 col;
    float2 uv;
    nointerpolation uint materialID;
    nointerpolation uint colorTextureID;
    nointerpolation uint colorSamplerID;
    nointerpolation uint metalRoughTextureID;
    nointerpolation uint metalRoughSamplerID;
};

[shader("vertex")]
VSOutput vertexMain(uint vertexID : SV_VulkanVertexID, uint instanceID : SV_VulkanInstanceID)
{
    Vertex vertex = pushConstants.vertices[vertexID];
    Instance instance = pushConstants.instances[instanceID];
    SceneData scene = *pushConstants.sceneData;
    VSOutput output;
    float4 vertexPos = float4(vertex.pos, 1.0);
    output.pos = mul(scene.projView, mul(instance.worldTransform, vertexPos));
    output.worldPos = mul(instance.worldTransform, vertexPos);
    output.col = vertex.col.xyz;
    // Shader supports only uniform scaling
    float3x3 normalMatrix = (float3x3)instance.worldTransform;
    output.norm = normalize(mul(normalMatrix, vertex.norm));
    output.uv = float2(vertex.u, vertex.v);

    output.colorTextureID = instance.colorTextureID;
    output.metalRoughTextureID = instance.metalRoughTextureID;
    output.colorSamplerID = instance.colorSamplerID;
    output.metalRoughSamplerID = instance.metalRoughSamplerID;
    return output;
};

struct PSOutput {
    float4 pos : SV_Target0;
    float4 normal : SV_Target1;
    float4 diffuse : SV_Target2;
    float4 specular : SV_Target3;
}


[shader("pixel")]
PSOutput fragmentMain(VSOutput input)
{
  SceneData scene = *pushConstants.sceneData;
  GLTFMaterialData materialData = materials[NonUniformResourceIndex(input.materialID)];

  float4 baseColor = textures[NonUniformResourceIndex(input.colorTextureID)].Sample( samplers[NonUniformResourceIndex(input.colorSamplerID)], input.uv);

  float4 metalRoughness = textures[NonUniformResourceIndex(input.metalRoughTextureID)].Sample( samplers[NonUniformResourceIndex(input.metalRoughSamplerID)], input.uv);



  
  PSOutput output;
  output.pos = input.worldPos;
  output.normal = float4(normalize(input.norm), 0.0f); 
  output.diffuse = materialData.colorFactors * float4(input.col, 1.0f) * baseColor;
  if (output.diffuse.w < 0.1)
      discard;
  output.specular = materialData.metalRoughnessFactors * metalRoughness;
  return output;
};
