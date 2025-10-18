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
}

struct SceneData 
{
    float4x4 view;
    float4x4 proj;
    float4x4 projView;
    float4 ambientColor;
    float4 sunlightDirection; // w for sun power
    float4 sunlightColor;
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
[[vk::binding(0, 1)]] ConstantBuffer<GLTFMaterialData> materials[];
[[vk::binding(1, 1)]] SamplerState samplers[];
[[vk::binding(2, 1)]] Texture2D textures[];

[[vk::push_constant]]
PushConstants pushConstants;

struct VSOutput
{
    float4 pos : SV_Position;
    float3 norm;
    float3 col;
    float2 uv;
    uint colorTextureID;
    uint colorSamplerID;
    uint metalRoughTextureID;
    uint metalRoughSamplerID;
};

[shader("vertex")]
VSOutput vertexMain(uint vertexID : SV_VulkanVertexID, uint instanceID : SV_VulkanInstanceID)
{
    Vertex vertex = pushConstants.vertices[vertexID];
    Instance instance = pushConstants.instances[instanceID];
    SceneData scene = *pushConstants.sceneData;
    GLTFMaterialData materialData = materials[NonUniformResourceIndex(instance.materialID)];
    VSOutput output;
    float4 vertexPos = float4(vertex.pos, 1.0);
    output.pos = mul(scene.projView, mul(instance.worldTransform, vertexPos));
    output.col = materialData.colorFactors.xyz * vertex.col.xyz;
    // Shader supports only uniform scaling
    output.norm = mul(instance.worldTransform, float4(vertex.norm, 0.0)).xyz;
    output.uv = float2(vertex.u, vertex.v);
    output.colorTextureID = instance.colorTextureID;
    output.metalRoughTextureID = instance.metalRoughTextureID;
    return output;
};


[shader("pixel")]
float4 fragmentMain(VSOutput input) : SV_Target0
{
  SceneData scene = *pushConstants.sceneData;
  float3 norm = normalize(input.norm);
  float3 lightDir = normalize(-scene.sunlightDirection.xyz);
  float lightValue = max(dot(lightDir, norm), 0.0);

  float3 color = input.col * 
  textures[NonUniformResourceIndex(input.colorTextureID)].Sample(samplers[NonUniformResourceIndex(input.colorSamplerID)], input.uv).xyz;
  float3 ambientColor = scene.ambientColor.xyz * color; 
  float3 diffuseColor = scene.sunlightDirection.w * lightValue * color;
  float4 outColor = float4(ambientColor + diffuseColor, 1.0); 
  return outColor;
};
