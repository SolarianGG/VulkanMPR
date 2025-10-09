
struct Vertex
{
    float3 pos;
    float u;
    float3 norm;
    float v;
    float4 col;
};

struct VSOutput
{
    float4 pos : SV_Position;
    float3 norm;
    float3 col;
    float2 uv;
};

struct PushConstants {
    float4x4 transform;
    Vertex *vertices;
};

struct SceneData 
{
    float4x4 view;
    float4x4 proj;
    float4x4 projView;
    float4 ambientColor;
    float4 sunlightDirection; // w for sun power
    float4 sunlightColor;
};

[[vk::binding(0, 0)]] ConstantBuffer<SceneData> sceneData;

struct GLTFMaterialData {
    float4 colorFactors;
    float4 metalRoughnessFactors;
};

[[vk::binding(0, 1)]] ConstantBuffer<GLTFMaterialData> materialData;

[[vk::binding(1, 1)]] Sampler2D colorTexture;
[[vk::binding(2, 1)]] Sampler2D metalRoughTexture;

[[vk::push_constant]]
PushConstants pushConstants;

[shader("vertex")]
VSOutput vertexMain(uint vertexID : SV_VulkanVertexID)
{
    Vertex vertex = pushConstants.vertices[vertexID];
    VSOutput output;
    float4 vertexPos = float4(vertex.pos, 1.0);
    output.pos = mul(sceneData.projView, mul(pushConstants.transform, vertexPos));
    output.col = materialData.colorFactors.xyz * vertex.col.xyz;
    output.norm = mul(pushConstants.transform, float4(vertex.norm, 0.0)).xyz;
    output.uv = float2(vertex.u, vertex.v);
    return output;
};


[shader("pixel")]
float4 fragmentMain(VSOutput input) : SV_Target0
{
  float3 norm = normalize(input.norm);
  float3 lightDir = normalize(-sceneData.sunlightDirection.xyz);
  float lightValue = max(dot(lightDir, norm), 0.0);



  float3 color = input.col * colorTexture.Sample(input.uv).xyz;
  float3 ambientColor = sceneData.ambientColor.xyz * color; 
  float3 diffuseColor = sceneData.sunlightDirection.w * lightValue * color;
  float4 outColor = float4(ambientColor + diffuseColor, 1.0); 
  return outColor;
};
