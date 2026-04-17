import utils;
import types;

struct PushConstants {
    VertexPosition *positions;
    VertexAttributes *attributes;
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
    float2 uv;
    float3x3 TBN;
    nointerpolation uint materialID;
    nointerpolation uint colorTextureID;
    nointerpolation uint colorSamplerID;
    nointerpolation uint metalRoughTextureID;
    nointerpolation uint metalRoughSamplerID;
    nointerpolation uint normalTextureID;
    nointerpolation uint normalSamplerID;
    nointerpolation uint emissiveTextureID;
    nointerpolation uint emissiveSamplerID;
};

[shader("vertex")]
VSOutput vertexMain(uint vertexID : SV_VulkanVertexID, uint instanceID : SV_VulkanInstanceID, uint offset : SVDraw)
{
    float3 pos = pushConstants.positions[vertexID].pos;
    VertexAttributes attr = pushConstants.attributes[vertexID];
    Instance instance = pushConstants.instances[instanceID];
    SceneData scene = *pushConstants.sceneData;
    VSOutput output;
    float4 vertexPos = float4(pos, 1.0);
    output.pos = mul(scene.projView, mul(instance.worldTransform, vertexPos));
    // Normal mapping TBN
    vector<float16_t, 4> tangent = DecodeTangent(attr.tangent);
    float3 T = normalize(mul((float3x3)instance.worldTransform, tangent.xyz));
    float3 N = normalize(mul((float3x3)instance.worldTransform, DecodeNormal(attr.norm)));
    T = normalize(T - dot(T, N) * N);
    float3 B = cross(N, T) * tangent.w;
    float3x3 TBN = float3x3(T, B, N);
    output.TBN = TBN;

    output.uv = float2(attr.u, attr.v);

    output.materialID = instance.materialID;
    output.colorTextureID = instance.colorTextureID;
    output.metalRoughTextureID = instance.metalRoughTextureID;
    output.colorSamplerID = instance.colorSamplerID;
    output.metalRoughSamplerID = instance.metalRoughSamplerID;
    output.normalTextureID = instance.normalTextureID;
    output.normalSamplerID = instance.normalSamplerID;
    output.emissiveTextureID = instance.emissiveTextureID;
    output.emissiveSamplerID = instance.emissiveSamplerID;
    return output;
};

struct PSOutput {
    uint normal : SV_Target0;
    float4 albedo : SV_Target1;
    float4 metalRoughness : SV_Target2;
    float4 emissive : SV_Target3;
}


[shader("pixel")]
PSOutput fragmentMain(VSOutput input)
{
  GLTFMaterialData materialData = materials[NonUniformResourceIndex(input.materialID)];

  float4 baseColor = textures[NonUniformResourceIndex(input.colorTextureID)].Sample( samplers[NonUniformResourceIndex(input.colorSamplerID)], input.uv);

  float4 metalRoughness = textures[NonUniformResourceIndex(input.metalRoughTextureID)].Sample( samplers[NonUniformResourceIndex(input.metalRoughSamplerID)], input.uv);

  float2 normalRG = textures[NonUniformResourceIndex(input.normalTextureID)].Sample(samplers[NonUniformResourceIndex(input.normalSamplerID)], input.uv).rg;
  float2 normalXY = normalRG * 2.0f - 1.0f;
  float normalZ = sqrt(max(0.0f, 1.0f - dot(normalXY, normalXY)));
  float3 bumpMapNormal = normalize(float3(normalXY, normalZ));
  
  PSOutput output;
  output.normal = PackNormalHalf2(QuantizeNormal(mul(bumpMapNormal, input.TBN)));
  output.albedo = materialData.colorFactors * baseColor;
  if (output.albedo.a < materialData.alphaCutoff.r)
      discard;
  // G = roughness, B = metalness
  output.metalRoughness = float4(
      materialData.metalRoughnessFactors.r * metalRoughness.b,
      materialData.metalRoughnessFactors.g * metalRoughness.g,
      0.0f, 0.0f);

  float4 emissiveSample = textures[NonUniformResourceIndex(input.emissiveTextureID)]
      .Sample(samplers[NonUniformResourceIndex(input.emissiveSamplerID)], input.uv);
  output.emissive = float4(emissiveSample.rgb * materialData.emissiveFactors.rgb, 1.0f);
  return output;
};
