import utils;
import types;
import lights;
import shadows;

struct PushConstants {
    VertexPosition *positions;
    VertexAttributes *attributes;
    Instance *instances;
    SceneData *sceneData;
    DirectionalLight *directionalLight;
    PointLight *visiblePointLights;
    uint32_t *visiblePointLightsCount;
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
VSOutput vertexMain(uint vertexID : SV_VulkanVertexID, uint instanceID : SV_VulkanInstanceID)
{
    float3 pos = pushConstants.positions[vertexID].pos;
    VertexAttributes attr = pushConstants.attributes[vertexID];
    Instance instance = pushConstants.instances[instanceID];
    SceneData scene = *pushConstants.sceneData;
    VSOutput output;
    float4 vertexPos = float4(pos, 1.0);
    output.pos = mul(scene.projView, mul(instance.worldTransform, vertexPos));
    output.worldPos = mul(instance.worldTransform, vertexPos);
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
    float4 color : SV_Target0;
    float reveal : SV_Target1;
}


[shader("pixel")]
PSOutput fragmentMain(VSOutput input, float4 fragCoord : SV_Position)
{
  GLTFMaterialData materialData = materials[NonUniformResourceIndex(input.materialID)];

  float4 baseColor = textures[NonUniformResourceIndex(input.colorTextureID)].Sample( samplers[NonUniformResourceIndex(input.colorSamplerID)], input.uv);

  float4 metalRoughness = textures[NonUniformResourceIndex(input.metalRoughTextureID)].Sample( samplers[NonUniformResourceIndex(input.metalRoughSamplerID)], input.uv);

  float3 bumpMapNormal = textures[NonUniformResourceIndex(input.normalTextureID)] .Sample(samplers[NonUniformResourceIndex(input.normalSamplerID)], input.uv) .rgb;
  bumpMapNormal = normalize(bumpMapNormal * 2.0f - float3(1.0f, 1.0f, 1.0f));
  
  float3 worldPos = input.worldPos.xyz;
  float3 normal = normalize(mul(bumpMapNormal, input.TBN));
  float4 diffuse = materialData.colorFactors * baseColor;
  // G = roughness, B = metalness
  float metallic = materialData.metalRoughnessFactors.r * metalRoughness.b;
  float roughness = materialData.metalRoughnessFactors.g * metalRoughness.g;
  roughness = max(roughness, 0.04f); 

  float4 color = 0.0f;

  {
   DirectionalLight dirLight = *pushConstants.directionalLight;
   color.xyz += CalculateDirLightPBR(
                worldPos, pushConstants.sceneData.cameraPos, normal, diffuse.xyz, metallic, roughness, 
                dirLight);
  }

  uint pointLightsCount = *pushConstants.visiblePointLightsCount;
  for (uint i = 0; i < pointLightsCount; ++i) {
      PointLight pointLight = pushConstants.visiblePointLights[i];
      color.xyz += CalculatePointLightPBR(worldPos, 
            pushConstants.sceneData.cameraPos, normal, diffuse.xyz, metallic, roughness, pointLight);
  }
  color.a = diffuse.a;

  float4 emissiveSample = textures[NonUniformResourceIndex(input.emissiveTextureID)]
      .Sample(samplers[NonUniformResourceIndex(input.emissiveSamplerID)], input.uv);
  color.rgb += emissiveSample.rgb * materialData.emissiveFactors.rgb;

  float weight =
    max(min(1.0, max(max(color.r, color.g), color.b) * color.a), color.a) *
                 clamp(0.03 / (1e-5 + pow(fragCoord.z / 200, 4.0)), 1e-2, 3e3);
  PSOutput output;

  output.color = float4(color.rgb * color.a, color.a) * weight;

  output.reveal = color.a;
  return output;
};
