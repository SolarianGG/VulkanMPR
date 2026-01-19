import utils;
import light_utils;

struct PushConstants {
    Vertex *vertices;
    Instance *instances;
    SceneData *sceneData;
    LightData *lights;
    uint lightCount;
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
    float4 col;
    float2 uv;
    float3x3 TBN;
    nointerpolation uint materialID;
    nointerpolation uint colorTextureID;
    nointerpolation uint colorSamplerID;
    nointerpolation uint metalRoughTextureID;
    nointerpolation uint metalRoughSamplerID;
    nointerpolation uint normalTextureID;
    nointerpolation uint normalSamplerID;
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
    output.col = vertex.col;
    // Normal mapping TBN
    float3 T = normalize(mul((float3x3)instance.worldTransform, vertex.tangent.xyz));
    float3 N = normalize(mul((float3x3)instance.worldTransform, vertex.norm));
    T = normalize(T - dot(T, N) * N);
    float3 B = cross(N, T);
    float3x3 TBN = float3x3(T, B, N);
    output.TBN = TBN;

    output.uv = float2(vertex.u, vertex.v);

    output.materialID = instance.materialID;
    output.colorTextureID = instance.colorTextureID;
    output.metalRoughTextureID = instance.metalRoughTextureID;
    output.colorSamplerID = instance.colorSamplerID;
    output.metalRoughSamplerID = instance.metalRoughSamplerID;
    output.normalTextureID = instance.normalTextureID;
    output.normalSamplerID = instance.normalSamplerID;
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
  
  float4 mr = materialData.metalRoughnessFactors * metalRoughness;

  float3 worldPos = input.worldPos.xyz;
  float3 normal = normalize(mul(bumpMapNormal, input.TBN));
  float4 diffuse = materialData.colorFactors * input.col * baseColor;
  float metallic = mr.r;
  float roughness = mr.g;
  roughness = max(roughness, 0.04f); 

  float4 color = 0.0f;

  for (uint i = 0; i < pushConstants.lightCount; ++i)
  {
      LightData light = pushConstants.lights[i];
      if (light.type.x == 0)
      {
          DirectionalLight dirLight;
          dirLight.direction = light.data0.xyz;
          dirLight.color = light.data1.xyz;
          dirLight.intensity = light.data1.w;
          color.rgb += CalculateDirLightPBR(worldPos, pushConstants.sceneData.cameraPos, normal, diffuse.rgb, metallic, roughness, dirLight);
      }
      else if (light.type.x == 1)
      {
          PointLight pointLight;
          pointLight.position = light.data0.xyz;
          pointLight.range = light.data0.w;
          pointLight.color = light.data1.xyz;
          pointLight.intensity = light.data1.w;
          color.rgb += CalculatePointLightPBR(worldPos, pushConstants.sceneData.cameraPos, normal, diffuse.rgb, metallic, roughness, pointLight);
      }
  }
  color.a = diffuse.a;


  float weight =
    max(min(1.0, max(max(color.r, color.g), color.b) * color.a), color.a) *
                 clamp(0.03 / (1e-5 + pow(fragCoord.z / 200, 4.0)), 1e-2, 3e3);
  PSOutput output;

  // blend func: GL_ONE, GL_ONE
  // switch to pre-multiplied alpha and weight
  output.color = float4(color.rgb * color.a, color.a) * weight;

  // blend func: GL_ZERO, GL_ONE_MINUS_SRC_ALPHA
  output.reveal = color.a;
  return output;
};
