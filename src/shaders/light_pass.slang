import utils;
import lights;
import shadows;
import types;

[[vk::binding(0, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> outputTexture;

[[vk::binding(0, 1)]]
Texture2D<float> depthTexture;
[[vk::binding(1, 1)]]
Texture2D<uint> normalTexture;
[[vk::binding(2, 1)]]
Texture2D<float4> albedoTexture;
[[vk::binding(3, 1)]]
Texture2D<float4> metalRoughTexture;

[[vk::binding(4, 1)]]
Texture2DArray<float> shadowMapArray;
[[vk::binding(5, 1)]]
SamplerComparisonState shadowSampler;
[[vk::binding(6, 1)]]
Texture2D<float> pointLightsTiledShadowMap;
[[vk::binding(7, 1)]]
Texture2D<float4> emissiveTexture;

[[vk::binding(8, 1)]]
Texture2D<float4> DDGIOutput;

struct PushConstants {
    SceneData *sceneData;
    DirectionalLight *directionalLight;
    PointLight *visiblePointLights;
    uint32_t *visiblePointLightsCount;
    TetrahedronData *tetraData;
    float cameraNear;
    float cameraFar;
    float4 skyRadiance;

    float4x4 inverseCameraViewProj;
};

[[vk::push_constant]]
PushConstants pushConstants;

groupshared DirectionalLight dirLight;

[shader("compute")]
[numthreads(16, 16, 1)]
void computeMain(uint3 threadID : SV_DispatchThreadID, uint localID : SV_GroupIndex)
{
    bool hasDirectionalLight = pushConstants.directionalLight != nullptr;
    if (localID == 0 && hasDirectionalLight)
        dirLight = *pushConstants.directionalLight;
    GroupMemoryBarrierWithGroupSync();
    uint width, height;
    outputTexture.GetDimensions(width, height);
    if (threadID.x >= width || threadID.y >= height) return;

    int2 uv = int2(threadID.xy);

    float depth = depthTexture[uv];
    if (depth >= 1.0f || depth <= 0.0f) {
        outputTexture[uv] = float4(pushConstants.skyRadiance.xyz * pushConstants.skyRadiance.a, 1.0f);
        return;
    }
    float3 normal = DecodeGBufferNormal(normalTexture[uv]);
    float4 albedo = albedoTexture[uv];
    float4 mr = metalRoughTexture[uv];

    float3 worldPos = ReconstructWorldPos(threadID.xy, uint2(width, height), pushConstants.inverseCameraViewProj, depth);
    float3 normalWS = normalize(normal.xyz);
    float3 baseColor = albedo.rgb;
    float metallic = mr.r;
    float roughness = mr.g;
    roughness = max(roughness, 0.04f);

    bool hasGeometry = dot(normal.xyz, normal.xyz) > 1e-6f;

    float linearDepth = LinearizeDepth(depth, pushConstants.cameraNear, pushConstants.cameraFar);
    float3 color = 0.0f;
    if (hasDirectionalLight)
    {
        float shadowFactor = CalculateDirectionalShadow(dirLight,
                worldPos, normalWS, linearDepth, shadowMapArray, shadowSampler,
                dirLight.normalBias, dirLight.constantBias);
        color += shadowFactor * CalculateDirLightPBR(
                worldPos, pushConstants.sceneData.cameraPos, normalWS, baseColor, metallic, roughness,
                dirLight);
    }

    uint pointLightsCount = *pushConstants.visiblePointLightsCount;
    for (uint i = 0; i < pointLightsCount; ++i) {
        PointLight pointLight = pushConstants.visiblePointLights[i];
        float shadowFactor = CalculatePointShadow(pointLight, worldPos, normalWS, pushConstants.tetraData, pointLightsTiledShadowMap, shadowSampler,
                pointLight.normalBias, pointLight.constantBias);
        color += shadowFactor * CalculatePointLightPBR(worldPos, 
            pushConstants.sceneData.cameraPos, normalWS, baseColor, metallic, roughness, pointLight);
    }
    color += emissiveTexture[uv].rgb;

    // DDGI Indirect
    color += DDGIOutput[uv].rgb;

    outputTexture[uv] = float4(color, 1.0f);
}
