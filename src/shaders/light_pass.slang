import utils;
import light_utils;
import shadows;
import types;

[[vk::binding(0, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> outputTexture;

[[vk::binding(0, 1)]]
Texture2D<float> depthTexture;
[[vk::binding(1, 1)]]
Texture2D<float4> normalTexture;
[[vk::binding(2, 1)]]
Texture2D<float4> albedoTexture;
[[vk::binding(3, 1)]]
Texture2D<float4> metalRoughTexture;

[[vk::binding(4, 1)]]
Texture2DArray<float> shadowMapArray;
[[vk::binding(5, 1)]]
SamplerComparisonState shadowSampler;

struct PushConstants {
    SceneData *sceneData;
    DirectionalLight *directionalLights;
    uint directionalLightsCount;
    PointLight *pointLights;
    uint pointLightsCount;

    float4x4 inverseCameraViewProj;
};

[[vk::push_constant]]
PushConstants pushConstants;

[shader("compute")]
[numthreads(16, 16, 1)]
void computeMain(uint3 threadID : SV_DispatchThreadID)
{
    uint width, height;
    outputTexture.GetDimensions(width, height);
    if (threadID.x >= width || threadID.y >= height) return;

    int2 uv = int2(threadID.xy);

    float depth = depthTexture[uv];
    if (depth >= 1.0f || depth <= 0.0f) {
        outputTexture[uv] = float4(0.33f, 0.66f, 0.99f, 1.0f);
        return;
    }
    float4 normal = normalTexture[uv];
    float4 albedo = albedoTexture[uv];
    float4 mr = metalRoughTexture[uv];

    float3 worldPos = ReconstructWorldPos(threadID.xy, uint2(width, height), pushConstants.inverseCameraViewProj, depth);
    float3 normalWS = normalize(normal.xyz);
    float3 baseColor = albedo.rgb;
    float metallic = mr.r;
    float roughness = mr.g;
    roughness = max(roughness, 0.04f);

    bool hasGeometry = dot(normal.xyz, normal.xyz) > 1e-6f;

    float3 color = 0.0f;

    uint dirLightsCount = pushConstants.directionalLightsCount;
    for (uint i = 0; i < dirLightsCount; ++i)
    {
        DirectionalLight dirLight = pushConstants.directionalLights[i];
        float linearDepth = LinearizeDepth(depth, 0.1f, 100.0f);
        float shadowFactor = CalculateDirectionalShadow(dirLight,
                worldPos, linearDepth, shadowMapArray, shadowSampler);
        color += shadowFactor * CalculateDirLightPBR(
                    worldPos, pushConstants.sceneData.cameraPos, normalWS, baseColor, metallic, roughness, 
                    dirLight);
    }

    uint pointLightsCount = pushConstants.pointLightsCount;
    for (uint i = 0; i < pointLightsCount; ++i) {
        PointLight pointLight = pushConstants.pointLights[i];
        color += CalculatePointLightPBR(worldPos, 
            pushConstants.sceneData.cameraPos, normalWS, baseColor, metallic, roughness, pointLight);
    }
    outputTexture[uv] = float4(color, 1.0f);
}
