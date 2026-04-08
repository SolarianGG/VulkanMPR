module shadows;

import types;

public float CalculateDirectionalShadow(
    DirectionalLight light, float3 worldPos, float3 normalWS, float depth,
    Texture2DArray<float> shadowMaps, SamplerComparisonState shadowSampler,
    float normalBias, float constantBias)
{

    uint cascadeIndex = 0;
    for (uint j = 1; j < light.cascadeCount; ++j) {
        if (depth < light.splitDistances[j]) break;
        cascadeIndex = j;
    }

    float NdotL = dot(normalWS, normalize(-light.direction.xyz));
    float3 offsetPos = worldPos + normalWS * normalBias * (1.0 - saturate(NdotL));

    float4 lsPos = mul(light.cascadeVPs[cascadeIndex], float4(offsetPos, 1.0));
    float3 proj = lsPos.xyz / lsPos.w;
    float2 uv = proj.xy * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;

    if (any(uv < 0.0) || any(uv > 1.0)) return 1.0;

    float currentDepth = proj.z;
    if (currentDepth < 0.0 || currentDepth > 1.0) return 1.0;

    float bias = constantBias;

    uint w, h, layers;
    shadowMaps.GetDimensions(w, h, layers);
    float2 texelSize = 1.0 / float2(w, h);

    // PCF 3x3 + Hardware 2x2 PCF
    float shadow = 0.0;
    for (int x = -1; x <= 1; ++x)
        for (int y = -1; y <= 1; ++y)
            shadow += shadowMaps.SampleCmpLevelZero(
                shadowSampler,
                float3(uv + float2(x, y) * texelSize, float(cascadeIndex)),
                currentDepth - bias);
    return shadow / 9.0;
}

public float CalculatePointShadow(
    PointLight light, float3 worldPos, float3 normalWS,
    TetrahedronData *tetraData,
    Texture2D<float> tiledShadowMap, SamplerComparisonState shadowSampler,
    float normalBias, float constantBias)
{
    float3 lightDir = normalize(light.position - worldPos);
    float NdotL = dot(normalWS, lightDir);
    float3 offsetPos = worldPos + normalWS * normalBias * (1.0 - saturate(NdotL));

    // Face selection via dot product with face vectors
    float3 lightVecN = normalize(offsetPos - light.position);
    float maxDot = -1e30;
    uint index = 0;
    for (uint i = 0; i < 4; i++)
    {
        float d = dot(lightVecN, tetraData.faceVectors[i].xyz);
        if (d > maxDot) { maxDot = d; index = i; }
    }

    // Shadow matrix already includes bias + tile transform
    float4 lsPos = mul(light.lightMatrices[index], float4(offsetPos, 1.0));
    float3 proj = lsPos.xyz / lsPos.w;
    // lightMatrices map world → tile clip-NDC (not UV space).
    // Convert clip-NDC to tile UV: x = (NDC+1)/2, y = (1-NDC)/2 (Vulkan y-flip).
    float2 uv;
    uv.x = proj.x * 0.5 + 0.5;
    uv.y = 1.0 - (proj.y * 0.5 + 0.5);
    float currentDepth = proj.z;

    if (any(uv < 0.0) || any(uv > 1.0)) return 1.0;
    if (currentDepth < 0.0 || currentDepth > 1.0) return 1.0;

    float bias = constantBias;

    // PCF 3x3
    uint w, h;
    tiledShadowMap.GetDimensions(w, h);
    float2 texelSize = 1.0 / float2(w, h);

    float shadow = 0.0;
    for (int x = -1; x <= 1; ++x)
        for (int y = -1; y <= 1; ++y)
            shadow += tiledShadowMap.SampleCmpLevelZero(
                shadowSampler,
                uv + float2(x, y) * texelSize,
                currentDepth - bias);
    return shadow / 9.0;
}
