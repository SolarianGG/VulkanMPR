module shadows;

import types;

public float CalculateDirectionalShadow(
    DirectionalLight light, float3 worldPos, float depth,
    Texture2DArray<float> shadowMaps, SamplerComparisonState shadowSampler,
    )
{

    uint cascadeIndex = 0;
    for (uint j = 1; j < light.cascadeCount.r; ++j) {
        if (depth < light.splitDistances[j]) break;
        cascadeIndex = j;
    }

    float4 lsPos = mul(light.cascadeVPs[cascadeIndex], float4(worldPos, 1.0));
    float3 proj = lsPos.xyz / lsPos.w;
    float2 uv = proj.xy * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;

    if (any(uv < 0.0) || any(uv > 1.0)) return 0.0;

    float currentDepth = proj.z;
    // If depth is outside [0,1] the fragment is outside the light frustum depth range.
    // Treat as fully in shadow (conservative) rather than passing a garbage value to SampleCmp.
    if (currentDepth < 0.0 || currentDepth > 1.0) return 0.0;
    // TODO: Make it specialized constant
    // TODO: Add normal + plane offset bias
    float bias = 0.005;

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
    PointLight light, float3 worldPos,
    TetrahedronData *tetraData,
    Texture2D<float> tiledShadowMap, SamplerComparisonState shadowSampler)
{
    // Face selection via dot product with face vectors
    float3 lightVecN = normalize(worldPos - light.position);
    float maxDot = -1e30;
    uint index = 0;
    for (uint i = 0; i < 4; i++)
    {
        float d = dot(-lightVecN, tetraData.faceVectors[i].xyz);
        if (d > maxDot) { maxDot = d; index = i; }
    }

    // Shadow matrix already includes bias + tile transform
    float4 lsPos = mul(light.lightMatrices[index], float4(worldPos, 1.0));
    float3 proj = lsPos.xyz / lsPos.w;
    // lightMatrices map world → tile clip-NDC (not UV space).
    // Convert clip-NDC to tile UV: x = (NDC+1)/2, y = (1-NDC)/2 (Vulkan y-flip).
    float2 uv;
    uv.x = proj.x * 0.5 + 0.5;
    uv.y = 1.0 - (proj.y * 0.5 + 0.5);
    float currentDepth = proj.z;

    if (any(uv < 0.0) || any(uv > 1.0)) return 0.0;
    if (currentDepth < 0.0 || currentDepth > 1.0) return 0.0;

    float bias = 0.005;

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
