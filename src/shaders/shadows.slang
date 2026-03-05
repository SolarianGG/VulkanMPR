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

    if (any(uv < 0.0) || any(uv > 1.0)) return 1.0;

    float currentDepth = proj.z;
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
