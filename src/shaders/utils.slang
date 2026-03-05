module utils;

public float3 ReconstructWorldPos(
    uint2 pixel,
    uint2 resolution,
    float4x4 invViewProj,
    float depth)
{
    float2 uv =
        (float2(pixel) + 0.5) / float2(resolution);

    float2 ndc;
    ndc.x = uv.x * 2.0 - 1.0;
    ndc.y = uv.y * 2.0 - 1.0;

    float4 clip =
        float4(ndc, depth, 1.0);

    float4 world =
        mul(invViewProj, clip);

    return world.xyz / world.w;
}

[[ForceInline]]
public float LinearizeDepth(float d, float zNear, float zFar)
{
    return zNear * zFar / (zFar + d * (zNear - zFar));
}

public void AtomicMinFloat(uint *addr, float value)
{
    uint newVal = asuint(value);
    uint oldVal;

    while (true)
    {
        InterlockedCompareExchange(*addr, ~0u, ~0u, oldVal);

        float oldFloat = asfloat(oldVal);

        if (oldFloat <= value)
            break;

        uint expected = oldVal;
        InterlockedCompareExchange(*addr, expected, newVal, oldVal);

        if (oldVal == expected)
            break;
    }
}

public void AtomicMaxFloat(uint *addr, float value)
{
    uint newVal = asuint(value);
    uint oldVal;

    while (true)
    {
        InterlockedCompareExchange(*addr, ~0u, ~0u, oldVal);

        float oldFloat = asfloat(oldVal);

        if (oldFloat >= value)
            break;

        uint expected = oldVal;
        InterlockedCompareExchange(*addr, expected, newVal, oldVal);

        if (oldVal == expected)
            break;
    }
}
