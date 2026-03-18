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
    ndc.y = 1.0 - uv.y * 2.0;

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

bool within(float lo, float x, float hi) {
    return (lo <= x) && (x <= hi);
}

public bool IsInFrustum(float3 min, float3 max, float4x4 viewProj) {
    float4 corners[8] = {
        { min.x, min.y, min.z, 1.0 }, // x y z
        { max.x, min.y, min.z, 1.0 }, // X y z
        { min.x, max.y, min.z, 1.0 }, // x Y z
        { max.x, max.y, min.z, 1.0 }, // X Y z

        { min.x, min.y, max.z, 1.0 }, // x y Z
        { max.x, min.y, max.z, 1.0 }, // X y Z
        { min.x, max.y, max.z, 1.0 }, // x Y Z
        { max.x, max.y, max.z, 1.0 }, // X Y Z
    };

    [[unroll]]
    for (int i = 0; i < 8; ++i) {
        corners[i] = mul(viewProj, corners[i]);
    }
    bool isInside;
    // Checking against each plane, aabb in only outside the frustrum if all of it 8 corners
    // is outside a plane
#define ALL_OUTSIDE_X(check)                                                                                           \
    isInside = false;                                                                                                  \
    [[unroll]]                                                                                                         \
    for (int i = 0; i < 8; ++i) {                                                                                                                  \
        if (!(check)) { isInside = true; break; }                                                                                                              \
    }                                                                                                                  \
    if (!isInside) return false;

    ALL_OUTSIDE_X(corners[i].x < -corners[i].w)
    ALL_OUTSIDE_X(corners[i].x > corners[i].w)
    ALL_OUTSIDE_X(corners[i].y < -corners[i].w)
    ALL_OUTSIDE_X(corners[i].y > corners[i].w)
    ALL_OUTSIDE_X(corners[i].z < 0)
    ALL_OUTSIDE_X(corners[i].z > corners[i].w)
#undef ALL_OUTSIDE_X

    return true;
}
