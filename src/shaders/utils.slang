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

float2 OctahedronWrap(float2 v)
{
    float2 w = 1.0f - abs(float2(v.y, v.x));
    if (v.x < 0.0f)
        w.x = -w.x;
    if (v.y < 0.0f)
        w.y = -w.y;
    return w;
}

public float2 QuantizeNormal(float3 n)
{
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    n = float3(n.z > 0.0f ? float2(n.x, n.y) : OctahedronWrap(float2(n.x, n.y)), n.z);

    return float2(n.x, n.y);
}

public float3 DecodeGBufferNormal(float2 f)
{
    float3 n = float3(f.x, f.y, 1.0f - abs(f.x) - abs(f.y));
    float t = max(-n.z, 0.0f);
    n.x += n.x >= 0.0f ? -t : t;
    n.y += n.y >= 0.0f ? -t : t;

    return normalize(n);
}

public vector<float16_t, 3> DecodeNormal(vector<float16_t, 2> f)
{
    f = f * 2.0hf - 1.0hf;

    vector<float16_t, 3> n = vector<float16_t, 3>(f.x, f.y, 1.0hf - abs(f.x) - abs(f.y));
    float16_t t = max(-n.z, 0.0hf);
    n.x += n.x >= 0.0hf ? -t : t;
    n.y += n.y >= 0.0hf ? -t : t;

    return normalize(n);
}

public vector<float16_t, 4> DecodeTangent(vector<float16_t, 2> f)
{
    vector<float16_t, 4> t;
    t.xyz = DecodeNormal(f);

    // Check if lowest bit of Y's mantissa is set. If so, bitangent sign is positive, otherwise negative.
    t.w = bool(bit_cast<uint16_t>(f.y) & 1u) ? 1.0hf : -1.0hf;

    return t;
}
