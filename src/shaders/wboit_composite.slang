struct VSOutput {
    float4 pos : SV_Position;
};

[[vk::binding(0, 0)]]
[[vk::image_format("rgba16f")]]
Texture2D<float4> AccumImage;
[[vk::binding(1, 0)]]
[[vk::image_format("r16f")]]
Texture2D<float> RevealImage;

[shader("vertex")]
VSOutput vertexMain(uint vertexIndex: SV_VulkanVertexID) {
    const float2 vertices[] = {
    float2(-1.0f, -1.0f),
    float2(1.0f, -1.0f),
    float2(1.0f, 1.0f),
    float2(-1.0f, -1.0f),
    float2(1.0f, 1.0f),
    float2(-1.0f, 1.0f)
};

    VSOutput output;
    output.pos = float4(vertices[vertexIndex], 0.0f, 1.0f);
    return output;
}

struct PSOutput {
    float4 color : SV_Target0;
};

#define EPSILON 0.00001f

bool isApproximatelyEqual(float a, float b)
{
    return abs(a - b) <= (abs(a) < abs(b) ? abs(b) : abs(a)) * EPSILON;
}

float max3(float3 v)
{
    return max(max(v.x, v.y), v.z);
}

[shader("pixel")]
PSOutput pixelMain(VSOutput input, float4 fragCoord: SV_Position) {
    // fragment coordination
    int2 coords = int2(fragCoord.xy);

    // fragment revealage
    float revealage = RevealImage[coords].r;

    if (isApproximatelyEqual(revealage, 1.0f))
        discard;

    float4 accumulation = AccumImage[coords];

    if (isinf(max3(abs(accumulation.rgb))))
        accumulation.rgb = float3(accumulation.a);

    float3 average_color = accumulation.rgb / max(accumulation.a, EPSILON);

    PSOutput output;
    output.color = float4(average_color, 1.0f - revealage);
    return output;
}