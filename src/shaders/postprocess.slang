[[vk::binding(0, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> outputTexture;



[shader("compute")]
[numthreads(16, 16, 1)]
void computeMain(uint3 threadID : SV_DispatchThreadID) {
    uint width, height;
    outputTexture.GetDimensions(width, height);
    if (threadID.x >= width || threadID.y >= height) return;

    int2 uv = int2(threadID.xy);

    outputTexture[uv].rgb = pow(outputTexture[uv].rgb, float3(1.0f / 2.2f));
}