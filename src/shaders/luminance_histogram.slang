import utils;

[[vk::binding(0, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> drawImage;

struct PushConstants
{
    uint *histogram;
    float minLogLum;
    float invLogLumRange;
};

[[vk::push_constant]]
PushConstants pushConstants;

groupshared uint histogramShared[256];

uint luminanceToBin(float3 color, float minLogLum, float invLogLumRange)
{
    float lum = LinearRGBToLuminance(color);
    if (lum < 1e-5f)
        return 0;
    float logLum = clamp((log2(lum) - minLogLum) * invLogLumRange, 0.0f, 1.0f);
    return uint(logLum * 254.0f + 1.0f);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 threadID: SV_DispatchThreadID, uint localIndex : SV_GroupIndex)
{
    histogramShared[localIndex] = 0;
    GroupMemoryBarrierWithGroupSync();

    uint width, height;
    drawImage.GetDimensions(width, height);

    if (threadID.x < width && threadID.y < height)
    {
        float3 color = drawImage[int2(threadID.xy)].rgb;
        uint bin = luminanceToBin(color, pushConstants.minLogLum, pushConstants.invLogLumRange);

        uint4 groupMask = WaveMatch(bin);

        uint binValue = dot(countbits(groupMask), uint4(1, 1, 1, 1));
        if (IsGroupLeader(groupMask))
        {
            InterlockedAdd(histogramShared[bin], binValue);
        }
    }

    GroupMemoryBarrierWithGroupSync();

    InterlockedAdd(pushConstants.histogram[localIndex], histogramShared[localIndex]);
}
