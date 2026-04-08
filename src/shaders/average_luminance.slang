struct PushConstants
{
    uint *histogram;
    float *avgLum;
    float minLogLum;
    float logLumRange;
    float timeCoeff;
    uint numPixels;
};

[[vk::push_constant]]
PushConstants pushConstants;

groupshared float shared[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 threadID: SV_DispatchThreadID)
{
    uint bin = threadID.x;

    float binCenter = (float(bin) + 0.5f) / 255.0f;
    float logLum = binCenter * pushConstants.logLumRange + pushConstants.minLogLum;
    shared[bin] = (bin == 0) ? 0.0f : float(pushConstants.histogram[bin]) * logLum;

    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint stride = 128; stride > 0; stride >>= 1)
    {
        if (bin < stride)
            shared[bin] += shared[bin + stride];
        GroupMemoryBarrierWithGroupSync();
    }

    if (bin == 0)
    {
        uint blackCount = pushConstants.histogram[0];
        float validPixels = max(float(pushConstants.numPixels) - float(blackCount), 1.0f);
        float newLum = exp2(shared[0] / validPixels);

        float prev = *pushConstants.avgLum;
        *pushConstants.avgLum = prev + (newLum - prev) * pushConstants.timeCoeff;
    }
}
