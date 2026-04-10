import types;
import utils;
[[vk::binding(0, 0)]]
Texture2D<float> depthMap;

struct PushConstants {
    MinMax *minMax;
    float near;
    float far;
};

[[vk::push_constant]]
PushConstants pushConstants;

groupshared uint groupMin;
groupshared uint groupMax;
groupshared uint width;
groupshared uint height;
groupshared uint minLocalIndex;

[shader("compute")]
[numthreads(16, 16, 1)]
void depthReductionCS(uint3 threadID : SV_DispatchThreadID, uint localIndex : SV_GroupIndex)
{
    if (localIndex == 0)
    {
        groupMax = asuint(0.0f);
        groupMin = asuint(float ::maxValue);
        minLocalIndex = ~0u;

        depthMap.GetDimensions(width, height);
    }
    GroupMemoryBarrierWithGroupSync();

    bool valid = (threadID.x < width && threadID.y < height);
    float z_initial = valid ? depthMap[threadID.xy] : 1.0f;
    valid = valid && (z_initial < 1.0f && z_initial >= 0.0f);

    if (valid)
    {
        float linearizedDepth = LinearizeDepth(z_initial, pushConstants.near, pushConstants.far);
        float waveMin = WaveActiveMin(linearizedDepth);
        float waveMax = WaveActiveMax(linearizedDepth);

        uint minWaveLocalIndex = WaveActiveMin(localIndex);
        if (WaveIsFirstLane())
        {
            InterlockedMin(groupMin, asuint(waveMin));
            InterlockedMax(groupMax, asuint(waveMax));
            InterlockedMin(minLocalIndex, minWaveLocalIndex);
        }
    }

    GroupMemoryBarrierWithGroupSync();
    if (localIndex == minLocalIndex)
    {
        InterlockedMin(pushConstants.minMax.min, groupMin);
        InterlockedMax(pushConstants.minMax.max, groupMax);
    }
}
