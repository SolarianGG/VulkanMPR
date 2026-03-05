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

[shader("compute")]
[numthreads(16, 16, 1)]
void depthReductionCS(uint3 threadID : SV_DispatchThreadID)
{
    uint width, height;
    depthMap.GetDimensions(width, height);
    if (threadID.x >= width || threadID.y >= height) return;

    const float z_initial = depthMap[threadID.xy].r;
    if (z_initial >= 1.0 || z_initial < 0.0) return;

    float linearizedDepth = LinearizeDepth(z_initial, pushConstants.near, pushConstants.far);
    float min = WaveActiveMin(linearizedDepth);
    float max = WaveActiveMax(linearizedDepth);
    if (WaveIsFirstLane())
    {
        AtomicMinFloat(&pushConstants.minMax.min, min);
        AtomicMaxFloat(&pushConstants.minMax.max, max);
    }
}
