import utils;
import types;

struct PushConstants {
    PointLight *pointLights;
    PointLight *visiblePointLights; 
    uint *pointLightsVisibleCount;
    uint pointLightsCount;
    uint _padding0;
    float4x4 viewProj;
};

[[vk::push_constant]]
PushConstants pushConstants;

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 threadID: SV_DispatchThreadID) {
    if (threadID.x >= pushConstants.pointLightsCount) {
        return;
    }
    PointLight light = pushConstants.pointLights[threadID.x];

    float3 lightMax = light.position + light.range;
    float3 lightMin = light.position - light.range;
    bool bNeedToAdd = IsInFrustum(lightMin, lightMax, pushConstants.viewProj);

    uint laneOffset = WavePrefixCountBits(bNeedToAdd);
    uint amountToAdd = WaveActiveCountBits(bNeedToAdd);

    uint commandIndexOffset;
    if (WaveIsFirstLane()) {
        commandIndexOffset = __atomic_add(*pushConstants.pointLightsVisibleCount, amountToAdd, MemoryOrder::Relaxed);
    }

    commandIndexOffset = WaveReadLaneFirst(commandIndexOffset);
    commandIndexOffset += laneOffset;

    if (bNeedToAdd) {
        pushConstants.visiblePointLights[commandIndexOffset] = light;
    }
}