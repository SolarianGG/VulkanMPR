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
        const float s = float(POINT_LIGHT_TILE_SIZE) / float(POINT_LIGHT_SHADOW_MAP_SIZE);
        const float px = (commandIndexOffset % POINT_LIGHT_TILES_PER_ROW) * s;
        const float py = (commandIndexOffset / POINT_LIGHT_TILES_PER_ROW) * s;

        // Tile center in NDC [-1,1] (viewport uses negative height for Y-flip)
        const float cx = 2.0f * px + s - 1.0f;
        const float cy = 1.0f - 2.0f * py - s;

        {
            float4x4 m = 0.0f;
            m[0][0] = s;
            m[1][1] = s / 2.0f;
            m[2][2] = 1.0f;
            m[3][3] = 1.0f;
            m[0][3] = cx;
            m[1][3] = cy - s / 2.0f;

           light.lightMatrices[0] = mul(m, light.lightMatrices[0]);
        }
        {
            float4x4 m = 0.0f;
            m[0][0] = s / 2.0f;
            m[1][1] = s;
            m[2][2] = 1.0f;
            m[3][3] = 1.0f;
            m[0][3] = cx + s / 2.0f;
            m[1][3] = cy;

            light.lightMatrices[1] = mul(m, light.lightMatrices[1]);
        }
        {
            float4x4 m = 0.0f;
            m[0][0] = s;
            m[1][1] = s / 2.0f;
            m[2][2] = 1.0f;
            m[3][3] = 1.0f;
            m[0][3] = cx;
            m[1][3] = cy + s / 2.0f;

            light.lightMatrices[2] = mul(m, light.lightMatrices[2]);
        }
        {
            float4x4 m = 0.0f;
            m[0][0] = s / 2.0f;
            m[1][1] = s;
            m[2][2] = 1.0f;
            m[3][3] = 1.0f;
            m[0][3] = cx - s / 2.0f;
            m[1][3] = cy;

            light.lightMatrices[3] = mul(m, light.lightMatrices[3]);
        }
        pushConstants.visiblePointLights[commandIndexOffset] = light;
    }
}