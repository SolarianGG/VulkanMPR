import utils;
import types;
import shadows;

[[vk::binding(0, 0)]]
Texture2D<float> depthMap;

struct CascadesAABB {
    AABB bounds[MAX_CASCADE_COUNT];
};

struct PushConstants {
    MinMax *minMax;
    CascadesAABB *splitsAABB;
    DirectionalLight *dirLight;
    float near;
    float far;
    float4x4 inverseCameraViewProj;
    float4x4 lightViewMatrix;
};

[[vk::push_constant]]
PushConstants pushConstants;

[shader("compute")]
[numthreads(16, 16, 1)]
void depthPartitionCS(uint3 threadID : SV_DispatchThreadID)
{
    uint width, height;
    depthMap.GetDimensions(width, height);
    if (threadID.x >= width || threadID.y >= height) return;

    const float z_initial = depthMap[threadID.xy].r;
    if (z_initial >= 1.0) return;

    const float depthMin = asfloat(pushConstants.minMax.min);
    const float depthMax = asfloat(pushConstants.minMax.max);
    const float ratio = depthMax / depthMin;

    uint cascadeCount = pushConstants.dirLight.cascadeCount;
    float splitDistances[MAX_CASCADE_COUNT];
    for (uint j = 0; j < cascadeCount; ++j) {
        if (cascadeCount > 1) {
            splitDistances[j] = depthMin * pow(ratio, float(j) / float(cascadeCount));
        } else {
            splitDistances[j] = depthMax;
        }
    }
    for (uint j = cascadeCount; j < MAX_CASCADE_COUNT; ++j) {
         splitDistances[j] = splitDistances[j - 1];
    }

    float3 worldSpace = ReconstructWorldPos(threadID.xy, uint2(width, height), pushConstants.inverseCameraViewProj, z_initial);
    float3 lightSpace = mul(pushConstants.lightViewMatrix, float4(worldSpace, 1.0f)).xyz;

    const float cameraViewDepth = LinearizeDepth(z_initial, pushConstants.near, pushConstants.far);
    uint cascadeIndex = 0;
    for (uint j = 1; j < cascadeCount; ++j) {
        if (cameraViewDepth < splitDistances[j]) break;
            cascadeIndex = j;
    }

    uint4 groupMask = WaveMatch(cascadeIndex);

    int4 highLanes = (int4)(firstbithigh(groupMask) | uint4(0, 0x20, 0x40, 0x60));
    // The signed max should be the highest lane index in the group.
    uint highLane = (uint)max(max(max(highLanes.x, highLanes.y), highLanes.z), highLanes.w);
    bool leader = WaveGetLaneIndex() == highLane;

    float3 groupMin = WaveMultiMin(lightSpace, groupMask);
    float3 groupMax = WaveMultiMax(lightSpace, groupMask);
    
    if (leader) {
        AtomicMinFloat(&pushConstants.splitsAABB.bounds[cascadeIndex].minX, groupMin.x);
        AtomicMinFloat(&pushConstants.splitsAABB.bounds[cascadeIndex].minY, groupMin.y);
        AtomicMinFloat(&pushConstants.splitsAABB.bounds[cascadeIndex].minZ, groupMin.z);
        AtomicMaxFloat(&pushConstants.splitsAABB.bounds[cascadeIndex].maxX, groupMax.x);
        AtomicMaxFloat(&pushConstants.splitsAABB.bounds[cascadeIndex].maxY, groupMax.y);
        AtomicMaxFloat(&pushConstants.splitsAABB.bounds[cascadeIndex].maxZ, groupMax.z);
    }
    if (WaveIsFirstLane())
    {
        pushConstants.dirLight.splitDistances = splitDistances;
    }
}
