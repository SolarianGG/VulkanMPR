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

groupshared uint3 ldsMin[MAX_CASCADE_COUNT];
groupshared uint3 ldsMax[MAX_CASCADE_COUNT];
groupshared uint cascadeCount;
groupshared float depthMin;
groupshared float depthMax;
groupshared uint minLocalIndex;

[shader("compute")]
[numthreads(16, 16, 1)]
void depthPartitionCS(uint3 threadID: SV_DispatchThreadID, uint localIndex: SV_GroupIndex)
{
    if (localIndex == 0)
    {
        [unroll]
        for (uint i = 0; i < MAX_CASCADE_COUNT; ++i)
        {
            ldsMin[i] = asuint(float ::maxValue);
            ldsMax[i] = asuint(-float ::maxValue);
        }
        cascadeCount = pushConstants.dirLight.cascadeCount;
        minLocalIndex = ~0u;
        depthMin = asfloat(pushConstants.minMax.min);
        depthMax = asfloat(pushConstants.minMax.max);
    }
    GroupMemoryBarrierWithGroupSync();

    uint width, height;
    depthMap.GetDimensions(width, height);

    bool valid = (threadID.x < width && threadID.y < height);
    float z_initial = valid ? depthMap[threadID.xy].r : 1.0f;
    valid = valid && (z_initial < 1.0f);

    float splitDistances[MAX_CASCADE_COUNT];

    if (valid)
    {
        const float ratio = depthMax / depthMin;

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

        float3 groupMin = WaveMultiMin(lightSpace, groupMask);
        float3 groupMax = WaveMultiMax(lightSpace, groupMask);

        if (IsGroupLeader(groupMask)) {
            AtomicMinFloat(&ldsMin[cascadeIndex].x, groupMin.x);
            AtomicMinFloat(&ldsMin[cascadeIndex].y, groupMin.y);
            AtomicMinFloat(&ldsMin[cascadeIndex].z, groupMin.z);
            AtomicMaxFloat(&ldsMax[cascadeIndex].x, groupMax.x);
            AtomicMaxFloat(&ldsMax[cascadeIndex].y, groupMax.y);
            AtomicMaxFloat(&ldsMax[cascadeIndex].z, groupMax.z);
        }

        uint minWaveIndex = WaveActiveMin(localIndex);
        if (WaveIsFirstLane())
        {
            InterlockedMin(minLocalIndex, minWaveIndex);
        }
    }

    GroupMemoryBarrierWithGroupSync();

    if (localIndex == minLocalIndex)
    {
        for (uint i = 0; i < cascadeCount; ++i)
        {
            AtomicMinFloat(&pushConstants.splitsAABB.bounds[i].minX, asfloat(ldsMin[i].x));
            AtomicMinFloat(&pushConstants.splitsAABB.bounds[i].minY, asfloat(ldsMin[i].y));
            AtomicMinFloat(&pushConstants.splitsAABB.bounds[i].minZ, asfloat(ldsMin[i].z));
            AtomicMaxFloat(&pushConstants.splitsAABB.bounds[i].maxX, asfloat(ldsMax[i].x));
            AtomicMaxFloat(&pushConstants.splitsAABB.bounds[i].maxY, asfloat(ldsMax[i].y));
            AtomicMaxFloat(&pushConstants.splitsAABB.bounds[i].maxZ, asfloat(ldsMax[i].z));
        }
        pushConstants.dirLight.splitDistances = splitDistances;
    }
}
