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
    DirectionalLight *dirLights;
    uint directionalLightCount;
    float near;
    float far;
    uint padding0;
    uint padding1;
    uint padding2;
    float4x4 inverseCameraViewProj;
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
    float3 worldSpace = ReconstructWorldPos(threadID.xy, uint2(width, height), pushConstants.inverseCameraViewProj, z_initial);
    const float cameraViewDepth = LinearizeDepth(z_initial, pushConstants.near, pushConstants.far);

    uint dirLightCount = pushConstants.directionalLightCount;

    for (uint i = 0; i < dirLightCount; ++i) {
        uint cascadeCount = pushConstants.dirLights[i].cascadeCount.r;
        float splitDistances[MAX_CASCADE_COUNT];
        // Compute logarithmic cascade splits
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

        // NOTE: At this point there are no view project matrices in cascade splits,
        // so we calculate view matrix and place it at the first cascade vp matrix
        // we will calculate proper vp's in the next stage
        float3 lightSpace = mul(pushConstants.dirLights[i].cascadeVPs[0], float4(worldSpace, 1.0f)).xyz;

        uint cascadeIndex = 0;
        for (uint j = 1; j < cascadeCount; ++j) {
            if (cameraViewDepth < splitDistances[j]) break;
                cascadeIndex = j;
        }

        AtomicMinFloat(&pushConstants.splitsAABB[i].bounds[cascadeIndex].minX, lightSpace.x);
        AtomicMinFloat(&pushConstants.splitsAABB[i].bounds[cascadeIndex].minY, lightSpace.y);
        AtomicMinFloat(&pushConstants.splitsAABB[i].bounds[cascadeIndex].minZ, lightSpace.z);
        AtomicMaxFloat(&pushConstants.splitsAABB[i].bounds[cascadeIndex].maxX, lightSpace.x);
        AtomicMaxFloat(&pushConstants.splitsAABB[i].bounds[cascadeIndex].maxY, lightSpace.y);
        AtomicMaxFloat(&pushConstants.splitsAABB[i].bounds[cascadeIndex].maxZ, lightSpace.z);
        pushConstants.dirLights[i].splitDistances = splitDistances;
    }
}
