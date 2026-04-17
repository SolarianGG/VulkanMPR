import types;

struct PushConstants {
    Mesh *meshes;
    Instance *instances;
    Command *meshDrawCommands;
    Count *meshDrawCommandsCount;
    PointLight *visiblePointLights;
    Count *visiblePointLightsCount;
    uint32_t instanceOffset;
    uint32_t *pointLightIndices;
    uint32_t *pointLightOffsets;
    uint32_t *pointLightOffsetsCounter;
};

[[vk::push_constant]]
PushConstants pushConstants;

static const int WORKGROUP_THREAD_COUNT = 256;

groupshared uint groupCounter;
groupshared uint groupLightIndices[MAX_POINT_LIGHTS];
groupshared uint startLightIndex;

[shader("compute")]
[numthreads(WORKGROUP_THREAD_COUNT, 1, 1)]
void computeMain(uint3 threadID: SV_DispatchThreadID, uint workgroupID : SV_GroupID, uint localID : SV_GroupIndex)
{
    if (localID == 0) {
        groupCounter = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    const uint instanceIndex = workgroupID + pushConstants.instanceOffset;
    Instance instance = pushConstants.instances[instanceIndex];
    Mesh mesh = pushConstants.meshes[instance.meshIndex];

    float4x4 M = instance.worldTransform;
    float3 col0 = float3(M[0][0], M[1][0], M[2][0]);
    float3 col1 = float3(M[0][1], M[1][1], M[2][1]);
    float3 col2 = float3(M[0][2], M[1][2], M[2][2]);
    float3 translation = float3(M[0][3], M[1][3], M[2][3]);

    float3 xa = col0 * mesh.min.x;
    float3 xb = col0 * mesh.max.x;
    float3 ya = col1 * mesh.min.y;
    float3 yb = col1 * mesh.max.y;
    float3 za = col2 * mesh.min.z;
    float3 zb = col2 * mesh.max.z;

    float3 minAABB = translation + min(xa, xb) + min(ya, yb) + min(za, zb);
    float3 maxAABB = translation + max(xa, xb) + max(ya, yb) + max(za, zb);
    uint visiblePointLightsCount = pushConstants.visiblePointLightsCount.commandCount;
    for (uint i = 0; i < visiblePointLightsCount; i += WORKGROUP_THREAD_COUNT) 
    {
        uint lightIndex = localID + i;

        if (lightIndex < visiblePointLightsCount)
        {
            PointLight light = pushConstants.visiblePointLights[lightIndex];

            float3 distances = max(minAABB - light.position, 0.0f) + max(light.position - maxAABB, 0.0f);
            bool bNeedToAdd = dot(distances, distances) <= (light.range * light.range);

            uint count = WaveActiveCountBits(bNeedToAdd);
            uint offset = WavePrefixCountBits(bNeedToAdd);

            uint index;
            if (WaveIsFirstLane()) {
                index = __atomic_add(groupCounter, count, MemoryOrder::Relaxed);
            }
            index = WaveReadLaneFirst(index);

            if (bNeedToAdd)
            {
                groupLightIndices[index + offset] = lightIndex; 
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (localID == 0)
    {
        if (groupCounter > 0)
        {
            uint index = __atomic_add(pushConstants.meshDrawCommandsCount.commandCount, 1, MemoryOrder::Relaxed);
            Command command;
            command.firstIndex = mesh.firstIndex;
            command.indexCount = mesh.indexCount;
            command.vertexOffset = mesh.vertexOffset;
            command.firstInstance = instanceIndex;
            command.instanceCount = groupCounter;
            pushConstants.meshDrawCommands[index] = command;

            startLightIndex = __atomic_add(*pushConstants.pointLightOffsetsCounter, groupCounter);
            pushConstants.pointLightOffsets[instanceIndex] = startLightIndex;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    for (uint i = localID; i < groupCounter; i += WORKGROUP_THREAD_COUNT)
    {
        pushConstants.pointLightIndices[startLightIndex + i] = groupLightIndices[i];
    }
}