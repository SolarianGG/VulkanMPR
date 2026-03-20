import types;

struct PushConstants {
    Mesh *meshes;
    Instance *instances;
    Command *meshDrawCommands;
    Count *meshDrawCommandsCount;
    PointLight *visiblePointLights;
    Count *visiblePointLightsCount;
    uint32_t _padding0;
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

    Instance instance = pushConstants.instances[workgroupID];
    Mesh mesh = pushConstants.meshes[instance.meshIndex];

    float3 minAABB = mul(instance.worldTransform, float4(mesh.min, 1.0f)).xyz;
    float3 maxAABB = mul(instance.worldTransform, float4(mesh.max, 1.0f)).xyz;
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
            command.firstInstance = workgroupID;
            command.instanceCount = groupCounter;
            pushConstants.meshDrawCommands[index] = command;

            startLightIndex = __atomic_add(*pushConstants.pointLightOffsetsCounter, groupCounter);
            pushConstants.pointLightOffsets[workgroupID] = startLightIndex;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    for (uint i = localID; i < groupCounter; i += WORKGROUP_THREAD_COUNT)
    {
        pushConstants.pointLightIndices[startLightIndex + i] = groupLightIndices[i];
    }
}