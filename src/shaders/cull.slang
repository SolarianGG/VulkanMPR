import utils;
import types;

struct PushConstants {
    Mesh *meshes;
    Instance *objects;
    Command *commands;
    Count *count;
    float4x4 viewProj;
    uint32_t objectCount;
    uint32_t objectOffset;
};

[[vk::push_constant]]
PushConstants pushConstants;

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 threadID: SV_DispatchThreadID) {
    static_assert(sizeof(Mesh) == 48, "Size of mesh should be equal to 48");
    uint objectIndex = threadID.x;

    if (objectIndex >= pushConstants.objectCount)
        return;
    Instance object = pushConstants.objects[objectIndex + pushConstants.objectOffset];

    Mesh mesh = pushConstants.meshes[object.meshIndex];

    // Now we're using subgroups in order to remove atomic increment memory pressure
    float3 worldMin = mul(object.worldTransform, float4(mesh.min, 1.0f)).xyz;
    float3 worldMax = mul(object.worldTransform, float4(mesh.max, 1.0f)).xyz;
    bool bNeedToAdd = IsInFrustum(worldMin, worldMax, pushConstants.viewProj);
    uint laneOffset = WavePrefixCountBits(bNeedToAdd);
    uint amountToAdd = WaveActiveCountBits(bNeedToAdd);

    uint commandIndexOffset;
    if (WaveIsFirstLane()) {
      commandIndexOffset = __atomic_add(pushConstants.count.commandCount, amountToAdd, MemoryOrder::Relaxed);
    }

    commandIndexOffset = WaveReadLaneFirst(commandIndexOffset);
    commandIndexOffset += laneOffset;


    // TODO: Right now we generate only one command per instance,
    // later I'll try one command per visible mesh with N instance count
    // For that I will probably need second pass
    if (bNeedToAdd) {
      Command command;
      command.firstIndex = mesh.firstIndex;
      command.indexCount = mesh.indexCount;
      command.vertexOffset = mesh.vertexOffset;
      command.firstInstance = objectIndex + pushConstants.objectOffset;
      command.instanceCount = 1;

      pushConstants.commands[commandIndexOffset] = command;
    }
}