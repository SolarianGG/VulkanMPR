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

    float4x4 M = object.worldTransform;
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
    bool bNeedToAdd = IsInFrustum(minAABB, maxAABB, pushConstants.viewProj);
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