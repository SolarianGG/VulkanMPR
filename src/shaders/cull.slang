import utils;
struct Mesh {
    uint32_t indexCount;
    uint32_t firstIndex;
    int32_t vertexOffset;
    #if 0
    float3 min;
    float3 max;
    #endif
};

struct Command {
    uint32_t indexCount;
    uint32_t instanceCount;
    uint32_t firstIndex;
    int32_t vertexOffset;
    uint32_t firstInstance;
};

struct Count {
    uint commandCount;
};

struct PushConstants {
    Mesh *meshes;
    Instance *objects;
    #if 0
    Frustum *frustum;
    #endif
    Command *commands;
    Count *count;
    uint objectCount;
    uint objectOffset;
};

bool IsInFrustum(Instance object, Mesh mesh) {
  // TODO: Implement frustrum culling visibility test
    return true; 
}

[[vk::push_constant]]
PushConstants pushConstants;

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 threadID: SV_DispatchThreadID) {
    uint objectIndex = threadID.x;

    if (objectIndex < pushConstants.objectCount) {
        Instance object = pushConstants.objects[objectIndex + pushConstants.objectOffset];

        Mesh mesh = pushConstants.meshes[object.meshIndex];

        if (IsInFrustum(object, mesh)) {
            const uint maxCommands = 100000;
            uint commandIndex = __atomic_add(pushConstants.count.commandCount, 1, MemoryOrder::Relaxed);
            
            if (commandIndex < maxCommands) {
                pushConstants.commands[commandIndex].firstIndex = mesh.firstIndex;
                pushConstants.commands[commandIndex].indexCount = mesh.indexCount;
                pushConstants.commands[commandIndex].vertexOffset = mesh.vertexOffset;
                pushConstants.commands[commandIndex].firstInstance = objectIndex + pushConstants.objectOffset;
                pushConstants.commands[commandIndex].instanceCount = 1;
            }
        }
    }
}