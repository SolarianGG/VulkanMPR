import utils;
struct Mesh {
    uint32_t indexCount;
    uint32_t firstIndex;
    int32_t vertexOffset;
    uint32_t padding;
    float3 min;
    float padding1;
    float3 max;
    float padding2;
};

struct Command {
    uint32_t indexCount;
    uint32_t instanceCount;
    uint32_t firstIndex;
    int32_t vertexOffset;
    uint32_t firstInstance;
};

struct Count {
    uint32_t commandCount;
};


struct PushConstants {
    Mesh *meshes;
    Instance *objects;
    Command *commands;
    Count *count;
    float4x4 viewProj;
    uint32_t objectCount;
    uint32_t objectOffset;
};

bool within(float lo, float x, float hi) {
    return (lo <= x) && (x <= hi);
}

bool IsInFrustum(Instance object, Mesh mesh) {
    float4 corners[8] = {
        { mesh.min.x, mesh.min.y, mesh.min.z, 1.0 }, // x y z
        { mesh.max.x, mesh.min.y, mesh.min.z, 1.0 }, // X y z
        { mesh.min.x, mesh.max.y, mesh.min.z, 1.0 }, // x Y z
        { mesh.max.x, mesh.max.y, mesh.min.z, 1.0 }, // X Y z

        { mesh.min.x, mesh.min.y, mesh.max.z, 1.0 }, // x y Z
        { mesh.max.x, mesh.min.y, mesh.max.z, 1.0 }, // X y Z
        { mesh.min.x, mesh.max.y, mesh.max.z, 1.0 }, // x Y Z
        { mesh.max.x, mesh.max.y, mesh.max.z, 1.0 }, // X Y Z
    };

    float4x4 MVP = mul(pushConstants.viewProj, object.worldTransform);
    [[unroll]]
    for (int i = 0; i < 8; ++i) {
        corners[i] = mul(MVP, corners[i]);
    }
    bool isInside;
    // Checking against each plane, aabb in only outside the frustrum if all of it 8 corners
    // is outside a plane
#define ALL_OUTSIDE_X(check)                                                                                           \
    isInside = false;                                                                                                  \
    [[unroll]] \
    for (int i = 0; i < 8; ++i) {                                                                                                                  \
        if (!(check)) { isInside = true; break; }                                                                                                     \
    }                                                                                                                  \
    if (!isInside) return false; \

    
    ALL_OUTSIDE_X(corners[i].x < -corners[i].w)
    ALL_OUTSIDE_X(corners[i].x > corners[i].w)
    ALL_OUTSIDE_X(corners[i].y < -corners[i].w)
    ALL_OUTSIDE_X(corners[i].y > corners[i].w)
    ALL_OUTSIDE_X(corners[i].z < 0)
    ALL_OUTSIDE_X(corners[i].z > corners[i].w)
#undef ALL_OUTSIDE_X


    return true;

}

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
    bool bNeedToAdd = IsInFrustum(object, mesh);
    uint laneOffset = WavePrefixCountBits(bNeedToAdd);
    uint amountToAdd = WaveActiveCountBits(bNeedToAdd);

    uint commandIndexOffset;
    if (WaveIsFirstLane()) {
      commandIndexOffset = __atomic_add(pushConstants.count.commandCount, amountToAdd, MemoryOrder::Relaxed);
    }

    commandIndexOffset = WaveReadLaneFirst(commandIndexOffset);
    commandIndexOffset += laneOffset;

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