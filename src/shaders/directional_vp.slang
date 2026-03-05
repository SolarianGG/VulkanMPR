import types;
import utils;

struct CascadesAABB {
    AABB bounds[MAX_CASCADE_COUNT];
};

struct PushConstants {
    CascadesAABB *splitsAABB;
    DirectionalLight *dirLights;	
};

[[vk::push_constant]]
PushConstants pushConstants;

float4x4 orthoRH_ZO(float left, float right, float bottom, float top, float zNear, float zFar)
{
    float4x4 res = 0;
    res[0][0] = 2.0 / (right - left);
    res[1][1] = 2.0 / (top - bottom);
    res[2][2] = 1.0 / (zFar - zNear);

    res[0][3] = -(right + left) / (right - left);
    res[1][3] = -(top + bottom) / (top - bottom);
    res[2][3] = -zNear / (zFar - zNear);
    res[3][3] = 1.0;

    return res;
}

// Bad distribution with (1, 1, 1), meaning  we are guaranteed to have inactive invocations in every subgroup
// Fix it later by using Wave Intrinsics
[shader("compute")]
[numthreads(1, 1, 1)] 
void dirVpComputeMain(uint3 threadID : SV_DispatchThreadID)
{
    uint cascadeCount = pushConstants.dirLights[threadID.x].cascadeCount.r;
    float4x4 lightView = pushConstants.dirLights[threadID.x].cascadeVPs[0];
    CascadesAABB cascadesBounds = pushConstants.splitsAABB[threadID.x];
    for (uint i = 0; i < cascadeCount; ++i) {
        AABB bounds = cascadesBounds.bounds[i];
        float minX = asfloat(bounds.minX);
        float maxX = asfloat(bounds.maxX);
        float minY = asfloat(bounds.minY);
        float maxY = asfloat(bounds.maxY);
        float minZ = asfloat(bounds.minZ);
        float maxZ = asfloat(bounds.maxZ);
        float4x4 lightProj = orthoRH_ZO(minX, maxX, minY, maxY, minZ, maxZ);
        float4x4 lightProjView = mul(lightProj, lightView);
        pushConstants.dirLights[threadID.x].cascadeVPs[i] = mul(lightView, lightProj);
    }
}