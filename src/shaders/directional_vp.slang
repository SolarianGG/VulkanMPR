import types;
import utils;

struct CascadesAABB {
    AABB bounds[MAX_CASCADE_COUNT];
};

struct PushConstants {
    CascadesAABB *splitsAABB;
    DirectionalLight *dirLights;
    float3 sceneMin;
    float boundsModifier;
    float3 sceneMax;
    uint shadowMapSize;

    float4x4 lightView;
};

[[vk::push_constant]]
PushConstants pushConstants;

float4x4 orthoRH_ZO(float left, float right, float bottom, float top, float zNear, float zFar)
{
    float4x4 res = 0;
    res[0][0] = 2.0 / (right - left);
    res[1][1] = 2.0 / (top - bottom);
    res[2][2] = -1.0 / (zFar - zNear);

    res[0][3] = -(right + left) / (right - left);
    res[1][3] = -(top + bottom) / (top - bottom);
    res[2][3] = -zNear / (zFar - zNear);
    res[3][3] = 1.0;

    return res;
}

// This value is used to avoid shadow flickering on flat surfaces
// Without it, surfaces with 90 degrees angle on the border of a cascade to the light with appear in light
static const float BOUNDS_MODIFIER = 0.1f; 

// Bad distribution with (1, 1, 1), meaning  we are guaranteed to have inactive invocations in every subgroup
// Fix it later by using Wave Intrinsics
[shader("compute")]
[numthreads(1, 1, 1)] 
void dirVpComputeMain(uint3 threadID : SV_DispatchThreadID)
{
    uint cascadeCount = pushConstants.dirLights[threadID.x].cascadeCount;
    float4x4 lightView = pushConstants.lightView;
    CascadesAABB cascadesBounds = pushConstants.splitsAABB[threadID.x];

    // Extend near plane to cover off-camera occluders between the light and the camera frustum.
    float3 s0 = pushConstants.sceneMin;
    float3 s1 = pushConstants.sceneMax;
    float sceneMaxZ = -float::maxValue;
    for (uint ci = 0; ci < 8; ++ci) {
        float3 corner = float3(
            ((ci & 1) != 0) ? s1.x : s0.x,
            ((ci & 2) != 0) ? s1.y : s0.y,
            ((ci & 4) != 0) ? s1.z : s0.z);
        const float depth = mul(lightView, float4(corner, 1.0f)).z;
        sceneMaxZ = max(sceneMaxZ, depth);
    }

    for (uint i = 0; i < cascadeCount; ++i) {
        AABB bounds = cascadesBounds.bounds[i];
        float minX = asfloat(bounds.minX) - BOUNDS_MODIFIER;
        float maxX = asfloat(bounds.maxX) + BOUNDS_MODIFIER;
        float minY = asfloat(bounds.minY) - BOUNDS_MODIFIER;
        float maxY = asfloat(bounds.maxY) + BOUNDS_MODIFIER;
        float minZ = asfloat(bounds.minZ) - BOUNDS_MODIFIER;
        float maxZ = asfloat(bounds.maxZ);
        maxZ = max(sceneMaxZ, maxZ);
        float4x4 lightProj = orthoRH_ZO(minX, maxX, minY, maxY, -maxZ, -minZ);
        float4x4 lightProjView = mul(lightProj, lightView);

        // Texel snapping: align projection to shadow map texel grid to prevent edge shimmering.
        // Project world origin into shadow NDC, snap to nearest texel, feed offset back.
        float4 shadowOrigin = mul(lightProjView, float4(0.0f, 0.0f, 0.0f, 1.0f));
        float halfSize = float(pushConstants.shadowMapSize) * 0.5f;
        float2 texelOrigin = shadowOrigin.xy * halfSize;
        float2 roundOffset = (round(texelOrigin) - texelOrigin) * (2.0f / float(pushConstants.shadowMapSize));
        lightProjView[0][3] += roundOffset.x;
        lightProjView[1][3] += roundOffset.y;

        pushConstants.dirLights[threadID.x].cascadeVPs[i] = lightProjView;
    }
}