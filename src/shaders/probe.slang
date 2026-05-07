import types;
import ddgi_common;
import utils;
import raytracing;
import lights;


[[vk::binding(0, 0)]]
RaytracingAccelerationStructure SceneTLAS;

[[vk::binding(1, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2DArray<float4> RayDatas[];

[[vk::binding(2, 0)]]
[[vk::image_format("rgba16f")]]
Texture2DArray<float4> Irradiance[];

[[vk::binding(3, 0)]]
[[vk::image_format("rg32f")]]
Texture2DArray<float2> Distance[];

[[vk::binding(0, 1)]] ConstantBuffer<GLTFMaterialData> materials[];
[[vk::binding(1, 1)]] SamplerState samplers[];
[[vk::binding(2, 1)]] Texture2D textures[];

struct PushConstants {
    DDGIVolume *volumes;
    uint currentVolumeIndex;
    DirectionalLight *dirLight;
    PointLight *pointLights;
    uint pointLightsCount;
    VertexPosition *vPositions;
    VertexAttributes *vAttributes;
    uint32_t *indices;
    Instance *instances;
    Mesh *meshes;
    float rayNormalBias;
    float rayViewBias;
};

[[vk::push_constant]]
PushConstants pushConstants;

[shader("raygeneration")]
void RGSMain()
{
    DDGIVolume volume = pushConstants.volumes[pushConstants.currentVolumeIndex];
    RWTexture2DArray<float4> RayData = RayDatas[pushConstants.currentVolumeIndex];

    // Compute the probe index for this thread
    int rayIndex = DispatchRaysIndex().x;        // index of the ray to trace for this probe
    int probePlaneIndex = DispatchRaysIndex().y; // index of this probe within the plane of probes
    int planeIndex = DispatchRaysIndex().z;      // index of the plane this probe is part of
    int probesPerPlane = DDGIGetProbesPerPlane(volume.probeCounts);

    int probeIndex = (planeIndex * probesPerPlane) + probePlaneIndex;

    // Get the probe's grid coordinates
    int3 probeCoords = DDGIGetProbeCoords(probeIndex, volume);

    // Get the probe's world position
    // Note: world positions are computed from probe coordinates *not* adjusted for infinite scrolling
    float3 probeWorldPosition = DDGIGetProbeWorldPosition(probeCoords, volume);

    // Get a random normalized ray direction to use for a probe ray
    float3 probeRayDirection = DDGIGetProbeRayDirection(rayIndex, volume);

    uint3 outputCoords = DDGIGetRayDataTexelCoords(rayIndex, probeIndex, volume);

    RayDesc ray;
    ray.Origin = probeWorldPosition;
    ray.Direction = probeRayDirection;
    ray.TMin = 0.f;
    ray.TMax = volume.probeMaxRayDistance;

    Payload payl = (Payload)0;
    TraceRay(SceneTLAS, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payl);

    // Miss
    if (payl.hitT < 0.f)
    {
        // TODO: Sample sky irradiance diffuse map
        RayData[outputCoords] = float4(0.33f, 0.66f, 0.99f, 1e27f);
        return;
    }

    // Probe inside a wall
    if (payl.hitKind == HIT_KIND_TRIANGLE_BACK_FACE)
    {
        RayData[outputCoords].w = -payl.hitT * 0.2f;
        return;
    }

    float3 diffuse = DirectDiffuseLighting(payl, pushConstants.rayNormalBias, pushConstants.rayViewBias, SceneTLAS, pushConstants.dirLight, pushConstants.pointLights, pushConstants.pointLightsCount);

    // TODO: Add irradiance evalution
    float3 irradiance = 0.5f;
    float3 surfaceBias = DDGIGetSurfaceBias(payl.normal, ray.Direction, volume);

    float volumeBlendWeight = DDGIGetVolumeBlendWeight(payl.worldPosition, volume);

    // Default trillinear samplers that works as billinear because irradiance and distance textures have only 1 mip
    SamplerState defaultSampler = samplers[0];
    Texture2DArray<float4> irradianceTexture = Irradiance[pushConstants.currentVolumeIndex];
    Texture2DArray<float2> distanceTexture = Distance[pushConstants.currentVolumeIndex];
    if (volumeBlendWeight > 0.f)
    {
        irradiance = DDGIGetVolumeIrradiance(payl.worldPosition, surfaceBias, payl.normal, volume, defaultSampler, irradianceTexture, distanceTexture);
        irradiance *= volumeBlendWeight;
    }

    float maxAlbedo = 0.9f;
    float3 radiance = diffuse + ((min(payl.albedo, float3(maxAlbedo, maxAlbedo, maxAlbedo)) / float ::getPi()) * irradiance);
    RayData[outputCoords] = float4(saturate(radiance), payl.hitT);
}

[shader("anyhit")]
void AHSMain(inout Payload payload, BuiltInTriangleIntersectionAttributes attr)
{
    // Get barycentric coordinates
    float3 barycentrics = float3(1 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);

    // Get instance and primitive information
    uint instanceID = InstanceIndex();
    uint meshID = InstanceID();
    uint triID = PrimitiveIndex();

    Instance instance = pushConstants.instances[instanceID];
    Mesh mesh = pushConstants.meshes[instance.meshIndex];
    GLTFMaterialData material = materials[NonUniformResourceIndex(instance.materialID)];

    float alpha = material.colorFactors.w;

    // Load the vertices
    Vertex vertices[3] = LoadVertices(mesh, triID, pushConstants.vPositions, pushConstants.vAttributes, pushConstants.indices);

    // Interpolate uv
    float2 uv = float2(0.0f, 0.0f);
    [unroll]
    for (uint i = 0; i < 3; ++i)
    {
        uv += float2(vertices[i].u, vertices[i].v) * barycentrics[i];
    }

    // TODO: Compute differentials instead of using level 0

    // Sample the texture
    alpha *= textures[NonUniformResourceIndex(instance.colorTextureID)]
        .SampleLevel(samplers[NonUniformResourceIndex(instance.colorSamplerID)], uv, 0).a;

    if (alpha < material.alphaCutoff.r) IgnoreHit();
}

[shader("closesthit")]
void CHSMain(inout Payload payload, BuiltInTriangleIntersectionAttributes attr)
{
    payload.hitT = RayTCurrent();
    payload.hitKind = HitKind();

    // Load the intersected mesh geometry's data
    // Get barycentric coordinates
    float3 barycentrics = float3(1 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);

    // Get instance and primitive information
    uint instanceID = InstanceIndex();
    uint meshID = InstanceID();
    uint triID = PrimitiveIndex();

    Instance instance = pushConstants.instances[instanceID];
    Mesh mesh = pushConstants.meshes[instance.meshIndex];
    GLTFMaterialData material = materials[NonUniformResourceIndex(instance.materialID)];
    // Load the triangle's vertices
    Vertex vertices[3] = LoadVertices(mesh, triID, pushConstants.vPositions, pushConstants.vAttributes, pushConstants.indices);

    Vertex v = InterpolateVertex(vertices, barycentrics);

    // World position
    payload.worldPosition = v.pos;
    payload.worldPosition = mul(ObjectToWorld3x4(), float4(payload.worldPosition, 1.f)).xyz; // instance transform

    // Geometric normal
    payload.normal = v.norm;
    payload.normal = normalize(mul(ObjectToWorld3x4(), float4(payload.normal, 0.f)).xyz);
    payload.shadingNormal = payload.normal;

    // Load the surface material
    payload.albedo = material.colorFactors.xyz;
    payload.opacity = material.colorFactors.w;

    
    // TODO: Compute differentials instead of using level 0

    // Albedo and Opacity
    {
        float4 bco = textures[NonUniformResourceIndex(instance.colorTextureID)]
            .SampleLevel(samplers[NonUniformResourceIndex(instance.colorSamplerID)], float2(v.u, v.v), 0);
        payload.albedo *= bco.rgb;
        payload.opacity *= bco.a;
    }

    // Shading normal
    {
        float3 tangent = normalize(mul(ObjectToWorld3x4(), float4(v.tangent.xyz, 0.f)).xyz);
        float3 bitangent = cross(payload.normal, tangent) * v.tangent.w;
        float3x3 TBN = { tangent, bitangent, payload.normal };
        payload.shadingNormal = textures[NonUniformResourceIndex(instance.normalTextureID)]
            .SampleLevel(samplers[NonUniformResourceIndex(instance.normalSamplerID)], float2(v.u, v.v), 0).xyz;
        payload.shadingNormal = (payload.shadingNormal * 2.f) - 1.f; // Transform to [-1, 1]
        payload.shadingNormal = mul(payload.shadingNormal, TBN);     // Transform tangent-space normal to world-space
    }

    // Roughness and Metallic
    {
        float3 rm = textures[NonUniformResourceIndex(instance.metalRoughTextureID)]
            .SampleLevel(samplers[NonUniformResourceIndex(instance.metalRoughSamplerID)], float2(v.u, v.v), 0).xyz;
        payload.metallic = material.metalRoughnessFactors.r * rm.b;
        payload.roughness = material.metalRoughnessFactors.g * rm.g;
    }

    // Emissive
    {
         float3 emissiveSample = textures[NonUniformResourceIndex(instance.emissiveTextureID)]
            .SampleLevel(samplers[NonUniformResourceIndex(instance.emissiveSamplerID)],float2(v.u, v.v), 0).rgb;

         payload.albedo += emissiveSample * material.emissiveFactors.xyz;
    }

}


[shader("miss")]
void MissMain(inout Payload packedPayload)
{
    packedPayload.hitT = -1.f;
}
