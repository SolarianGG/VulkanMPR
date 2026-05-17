import types;
import ddgi_common;
import utils;
import raytracing;
import lights;


[[vk::binding(0, 0)]]
RaytracingAccelerationStructure SceneTLAS;

[[vk::binding(0, 1)]]
[[vk::image_format("rgba32f")]]
RWTexture2DArray<float4> RayDatas[];

[[vk::binding(0, 2)]]
Texture2DArray<float4> Irradiances[];

[[vk::binding(1, 2)]]
Texture2DArray<float2> Distances[];

[[vk::binding(2, 2)]]
Texture2DArray<float4> ProbeDatas[];

[[vk::binding(3, 2)]]
SamplerState BillinearWrapSampler;

[[vk::binding(0, 3)]] ConstantBuffer<GLTFMaterialData> materials[];
[[vk::binding(1, 3)]] SamplerState samplers[];
[[vk::binding(2, 3)]] Texture2D textures[];

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
    float _pad0;
    float _pad1;
    float4 skyRadiance;
};

[[vk::push_constant]]
PushConstants pushConstants;

[shader("raygeneration")]
void RGSMain()
{
    DDGIVolume volume = pushConstants.volumes[pushConstants.currentVolumeIndex];
    RWTexture2DArray<float4> RayData = RayDatas[pushConstants.currentVolumeIndex];
    Texture2DArray<float4> ProbeData = ProbeDatas[pushConstants.currentVolumeIndex];

    // Compute the probe index for this thread
    int rayIndex = DispatchRaysIndex().x;        // index of the ray to trace for this probe
    int probePlaneIndex = DispatchRaysIndex().y; // index of this probe within the plane of probes
    int planeIndex = DispatchRaysIndex().z;      // index of the plane this probe is part of
    int probesPerPlane = DDGIGetProbesPerPlane(volume.probeCounts);

    int probeIndex = (planeIndex * probesPerPlane) + probePlaneIndex;

    // Get the probe's grid coordinates
    int3 probeCoords = DDGIGetProbeCoords(probeIndex, volume);

    // Get the probe's world position
    float3 probeWorldPosition = DDGIGetProbeWorldPosition(probeCoords, volume, ProbeData);

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
        float3 skyRadiance = pushConstants.skyRadiance.xyz * pushConstants.skyRadiance.a;
        RayData[outputCoords] = float4(skyRadiance, 1e27f);
        return;
    }

    // Probe inside a wall
    if (payl.hitKind == HIT_KIND_TRIANGLE_BACK_FACE)
    {
        RayData[outputCoords] = float4(0.0f, 0.0f, 0.0f, -payl.hitT * 0.2f);
        return;
    }

    if ((volume.probeRelocationEnabled == 1) && rayIndex < DDGI_NUM_FIXED_RAYS)
    {
        RayData[outputCoords] = float4(0.0f, 0.0f, 0.0f, payl.hitT);
    }

    float3 diffuse = DirectDiffuseLighting(payl, pushConstants.rayNormalBias, pushConstants.rayViewBias, SceneTLAS, pushConstants.dirLight, pushConstants.pointLights, pushConstants.pointLightsCount);

    float3 irradiance = 0.f;
    float3 surfaceBias = DDGIGetSurfaceBias(payl.normal, ray.Direction, volume);

    float volumeBlendWeight = DDGIGetVolumeBlendWeight(payl.worldPosition, volume);

    Texture2DArray<float4> Irradiance = Irradiances[pushConstants.currentVolumeIndex];
    Texture2DArray<float2> Distance = Distances[pushConstants.currentVolumeIndex];
    if (volumeBlendWeight > 0.f)
    {
        irradiance = 
            DDGIGetVolumeIrradiance(payl.worldPosition, surfaceBias, payl.normal, volume, 
                BillinearWrapSampler, Irradiance, Distance, ProbeData);
        irradiance *= volumeBlendWeight;
    }

    float maxAlbedo = 0.9f;
    float3 radiance = payl.emissive + diffuse + ((min(payl.albedo, float3(maxAlbedo, maxAlbedo, maxAlbedo)) / float ::getPi()) * irradiance);
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
    payload.worldPosition = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();

    float3 e1 = vertices[1].pos - vertices[0].pos;
    float3 e2 = vertices[2].pos - vertices[0].pos;
    float3 geomNormal = normalize(cross(e1, e2));
    payload.normal = normalize(mul(ObjectToWorld3x4(), float4(geomNormal, 0.f)).xyz);

    // Handle backface case — flip if pointing away from ray
    if (dot(payload.normal, WorldRayDirection()) > 0.0f)
        payload.normal = -payload.normal;
    payload.shadingNormal = payload.normal;

    // Load the surface material
    payload.albedo = material.colorFactors.xyz;
    payload.opacity = material.colorFactors.w;

    
    // Albedo and Opacity
    {

        float4 bco = textures[NonUniformResourceIndex(instance.colorTextureID)]
            .SampleLevel( samplers[NonUniformResourceIndex(instance.colorSamplerID)], float2(v.u, v.v), 0);
        payload.albedo *= bco.rgb;
        payload.opacity *= bco.a;
    }

    // Shading normal
    {

        float3 tangent = normalize(mul(ObjectToWorld3x4(), float4(v.tangent.xyz, 0.f)).xyz);
        float3 bitangent = cross(payload.normal, tangent) * v.tangent.w;
        float3x3 TBN = { tangent, bitangent, payload.normal };
        float2 normalRG = textures[NonUniformResourceIndex(instance.normalTextureID)]
                              .SampleLevel(samplers[NonUniformResourceIndex(instance.normalSamplerID)], float2(v.u, v.v), 0) .xy;
        float2 normalXY = normalRG * 2.0f - 1.0f;
        float normalZ = sqrt(max(0.0f, 1.0f - dot(normalXY, normalXY)));
        payload.shadingNormal = mul(normalize(float3(normalXY, normalZ)), TBN);     // Transform tangent-space normal to world-space
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
            .SampleLevel(samplers[NonUniformResourceIndex(instance.emissiveSamplerID)], float2(v.u, v.v), 0).rgb;

         payload.emissive = emissiveSample * material.emissiveFactors.xyz;
    }

}


[shader("miss")]
void MissMain(inout Payload packedPayload)
{
    packedPayload.hitT = -1.f;
}
