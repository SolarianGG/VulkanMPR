import types;
import utils;
import ddgi_common;

[[vk::binding(0, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2DArray<float4> RayDatas[];

[[vk::binding(0, 1)]]
[[vk::image_format("rg32f")]]
RWTexture2DArray<float2> DistanceDatas[];

[[vk::binding(0, 2)]]
[[vk::image_format("rgba16f")]]
RWTexture2DArray<float4> ProbeDatas[];

struct PushConstants {
    DDGIVolume *volumes;
    uint currentVolumeIndex;
};

groupshared float RayDistance[BLEND_RAYS_PER_PROBE];
groupshared float3 RayDirection[BLEND_RAYS_PER_PROBE];

[[vk::push_constant]]
PushConstants pushConstants;

// Cooperatively load the ray radiance and hit distance values into shared memory
// Cooperatively compute probe ray directions
void LoadSharedMemory(int probeIndex, uint GroupIndex, RWTexture2DArray<float4> RayData, DDGIVolume volume)
{
    int totalIterations = int(ceil(float(BLEND_RAYS_PER_PROBE) / float(MAX_DISTANCE_TEXELS * MAX_DISTANCE_TEXELS)));
    for (int iteration = 0; iteration < totalIterations; iteration++)
    {
        int rayIndex = (GroupIndex * totalIterations) + iteration;
        if (rayIndex >= BLEND_RAYS_PER_PROBE) break;

        // Get the coordinates for the probe ray in the RayData texture array
        uint3 rayDataTexCoords = DDGIGetRayDataTexelCoords(rayIndex, probeIndex, volume);

        // Load the ray hit distance and store it in shared memory
        RayDistance[rayIndex] = RayData[rayDataTexCoords].a;

        // Get a random normalized probe ray direction and store it in shared memory
        RayDirection[rayIndex] = DDGIGetProbeRayDirection(rayIndex, volume);
    }

    // Wait for all threads in the group to finish their shared memory operations
    GroupMemoryBarrierWithGroupSync();
}

// When the thread maps to a border texel, update it with the latest blended information for later use in bilinear filtering
void UpdateBorderTexel(uint3 DispatchThreadID, uint3 GroupThreadID, uint3 GroupID, uint probeNumTexels, RWTexture2DArray<float2> Output, DDGIVolume volume)
{
    uint probeNumInteriorTexels = probeNumTexels - 2;
    bool isCornerTexel = (GroupThreadID.x == 0 || GroupThreadID.x == (probeNumTexels - 1)) && (GroupThreadID.y == 0 || GroupThreadID.y == (probeNumTexels - 1));
    bool isRowTexel = (GroupThreadID.x > 0 && GroupThreadID.x < (probeNumTexels - 1));

    uint3 copyCoordinates = uint3(GroupID.x * probeNumTexels, GroupID.y * probeNumTexels, DispatchThreadID.z);

    if (isCornerTexel)
    {
        copyCoordinates.x += GroupThreadID.x > 0 ? 1 : probeNumInteriorTexels;
        copyCoordinates.y += GroupThreadID.y > 0 ? 1 : probeNumInteriorTexels;
    }
    else if (isRowTexel)
    {
        copyCoordinates.x += (probeNumTexels - 1) - GroupThreadID.x;
        copyCoordinates.y += GroupThreadID.y + ((GroupThreadID.y > 0) ? -1 : 1);
    }
    else // Column Texel
    {
        copyCoordinates.x += GroupThreadID.x + ((GroupThreadID.x > 0) ? -1 : 1);
        copyCoordinates.y += (probeNumTexels - 1) - GroupThreadID.y;
    }

    Output[DispatchThreadID] = Output[copyCoordinates];
}

[shader("compute")]
[numthreads(MAX_DISTANCE_TEXELS, MAX_DISTANCE_TEXELS, 1)]
void DDGIProbeBlendingCS(
    uint3 DispatchThreadID: SV_DispatchThreadID,
    uint3 GroupThreadID: SV_GroupThreadID,
                         uint3 GroupID: SV_GroupID,
    uint GroupIndex: SV_GroupIndex)
{
    DDGIVolume volume = pushConstants.volumes[pushConstants.currentVolumeIndex];
    const int probeDistnaceTexels = MAX_DISTANCE_TEXELS; 
    const int probeDistnaceInteriorTexels = probeDistnaceTexels - 2;

    // Determine if this thread maps to a probe border texel
    bool bIsBorderTexel = (GroupThreadID.x == 0 || GroupThreadID.x == (probeDistnaceTexels - 1)); // Border Columns
    bIsBorderTexel |= (GroupThreadID.y == 0 || GroupThreadID.y == (probeDistnaceTexels - 1)); // Border Rows

    RWTexture2DArray<float4> RayData = RayDatas[pushConstants.currentVolumeIndex];
    RWTexture2DArray<float4> ProbeData = ProbeDatas[pushConstants.currentVolumeIndex];
    RWTexture2DArray<float2> Output = DistanceDatas[pushConstants.currentVolumeIndex];

    int probeIndex = DDGIGetProbeIndex(DispatchThreadID, probeDistnaceTexels, volume);

    uint numProbes = (volume.probeCounts.x * volume.probeCounts.y * volume.probeCounts.z);

    if (probeIndex >= numProbes || probeIndex < 0) return;

    LoadSharedMemory(probeIndex, GroupIndex, RayData, volume);

    if (!bIsBorderTexel)
    {
        // Remap thread coordinates to not include the border texels
        int3 threadCoords = 
            int3(GroupID.x * probeDistnaceInteriorTexels, GroupID.y * probeDistnaceInteriorTexels, DispatchThreadID.z) 
            + GroupThreadID - int3(1, 1, 0);

        // Early out: don't blend rays for probes that are inactive
        float probeState = DDGILoadProbeState(probeIndex, ProbeData, volume);
        if (probeState == DDGI_PROBE_STATE_INACTIVE)
        {
            return;
        }

        // Get the probe ray direction associated with this thread
        float2 probeOctantUV = DDGIGetNormalizedOctahedralCoordinates(int2(threadCoords.xy), probeDistnaceInteriorTexels);
        float3 probeRayDirection = DDGIGetOctahedralDirection(probeOctantUV);

        int rayIndex = 0;
        if (volume.probeRelocationEnabled == 1 || volume.probeClassificationEnabled == 1)
        {
            rayIndex = DDGI_NUM_FIXED_RAYS;
        }

        // Blend each ray's radiance or distance values to compute irradiance or fitered distance
        float4 result = float4(0.f, 0.f, 0.f, 0.f);
        for (; rayIndex < volume.probeNumRays; rayIndex++)
        {
            // Get the direction for this probe ray
            float3 rayDirection = RayDirection[rayIndex];

            // Find the weight of the contribution for this ray
            // Weight is based on the cosine of the angle between the ray direction and the direction of the probe octant's texel
            float weight = max(0.f, dot(probeRayDirection, rayDirection));

            // Get the coordinates for the probe ray in the RayData texture array
            uint3 rayDataTexCoords = DDGIGetRayDataTexelCoords(rayIndex, probeIndex, volume);

            // Initialize the max probe hit distance to 50% larger the maximum distance between probe grid cells
            float probeMaxRayDistance = length(volume.probeSpacing) * 1.5f;

            // Increase or decrease the filtered distance value's "sharpness"
            weight = pow(weight, volume.probeDistanceExponent);

            // Load the ray traced distance
            // Hit distance is negative on backface hits (for probe relocation), so take the absolute value of the loaded data
            float probeRayDistance = 0.f;
            probeRayDistance = min(abs(RayDistance[rayIndex]), probeMaxRayDistance);

            // Filter the ray hit distance
            result += float4(probeRayDistance * weight, (probeRayDistance * probeRayDistance) * weight, 0.f, weight);
        }

        float epsilon = float(volume.probeNumRays);
        if (volume.probeRelocationEnabled == 1 || volume.probeClassificationEnabled == 1)
        {
            // If relocation or classification are enabled, fixed rays aren't blended since they will bias the result
            epsilon -= DDGI_NUM_FIXED_RAYS;
        }
        epsilon *= 1e-9f;

        // Normalize the blended irradiance (or filtered distance), if the combined weight is not close to zero.
        // To match the Monte Carlo Estimator of Irradiance, we should divide by N (the number of radiance samples).
        // Instead, we are dividing by sum(cos(theta)) (i.e. the sum of cosine weights) to reduce variance. To account
        // for this, we must multiply in a factor of 1/2. See the Math Guide in the documentation for more information.
        // For distance, note that we are *not* dividing by the sum of the cosine weights, but to avoid branching here
        // we are still dividing by 2. This means distance values sampled from texture need to be multiplied by 2
        result.rgb *= 1.f / (2.f * max(result.a, epsilon));
        result.a = 1.f;

        // Get the irradiance mean stored in the probe
        float2 probeDistanceMean = Output[DispatchThreadID].rg;

        // Get the history weight (hysteresis) to use for the probe texel's previous value
        // If the probe was previously cleared to completely black, set the hysteresis to zero
        float hysteresis = volume.probeHysteresis;
        if (dot(probeDistanceMean, probeDistanceMean) == 0) hysteresis = 0.f;

        // Interpolate the new filtered distance with the existing filtered distance in the probe.
        // A high hysteresis value emphasizes the existing probe filtered distance.
        result.xy = float2(lerp(result.rg, probeDistanceMean, hysteresis));

        Output[DispatchThreadID] = result.xy;
        return;
    }

    AllMemoryBarrierWithGroupSync();

    UpdateBorderTexel(DispatchThreadID, GroupThreadID, GroupID, probeDistnaceTexels, Output, volume);
}
