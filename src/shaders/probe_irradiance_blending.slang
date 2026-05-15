import types;
import utils;
import ddgi_common;

[[vk::binding(0, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2DArray<float4> RayDatas[];

[[vk::binding(0, 1)]]
[[vk::image_format("rgba16f")]]
RWTexture2DArray<float4> IrradianceDatas[];

struct PushConstants {
    DDGIVolume *volumes;
    uint currentVolumeIndex;
};

static const int MAX_IRRADIANCE_TEXELS = 8;
static const int BLEND_RAYS_PER_PROBE = 256;

groupshared float3 RayRadiance[BLEND_RAYS_PER_PROBE];
groupshared float RayDistance[BLEND_RAYS_PER_PROBE];
groupshared float3 RayDirection[BLEND_RAYS_PER_PROBE];

[[vk::push_constant]]
PushConstants pushConstants;

// Cooperatively load the ray radiance and hit distance values into shared memory
// Cooperatively compute probe ray directions
void LoadSharedMemory(int probeIndex, uint GroupIndex, RWTexture2DArray<float4> RayData, DDGIVolume volume)
{
    int totalIterations = int(ceil(float(BLEND_RAYS_PER_PROBE) / float(MAX_IRRADIANCE_TEXELS * MAX_IRRADIANCE_TEXELS)));
    for (int iteration = 0; iteration < totalIterations; iteration++)
    {
        int rayIndex = (GroupIndex * totalIterations) + iteration;
        if (rayIndex >= BLEND_RAYS_PER_PROBE) break;

        // Get the coordinates for the probe ray in the RayData texture array
        uint3 rayDataTexCoords = DDGIGetRayDataTexelCoords(rayIndex, probeIndex, volume);

        // Load the ray radiance and store it in shared memory
        RayRadiance[rayIndex] = RayData[rayDataTexCoords].rgb;
        // Load the ray hit distance and store it in shared memory
        RayDistance[rayIndex] = RayData[rayDataTexCoords].a;

        // Get a random normalized probe ray direction and store it in shared memory
        RayDirection[rayIndex] = DDGIGetProbeRayDirection(rayIndex, volume);
    }

    // Wait for all threads in the group to finish their shared memory operations
    GroupMemoryBarrierWithGroupSync();
}

// When the thread maps to a border texel, update it with the latest blended information for later use in bilinear filtering
void UpdateBorderTexel(uint3 DispatchThreadID, uint3 GroupThreadID, uint3 GroupID, uint probeNumTexels, RWTexture2DArray<float4> Output, DDGIVolume volume)
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
[numthreads(MAX_IRRADIANCE_TEXELS, MAX_IRRADIANCE_TEXELS, 1)]
void DDGIProbeBlendingCS(
    uint3 DispatchThreadID: SV_DispatchThreadID,
    uint3 GroupThreadID: SV_GroupThreadID,
                         uint3 GroupID: SV_GroupID,
    uint GroupIndex: SV_GroupIndex)
{
    DDGIVolume volume = pushConstants.volumes[pushConstants.currentVolumeIndex];
    const int probeIrradianceTexels = volume.probeIrradianceTexels;
    const int probeIrradianceInteriorTexels = probeIrradianceTexels - 2;

    if (GroupThreadID.x >= probeIrradianceTexels || GroupThreadID.y >= probeIrradianceTexels) return;
    // Determine if this thread maps to a probe border texel
    bool bIsBorderTexel = (GroupThreadID.x == 0 || GroupThreadID.x == (probeIrradianceTexels - 1)); // Border Columns
    bIsBorderTexel |= (GroupThreadID.y == 0 || GroupThreadID.y == (probeIrradianceTexels - 1)); // Border Rows

    RWTexture2DArray<float4> RayData = RayDatas[pushConstants.currentVolumeIndex];
    RWTexture2DArray<float4> Output = IrradianceDatas[pushConstants.currentVolumeIndex];

    int probeIndex = DDGIGetProbeIndex(DispatchThreadID, probeIrradianceTexels, volume);

    uint numProbes = (volume.probeCounts.x * volume.probeCounts.y * volume.probeCounts.z);

    if (probeIndex >= numProbes || probeIndex < 0) return;

    LoadSharedMemory(probeIndex, GroupIndex, RayData, volume);

    if (!bIsBorderTexel)
    {
        // Remap thread coordinates to not include the border texels
        int3 threadCoords = 
            int3(GroupID.x * probeIrradianceInteriorTexels, GroupID.y * probeIrradianceInteriorTexels, DispatchThreadID.z) 
            + GroupThreadID - int3(1, 1, 0);

         // Get the probe ray direction associated with this thread
        float2 probeOctantUV = DDGIGetNormalizedOctahedralCoordinates(int2(threadCoords.xy), probeIrradianceInteriorTexels);
        float3 probeRayDirection = DDGIGetOctahedralDirection(probeOctantUV);

        int rayIndex = 0;

        // Backface hits are ignored when blending radiance
        // If more than the backface threshold of the rays hit backfaces, the probe is probably inside geometry
        // In this case, don't blend anything into the probe
        uint backfaces = 0;

        uint maxBackfaces = uint((volume.probeNumRays - rayIndex) * volume.probeRandomRayBackfaceThreshold);

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

            // Load the ray traced radiance and hit distance
            float3 probeRayRadiance = 0.f;
            float probeRayDistance = 0.f;

            probeRayRadiance = RayRadiance[rayIndex];
            probeRayDistance = RayDistance[rayIndex];

            // Backface hit, don't blend this sample
            if (probeRayDistance < 0.f)
            {
                backfaces++;

                // Early out: only blend ray radiance into the probe if the backface threshold hasn't been exceeded
                if (backfaces >= maxBackfaces) return;

                continue;
            }

            // Blend the ray's radiance
            result += float4(probeRayRadiance * weight, weight);
        }

        float epsilon = float(volume.probeNumRays);
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
        float3 probeIrradianceMean = Output[DispatchThreadID].rgb;

        // Get the history weight (hysteresis) to use for the probe texel's previous value
        // If the probe was previously cleared to completely black, set the hysteresis to zero
        float hysteresis = volume.probeHysteresis;
        if (dot(probeIrradianceMean, probeIrradianceMean) == 0) hysteresis = 0.f;
        // Tone-mapping gamma adjustment
         result.rgb = pow(result.rgb, (1.f / volume.probeIrradianceEncodingGamma));

        // Get the difference between the current irradiance and the irradiance mean stored in the probe
        float3 delta = (result.rgb - probeIrradianceMean.rgb);

        // Store the current irradiance (before interpolation) for use in probe variability
        float3 irradianceSample = result.rgb;

        if (MaxComponent(probeIrradianceMean.rgb - result.rgb) > volume.probeIrradianceThreshold)
        {
            // Lower the hysteresis when a large lighting change is detected
            hysteresis = max(0.f, hysteresis - 0.75f);
        }

        if (LinearRGBToLuminance(delta) > volume.probeBrightnessThreshold)
        {
            // Clamp the maximum per-update change in irradiance when a large brightness change is detected
            delta *= 0.25f;
        }

        // Interpolate the new blended irradiance with the existing irradiance in the probe.
        // A high hysteresis value emphasizes the existing probe irradiance.
        //
        // When using lower bit depth formats for irradiance, the difference between lerped values
        // may be smaller than what the texture format can represent. This can stop progress towards
        // the target value when going from high to low values. When darkening, step at least the minimum
        // value the texture format can represent to ensure the target value is reached. The threshold value
        // for 10-bit/channel formats is always used (even for 32-bit/channel formats) to speed up light to
        // dark convergence.
        static const float c_threshold = 1.f / 1024.f;
        float3 lerpDelta = (1.f - hysteresis) * delta;
        if (MaxComponent(result.rgb) < MaxComponent(probeIrradianceMean.rgb))
        {
            lerpDelta = min(max(c_threshold, abs(lerpDelta)), abs(delta)) * sign(lerpDelta);
        }
        result = float4(probeIrradianceMean.rgb + lerpDelta, 1.f);

        Output[DispatchThreadID] = result;
        return;
    }

    AllMemoryBarrierWithGroupSync();

    UpdateBorderTexel(DispatchThreadID, GroupThreadID, GroupID, probeIrradianceTexels, Output, volume);
}
