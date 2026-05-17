import types;
import utils;
import ddgi_common;


[[vk::binding(0, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2DArray<float4> RayDatas[];

[[vk::binding(0, 1)]]
[[vk::image_format("rgba16f")]]
RWTexture2DArray<float4> ProbeDatas[];

struct PushConstants {
    DDGIVolume *volumes;
    uint currentVolumeIndex;
};

[[vk::push_constant]]
PushConstants pushConstants;

[shader("compute")]
[numthreads(32, 1, 1)]
void CSMain(uint3 threadID : SV_DispatchThreadID)
{
    DDGIVolume volume = pushConstants.volumes[pushConstants.currentVolumeIndex];
    uint probeIndex = threadID.x;

    // Early out: if this thread maps past the number of probes in the volume
    int numProbes = (volume.probeCounts.x * volume.probeCounts.y * volume.probeCounts.z);
    if (probeIndex >= numProbes) return;

    RWTexture2DArray<float4> RayData = RayDatas[pushConstants.currentVolumeIndex];
    RWTexture2DArray<float4> ProbeData = ProbeDatas[pushConstants.currentVolumeIndex];

    // Get the probe's texel coordinates in the Probe Data texture array
    uint3 outputCoords = DDGIGetProbeTexelCoords(probeIndex, volume);

    // Read the current world position offset
    float3 offset = DDGILoadProbeDataOffset(ProbeData, outputCoords, volume);

    // Initialize variables
    int closestBackfaceIndex = -1;
    int closestFrontfaceIndex = -1;
    int farthestFrontfaceIndex = -1;
    float closestBackfaceDistance = 1e27f;
    float closestFrontfaceDistance = 1e27f;
    float farthestFrontfaceDistance = 0.f;
    float backfaceCount = 0.f;

    // Get the number of rays to inspect
    int numRays = min(volume.probeNumRays, DDGI_NUM_FIXED_RAYS);

    // Iterate over the rays cast for this probe to find the number of backfaces and closest/farthest distances to the probe
    for (int rayIndex = 0; rayIndex < numRays; rayIndex++)
    {
        // Get the coordinates for the probe ray in the RayData texture array
        int3 rayDataTexCoords = DDGIGetRayDataTexelCoords(rayIndex, probeIndex, volume);

        // Load the hit distance for the ray
        float hitDistance = RayData[rayDataTexCoords].a;

        if (hitDistance < 0.f)
        {
            // Found a backface
            backfaceCount++;

            // Negate the hit distance on a backface hit and scale back to the full distance
            hitDistance = hitDistance * -5.f;
            if (hitDistance < closestBackfaceDistance)
            {
                // Store the closest backface distance and ray index
                closestBackfaceDistance = hitDistance;
                closestBackfaceIndex = rayIndex;
            }
        }
        else
        {
            // Found a frontface
            if (hitDistance < closestFrontfaceDistance)
            {
                // Store the closest frontface distance and ray index
                closestFrontfaceDistance = hitDistance;
                closestFrontfaceIndex = rayIndex;
            }
            else if (hitDistance > farthestFrontfaceDistance)
            {
                // Store the farthest frontface distance and ray index
                farthestFrontfaceDistance = hitDistance;
                farthestFrontfaceIndex = rayIndex;
            }
        }
    }

    float3 fullOffset = float3(1e27f, 1e27f, 1e27f);

    if (closestBackfaceIndex != -1 && ((float)backfaceCount / numRays) > volume.probeFixedRayBackfaceThreshold)
    {
        // If at least one backface triangle is hit AND backfaces are hit by enough probe rays,
        // assume the probe is inside geometry and move it outside of the geometry.
        float3 closestBackfaceDirection = DDGIGetProbeRayDirection(closestBackfaceIndex, volume);
        fullOffset = offset + (closestBackfaceDirection * (closestBackfaceDistance + volume.probeMinFrontfaceDistance * 0.5f));
    }
    else if (closestFrontfaceDistance < volume.probeMinFrontfaceDistance)
    {
        // Don't move the probe if moving towards the farthest frontface will also bring us closer to the nearest frontface
        float3 closestFrontfaceDirection = DDGIGetProbeRayDirection(closestFrontfaceIndex, volume);
        float3 farthestFrontfaceDirection = DDGIGetProbeRayDirection(farthestFrontfaceIndex, volume);

        if (dot(closestFrontfaceDirection, farthestFrontfaceDirection) <= 0.f)
        {
            // Ensures the probe never moves through the farthest frontface
            farthestFrontfaceDirection *= min(farthestFrontfaceDistance, 1.f);
            fullOffset = offset + farthestFrontfaceDirection;
        }
    }
    else if (closestFrontfaceDistance > volume.probeMinFrontfaceDistance)
    {
        // Probe isn't near anything, try to move it back towards zero offset
        float moveBackMargin = min(closestFrontfaceDistance - volume.probeMinFrontfaceDistance, length(offset));
        float3 moveBackDirection = normalize(-offset);
        fullOffset = offset + (moveBackMargin * moveBackDirection);
    }

    // Absolute maximum distance that probe could be moved should satisfy ellipsoid equation:
    // x^2 / probeGridSpacing.x^2 + y^2 / probeGridSpacing.y^2 + z^2 / probeGridSpacing.y^2 < (0.5)^2
    // Clamp to less than maximum distance to avoid degenerate cases
    float3 normalizedOffset = fullOffset / volume.probeSpacing;
    if (dot(normalizedOffset, normalizedOffset) < 0.2025f) // 0.45 * 0.45 == 0.2025
    {
        offset = fullOffset;
    }

    // Write the probe offsets
    DDGIStoreProbeDataOffset(ProbeData, outputCoords, offset, volume);
}