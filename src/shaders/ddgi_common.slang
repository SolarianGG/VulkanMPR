module ddgi_common;

import types;
import utils;

/**
 * Computes the surfaceBias parameter used by DDGIGetVolumeIrradiance().
 * The surfaceNormal and cameraDirection arguments are expected to be normalized.
 */
public float3 DDGIGetSurfaceBias(float3 surfaceNormal, float3 cameraDirection, DDGIVolume volume)
{
    return (surfaceNormal * volume.probeNormalBias) + (-cameraDirection * volume.probeViewBias);
}

/**
 * Computes a weight value in the range [0, 1] for a world position and volume pair.
 * All positions inside the given volume recieve a weight of 1.
 * Positions outside the volume receive a weight in [0, 1] that
 * decreases as the position moves away from the volume.
 */
public float DDGIGetVolumeBlendWeight(float3 worldPosition, DDGIVolume volume)
{
    // Get the volume's origin and extent
    float3 origin = volume.origin;
    float3 extent = (volume.probeSpacing * (volume.probeCounts - 1)) * 0.5f;

    float3 position = (worldPosition - origin);
    position = abs(QuaternionRotate(position, QuaternionConjugate(volume.rotation)));

    float3 delta = position - extent;
    if (all(delta < 0)) return 1.f;

    // Adjust the blend weight for each axis
    float volumeBlendWeight = 1.f;
    volumeBlendWeight *= (1.f - saturate(delta.x / volume.probeSpacing.x));
    volumeBlendWeight *= (1.f - saturate(delta.y / volume.probeSpacing.y));
    volumeBlendWeight *= (1.f - saturate(delta.z / volume.probeSpacing.z));

    return volumeBlendWeight;
}



public int DDGIGetProbesPerPlane(int3 probeCounts)
{
    return (probeCounts.x * probeCounts.z);
}

public int3 DDGIGetProbeCoords(int probeIndex, DDGIVolume volume)
{
    int3 probeCoords;

    probeCoords.x = probeIndex % volume.probeCounts.x;
    probeCoords.y = probeIndex / (volume.probeCounts.x * volume.probeCounts.z);
    probeCoords.z = (probeIndex / volume.probeCounts.x) % volume.probeCounts.z;

    return probeCoords;
}

public float3 DDGIGetProbeWorldPosition(int3 probeCoords, DDGIVolume volume)
{
    // Multiply the grid coordinates by the probe spacing
    float3 probeGridWorldPosition = probeCoords * volume.probeSpacing;

    // Shift the grid of probes by half of each axis extent to center the volume about its origin
    float3 probeGridShift = (volume.probeSpacing * (volume.probeCounts - 1)) * 0.5f;

    // Center the probe grid about the origin
    float3 probeWorldPosition = (probeGridWorldPosition - probeGridShift);

    // Rotate the probe grid if infinite scrolling is not enabled
    probeWorldPosition = QuaternionRotate(probeWorldPosition, volume.rotation);

    // Translate the grid to the volume's center
    probeWorldPosition += volume.origin;

    return probeWorldPosition;
}

public float3 SphericalFibonacci(float sampleIndex, float numSamples)
{
    const float b = (sqrt(5.f) * 0.5f + 0.5f) - 1.f;
    float phi = float ::getPi() * 2.0f * frac(sampleIndex * b);
    float cosTheta = 1.f - (2.f * sampleIndex + 1.f) * (1.f / numSamples);
    float sinTheta = sqrt(saturate(1.f - (cosTheta * cosTheta)));

    return float3((cos(phi) * sinTheta), (sin(phi) * sinTheta), cosTheta);
}

public float3 DDGIGetProbeRayDirection(int rayIndex, DDGIVolume volume)
{
    int sampleIndex = rayIndex;
    int numRays = volume.probeNumRays;

    // Get a ray direction on the sphere
    float3 direction = SphericalFibonacci(sampleIndex, numRays);

    // Apply a random rotation and normalize the direction
    return normalize(QuaternionRotate(direction, QuaternionConjugate(volume.probeRayRotation)));
}

public uint3 DDGIGetRayDataTexelCoords(int rayIndex, int probeIndex, DDGIVolume volume)
{
    int probesPerPlane = DDGIGetProbesPerPlane(volume.probeCounts);

    uint3 coords;
    coords.x = rayIndex;
    coords.z = probeIndex / probesPerPlane;
    coords.y = probeIndex - (coords.z * probesPerPlane);

    return coords;
}

int3 DDGIGetBaseProbeGridCoords(float3 worldPosition, DDGIVolume volume)
{
    // Get the vector from the volume origin to the surface point
    float3 position = worldPosition - volume.origin;

    position = QuaternionRotate(position, QuaternionConjugate(volume.rotation));

    // Shift from [-n/2, n/2] to [0, n] (grid space)
    position += (volume.probeSpacing * (volume.probeCounts - 1)) * 0.5f;

    // Quantize the position to grid space
    int3 probeCoords = int3(position / volume.probeSpacing);

    // Clamp to [0, probeCounts - 1]
    // Snaps positions outside of grid to the grid edge
    probeCoords = clamp(probeCoords, int3(0, 0, 0), (volume.probeCounts - int3(1, 1, 1)));

    return probeCoords;
}

int DDGIGetProbeIndexInPlane(int3 probeCoords, int3 probeCounts)
{
    return probeCoords.x + (probeCounts.x * probeCoords.z);
}
/**
 * Get the index of a probe within a horizontal plane (i.e. Texture2DArray slice) that the
 * given texel coordinates map to, in the active coordinate system. Provided 2D texel coordinates
 * should *not* include the octahedral texture's 1-texel border.
 */
int DDGIGetProbeIndexInPlane(uint3 texCoords, int3 probeCounts, int probeNumTexels)
{
    return int(texCoords.x / probeNumTexels) + (probeCounts.x * int(texCoords.y / probeNumTexels));
}

int DDGIGetPlaneIndex(int3 probeCoords)
{
    return probeCoords.y;
}

/**
 * Computes the probe index from 3D grid coordinates.
 * The opposite of DDGIGetProbeCoords(probeIndex,...).
 */
public int DDGIGetProbeIndex(int3 probeCoords, DDGIVolume volume)
{
    int probesPerPlane = DDGIGetProbesPerPlane(volume.probeCounts);
    int planeIndex = DDGIGetPlaneIndex(probeCoords);
    int probeIndexInPlane = DDGIGetProbeIndexInPlane(probeCoords, volume.probeCounts);

    return (planeIndex * probesPerPlane) + probeIndexInPlane;
}
public int DDGIGetProbeIndex(uint3 texCoords, int probeNumTexels, DDGIVolume volume)
{
    int probesPerPlane = DDGIGetProbesPerPlane(volume.probeCounts);
    int probeIndexInPlane = DDGIGetProbeIndexInPlane(texCoords, volume.probeCounts, probeNumTexels);

    return (texCoords.z * probesPerPlane) + probeIndexInPlane;
}

/**
 * Computes the octant coordinates in the normalized [-1, 1] square, for the given a unit direction vector.
 * The opposite of DDGIGetOctahedralDirection().
 * Used by GetDDGIVolumeIrradiance() in Irradiance.hlsl.
 */
public float2 DDGIGetOctahedralCoordinates(float3 direction)
{
    float l1norm = abs(direction.x) + abs(direction.y) + abs(direction.z);
    float2 uv = direction.xy * (1.f / l1norm);
    if (direction.z < 0.f)
    {
        uv = (1.f - abs(uv.yx)) * float2(SignNotZero(uv.x), SignNotZero(uv.y));
    }
    return uv;
}

uint3 DDGIGetProbeTexelCoords(int probeIndex, DDGIVolume volume)
{
    // Find the probe's plane index
    int probesPerPlane = DDGIGetProbesPerPlane(volume.probeCounts);
    int planeIndex = int(probeIndex / probesPerPlane);

    int x = (probeIndex % volume.probeCounts.x);
    int y = (probeIndex / volume.probeCounts.x) % volume.probeCounts.z;

    return uint3(x, y, planeIndex);
}

public float3 DDGIGetProbeUV(int probeIndex, float2 octantCoordinates, int numProbeInteriorTexels, DDGIVolume volume)
{
    // Get the probe's texel coordinates, assuming one texel per probe
    uint3 coords = DDGIGetProbeTexelCoords(probeIndex, volume);

    // Add the border texels to get the total texels per probe
    float numProbeTexels = (numProbeInteriorTexels + 2.f);

    #if 0
    float textureWidth = numProbeTexels * volume.probeCounts.x;
    float textureHeight = numProbeTexels * volume.probeCounts.z;
    #endif

    float textureWidth = numProbeTexels * kMaxDDGIProbesX;
    float textureHeight = numProbeTexels * kMaxDDGIProbesZ;

    // Move to the center of the probe and move to the octant texel before normalizing
    float2 uv = float2(coords.x * numProbeTexels, coords.y * numProbeTexels) + (numProbeTexels * 0.5f);
    uv += octantCoordinates.xy * ((float)numProbeInteriorTexels * 0.5f);
    uv /= float2(textureWidth, textureHeight);
    return float3(uv, coords.z);
}

[ForceInline]
public float3 DDGIGetVolumeIrradiance(float3 worldPosition, float3 surfaceBias, float3 direction,
                                      DDGIVolume volume, SamplerState sampler, Texture2DArray<float4> Irradiance, Texture2DArray<float2> Distance)
{
    float3 irradiance = float3(0.0f, 0.0f, 0.0f);
    float accumulatedWeights = 0.0f;

    float3 biasedWorldPosition = worldPosition + surfaceBias;

    // Get the 3D grid coordinates of the probe nearest the biased world position (i.e. the "base" probe)
    int3 baseProbeCoords = DDGIGetBaseProbeGridCoords(biasedWorldPosition, volume);

    // Get the world-space position of the base probe
    float3 baseProbeWorldPosition = DDGIGetProbeWorldPosition(baseProbeCoords, volume);

    // Clamp the distance (in grid space) between the given point and the base probe's world position (on each axis) to [0, 1]
    float3 gridSpaceDistance = (biasedWorldPosition - baseProbeWorldPosition);
    gridSpaceDistance = QuaternionRotate(gridSpaceDistance, QuaternionConjugate(volume.rotation));
    float3 alpha = clamp((gridSpaceDistance / volume.probeSpacing), float3(0.0f,0.0f,0.0f), float3(1.f,1.f,1.f));

    // Iterate over the 8 closest probes and accumulate their contributions
    for (int probeIndex = 0; probeIndex < 8; ++probeIndex)
    {
        // Compute the offset to the adjacent probe in grid coordinates by
        // sourcing the offsets from the bits of the loop index: x = bit 0, y = bit 1, z = bit 2
        int3 adjacentProbeOffset = int3(probeIndex, probeIndex >> 1, probeIndex >> 2) & int3(1, 1, 1);

        // Get the 3D grid coordinates of the adjacent probe by adding the offset to
        // the base probe and clamping to the grid boundaries
        int3 adjacentProbeCoords = clamp(baseProbeCoords + adjacentProbeOffset, int3(0, 0, 0), volume.probeCounts - int3(1, 1, 1));

        // Get the adjacent probe's index, adjusting the adjacent probe index for scrolling offsets (if present)
        int adjacentProbeIndex = DDGIGetProbeIndex(((adjacentProbeCoords + volume.probeCounts) % volume.probeCounts), volume);;

        // Get the adjacent probe's world position
        float3 adjacentProbeWorldPosition = DDGIGetProbeWorldPosition(adjacentProbeCoords, volume);

        // Compute the distance and direction from the (biased and non-biased) shading point and the adjacent probe
        float3 worldPosToAdjProbe = normalize(adjacentProbeWorldPosition - worldPosition);
        float3 biasedPosToAdjProbe = normalize(adjacentProbeWorldPosition - biasedWorldPosition);
        float biasedPosToAdjProbeDist = length(adjacentProbeWorldPosition - biasedWorldPosition);

        // Compute trilinear weights based on the distance to each adjacent probe
        // to smoothly transition between probes. adjacentProbeOffset is binary, so we're
        // using a 1-alpha when adjacentProbeOffset = 0 and alpha when adjacentProbeOffset = 1.
        float3 trilinear = max(0.001f, lerp(1.f - alpha, alpha, adjacentProbeOffset));
        float trilinearWeight = (trilinear.x * trilinear.y * trilinear.z);
        float weight = 1.f;

        // A naive soft backface weight would ignore a probe when
        // it is behind the surface. That's good for walls, but for
        // small details inside of a room, the normals on the details
        // might rule out all of the probes that have mutual visibility
        // to the point. We instead use a "wrap shading" test. The small
        // offset at the end reduces the "going to zero" impact.
        float wrapShading = (dot(worldPosToAdjProbe, direction) + 1.f) * 0.5f;
        weight *= (wrapShading * wrapShading) + 0.2f;

        // Compute the octahedral coordinates of the adjacent probe
        float2 octantCoords = DDGIGetOctahedralCoordinates(-biasedPosToAdjProbe);

        // Get the texture array coordinates for the octant of the probe
        float3 probeTextureUV = DDGIGetProbeUV(adjacentProbeIndex, octantCoords, MAX_DISTANCE_TEXELS - 2, volume);

        // Sample the probe's distance texture to get the mean distance to nearby surfaces
        float2 filteredDistance = 2.f * Distance.SampleLevel(sampler, probeTextureUV, 0).rg;

        // Find the variance of the mean distance
        float variance = abs((filteredDistance.x * filteredDistance.x) - filteredDistance.y);

        // Occlusion test
        float chebyshevWeight = 1.f;
        if (biasedPosToAdjProbeDist > filteredDistance.x) // occluded
        {
            // v must be greater than 0, which is guaranteed by the if condition above.
            float v = biasedPosToAdjProbeDist - filteredDistance.x;
            chebyshevWeight = variance / (variance + (v * v));

            // Increase the contrast in the weight
            chebyshevWeight = max((chebyshevWeight * chebyshevWeight * chebyshevWeight), 0.f);
        }

        // Avoid visibility weights ever going all the way to zero because
        // when *no* probe has visibility we need a fallback value
        weight *= max(0.05f, chebyshevWeight);

        // Avoid a weight of zero
        weight = max(0.000001f, weight);

        // A small amount of light is visible due to logarithmic perception, so
        // crush tiny weights but keep the curve continuous
        const float crushThreshold = 0.2f;
        if (weight < crushThreshold)
        {
            weight *= (weight * weight) * (1.f / (crushThreshold * crushThreshold));
        }

        // Apply the trilinear weights
        weight *= trilinearWeight;

        // Get the octahedral coordinates for the sample direction
        octantCoords = DDGIGetOctahedralCoordinates(direction);

        // Get the probe's texture coordinates
        probeTextureUV = DDGIGetProbeUV(adjacentProbeIndex, octantCoords, MAX_IRRADIANCE_TEXELS - 2, volume);

        // Sample the probe's irradiance
        float3 probeIrradiance = Irradiance.SampleLevel(sampler, probeTextureUV, 0).rgb;

        // Decode the tone curve, but leave a gamma = 2 curve to approximate sRGB blending
        float3 exponent = volume.probeIrradianceEncodingGamma * 0.5f;
        probeIrradiance = pow(probeIrradiance, exponent);

        // Accumulate the weighted irradiance
        irradiance += (weight * probeIrradiance);
        accumulatedWeights += weight;
    }

    if (accumulatedWeights == 0.f) return float3(0.0f, 0.0f, 0.0f);

    irradiance *= (1.f / accumulatedWeights); // Normalize by the accumulated weights
    irradiance *= irradiance;                 // Go back to linear irradiance
    // Multiply by the area of the integration domain (hemisphere) to complete the Monte Carlo Estimator equation
    irradiance *= float::getPi() * 2.0f;

    return irradiance;
}

/**
 * Computes normalized octahedral coordinates for the given texel coordinates.
 * Maps the top left texel to (-1,-1).
 * Used by DDGIProbeBlendingCS() in ProbeBlending.hlsl.
 */
public float2 DDGIGetNormalizedOctahedralCoordinates(int2 texCoords, int numTexels)
{
    // Map 2D texture coordinates to a normalized octahedral space
    float2 octahedralTexelCoord = float2(texCoords.x % numTexels, texCoords.y % numTexels);

    // Move to the center of a texel
    octahedralTexelCoord.xy += 0.5f;

    // Normalize
    octahedralTexelCoord.xy /= float(numTexels);

    // Shift to [-1, 1);
    octahedralTexelCoord *= 2.f;
    octahedralTexelCoord -= float2(1.f, 1.f);

    return octahedralTexelCoord;
}

/**
 * Computes the normalized octahedral direction that corresponds to the
 * given normalized coordinates on the [-1, 1] square.
 * The opposite of DDGIGetOctahedralCoordinates().
 * Used by DDGIProbeBlendingCS() in ProbeBlending.hlsl.
 */
public float3 DDGIGetOctahedralDirection(float2 coords)
{
    float3 direction = float3(coords.x, coords.y, 1.f - abs(coords.x) - abs(coords.y));
    if (direction.z < 0.f)
    {
        direction.xy = (1.f - abs(direction.yx)) * float2(SignNotZero(direction.x), SignNotZero(direction.y));
    }
    return normalize(direction);
}

