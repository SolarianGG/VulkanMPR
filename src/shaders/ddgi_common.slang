module ddgi_common;

import types;
import utils;



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