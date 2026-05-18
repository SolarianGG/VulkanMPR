import types;
import utils;
import ddgi_common;

[[vk::binding(0, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2DArray<float4> ProbeDatas[];

struct PushConstants {
    DDGIVolume *volumes;
    uint currentVolumeIndex;
}

[[vk::push_constant]]
PushConstants pushConstants;

[shader("compute")]
[numthreads(32, 1, 1)]
void CSMain(uint3 threadID: SV_DispatchThreadID)
{
    uint volumeIndex = pushConstants.currentVolumeIndex;
    DDGIVolume volume = pushConstants.volumes[volumeIndex];

    if (threadID.x >= (volume.probeSpacing.x * volume.probeSpacing.y * volume.probeSpacing.z)) return;

    RWTexture2DArray<float4> ProbeData = ProbeDatas[volumeIndex];

    // Get the probe's texel coordinates in the Probe Data texture
    uint3 outputCoords = DDGIGetProbeTexelCoords(threadID.x, volume);

    // Write the probe offset
    ProbeData[outputCoords].xyz = float3(0.f, 0.f, 0.f);
}
