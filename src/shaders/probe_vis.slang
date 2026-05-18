import types;
import ddgi_common;

[[vk::binding(0, 0)]]
Texture2DArray<float4> IrradianceDatas[];

[[vk::binding(1, 0)]]
Texture2DArray<float2> DistanceDatas[];

[[vk::binding(2, 0)]]
Texture2DArray<float4> ProbeDatas[];

[[vk::binding(3, 0)]]
SamplerState BilinearSampler;

struct PushConstants {
    DDGIVolume*     volumes;
    VertexPosition* sphereVertices;
    SceneData*      sceneData;
    uint            volumeIndex;
    float           probeRadius;
    uint            visMode;
    uint            _pad;
};

[[vk::push_constant]]
PushConstants pushConstants;

struct VSOutput {
    float4               position    : SV_Position;
    float3               color;
    float3               sphereNormal;
    nointerpolation uint probeIndex;
    nointerpolation float probeState;
}

[shader("vertex")]
VSOutput VSMain(uint vertexId : SV_VertexID, uint instanceId : SV_InstanceID)
{
    DDGIVolume volume = pushConstants.volumes[pushConstants.volumeIndex];

    Texture2DArray<float4> ProbeData = ProbeDatas[pushConstants.volumeIndex];

    int3 coords = DDGIGetProbeCoords(int(instanceId), volume);

    float probeState = DDGILoadProbeState(int(instanceId), ProbeData, volume);
  
    float3 probeWorldPos = DDGIGetProbeWorldPosition(coords, volume, ProbeData);

    float3 sphereVtx = pushConstants.sphereVertices[vertexId].pos;
    float3 worldPos = probeWorldPos + sphereVtx * pushConstants.probeRadius;

    float3 maxCoords = float3(max(volume.probeCounts - int3(1), int3(1)));

    VSOutput output;
    output.position    = mul(pushConstants.sceneData.projView, float4(worldPos, 1.0f));
    output.color       = float3(coords) / maxCoords;
    output.sphereNormal = sphereVtx;
    output.probeIndex = instanceId;
    output.probeState = probeState;
    return output;
}

[shader("fragment")]
float4 FSMain(VSOutput input) : SV_Target
{
    if (input.probeState == DDGI_PROBE_STATE_INACTIVE)
        discard;
    if (pushConstants.visMode == 0)
        return float4(input.color, 1.0f);

    DDGIVolume volume = pushConstants.volumes[pushConstants.volumeIndex];

    float2 octUV = DDGIGetOctahedralCoordinates(input.sphereNormal);

    if (pushConstants.visMode == 1) // Irradiance
    {
        float3 probeUV = DDGIGetProbeUV(int(input.probeIndex), octUV,
                                        MAX_IRRADIANCE_TEXELS - 2, volume);
        float3 enc = IrradianceDatas[pushConstants.volumeIndex]
                         .SampleLevel(BilinearSampler, probeUV, 0).rgb;
        float3 irr = pow(enc, volume.probeIrradianceEncodingGamma * 0.5f);
        irr *= irr;
        return float4(irr, 1.0f);
    }
    else // Distance
    {
        float3 probeUV = DDGIGetProbeUV(int(input.probeIndex), octUV,
                                        MAX_DISTANCE_TEXELS - 2, volume);
        float meanDist = 2.0f * DistanceDatas[pushConstants.volumeIndex]
                                    .SampleLevel(BilinearSampler, probeUV, 0).r;
        float maxDist = length(volume.probeSpacing) * 1.5f;
        float t = saturate(meanDist / maxDist);
        return float4(t, t, t, 1.0f);
    }
}
