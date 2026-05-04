import types;
import ddgi_common;

struct PushConstants {
    DDGIVolume*     volumes;
    VertexPosition* sphereVertices;
    SceneData*      sceneData;
    uint            volumeIndex;
    float           probeRadius;
}

[[vk::push_constant]]
PushConstants pushConstants;

struct VSOutput {
    float4 position : SV_Position;
    float3 color    : COLOR0;
}

[shader("vertex")]
VSOutput VSMain(uint vertexId : SV_VertexID, uint instanceId : SV_InstanceID)
{
    DDGIVolume volume = pushConstants.volumes[pushConstants.volumeIndex];

    int3 coords = DDGIGetProbeCoords(int(instanceId), volume);
    float3 probeWorldPos = DDGIGetProbeWorldPosition(coords, volume);

    float3 sphereVtx = pushConstants.sphereVertices[vertexId].pos;
    float3 worldPos = probeWorldPos + sphereVtx * pushConstants.probeRadius;

    float3 maxCoords = float3(max(volume.probeCounts - int3(1), int3(1)));

    VSOutput output;
    output.position = mul(pushConstants.sceneData.projView, float4(worldPos, 1.0f));
    output.color = float3(coords) / maxCoords;
    return output;
}

[shader("fragment")]
float4 FSMain(VSOutput input) : SV_Target
{
    return float4(input.color, 1.0f);
}
