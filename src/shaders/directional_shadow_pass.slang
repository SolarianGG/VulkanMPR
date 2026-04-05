import utils;
import types;

struct PushConstants {
    VertexPosition *positions;
    Instance *instances;
    DirectionalLight *dirLight;
    uint cascadeCount;
};

[[vk::push_constant]]
PushConstants pushConstants;

struct VSOutput {
    float4 pos : SV_Position;
    uint layer : SV_RenderTargetArrayIndex;
}

[shader("vertex")]
VSOutput vertexMain(uint vertexID: SV_VulkanVertexID, uint instanceID: SV_StartInstanceLocation, uint cascadeIndex: SV_InstanceID) {
    float3 pos = pushConstants.positions[vertexID].pos;
    Instance instance = pushConstants.instances[instanceID];
    VSOutput output;
    output.pos = mul(pushConstants.dirLight.cascadeVPs[cascadeIndex], mul(instance.worldTransform, float4(pos, 1.0f)));
    output.layer = cascadeIndex;
    return output;
}

[shader("pixel")]
void pixelMain(VSOutput input) {
}
