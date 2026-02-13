import utils;

struct PushConstants {
    Vertex *vertices;
    Instance *instances;
    float4x4 lightVP;
};

[[vk_push_constant]]
PushConstants pushConstants;

struct VSOutput {
    float4 pos : SV_Position;
}

[shader("vertex")]
VSOutput vertexMain(uint vertexID: SV_VulkanVertexID, uint instanceID: SV_VulkanInstanceID) : SV_Position {
    Vertex vertex = pushConstants.vertices[vertexID];
    Instance instance = pushConstants.instances[instanceID];
    float4x4 MVP = mul(pushConstants.lightVP, instance.worldTransform);
    VSOutput output;
    output.pos = mul(MVP, float4(vertex.pos, 1.0f));
    return output;
}

[shader("pixel")]
void pixelMain(VSOutput input) {
}