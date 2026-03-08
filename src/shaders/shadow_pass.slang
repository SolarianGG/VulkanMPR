import utils;
import types;

struct PushConstants {
    Vertex *vertices;
    Instance *instances;
    DirectionalLight *dirLight;
    uint cascadeCount;
};

[[vk_push_constant]]
PushConstants pushConstants;

struct VSOutput {
    float4 worldPos;
}

struct GSOutput {
    float4 pos : SV_Position;
    uint layer : SV_RenderTargetArrayIndex;
}

[shader("vertex")]
VSOutput vertexMain(uint vertexID: SV_VulkanVertexID, uint instanceID: SV_VulkanInstanceID) {
    Vertex vertex = pushConstants.vertices[vertexID];
    Instance instance = pushConstants.instances[instanceID];
    VSOutput output;
    output.worldPos = mul(instance.worldTransform, float4(vertex.pos, 1.0f));
    return output;
}

[shader("geometry")]
[maxvertexcount(12)]
void geometryMain(triangle VSOutput input[3], inout TriangleStream<GSOutput> output) {
    for (uint c = 0; c < pushConstants.cascadeCount; ++c)
    {
        for (uint v = 0; v < 3; ++v)
        {
            GSOutput o;
            o.pos = mul(pushConstants.dirLight[0].cascadeVPs[c], input[v].worldPos);
            o.layer = c;
            output.Append(o);
        }
        output.RestartStrip();
    }
}

[shader("pixel")]
void pixelMain(GSOutput input) {
}
