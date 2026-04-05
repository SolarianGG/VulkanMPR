import utils;
import types;

struct PushConstants {
    VertexPosition *positions;
    VertexAttributes *attributes;
    Instance *instances;
    SceneData *sceneData;
};

[[vk::binding(0, 0)]] ConstantBuffer<GLTFMaterialData> materials[];
[[vk::binding(1, 0)]] SamplerState samplers[];
[[vk::binding(2, 0)]] Texture2D textures[];

[[vk::push_constant]]
PushConstants pushConstants;

struct VSOutput {
    float4 pos : SV_Position;
    float2 uv;
    nointerpolation uint materialID;
    nointerpolation uint colorTextureID;
    nointerpolation uint colorSamplerID;
};

[shader("vertex")]
VSOutput vertexMain(uint vertexID : SV_VulkanVertexID, uint instanceID : SV_VulkanInstanceID) {
    float3 pos = pushConstants.positions[vertexID].pos;
    VertexAttributes attr = pushConstants.attributes[vertexID];
    Instance instance = pushConstants.instances[instanceID];
    SceneData scene = *pushConstants.sceneData;
    VSOutput output;
    output.pos = mul(scene.projView, mul(instance.worldTransform, float4(pos, 1.0)));
    output.uv = float2(attr.u, attr.v);
    output.materialID = instance.materialID;
    output.colorTextureID = instance.colorTextureID;
    output.colorSamplerID = instance.colorSamplerID;
    return output;
}

[shader("pixel")]
void fragmentMain(VSOutput input) {
    GLTFMaterialData materialData = materials[NonUniformResourceIndex(input.materialID)];
    float4 baseColor = textures[NonUniformResourceIndex(input.colorTextureID)]
        .Sample(samplers[NonUniformResourceIndex(input.colorSamplerID)], input.uv);
    if ((materialData.colorFactors * baseColor).a < materialData.alphaCutoff.r)
        discard;
}
