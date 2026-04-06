import utils;
import types;

[[vk::binding(0, 0)]] ConstantBuffer<GLTFMaterialData> materials[];
[[vk::binding(1, 0)]] SamplerState samplers[];
[[vk::binding(2, 0)]] Texture2D textures[];

struct PushConstants {
    VertexPosition *positions;
    VertexAttributes *attributes;
    Instance *instances;
    PointLight *visiblePointLights;
    uint32_t *pointLightIndices;
    uint32_t *pointLightOffsets;
    TetrahedronData *tetraData;
};

[[vk_push_constant]]
PushConstants pushConstants;

struct VSOut {
    float4 worldPos;
    uint lightIndex;
    float2 uv;
    uint materialID;
    uint colorTextureID;
    uint colorSamplerID;
};

struct GSOutput {
    float4 position : SV_Position;
    float clipDistance0 : SV_ClipDistance0;
    float clipDistance1 : SV_ClipDistance1;
    float clipDistance2 : SV_ClipDistance2;
    float2 uv : TEXCOORD0;
    nointerpolation uint materialID : BLENDINDICES0;
    nointerpolation uint colorTextureID : BLENDINDICES1;
    nointerpolation uint colorSamplerID : BLENDINDICES2;
};

[shader("vertex")]
VSOut vertexMain(uint vertexID: SV_VulkanVertexID, uint meshInstance: SV_StartInstanceLocation, uint lightIndex: SV_InstanceID) {
    uint32_t offset = pushConstants.pointLightOffsets[meshInstance];
    float3 pos = pushConstants.positions[vertexID].pos;
    VertexAttributes attr = pushConstants.attributes[vertexID];
    Instance instance = pushConstants.instances[meshInstance];

    VSOut res;
    res.worldPos = mul(instance.worldTransform, float4(pos, 1.0f));
    res.lightIndex = pushConstants.pointLightIndices[offset + lightIndex];
    res.uv = float2(attr.u, attr.v);
    res.materialID = instance.materialID;
    res.colorTextureID = instance.colorTextureID;
    res.colorSamplerID = instance.colorSamplerID;
    return res;
}

float GetClipDist(float3 lightPos, float3 vertexPos, uint planeIndex)
{
    float3 normal = pushConstants.tetraData.planeNormals[planeIndex].xyz;
    return dot(vertexPos - lightPos, normal);
}

[shader("geometry")]
[maxvertexcount(12)]
void main(
    triangle VSOut input[3],
    inout TriangleStream<GSOutput> outputStream)
{
    const uint lightIndex = input[0].lightIndex;
    PointLight light = pushConstants.visiblePointLights[lightIndex];
    const float3 lightPosition = light.position;

    // Back-face culling
    float3 normal = cross(
        input[2].worldPos.xyz - input[0].worldPos.xyz,
        input[0].worldPos.xyz - input[1].worldPos.xyz);

    float3 view = lightPosition - input[0].worldPos.xyz;
    if (dot(normal, view) < 0.0f)
        return;

    // Iterate over tetrahedron faces
    for (uint faceIndex = 0; faceIndex < 4; faceIndex++)
    {
        uint inside = 0;
        float clipDistances[9];

        for (uint sideIndex = 0; sideIndex < 3; sideIndex++)
        {
            const uint planeIndex = faceIndex * 3 + sideIndex;
            const uint bit = 1u << sideIndex;

            for (uint vertexIndex = 0; vertexIndex < 3; vertexIndex++)
            {
                uint clipDistIndex = sideIndex * 3 + vertexIndex;
                clipDistances[clipDistIndex] = GetClipDist(
                    lightPosition,
                    input[vertexIndex].worldPos.xyz,
                    planeIndex);

                inside |= (clipDistances[clipDistIndex] > 0.001) ? bit : 0;
            }
        }

        // If triangle is inside volume, emit primitive
        if (inside == 0x7)
        {
            const float4x4 shadowMatrix = light.lightMatrices[faceIndex];

            for (uint vertexIndex = 0; vertexIndex < 3; vertexIndex++)
            {
                GSOutput output;
                output.position = mul(shadowMatrix, input[vertexIndex].worldPos);
                output.clipDistance0 = clipDistances[vertexIndex];
                output.clipDistance1 = clipDistances[3 + vertexIndex];
                output.clipDistance2 = clipDistances[6 + vertexIndex];
                output.uv = input[vertexIndex].uv;
                output.materialID = input[0].materialID;
                output.colorTextureID = input[0].colorTextureID;
                output.colorSamplerID = input[0].colorSamplerID;
                outputStream.Append(output);
            }
            outputStream.RestartStrip();
        }
    }
}

[shader("pixel")]
void pixelMain(GSOutput input) {
    GLTFMaterialData materialData = materials[NonUniformResourceIndex(input.materialID)];
    float4 baseColor = textures[NonUniformResourceIndex(input.colorTextureID)]
        .Sample(samplers[NonUniformResourceIndex(input.colorSamplerID)], input.uv);
    if ((materialData.colorFactors * baseColor).a < materialData.alphaCutoff.r)
        discard;
}
