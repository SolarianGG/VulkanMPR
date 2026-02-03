module utils;

public struct Vertex
{
    public float3 pos;
    public float u;
    public float3 norm;
    public float v;
    public float4 tangent;
    public float4 col;
};

public struct Instance
{
    public float4x4 worldTransform;
    public uint32_t meshIndex;
    public uint32_t materialID;
    public uint32_t colorTextureID;
    public uint32_t colorSamplerID;
    public uint32_t metalRoughTextureID;
    public uint32_t metalRoughSamplerID;
    public uint32_t normalTextureID;
    public uint32_t normalSamplerID;
};

public struct SceneData 
{
    public float4x4 view;
    public float4x4 proj;
    public float4x4 projView;
    public float3 cameraPos;
    public float padding;
};

public struct GLTFMaterialData {
    public float4 colorFactors;
    public float4 metalRoughnessFactors;
    public float4 alphaCutoff;
};
