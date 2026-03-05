module types;

public static const int MAX_CASCADE_COUNT = 4;

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
    public float4 metalRoughnessFactors; // r -> metal, g -> roughness
    public float4 alphaCutoff;           // r -> alpha cutoff
};

public struct DirectionalLight {
    public float4 direction; // w - padding
    public float4 color; // w - intensity

    public int4 cascadeCount; // r - cascadeCount

    public float splitDistances[MAX_CASCADE_COUNT];
    public float4x4 cascadeVPs[MAX_CASCADE_COUNT];
};

public struct PointLight {
    public float3 position;
    public float range;
    public float3 color;
    public float intensity;
};

public struct MinMax {
    public uint min;
    public uint max;
};

public struct AABB {
    public uint minX; public uint minY; public uint minZ; public uint _pad0;
    public uint maxX; public uint maxY; public uint maxZ; public uint _pad1;
};
