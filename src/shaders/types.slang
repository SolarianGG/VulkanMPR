module types;

public static const int MAX_CASCADE_COUNT = 4;
public static const int POINT_LIGHT_SHADOW_MAP_SIZE = 8192u;
public static const int POINT_LIGHT_TILE_SIZE = 1024u;
public static const int MAX_POINT_LIGHTS =
    (POINT_LIGHT_SHADOW_MAP_SIZE * POINT_LIGHT_SHADOW_MAP_SIZE) / (POINT_LIGHT_TILE_SIZE * POINT_LIGHT_TILE_SIZE);
public static const int POINT_LIGHT_TILES_PER_ROW = POINT_LIGHT_SHADOW_MAP_SIZE / POINT_LIGHT_TILE_SIZE;

public struct Vertex
{
    public float3 pos;
    public float u;
    public float3 norm;
    public float v;
    public float4 tangent;
};

public struct VertexPosition
{
    public float3 pos;
};

public struct VertexAttributes
{
    public float16_t u;
    public float16_t v;
    public vector<float16_t, 2> norm;
    public vector<float16_t, 2> tangent;
};

public struct Mesh {
    public uint32_t indexCount;
    public uint32_t firstIndex;
    public int32_t vertexOffset;
    public uint32_t padding;
    public float3 min;
    public float padding1;
    public float3 max;
    public float padding2;
};

public struct Command {
    public uint32_t indexCount;
    public uint32_t instanceCount;
    public uint32_t firstIndex;
    public int32_t vertexOffset;
    public uint32_t firstInstance;
};

public struct Count {
    public uint32_t commandCount;
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
    public uint32_t emissiveTextureID;
    public uint32_t emissiveSamplerID;
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
    public float4 emissiveFactors;       // rgb -> emissive factor
};

public struct DirectionalLight {
    public float3 direction;
    public int cascadeCount;
    public float4 color; // w - intensity
    public float normalBias;
    public float constantBias;
    public float2 _pad;

    public float splitDistances[MAX_CASCADE_COUNT];
    public float4x4 cascadeVPs[MAX_CASCADE_COUNT];
};

public struct TetrahedronData {
    public float4 faceVectors[4];
    public float4 planeNormals[12];
};

public struct PointLight {
    public float3 position;
    public float range;
    public float3 color;
    public float intensity;
    public float normalBias;
    public float constantBias;
    public float2 _pad;

    public float4x4 lightMatrices[4];
};

public struct MinMax {
    public uint min;
    public uint max;
};

public struct AABB {
    public uint minX; public uint minY; public uint minZ; public uint _pad0;
    public uint maxX; public uint maxY; public uint maxZ; public uint _pad1;
};

public struct DDGIVolume {
    public float3 origin;
    public int probeNumRays;
    public float4 rotation;
    public float3 probeSpacing;
    public float probeMaxRayDistance;
    public int3 probeCounts;
    public float rayNormalBias;
    public float4 probeRayRotation;
    float3 _padding0;
    public float rayViewBias;
};

public struct Payload {
    public float3 worldPosition;
    public float hitT;
    public float3 normal;
    public uint hitKind;
    public float3 albedo;
    public float opacity;
    public float3 shadingNormal;
    public float metallic;
    public float3 _padding0;
    public float roughness;
};
