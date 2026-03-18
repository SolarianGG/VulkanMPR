import utils;
import types;


[[vk::binding(0, 0)]]
[[vk::image_format("r8")]]
RWTexture2D<float> outputTexture;

[[vk::binding(1, 0)]]
Texture2D<float> depthTexture;

struct PushConstants {
    DirectionalLight *light;
    float4x4 cameraViewProj;
};

[[vk::push_constant]]
PushConstants pushConstants;

static const int NUM_THREADS = 128;

static const float EPSILON = 0.0001f;


[shader("compute")]
[numthreads(NUM_THREADS, 1, 1)]
void computeMain(uint3 groupID: SV_GroupID, uint groupIndex : SV_GroupIndex) {
    uint laneCount = WaveGetLaneCount();     
    uint waveIndex = groupIndex / laneCount;

    uint wavesPerGroup = NUM_THREADS / laneCount;
    uint globalWaveIndex = groupID.x * wavesPerGroup + waveIndex;

    uint imageWidth, imageHeight;
    outputTexture.GetDimensions(imageWidth, imageHeight);
    if (globalWaveIndex >= imageWidth * imageHeight)
        return;

    uint2 pixelCoord = uint2(globalWaveIndex % imageWidth, globalWaveIndex / imageWidth);

    float2 pixelUV = (float2(pixelCoord) + 0.5) / float2(imageWidth, imageHeight);
    pixelUV = 1.0f - pixelUV.y;

    float4 lightDirClipSpace = mul(pushConstants.cameraViewProj, float4(pushConstants.light.direction.xyz, 0.0f));
    if (abs(lightDirClipSpace.w) < EPSILON)
        lightDirClipSpace.w = sign(lightDirClipSpace.w) * EPSILON;

    float2 lightUV = lightDirClipSpace.xy / lightDirClipSpace.w;
    float lightDepthValue = lightDirClipSpace.z / lightDirClipSpace.w;

}