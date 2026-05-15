import types;
import utils;
import ddgi_common;

[[vk::binding(0, 0)]]
RWTexture2D<float4> DDGIOutput;

[[vk::binding(0, 1)]]
Texture2D<float> depthTexture;
[[vk::binding(1, 1)]]
Texture2D<uint> normalTexture;
[[vk::binding(2, 1)]]
Texture2D<float4> albedoTexture;

[[vk::binding(0, 2)]]
Texture2DArray<float4> IrradianceDatas[];

[[vk::binding(1, 2)]]
Texture2DArray<float2> DistanceDatas[];

[[vk::binding(2, 2)]]
SamplerState BillinearWrapSampler;

struct PushConstants {
    DDGIVolume *volumes;
    uint volumeCount;
    SceneData *sceneData;
};

[[vk::push_constant]]
PushConstants pushConstants;

[shader("compute")]
[numthreads(16, 16, 1)]
void IndirectCS(uint3 threadID: SV_DispatchThreadID)
{
    uint width, height;
    DDGIOutput.GetDimensions(width, height);
    if (threadID.x >= width || threadID.y >= height) return;
    float3 color = float3(0.0f, 0.0f, 0.0f);

    float4 albedo = albedoTexture[threadID.xy];

    if (albedo.a > 0.f)
    {
        float depth = depthTexture[threadID.xy];
        float3 worldPos = ReconstructWorldPos(threadID.xy, uint2(width, height), pushConstants.sceneData.inverseCameraViewProj, depth);
        float3 normal = normalize(DecodeGBufferNormal(normalTexture[threadID.xy]));



        uint volumesCount = pushConstants.volumeCount;
        float3 irradiance = 0.f;
        float3 cameraDirection = normalize(worldPos - pushConstants.sceneData.cameraPos);
        for (uint i = 0; i < volumesCount; ++i)
        {
            DDGIVolume volume = pushConstants.volumes[i];

            float3 surfaceBias = DDGIGetSurfaceBias(normal, cameraDirection, volume);

            float blendWeight = DDGIGetVolumeBlendWeight(worldPos, volume);
            if (blendWeight > 0)
            {
                irradiance += DDGIGetVolumeIrradiance(worldPos, surfaceBias, normal, volume, BillinearWrapSampler, IrradianceDatas[i], DistanceDatas[i]) * blendWeight;
            }
        }

        color = (albedo.rgb / float ::getPi()) * irradiance;
    }

    DDGIOutput[threadID.xy] = float4(color, 1.0f);
}