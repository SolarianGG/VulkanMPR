import utils;
import light_utils;

[[vk::binding(0, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> outputTexture;

[[vk::binding(0, 1)]]
[[vk::image_format("rgba32f")]]
Texture2D<float4> positionTexture;
[[vk::binding(1, 1)]]
[[vk::image_format("rgba32f")]]
Texture2D<float4> normalTexture;
[[vk::binding(2, 1)]]
[[vk::image_format("rgba8")]]
Texture2D<float4> albedoTexture;       
[[vk::binding(3, 1)]]
[[vk::image_format("rgba8")]]
Texture2D<float4> metalRoughTexture;   



struct PushConstants {
    SceneData* sceneData;
    LightData* lights; 
    uint lightCount;
};

[[vk::push_constant]]
PushConstants pushConstants;



[shader("compute")]
[numthreads(16, 16, 1)]
void computeMain(uint3 threadID : SV_DispatchThreadID)
{
    uint width, height;
    outputTexture.GetDimensions(width, height);
    if (threadID.x >= width || threadID.y >= height) return;

    int2 uv = int2(threadID.xy);
    float4 position = positionTexture[uv];
    float4 normal = normalTexture[uv];
    float4 albedo = albedoTexture[uv];
    float4 mr = metalRoughTexture[uv];

    float3 worldPos = position.xyz;
    float3 normalWS = normalize(normal.xyz);
    float3 baseColor = albedo.rgb;
    float metallic = mr.r;
    float roughness = mr.g;
    roughness = max(roughness, 0.04f); 

    float3 color = 0.0f;

    for (uint i = 0; i < pushConstants.lightCount; ++i)
    {
        LightData light = pushConstants.lights[i];
        if (light.type.x == 0)
        {
            DirectionalLight dirLight;
            dirLight.direction = light.data0.xyz;
            dirLight.color = light.data1.xyz;
            dirLight.intensity = light.data1.w;
            color += CalculateDirLightPBR(worldPos, pushConstants.sceneData.cameraPos, normalWS, baseColor, metallic, roughness, dirLight);
        }
        else if (light.type.x == 1)
        {
            PointLight pointLight;
            pointLight.position = light.data0.xyz;
            pointLight.range = light.data0.w;
            pointLight.color = light.data1.xyz;
            pointLight.intensity = light.data1.w;
            color += CalculatePointLightPBR(worldPos, pushConstants.sceneData.cameraPos, normalWS, baseColor, metallic, roughness, pointLight);
        }
    }

    outputTexture[uv] = float4(color, 1.0f);
}