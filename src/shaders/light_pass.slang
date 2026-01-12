
[[vk::binding(0, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> outputTexture;

[[vk::binding(1, 0)]]
[[vk::image_format("rgba32f")]]
Texture2D<float4> positionTexture;
[[vk::binding(2, 0)]]
[[vk::image_format("rgba32f")]]
Texture2D<float4> normalTexture;
[[vk::binding(3, 0)]]
[[vk::image_format("rgba8")]]
Texture2D<float4> albedoTexture;       
[[vk::binding(4, 0)]]
[[vk::image_format("rgba8")]]
Texture2D<float4> metalRoughTexture;   

struct SceneData
{
    float4x4 view;
    float4x4 proj;
    float4x4 projView;
    float3 cameraPos;
    float padding0;
};

struct DirectionalLight {
    float3 direction;
    float padding0;
    float3 color;
    float intensity;
};

struct PointLight {
    float3 position;
    float range;
    float3 color;
    float intensity;
};

struct LightData {
    int4 type;
    float4 data0; 
    float4 data1; 
};

struct PushConstants {
    SceneData* sceneData;
    LightData* lights; 
    uint lightCount;
};

[[vk::push_constant]]
PushConstants pushConstants;

float D_GGX(float NdotH, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
    return a2 / (3.1415926535f * denom * denom);
}

float G_Smith(float NdotL, float NdotV, float roughness)
{
    float r = roughness + 1.0f;
    float k = (r * r) / 8.0f;
    float GL = NdotL / (NdotL * (1.0f - k) + k);
    float GV = NdotV / (NdotV * (1.0f - k) + k);
    return GL * GV;
}

float3 F_Schlick(float3 F0, float HdotV)
{
    return F0 + (1.0f - F0) * pow(1.0f - HdotV, 5.0f);
}

float3 CalculateDirLightPBR(
    float3 worldPos,
    float3 normal,
    float3 albedo,
    float metallic,
    float roughness,
    DirectionalLight light
) {
    float3 lightDir = normalize(-light.direction);
    float3 viewDir = normalize(pushConstants.sceneData->cameraPos - worldPos);
    float3 halfDir = normalize(lightDir + viewDir);

    float NdotL = max(dot(normal, lightDir), 0.0f);
    float NdotV = max(dot(normal, viewDir), 0.001f); 
    float NdotH = max(dot(normal, halfDir), 0.0f);
    float HdotV = max(dot(halfDir, viewDir), 0.0f);

    const float3 F0 = lerp(0.04f, albedo, metallic);
    float3 F = F_Schlick(F0, HdotV);
    float D = D_GGX(NdotH, roughness);
    float G = G_Smith(NdotL, NdotV, roughness);

    float3 specular = (D * G * F) / (4.0f * NdotL * NdotV + 1e-5f);

    float3 kS = F;              
    float3 kD = 1.0f - kS;       
    kD *= 1.0f - metallic;       

    float3 diffuse = kD * albedo / 3.1415926535f;

    float3 radiance = light.color * light.intensity;
    return (diffuse + specular) * radiance * NdotL;
}

float3 CalculatePointLightPBR(
    float3 worldPos,
    float3 normal,
    float3 albedo,
    float metallic,
    float roughness,
    PointLight light
) {
    float3 lightDir = light.position - worldPos;
    float distance = length(lightDir);

    lightDir /= distance;
    float3 viewDir = normalize(pushConstants.sceneData->cameraPos - worldPos);
    float3 halfDir = normalize(lightDir + viewDir);

    float attenuation = pow(max(0.0, 1.0 - pow(distance / light.range, 2)), 2);

    float NdotL = max(dot(normal, lightDir), 0.0f);
    float NdotV = max(dot(normal, viewDir), 0.001f);
    float NdotH = max(dot(normal, halfDir), 0.0f);
    float HdotV = max(dot(halfDir, viewDir), 0.0f);

    const float3 F0 = lerp(0.04f, albedo, metallic);
    float3 F = F_Schlick(F0, HdotV);
    float D = D_GGX(NdotH, roughness);
    float G = G_Smith(NdotL, NdotV, roughness);

    float3 specular = (D * G * F) / (4.0f * NdotL * NdotV + 1e-5f);

    float3 kS = F;
    float3 kD = (1.0f - kS) * (1.0f - metallic);
    float3 diffuse = kD * albedo / 3.1415926535f;

    float3 radiance = light.color * light.intensity * attenuation;
    return (diffuse + specular) * radiance * NdotL;
}

float3 CalculateDirLightGooch(float3 worldPos,
    float3 normal,
    float3 albedo, DirectionalLight light) {
    float3 lightDir = normalize(-light.direction);
    float3 viewDir = normalize(pushConstants.sceneData->cameraPos - worldPos);

    float3 ccool = float3(0, 0, 0.55f) + 0.25f * albedo;
    float3 cwarm = float3(0.3f, 0.3f, 0.0f) + 0.25f * albedo;
    float3 chighlight = float3(1.0f, 1.0f, 1.0f);
    float t = (dot(normal, lightDir) + 1) / 2;
    float3 r = (2 * dot(normal, lightDir) * normal) - lightDir;
    float s = (100 * dot(r, viewDir) - 97);
    s = clamp(s, 0.0f, 1.0f);

    return s * chighlight + (1.0f - s) * (t * cwarm + (1 - t) * ccool);
}

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
#if 1
            color += CalculateDirLightPBR(worldPos, normalWS, baseColor, metallic, roughness, dirLight);
#else
            color += CalculateDirLightGooch(worldPos, normalWS, baseColor, dirLight);
#endif
        }
        else if (light.type.x == 1)
        {
            PointLight pointLight;
            pointLight.position = light.data0.xyz;
            pointLight.range = light.data0.w;
            pointLight.color = light.data1.xyz;
            pointLight.intensity = light.data1.w;
            color += CalculatePointLightPBR(worldPos, normalWS, baseColor, metallic, roughness, pointLight);
        }
    }

    outputTexture[uv] = float4(color, 1.0f);
}