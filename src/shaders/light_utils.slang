module utils;

#define PI 3.141592f;

public struct DirectionalLight {
    public float3 direction;
    public float padding0;
    public float3 color;
    public float intensity;
};

public struct PointLight {
    public float3 position;
    public float range;
    public float3 color;
    public float intensity;
};

public struct LightData {
    public int4 type;
    public float4 data0; // xyz: position/direction, w: range for punctual
    public float4 data1; // xyz: color, w: intensity
    public float4x4 vp;
};

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

public float3 CalculateDirLightPBR(
    float3 worldPos,
    float3 cameraPos,
    float3 normal,
    float3 albedo,
    float metallic,
    float roughness,
    DirectionalLight light
) {
    float3 lightDir = normalize(-light.direction);
    float3 viewDir = normalize(cameraPos - worldPos);
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

    float3 diffuse = kD * albedo / PI;

    float3 radiance = light.color * light.intensity;
    return (diffuse + specular) * radiance * NdotL;
}

public float3 CalculatePointLightPBR(
    float3 worldPos,
    float3 cameraPos,
    float3 normal,
    float3 albedo,
    float metallic,
    float roughness,
    PointLight light
) {
    float3 lightDir = light.position - worldPos;
    float distance = length(lightDir);

    lightDir /= distance;
    float3 viewDir = normalize(cameraPos - worldPos);
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
    float3 diffuse = kD * albedo / PI;

    float3 radiance = light.color * light.intensity * attenuation;
    return (diffuse + specular) * radiance * NdotL;
}

public float CalculateShadow(float4x4 lightVP, float3 worldPos, Texture2D<float> shadowMap, SamplerComparisonState shadowSampler) {
    float4 lightSpacePos = mul(lightVP, float4(worldPos, 1.0));
    float3 projCoords = lightSpacePos.xyz / lightSpacePos.w;

    float2 shadowUV = projCoords.xy * 0.5 + 0.5;
    shadowUV.y = 1.0 - shadowUV.y;

    if (shadowUV.x < 0 || shadowUV.x > 1 || shadowUV.y < 0 || shadowUV.y > 1)
        return 1.0;

    float currentDepth = projCoords.z;
    float bias = 0.005;

    float shadow = 0.0;
    float texelSize = 1.0 / 2048.0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            shadow += shadowMap.SampleCmpLevelZero(
                shadowSampler, shadowUV + float2(x, y) * texelSize, currentDepth - bias);
        }
    }
    return shadow / 9.0;
}

public float3 CalculateDirLightGooch(float3 worldPos, float3 cameraPos,
    float3 normal,
    float3 albedo, DirectionalLight light) {
    float3 lightDir = normalize(-light.direction);
    float3 viewDir = normalize(cameraPos - worldPos);

    float3 ccool = float3(0, 0, 0.55f) + 0.25f * albedo;
    float3 cwarm = float3(0.3f, 0.3f, 0.0f) + 0.25f * albedo;
    float3 chighlight = float3(1.0f, 1.0f, 1.0f);
    float t = (dot(normal, lightDir) + 1) / 2;
    float3 r = (2 * dot(normal, lightDir) * normal) - lightDir;
    float s = (100 * dot(r, viewDir) - 97);
    s = clamp(s, 0.0f, 1.0f);

    return s * chighlight + (1.0f - s) * (t * cwarm + (1 - t) * ccool);
}