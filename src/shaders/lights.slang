module lights;

import types;

float D_GGX(float NdotH, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
    return a2 / (float::getPi() * denom * denom);
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
    float3 lightDir = normalize(-light.direction.xyz);
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

    float3 diffuse = kD * albedo / float::getPi();

    float3 radiance = light.color.xyz * light.color.w;
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
    float3 diffuse = kD * albedo / float::getPi();

    float3 radiance = light.color * light.intensity * attenuation;
    return (diffuse + specular) * radiance * NdotL;
}

