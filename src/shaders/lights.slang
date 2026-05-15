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

float LightWindowing(float distanceToLight, float maxDistance) 
{ 
    return pow(saturate(1.f - pow((distanceToLight / maxDistance), 4)), 2); 
}

float LightFalloff(float distanceToLight)
{ 
    return 1.f / pow(max(distanceToLight, 1.f), 2);
}

public
float LightVisibility(
    Payload payload,
    float3 lightVector,
    float tmax,
    float normalBias,
    float viewBias,
    RaytracingAccelerationStructure bvh)
{
    RayDesc ray;
    ray.Origin = payload.worldPosition + (payload.normal * normalBias);
    ray.Direction = normalize(lightVector);
    ray.TMin = 0.0f;
    ray.TMax = tmax;

    // Trace a visibility ray
    // Skip the CHS to avoid evaluating materials
    Payload packedPayload = (Payload)0;
    TraceRay(
        bvh,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
        0xFF,
        0,
        0,
        0,
        ray,
        packedPayload);

    return (packedPayload.hitT < 0.f) ? 1.0f : 0.0f;
}

float3 EvaluateDirectionalLight(
    Payload payload,
    float normalBias,
    float viewBias,
    RaytracingAccelerationStructure bvh,
                                DirectionalLight light)
{
    float visibility = LightVisibility(payload, -light.direction.xyz, 1e27f, normalBias, viewBias, bvh);

    // Early out, the light isn't visible from the surface
    if (visibility <= 0.f) return float3(0.f, 0.f, 0.f);

    // Compute lighting
    float3 lightDirection = -normalize(light.direction.xyz);
    float nol = max(dot(payload.shadingNormal, lightDirection), 0.f);

    return light.color.w * light.color.xyz * nol * visibility;
}

/**
 * Evaluate direct lighting for the current surface and all influential point lights.
 */
float3 EvaluatePointLight(
    Payload payload,
    float normalBias,
    float viewBias,
    RaytracingAccelerationStructure bvh,
                          PointLight light)
{
    float3 color = 0;

    float3 lightVector = (light.position - payload.worldPosition);
    float lightDistance = length(lightVector);

    // Early out, light energy doesn't reach the surface
    if (lightDistance > light.range) return float3(0.f, 0.f, 0.f);

    float tmax = (lightDistance - viewBias);
    float visibility = LightVisibility(payload, lightVector, tmax, normalBias, viewBias, bvh);

    // Early out, this light isn't visible from the surface
    if (visibility <= 0.f) return float3(0.f, 0.f, 0.f);

    // Compute lighting
    float3 lightDirection = normalize(lightVector);
    float nol = max(dot(payload.shadingNormal, lightDirection), 0.f);
    float falloff = LightFalloff(lightDistance);
    float window = LightWindowing(lightDistance, light.range);

    color += light.intensity * light.color * nol * falloff * window * visibility;
    return color;
}


public float3 DirectDiffuseLighting(
    Payload payload,
    float normalBias,
    float viewBias,
    RaytracingAccelerationStructure bvh,
                             DirectionalLight* dirLight, PointLight *pointLights, uint pointLightsCount)
{
    float3 brdf = (payload.albedo / float::getPi());
    float3 lighting = 0.f;

    if (dirLight != nullptr)
    {
        lighting += EvaluateDirectionalLight(payload, normalBias, viewBias, bvh, *dirLight);
    }

    for (uint i = 0; i < pointLightsCount; ++i)
        lighting += EvaluatePointLight(payload, normalBias, viewBias, bvh, pointLights[i]);

    return (brdf * lighting);
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

    float attenuation = LightFalloff(distance) * LightWindowing(distance, light.range);

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

