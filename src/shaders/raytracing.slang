module raytracing;

import utils;
import types;

public uint3 LoadIndices(Mesh mesh, uint primitiveIndex, uint *indices)
{
    uint base = mesh.firstIndex + primitiveIndex * 3;
    return uint3(
        indices[base + 0],
        indices[base + 1],
        indices[base + 2]
    );
}

public Vertex[3] LoadVertices(Mesh mesh, uint primitiveIndex, VertexPosition *pos, VertexAttributes *attributes, uint *indices)
{
    uint3 indices = LoadIndices(mesh, primitiveIndex, indices);

    Vertex v[3];
    [unroll]
    for (int i = 0; i < 3; ++i)
    {
        v[i].pos = pos[indices[i]].pos;
        v[i].norm = DecodeNormal(attributes[indices[i]].norm);
        v[i].tangent = DecodeTangent(attributes[indices[i]].tangent);
        v[i].u = attributes[indices[i]].u;
        v[i].v = attributes[indices[i]].v;

        v[i].norm = normalize(v[i].norm);
        v[i].tangent.xyz = normalize(v[i].tangent.xyz);
    }
    return v;
}

public Vertex InterpolateVertex(Vertex vertices[3], float3 barycentrics)
{
    Vertex v = (Vertex)0;
    [unroll]
    for (uint i = 0; i < 3; i++)
    {
        v.pos += vertices[i].pos * barycentrics[i];
        v.norm += vertices[i].norm * barycentrics[i];
        v.tangent.xyz += vertices[i].tangent.xyz * barycentrics[i];
        v.u += vertices[i].u * barycentrics[i];
        v.v += vertices[i].v * barycentrics[i];
    }

    // Normalize normal and tangent vectors, set tangent direction component
    v.norm = normalize(v.norm);
    v.tangent.xyz = normalize(v.tangent.xyz);
    v.tangent.w = vertices[0].tangent.w;

    return v;
}