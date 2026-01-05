

[[vk::binding(0, 0)]]
[[vk_image_format("rgba16f")]]
RWTexture2D<float4> outputTexture;


[shader("compute")]
[numthreads(16, 16, 1)]
void computeMain(uint3 threadID: SV_DispatchThreadID) {
  uint width;
  uint height;
  outputTexture.GetDimensions(width, height);
  if (threadID.x < width && threadID.y < height) {
    outputTexture[threadID.xy] = float4(float(threadID.x) / width, float(threadID.y) / height, 0.0f, 1.0f);
  }
}