import types;


struct PushConstants {
  Command *commands;
  Count *count;
  int cascadesCount;
};

[[vk::push_constant]]
PushConstants pushConstants;

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 threadID: SV_DispatchThreadID) {
    uint commandIndex = threadID.x;

    if (commandIndex >= pushConstants.count.commandCount)
        return;

    pushConstants.commands[commandIndex].instanceCount = pushConstants.cascadesCount;
}
