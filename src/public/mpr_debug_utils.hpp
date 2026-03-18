#pragma once
// clang-format off
#include <volk.h>
#include <glm/glm.hpp>
// clang-format on

namespace mp::debug
{

#ifdef MPR_DEBUG

inline void set_object_name(VkDevice device, VkObjectType objectType,
                             uint64_t handle, const char* name)
{
    const VkDebugUtilsObjectNameInfoEXT nameInfo{
        .sType        = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
        .objectType   = objectType,
        .objectHandle = handle,
        .pObjectName  = name,
    };
    vkSetDebugUtilsObjectNameEXT(device, &nameInfo);
}

inline void cmd_begin_label(VkCommandBuffer cmd, const char* name,
                             glm::vec4 color = {1.f, 1.f, 1.f, 1.f})
{
    const VkDebugUtilsLabelEXT labelInfo{
        .sType      = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
        .pLabelName = name,
        .color      = {color.r, color.g, color.b, color.a},
    };
    vkCmdBeginDebugUtilsLabelEXT(cmd, &labelInfo);
}

inline void cmd_end_label(VkCommandBuffer cmd)
{
    vkCmdEndDebugUtilsLabelEXT(cmd);
}

inline void cmd_insert_label(VkCommandBuffer cmd, const char* name,
                              glm::vec4 color = {1.f, 1.f, 1.f, 1.f})
{
    const VkDebugUtilsLabelEXT labelInfo{
        .sType      = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
        .pLabelName = name,
        .color      = {color.r, color.g, color.b, color.a},
    };
    vkCmdInsertDebugUtilsLabelEXT(cmd, &labelInfo);
}

#else

inline void set_object_name(VkDevice, VkObjectType, uint64_t, const char*) {}
inline void cmd_begin_label(VkCommandBuffer, const char*, glm::vec4 = {}) {}
inline void cmd_end_label(VkCommandBuffer) {}
inline void cmd_insert_label(VkCommandBuffer, const char*, glm::vec4 = {}) {}

#endif

} // namespace mp::debug
