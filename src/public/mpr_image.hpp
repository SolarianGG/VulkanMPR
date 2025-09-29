#pragma once
#include <vulkan/vulkan.h>

namespace mp::utils {
void transition_image(VkCommandBuffer cmd, VkImage image,
                      VkImageLayout currentLayout, VkImageLayout newLayout);


void copy_to_image(VkCommandBuffer cmd, VkImage src, VkImage dest,
                   VkExtent2D srcSize, VkExtent2D destSize);


}  // namespace mp::utils