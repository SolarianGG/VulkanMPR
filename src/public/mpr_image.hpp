#pragma once

#include <volk.h>

#include <cmath>
#include <deque>
#include <vector>

namespace mp::utils
{

inline std::uint32_t calculate_mip_levels(const VkExtent2D extent)
{
    return static_cast<std::uint32_t>(std::floor(std::log2(std::max(extent.width, extent.height)))) + 1;
}

void generate_mipmaps(VkCommandBuffer cmd, VkImage image, VkExtent2D extent);
void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout, VkImageLayout newLayout);

void copy_to_image(VkCommandBuffer cmd, VkImage src, VkImage dest, VkExtent2D srcSize, VkExtent2D destSize);

struct BarrierBuilder
{
    std::vector<VkImageMemoryBarrier2> imageBarriers;
    std::vector<VkBufferMemoryBarrier2> bufferBarriers;
    std::vector<VkMemoryBarrier2> memoryBarriers;

    void clear()
    {
        memoryBarriers.clear();
        bufferBarriers.clear();
        imageBarriers.clear();
    }

    void add_image_barrier(const VkImageMemoryBarrier2 &imageBarrier)
    {
        imageBarriers.push_back(imageBarrier);
    }

    void add_buffer_barrier(const VkBufferMemoryBarrier2 &bufferBarrier)
    {
        bufferBarriers.push_back(bufferBarrier);
    }

    void add_memory_barrier(const VkMemoryBarrier2 &memoryBarrier)
    {
        memoryBarriers.push_back(memoryBarrier);
    }

    void barrier(const VkCommandBuffer cmd, const VkDependencyFlags dependencyFlags = 0);
};

} // namespace mp::utils