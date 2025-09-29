#include "mpr_error_check.hpp"

#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan.h>

#include <format>
#include <stdexcept>

namespace mp {

CheckerToken chk;
VkResultGrabber::VkResultGrabber(const VkResult& result,
                                 const std::source_location& location) noexcept
    : res(result), loc(location) {}

void operator>>(const VkResultGrabber& grab, CheckerToken) {
  if (grab.res) {
    throw std::runtime_error(
        std::format("Graphics error: {}\n {}({})", string_VkResult(grab.res),
                    grab.loc.file_name(), grab.loc.line()));
  }
}
}  // namespace mp
