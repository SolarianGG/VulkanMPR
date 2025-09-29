#pragma once

#include <source_location>

enum VkResult : int;

namespace mp {

  struct CheckerToken {};

  extern CheckerToken chk;

  struct VkResultGrabber {
    VkResultGrabber(const VkResult& res, const std::source_location& = std::source_location::current()) noexcept;
    const VkResult& res;
    std::source_location loc;
  };

  void operator>>(const VkResultGrabber&, CheckerToken);

}