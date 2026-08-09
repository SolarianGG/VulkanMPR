#include "public/mpr_engine.hpp"
#include <tracy/Tracy.hpp>

int main(int argc, char *argv[])
{
    ZoneScoped;
    std::uint32_t width = 1600;
    std::uint32_t height = 900;
    if (argc >= 3)
    {
        width = std::stoi(argv[1]);
        height = std::stoi(argv[2]);
    }
    mp::Engine engine(width, height);

    engine.run();
    return 0;
}
