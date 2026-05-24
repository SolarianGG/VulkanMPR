#include "public/mpr_engine.hpp"

int main(int argc, char *argv[])
{
    std::uint32_t width = 1920;
    std::uint32_t height = 1080;
    if (argc >= 3)
    {
        width = std::stoi(argv[1]);
        height = std::stoi(argv[2]);
    }
    mp::Engine engine(width, height);

    engine.run();
    return 0;
}
