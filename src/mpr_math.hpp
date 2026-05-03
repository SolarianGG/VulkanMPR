#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <limits>
#include <numeric>
#include <random>

namespace mp::math
{

inline bool is_nearly_zero(const float value)
{
    return FP_ZERO == fpclassify(value);
}

class RandomGenerator final
{
  public:
    static RandomGenerator &Get()
    {
        static RandomGenerator rd{};
        return rd;
    }

    template <typename T = float> T generate_float(T min = 0.0f, T max = 1.0f)
    {
        std::uniform_real_distribution<T> distance(min, max);
        return distance(m_Gen);
    }

    template <typename T = int> T generate_int(T min = 0, T max = 100)
    {
        std::uniform_int_distribution<T> distance(min, max);
        return distance(m_Gen);
    }

  private:
    RandomGenerator()
        : m_Gen(m_Rd()) {

          };

    std::random_device m_Rd{};
    std::mt19937 m_Gen;
};

inline glm::vec4 random_rotation_quaternion()
{
    const float u1 = RandomGenerator::Get().generate_float();
    const float u2 = RandomGenerator::Get().generate_float();
    const float u3 = RandomGenerator::Get().generate_float();

    float q1 = sqrt(1.0f - u1) * sin(2 * glm::pi<float>() * u2);
    float q2 = sqrt(1.0f - u1) * cos(2 * glm::pi<float>() * u2);
    float q3 = sqrt(u1) * sin(2.0f * glm::pi<float>() * u3);
    float q4 = sqrt(u1) * cos(2.0f * glm::pi<float>() * u3);
    return {q1, q2, q3, q4};
}
} // namespace mp::math