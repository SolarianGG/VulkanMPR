#define GLM_ENABLE_EXPERIMENTAL
#include "mpr_camera.hpp"

#include <SDL3/SDL.h>

#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>

namespace mp {
void Camera::update(const float deltaTime) {
  const glm::mat4 cameraRotation = get_rotation_matrix();
  position +=
      glm::vec3(cameraRotation *
                glm::vec4(velocity * 0.5f * deltaTime * cameraSpeed, 0.f));
}

void Camera::process_sdl_event(SDL_Event& e, SDL_Window* window) {
  if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
    if (e.button.button == SDL_BUTTON_RIGHT) {
      m_bIsRightMouseButtonClicked = true;
      SDL_SetWindowRelativeMouseMode(window, true);
    }
  }

  if (e.type == SDL_EVENT_MOUSE_BUTTON_UP) {
    if (e.button.button == SDL_BUTTON_RIGHT) {
      m_bIsRightMouseButtonClicked = false;
      SDL_SetWindowRelativeMouseMode(window, false);
    }
  }

  if (e.type == SDL_EVENT_KEY_DOWN && m_bIsRightMouseButtonClicked) {
    if (e.key.key == SDLK_W) {
      velocity.z = -1;
    }
    if (e.key.key == SDLK_S) {
      velocity.z = 1;
    }
    if (e.key.key == SDLK_A) {
      velocity.x = -1;
    }
    if (e.key.key == SDLK_D) {
      velocity.x = 1;
    }
  }

  if (e.type == SDL_EVENT_KEY_UP) {
    if (e.key.key == SDLK_W) {
      velocity.z = 0;
    }
    if (e.key.key == SDLK_S) {
      velocity.z = 0;
    }
    if (e.key.key == SDLK_A) {
      velocity.x = 0;
    }
    if (e.key.key == SDLK_D) {
      velocity.x = 0;
    }
  }

  if (e.type == SDL_EVENT_MOUSE_MOTION && m_bIsRightMouseButtonClicked) {
    yaw += (float)e.motion.xrel / 200.f;
    pitch -= (float)e.motion.yrel / 200.f;
  }
}

glm::mat4 Camera::get_view_matrix() {
  // to create a correct model view, we need to move the world in opposite
  // direction to the camera
  //  so we will create the camera model matrix and invert
  const glm::mat4 cameraTranslation = glm::translate(glm::mat4(1.f), position);
  const glm::mat4 cameraRotation = get_rotation_matrix();
  return glm::inverse(cameraTranslation * cameraRotation);
}

glm::mat4 Camera::get_rotation_matrix() {
  // fairly typical FPS style camera. we join the pitch and yaw rotations into
  // the final rotation matrix

  const glm::quat pitchRotation =
      glm::angleAxis(pitch, glm::vec3{1.f, 0.f, 0.f});
  const glm::quat yawRotation = glm::angleAxis(yaw, glm::vec3{0.f, -1.f, 0.f});

  return glm::toMat4(yawRotation) * glm::toMat4(pitchRotation);
}

}  // namespace mp