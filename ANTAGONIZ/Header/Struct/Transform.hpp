#ifndef __ENGINE_TRANSFORM__
#define __ENGINE_TRANSFORM__

#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"

struct Transform
{
    glm::vec3 position;
    glm::quat rotation;
    glm::vec3 scale;
};

struct LocalTransform
{
    glm::vec3 position;
    glm::quat rotation;
};

#endif //!__ENGINE_TRANSFORM__