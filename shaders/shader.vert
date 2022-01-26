#version 450

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;

layout (push_constant) uniform PushConstants {
    mat4 vp;
} pc;

layout (location = 0) out vec3 fragPosition;
layout (location = 1) out vec3 fragNormal;
layout (location = 2) out vec2 fragUv;

void main() {
    fragPosition = position;
    fragNormal = normal;
    fragUv = uv;
    gl_Position = pc.vp * vec4(position, 1.0);
}
