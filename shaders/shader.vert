#version 450

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;

layout (push_constant) uniform VertexPushConstants {
    mat4 m;
    mat4 vp;
} pc;

layout (location = 0) out vec4 worldPosition;
layout (location = 1) out vec4 worldNormal;
layout (location = 2) out vec2 fragUv;

void main() {
    worldPosition = pc.m * vec4(position, 1.0);
    worldNormal = normalize(pc.m * vec4(normal, 0.0));
    fragUv = uv;
    gl_Position = pc.vp * worldPosition;
}
