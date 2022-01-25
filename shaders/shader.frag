#version 450

layout (location = 0) in vec3 normal;
layout (location = 1) in vec2 uv;

layout (set = 0, binding = 0) uniform Brightness {
    float brightness;
} b;

layout (location = 0) out vec4 outColor;

void main() {
    vec3 directional_light = normalize(vec3(1.0, 1.0, -2.0));
    float litness = clamp(dot(-1.0 * directional_light, normal), 0.0, 1.0);

    vec3 color = vec3(abs(uv.r), abs(uv.r * uv.g), abs(uv.g));
    outColor = vec4(b.brightness * litness * color, 1.0);
}
