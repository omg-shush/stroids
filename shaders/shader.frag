#version 450

layout (location = 0) in vec3 color;

layout (set = 0, binding = 0) uniform Brightness {
    float brightness;
} b;

layout (location = 0) out vec4 outColor;

void main() {
    outColor = vec4(color * b.brightness, 1.0);
}
