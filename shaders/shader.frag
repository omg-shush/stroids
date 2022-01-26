#version 450

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;

layout (set = 0, binding = 0) uniform Brightness {
    float brightness;
} b;
layout (set = 1, binding = 0) uniform sampler2D tex;

layout (location = 0) out vec4 outColor;

void main() {
    vec3 directional_light = normalize(vec3(0.3, -0.5, 10.0));
    float directional = clamp(dot(-1.0 * directional_light, normal), 0.0, 1.0);

    vec3 point_light = vec3(1.5, -3.0, 0.5);
    vec3 to_light = point_light - position;
    float distance_weight = 1.0 / (length(to_light) * length(to_light));
    float point = clamp(dot(to_light, normal) * distance_weight, 0.0, 1.0);

    float litness = point * 0.0 + directional * 0.99 + 0.01;

    vec3 color = texture(tex, uv).rgb;
    outColor = vec4(litness * color, 1.0);
}
