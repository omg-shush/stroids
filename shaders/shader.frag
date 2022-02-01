#version 450

layout (location = 0) in vec4 position;
layout (location = 1) in vec4 normal;
layout (location = 2) in vec2 uv;

layout (set = 0, binding = 0) uniform Brightness {
    float brightness;
} b;
layout (set = 1, binding = 0) uniform sampler2D tex;
layout (push_constant) uniform FragmentPushConstant {
    layout (offset = 128) bool do_lighting;
} pc;

layout (location = 0) out vec4 outColor;

void main() {
    vec3 directional_light = normalize(vec3(0.0, -10.0, 5.0));
    float directional = clamp(dot(-1.0 * directional_light, normal.xyz), 0.0, 1.0);

    vec3 point_light = vec3(0.0, 0.0, 0.0);
    float strength = 10.0;
    vec3 to_light = point_light - position.xyz;
    float distance_weight = 1.0 / (length(to_light) * length(to_light));
    float point = clamp(dot(to_light, normal.xyz) * distance_weight * strength, 0.0, 1.0);

    float litness = point * 0.98 + directional * 0.0 + 0.02;

    if (!pc.do_lighting) {
        litness = 1.0;
    }

    vec4 color = texture(tex, uv);
    outColor = vec4(litness * color.rgb, color.a);
}
