#version 450

layout (location = 0) in vec3 color;
layout (location = 1) in float radius;
layout (location = 2) in float orbit;
layout (location = 3) in float year;

layout (push_constant) uniform UniversalTime {
    float time;
} ut;

layout (location = 0) out vec3 fragColor;

void main() {
    fragColor = color;
    gl_PointSize = radius;
    gl_Position = vec4(orbit * cos(ut.time / year + orbit), orbit * sin(ut.time / year + orbit), 0.0, 1.0);
}
