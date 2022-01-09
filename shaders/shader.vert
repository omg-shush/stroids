#version 450

layout (location = 0) in vec2 position;

layout (push_constant) uniform UniversalTime {
    float time;
} ut;

void main() {
    gl_PointSize = 10.0;
    gl_Position = vec4(position.x * cos(ut.time), position.x * sin(ut.time), 0.0, 1.0);
}
