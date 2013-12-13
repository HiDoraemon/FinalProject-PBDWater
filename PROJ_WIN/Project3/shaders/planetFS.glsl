#version 330

in vec3 WorldCoord;
in vec3 ToCam;
in vec3 Up;
in vec3 Right;
in vec2 TexCoord;
out vec4 FragColor;

void main()
{

    vec2 coord = 2.01 * (TexCoord - vec2(0.5));
    float r = length(coord);
    if (r >= .25) { discard; }

    float dist = length(vec3(0,0,25)-WorldCoord);
    vec3 N = Right*-coord.x + Up*coord.y + ToCam*sqrt(1-r*r);
    vec3 L = normalize(vec3(0,0,25)-WorldCoord);
    float light = 0.1 + 0.9*clamp(dot(N,L),0.0, 1.0);
    vec3 color = vec3(0.2, 0.6, 1.0);
    FragColor = vec4(color*light,1.0);
} 