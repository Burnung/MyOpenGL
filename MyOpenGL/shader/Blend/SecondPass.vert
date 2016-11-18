#version 450

in layout (location = 0)  vec3 PosAttr;
in layout (location = 1)  vec3 NorAttr;
in layout (location = 2)  vec2 UvAttr;

out layout (location = 0) vec3 Nor;
out layout (location = 1) vec2 Uv;

uniform  mat4 MVP;


void main(){

    Nor = NorAttr;//vec4(colAttr.x,colAttr.y,colAttr.z,1.0f);
    Uv = UvAttr;
    gl_Position = MVP * vec4(PosAttr.xyz,1.0f);
   // gl_Position  = pos;

}