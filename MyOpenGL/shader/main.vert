#version 440

in layout (location = 0)  vec3 posAttr;
in layout (location = 1)  vec3 colAttr;
out layout (location = 0)  vec3 col;
uniform  mat4 matrix;
void main(){

    col = colAttr;//vec4(colAttr.x,colAttr.y,colAttr.z,1.0f);
    vec4 pos  = vec4(posAttr.x,posAttr.y,posAttr.z,1.0f);
    gl_Position = matrix * pos;
   // gl_Position  = pos;

}