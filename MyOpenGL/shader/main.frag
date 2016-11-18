#version 430
in layout (location = 0) vec3 theColor;
out layout (location = 0)  vec4 color;

void main(){
   // gl_FragColor = ve4(1.0f,0.0f,0.0f,1.0f);
    color = vec4(theColor,0.5f);
    //color = vec4(1.0f,1.0f,0.0f,1.0f);

}