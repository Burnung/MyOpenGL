#version 430
in layout (location = 0) vec3 Nor;
in layout (location = 1) vec2 Uv;

out layout (location = 0)  vec4 color;

uniform sampler2D gColorMap;  

void main(){
   // gl_FragColor = ve4(1.0f,0.0f,0.0f,1.0f);
    color = texture(gColorMap,Uv);
    color.w = 1.0;
   ///color = vec4(Uv.xy,0.0f,1.0f);
  // color = vec4(Nor,0);
    //color = vec4(1.0f,1.0f,0.0f,1.0f);

}