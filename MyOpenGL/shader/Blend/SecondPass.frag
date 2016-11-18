#version 450
in layout (location = 0) vec3 Nor;
in layout (location = 1) vec2 Uv;

layout (binding = 0,r32ui) uniform uimage2D myFirstIndex;
layout (binding = 1,rgba32ui) uniform uimageBuffer myBuffer;

//申请存储结点的空间
#define MaxLenght 40
uvec4 TmpFragment[MaxLenght];

out layout (location = 0)  vec4 color0;

int getFragment(){
    int count = 0;
    uint nowPos = imageLoad(myFirstIndex,ivec2(gl_FragCoord.x,gl_FragCoord.y)).x;
    if(nowPos == 0) return count;
    while(nowPos != 0xFFFFFFFF && count <MaxLenght){
        uvec4 Titem = imageLoad(myBuffer,int(nowPos));
        nowPos = Titem.z;
       // count++;
        TmpFragment[count] = Titem;
        count++;
    }
    return count;
}

void sortFragment(int L){
    for(int i = 0;i<L;i++)
        for(int j = i+1;j<L;j++){
            float depth1 = uintBitsToFloat(TmpFragment[i].y);
            float depth2 = uintBitsToFloat(TmpFragment[j].y);
        if(depth2 <depth1){
            uvec4 tmpT = TmpFragment[j];
            TmpFragment[j] = TmpFragment[i];
            TmpFragment[i] = tmpT;
            }
        } 
}

vec4 getFinalColor(int L){
    vec4 FinalColor= vec4(0.0f);
    for(int i = 0; i<L; i++){
        vec4 tmpColor = unpackUnorm4x8(TmpFragment[i].x);
       // FinalColor = mix(FinalColor,tmpColor,tmpColor.a);
       FinalColor = tmpColor;// vec4(0.5f);
    }

    return FinalColor;

}
void main(){
   // gl_FragColor = ve4(1.0f,0.0f,0.0f,1.0f);
   int TmpLength = getFragment();

   if(TmpLength == 0){
        color0 = vec4(1.0f,0.0f,0.0f,0.0f);
        return;
   }
   if(TmpLength == MaxLenght){
        color0 = vec4(0.0f,1.0f,0.0f,0.0f);
        return;
   }
   sortFragment(TmpLength);
    //color = vec4(1.0f,1.0f,0.0f,1.0f);
   //color0 = getFinalColor(TmpLength);
   // color0 = vec4(TmpLength/40,0.0f,0.0f,0.0f);
   color0 = TmpFragment[0];

}