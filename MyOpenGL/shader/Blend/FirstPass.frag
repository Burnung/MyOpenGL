#version 440

in layout (location = 0) vec3 Nor;
in layout (location = 1) vec2 Uv;
in layout(location = 2) vec3 MPos;
uniform sampler2D Tex; 

layout (binding = 0, offset = 0) uniform atomic_uint List_pos;  //原子计数

layout (binding =  0, r32ui)  uniform uimage2D  FirstIndexImg;   //存储头结点位置

layout (binding = 1, rgba32ui)  uniform  uimageBuffer MyBuffer;   //存储颜色、深度等信息

out layout (location = 0)  vec4 color1 ;

//开启预先片原测试 这个很重要
layout (early_fragment_tests) in;

#define WIDTH 512

void main(){
   // gl_FragColor = ve4(1.0f,0.0f,0.0f,1.0f);
//得到颜色信息
   vec4 tmpColor = vec4(Nor,1.0f);//texture(Tex,Uv);
   tmpColor.w = 0.5f;

//原子计数器得到新的索引位置 atomicCounterIncrement
uint new_Pos = atomicCounterIncrement(List_pos);
   //逆序建表 得到之前的位置
   //imageStore(FirstIndexImg,ivec2(gl_FragCoord.xy),uvec4(5));
uint old_Pos = imageAtomicExchange(FirstIndexImg,ivec2(gl_FragCoord.xy),new_Pos);

//写入结点
uvec4 item;
item.x = packUnorm4x8(tmpColor);
item.y = floatBitsToUint(gl_FragCoord.z);
item.z = new_Pos;
imageStore(MyBuffer,int(new_Pos),item);

uvec4 tmpItem;
tmpItem = imageLoad(MyBuffer,int(old_Pos));
//tmpItem
//color1 = tmpColor;
uvec4 tmpP = imageLoad(FirstIndexImg,ivec2(gl_FragCoord.xy));
if(tmpP.r == new_Pos)
    color1 = vec4(1.0f,0.0f,0.0f,1.0f);
//else
    //color1 = unpackUnorm4x8(tmpItem.x);
//if(new_Pos == 0)//232767)
    //color1 = vec4(1.0f,0.0f,0.0f,0.0f);
//else
   // color1 = vec4(0.0f,1.0f,0.0f,0.0f);
//color1 = unpackUnorm4x8(tmpItem.x);
    //color = vec4(1.0f,1.0f,0.0f,1.0f);

}