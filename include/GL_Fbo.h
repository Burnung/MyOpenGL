
#include<gl\glew.h>
#include<Windows.h>

#ifndef TEXSTATE_H
#define TEXSTATE_H

class TexState{
public:
	TexState(){};

	TexState(int w, int h, int f = GL_LINEAR, int c = GL_CLAMP) :width(w), height(h), filterMode(f), clampMode(c){

	};
	inline int getWidth(){ return width; };
	inline int getHeight(){ return height; };
	inline int getClamp(){ return clampMode; };
	inline int getFilterMode(){ return filterMode; };

private:
	int width, height;
	int filterMode, clampMode;
};
#endif

#ifndef FBO_H
#define FBO_H
static GLenum mybuffers[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1_EXT, GL_COLOR_ATTACHMENT2_EXT, GL_COLOR_ATTACHMENT3_EXT,
                           GL_COLOR_ATTACHMENT4_EXT, GL_COLOR_ATTACHMENT5_EXT, GL_COLOR_ATTACHMENT6_EXT };
class Fbo {
private:

	GLuint DepthTex;
	GLuint depthbuffer;
	int beforeFboId;
	TexState texDescript;
	int num;           //绑定纹理的数量

public:
	//存储纹理
	GLuint TexId[8];
	GLuint fboId;   // 这个fbo的ID

	

	Fbo(){};
	Fbo(int num, int width, int height);
	~Fbo();
	inline TexState getTexState(){
		return texDescript;
	}
	//初始化 分配内存等
	void init();

	void init(int num, int width, int height);
	//启动fbo
	void begin();
	//指定渲染buffer
	void BindForWrite(int i);
	void BindForRead(int i, GLenum TextureUnit);

	//停用fbo
	void end();
	//static void SaveBMP(char *filename);
	void destory();
	//返回纹理指针
	GLuint getTexture(int id);
	//静态函数 存储bmp
	static void SaveBMP(const char *fileName, BYTE *buf, UINT width, UINT height);
	void SaveBMP(const char *fileName, int id);

	void SaveBuffToBMP(const char *fileName, int id);

	void SaveFloat(const char *fileName, int id);

	void CoutFloat(int);

	int ComputNum(int id, int lastNum);
	//glm::vec2 ComputError(int);
};

#endif