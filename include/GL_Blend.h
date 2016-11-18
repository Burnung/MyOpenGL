#ifndef GL_BLEND_H
#define GLBLEND_H

#include<GL_Shader.h>
#include<glm\vec4.hpp>

class Blend  {

public:
	Blend();// {};
	~Blend();

	void Clear();
	bool Init(int width, int height, int lenth);

	void DrawFirstPass();
	void DrawSecondPass();

	GLuint getFirstMVPUniform(){ return m_MvpUniform1; }
	GLuint getSecondMVPUniform(){ return m_MvpUniform2; }
private:
	// 窗口的大小

	struct Node{
		glm::vec4 color;
		float depth;
		unsigned int index;
	};

	int m_Width;
	int m_Height;
	int m_Length;
	int m_BufferSize;

	char *m_OnePassVerShaderFile, *m_OnePassFragShaderFile;
	char *m_TwoPassVerShaderFile, *m_TwoPassFragShaderFile;

	glslShader m_FirstPassShader, m_SecondShader;

	//存储需混合的颜色 深度 和索引
	GLuint m_LinkListBuffer;
	GLuint m_LinkListTex;
	//存储起始索引
	GLuint m_AtomicsBuffer;
	GLuint m_FirstIndexImg;
	GLuint m_FirstIndexBuffer;

	GLuint m_MvpUniform1, m_MvpUniform2;
	GLuint m_MeterialUniform;


};

#endif
