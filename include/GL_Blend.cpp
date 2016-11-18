#include"GL_Blend.h"
#include"gl_util.h"
#include<iostream>
Blend::Blend() :m_FirstIndexImg(INVALID_OGL_VALUE), m_LinkListBuffer(INVALID_OGL_VALUE), m_AtomicsBuffer(INVALID_OGL_VALUE)
, m_OnePassVerShaderFile(nullptr), m_OnePassFragShaderFile(nullptr), m_TwoPassVerShaderFile(nullptr), m_TwoPassFragShaderFile(nullptr)
{
}

Blend::~Blend(){
	Clear();
}


void Blend::Clear(){

	if (m_LinkListBuffer != INVALID_OGL_VALUE)
		glDeleteBuffers(1, &m_LinkListBuffer);

	if (m_AtomicsBuffer != INVALID_OGL_VALUE)
		glDeleteBuffers(1, &m_AtomicsBuffer);


}

bool Blend::Init(int Width, int Height, int passTime){
	Clear();
	m_Width = Width;
	m_Height = Height;
	m_Length = passTime;
	m_BufferSize = m_Width * m_Height * m_Length;

	m_OnePassVerShaderFile = "Shader/Blend/FirstPass.vert";
	m_OnePassFragShaderFile = "Shader/Blend/FirstPass.frag";
	m_TwoPassVerShaderFile = "Shader/Blend/SecondPass.vert";
	m_TwoPassFragShaderFile = "Shader/Blend/SecondPass.frag";

	//生成存储链表的buffer
	glGenBuffers(1, &m_LinkListBuffer);
	glBindBuffer(GL_TEXTURE_BUFFER, m_LinkListBuffer);
	glBufferData(GL_TEXTURE_BUFFER, sizeof(glm::uvec4)* m_BufferSize, NULL, GL_DYNAMIC_COPY);

	glGenTextures(1, &m_LinkListTex);
	glBindTexture(GL_TEXTURE_BUFFER, m_LinkListTex);
	glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32UI, m_LinkListBuffer);
	glBindBuffer(GL_TEXTURE_BUFFER, 0);
	glBindTexture(GL_TEXTURE_BUFFER, 0);

	glGenTextures(1, &m_FirstIndexImg);
	glBindTexture(GL_TEXTURE_2D, m_FirstIndexImg);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, m_Width, m_Height, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
	glBindTexture(GL_TEXTURE_2D, 0);
	//glTexStorage2D(GL_TEXTURE_2D, 0, GL_R32UI, m_Width, m_Height, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
	glGenBuffers(1, &m_FirstIndexBuffer);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_FirstIndexBuffer);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, m_Width * m_Height * sizeof(GLuint), NULL, GL_STATIC_DRAW);
	//将数据设为0xFFFFFFFF
	GLuint *tmpB = (GLuint*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
	memset(tmpB, 0xFF, sizeof(GLuint)* m_Width * m_Height);
	glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	//生成原子加的buffer 并初始化为0
	//	unsigned int tmpZero = 0;
	glGenBuffers(1, &m_AtomicsBuffer);
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, m_AtomicsBuffer);
	glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint), NULL, GL_DYNAMIC_COPY);
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);


	//初始化shader
	m_FirstPassShader.loadShader(m_OnePassVerShaderFile, 0, m_OnePassFragShaderFile);
	m_SecondShader.loadShader(m_TwoPassVerShaderFile, 0, m_TwoPassFragShaderFile);

	m_MvpUniform1 = m_FirstPassShader.getUniform("MVP");
	m_MvpUniform2 = m_SecondShader.getUniform("MVP");
	m_MeterialUniform = m_FirstPassShader.getUniform("Tex");
	//m_FirstPassShader->bind()
	return true;
}

void Blend::DrawFirstPass(){


	GLuint TmpZero = 0;
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, m_AtomicsBuffer);
	glBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint), &TmpZero);
	glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, m_AtomicsBuffer);


	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_FirstIndexBuffer);
	glBindTexture(GL_TEXTURE_2D, m_FirstIndexImg);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, m_Width, m_Height, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	m_FirstPassShader.useShder();
	//glBindImageTexture(0, m_FirstIndexImg, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
	glBindImageTexture(1, m_LinkListTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32UI);

	glUniform1i(m_MeterialUniform, 0);
}
void Blend::DrawSecondPass(){


	glBindTexture(GL_TEXTURE_2D, m_FirstIndexImg);
 	GLuint *tmpData = new GLuint[m_Width * m_Height];
	memset(tmpData, 0, sizeof(GLuint)* m_Width * m_Height);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, tmpData);
	std::ofstream tof("firstImg.txt");
	for (int i = 0; i < m_Height; i++) {
		for (int j = 0; j < m_Width; j++)
		if (tmpData[i * m_Width + j] != 0xffffffff)
			std::cout << tmpData[i * m_Width + j] << std::endl;
	}
	tof.close();

	delete[] tmpData;

	m_SecondShader.useShder();
	glBindImageTexture(0, m_FirstIndexImg, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI);
	//glBindTexture(GL_TEXTURE_BUFFER, m_LinkListTex);
	glBindImageTexture(1, m_LinkListTex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32UI);

}
