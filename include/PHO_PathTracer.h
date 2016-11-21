#ifndef PHO_PATHTRACER_H
#define PHO_PATHTRACER_H

#include"gl_util.h"
#include<glm\vec3.hpp>
#include"GL_BaseType.h"

#include"PHO_Random.h"
//#define PATHTRACER_USE_GPU   //是否使用GPU加速

class PHO_ViewPort;
class PHO_PahtTracer{
	GL_DECLARE_SINGLETON(PHO_PahtTracer);
public:
	~PHO_PahtTracer();

	void Init(PHO_ViewPort*);
	//void SetViewPort()
	void GoTrace(int samples);
	void SaveRet();

	void Update();

	int GetMaxDepth();

	void Save2BMP(const char* filename );

private:
	PHO_ViewPort *m_ViewPort;

	BYTE *m_RetBMP;

	int m_WindowWidth;
	int m_WindowHeight;
	glm::vec3 m_CamPos;
	glm::vec3 m_CamTarVec;
	glm::vec3 m_CamYVec;
	glm::vec3 m_CamXVec;

	float m_Width_recp;
	float m_Height_recp;
	float m_FovS;
	float m_Ratio;
	float m_X_Spacing;
	float m_X_Spacing_Half;
	float m_Y_Spacing;
	float m_Y_Spacing_Half;


private:
	void ClearRetBuffer();
	void Clear();
	GL_Ray GetRay(int x,int y);
	void WriteColor(glm::vec3 &col, int x, int y);
};


#endif
