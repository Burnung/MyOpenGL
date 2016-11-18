#ifndef PHO_PATHTRACER_H
#define PHO_PATHTRACER_H

#include"gl_util.h"
#include<glm\vec3.hpp>
#include"GL_BaseType.h"

class PHO_ViewPort;
class PHO_PahtTracer{
	GL_DECLARE_SINGLETON(PHO_PahtTracer);
public:
	~PHO_PahtTracer();

	void Init(PHO_ViewPort*);
	//void SetViewPort()
	void GoTrace();
	void SaveRet();

private:
	PHO_ViewPort *m_ViewPort;
	BYTE *m_RetBMP;
	int m_Width;
	int m_Height;
private:
	void Clear();
};


#endif
