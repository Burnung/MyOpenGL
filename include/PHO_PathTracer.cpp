#include"PHO_PathTracer.h"
#include"PHO_ViewPort.h"
#include"GL_Scene.h"

GL_DEFINE_SINGLETON(PHO_PahtTracer);

PHO_PahtTracer::~PHO_PahtTracer(){
	Clear();
}

void PHO_PahtTracer::Clear(){
	m_ViewPort = nullptr;
	SAFERELEASE(m_RetBMP);
}

void PHO_PahtTracer::Init(PHO_ViewPort *ViewPort){
	Clear();
	m_ViewPort = ViewPort;
}

void PHO_PahtTracer::GoTrace(){
	SAFERELEASE(m_RetBMP);
	if (!m_ViewPort){
		ERROROUT("Not Set ViewPort");
		exit(1);
	}

}