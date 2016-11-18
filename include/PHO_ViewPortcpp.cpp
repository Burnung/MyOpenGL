#include"PHO_ViewPort.h"
#include"Camera.h"
#include<glm/gtc/matrix_transform.hpp>

PHO_ViewPort::PHO_ViewPort() :m_Width(512), m_Height(512), m_Nearest(1.0f), m_Farthest(500.0f),m_Fovy(60.0f), m_Camera(nullptr), m_ViewPortChange(true){
	m_Camera = new Camera(glm::vec3(0, 50, -100), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0),m_Width,m_Height);
}

PHO_ViewPort::~PHO_ViewPort(){
	SAFERELEASE(m_Camera);
}

void PHO_ViewPort::SetHeight(int h){
	m_Height = h;
	m_ViewPortChange = true;
	if (!m_Camera)
		m_Camera = new Camera();
	m_Camera->SetHeithg(h);
}

void PHO_ViewPort::SetWidth(int w){
	m_Width = w;
	m_ViewPortChange = true;
	if (!m_Camera)
		m_Camera = new Camera();
	m_Camera->SetWidth(w);
}

void PHO_ViewPort::SetNearAndFar(float n, float f){
	m_Nearest = n;
	m_Farthest = f;
	m_ViewPortChange = true;
}
void PHO_ViewPort::SetFovy(float f){
	m_Fovy = glm::radians(f);
	m_ViewPortChange = true;
}
void PHO_ViewPort::SetCameraPos(glm::vec3& p){
	if (!m_Camera)
		m_Camera = new Camera();
	m_Camera->SetPostion(p);
}

void PHO_ViewPort::SetCameraLookvec(glm::vec3& lv){
	if (!m_Camera)
		m_Camera = new Camera();
	m_Camera->SetLookVec(lv);
}

void PHO_ViewPort::SetCameraUpVec(glm::vec3& up){
	if (!m_Camera)
		m_Camera = new Camera();
	m_Camera->SetUpVec(up);
}

void PHO_ViewPort::InitCamera(){
	if (!m_Camera)
		m_Camera = new Camera();
	m_Camera->Init();
}


glm::mat4 PHO_ViewPort::GetProMat(){
	if (m_ViewPortChange){
		m_ProMat = glm::perspective(glm::radians(m_Fovy), (m_Width * 1.0f) / (m_Height * 1.0f), m_Nearest, m_Farthest);
		m_ViewPortChange = false;
	}
	return m_ProMat;
}

glm::mat4 PHO_ViewPort::GetViewMat(){
	m_ViewMat = m_Camera->Look();
	return m_Camera->Look();
}

glm::mat4 PHO_ViewPort::GetProViewMat(){
	mat4 tmpView, tmpProMat;
	tmpView = GetViewMat();
	tmpProMat = GetProMat();
	m_ProViewMat = tmpProMat * tmpView;
	return m_ProViewMat;
}
glm::vec3 PHO_ViewPort::GetCameraPos(){
	return m_Camera->getPosition();
}
glm::vec3 PHO_ViewPort::GetCameraLookVec(){
	return m_Camera->GetLookvec();
}
glm::vec3 PHO_ViewPort::GetCameraUpVec(){
	return m_Camera->GetUpvec();
}