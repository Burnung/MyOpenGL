#ifndef GL_VIEWPOR_H

#define GL_VIEWPORT_H
#include"gl_util.h"
#include<glm\mat4x4.hpp>
#include<glm\vec3.hpp>
class Camera;

class PHO_ViewPort{

public:
	PHO_ViewPort();
	~PHO_ViewPort();

	void SetNearAndFar(float, float);
	void SetWidth(int);
	void SetHeight(int);
	void SetFovy(float);
	void SetCameraPos(glm::vec3&);
	void SetCameraLookvec(glm::vec3&);
	void SetCameraUpVec(glm::vec3&);
	void InitCamera();

	int GetWidth(){ return m_Width; };
	int GetHeight(){ return m_Height; };
	float GetNear(){ return m_Nearest; };
	float GetFar() { return m_Farthest; };
	glm::vec3 GetCameraPos();
	glm::vec3 GetCameraLookVec();
	glm::vec3 GetCameraUpvec();

	glm::mat4 GetProMat();
	glm::mat4 GetViewMat();
	glm::mat4 GetProViewMat();



private:
	Camera *m_Camera;
	int m_Width, m_Height;
	float m_Nearest, m_Farthest;
	float m_Fovy;

	bool m_ViewPortChange;

	glm::mat4 m_ProMat;
	glm::mat4 m_ViewMat;
	glm::mat4 m_ProViewMat;

};





#endif