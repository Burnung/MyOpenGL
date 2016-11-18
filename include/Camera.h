#ifndef Camera_H
#define Camera_H
//#include<iostream>
//#include"Constructs/Vec3f.h"
#include<glm\vec3.hpp>
#include<glm\geometric.hpp>
#include<glm\matrix.hpp>
#include"Quaternion.h"

//using namespace glm;


const float myStep = 0.5;

const int MARGIN = 10;

const float AngleStep = 5;

const float PI = 3.1415926;

class Camera{

private:

	glm::vec3 Oposition, Otarget, OupVec;
	glm::vec3 Nposition, Ntarget, NtargetVec, NupVec;
	glm::mat4 ViewMat;

	float m_AngleH, m_AngleV;

	int MouseX, MouseY;
	float MouseFx, MouseFy;

	bool m_OnUpperEdge;
	bool m_OnLowerEdge;
	bool m_OnLeftEdge;
	bool m_OnRightEdge;
	bool mouspush;
	bool mouseClick;

	int m_windowWidth;
	int m_windowHeight;


public:
	//Camera(float *tmp1,float *tmp2,float *tmp3);

	Camera();
	Camera(float* tmp,int ,int );
	Camera(glm::vec3&, glm::vec3&, glm::vec3&, int, int);

	void Init();

	void OnKeyBoard(int key);

	void MousePush(float ,float );
	void MouseUnPress(){ mouseClick = false; }
	void OnMouse(float x,float y);

	void Update();

	void OnRender();

	void Reset();

	glm::mat4 Look();

	void coutCamera();

	void MoveForward();
	void MoveBack();
	void MoveLeft();
	void MoveRight();
	void SetMousePos(float x, float y){ MouseFx = x; MouseFy = y; }
	void MouseClick(bool b){ mouseClick = b; }

	void SetWidth(int w);
	void SetHeithg(int h);

	void SetPostion(glm::vec3& P);
	void SetLookVec(glm::vec3& LV);
	void SetUpVec(glm::vec3& up);

	glm::vec3 getPosition(){
		return Nposition;
	}
};



glm::vec3 Rotate(glm::vec3 rource, float angle, glm::vec3 dest);



#endif