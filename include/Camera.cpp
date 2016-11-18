#include"Camera.h"
#include<glm/gtc/matrix_transform.hpp>
#include"gl_util.h"

//using namespace std;
Camera::Camera(float *tmp, int width, int height) :mouseClick(false){
	Oposition = glm::vec3(tmp[0], tmp[1], tmp[2]);
	Otarget = glm::vec3(tmp[3], tmp[4], tmp[5]);
	OupVec = normalize(glm::vec3(tmp[6], tmp[7], tmp[8]));
	m_windowWidth = width;
	m_windowHeight = height;

	Init();

}
Camera::Camera():Oposition(glm::vec3(0,0,0)),Otarget(glm::vec3(0,0,-1.0f)),OupVec(glm::vec3(0,1,0)),
                 m_windowHeight(512),m_windowWidth(512),mouseClick(false){
	Init();
}

Camera::Camera(glm::vec3 &position, glm::vec3 &target_, glm::vec3 &upve_, int w, int h):mouseClick(false){

	Oposition = position;
	Otarget = target_;
	OupVec = upve_;
	m_windowWidth = w;
	m_windowHeight = h;

	Init();
}

void Camera::Init()
{


	Nposition = Oposition;
	Ntarget = Otarget;
	NupVec = OupVec;

	NtargetVec = normalize(Ntarget - Nposition);

	MouseX = m_windowWidth / 2;
	MouseY = m_windowHeight / 2;

	mouspush = false;


	glm::vec3 HTarget(NtargetVec.x, 0.0, NtargetVec.z);
	HTarget = normalize(HTarget);

	if (HTarget.z >= 0.0f)
	{
		if (HTarget.x >= 0.0f)
		{
			m_AngleH = 2*PI - asin(HTarget.z);
		}
		else
		{
			m_AngleH = PI + asin(HTarget.z);
		}
	}
	else
	{
		if (HTarget.x >= 0.0f)
		{
			m_AngleH = asin(-HTarget.z);
		}
		else
		{
			m_AngleH = 0.5*PI + asin(-HTarget.z);
		}
	}

	m_AngleV = -asin(NtargetVec.y);


	//cout << m_AngleH << " " << m_AngleV << endl;

	m_OnUpperEdge = false;
	m_OnLowerEdge = false;
	m_OnLeftEdge = false;
	m_OnRightEdge = false;
	//MouseX.x = m_windowWidth / 2;
	//m_mousePos.y = m_windowHeight / 2;
	//coutCamera();
	Update();
	//coutCamera();
	//glutWarpPointer(MouseX, MouseY);
}



void Camera::OnKeyBoard(int key){
	glm::vec3 left = cross(NtargetVec, NupVec);
	switch (key){
	case 'w':
		Nposition += myStep*NtargetVec;
		break;
	case 's':
		Nposition -= NtargetVec*myStep;
		break;
	case 'a':

		left = normalize(left);
		Nposition += left*myStep;
		Ntarget = Nposition + NtargetVec;
		break;
	case 'd':
		//Vec3f left2 = NtargetVec.Cross(NupVec);
		left = normalize(left);
		Nposition -= left*myStep;
		Ntarget = Nposition + NtargetVec;
		break;
	case 'c':
		coutCamera();
	}


}

void Camera::MoveForward(){
	Nposition += myStep*NtargetVec;
}

void Camera::MoveBack(){
	Nposition -= myStep*NtargetVec;
}

void Camera::MoveLeft(){
	glm::vec3 left = cross(NtargetVec, NupVec);
	left = glm::normalize(left);

	Nposition += left*myStep;

	Ntarget = Nposition + NtargetVec; 

}

void Camera::MoveRight(){
	glm::vec3 left = cross(NtargetVec, NupVec);
	left = glm::normalize(left);

	Nposition -= left*myStep;

	Ntarget = Nposition + NtargetVec;

}


void Moveback();
void MoveLeft();
void MoveRight();


void Camera::OnMouse(float x, float y){

	//if (!mouseClick)
		//return;

	int DeltaX = (x - MouseFx);// *0.5;
	int DeltaY = (y - MouseFy);// *0.5;

	MouseFx = x;
    MouseFy = y;
	if (!mouseClick)
		return;

	m_AngleH += ((float)DeltaX / AngleStep)*PI/180.0;
	m_AngleV += ((float)DeltaY / AngleStep)*PI / 180.0;

	if (DeltaX == 0) {
		if (x <= MARGIN) {
			//    m_AngleH -= 1.0f;
			m_OnLeftEdge = true;
		}
		else if (x >= (m_windowWidth - MARGIN)) {
			//    m_AngleH += 1.0f;
			m_OnRightEdge = true;
		}
	}
	else {
		m_OnLeftEdge = false;
		m_OnRightEdge = false;
	}

	if (DeltaY == 0) {
		if (y <= MARGIN) {
			m_OnUpperEdge = true;
		}
		else if (y >= (m_windowHeight - MARGIN)) {
			m_OnLowerEdge = true;
		}
	}
	else {
		m_OnUpperEdge = false;
		m_OnLowerEdge = false;
	}

	Update();
	//MouseX = m_windowWidth / 2;
	//MouseY = m_windowHeight / 2;
	//glutWarpPointer(MouseX, MouseY);
}


void Camera::OnRender()
{
	bool ShouldUpdate = false;

	if (m_OnLeftEdge) {
		m_AngleH -= 0.1f/180.0*PI;
		ShouldUpdate = true;
	}
	else if (m_OnRightEdge) {
		m_AngleH += 0.1f / 180.0*PI;
		ShouldUpdate = true;
	}

	if (m_OnUpperEdge) {
		if (m_AngleV > -90.0f) {
			m_AngleV -= 0.1f / 180.0*PI;
			ShouldUpdate = true;
		}
	}
	else if (m_OnLowerEdge) {
		if (m_AngleV < 90.0f) {
			m_AngleV += 0.1f / 180.0*PI;
			ShouldUpdate = true;
		}
	}

	if (ShouldUpdate) {
		Update();
	}
}


void Camera::MousePush(float x,float y){
	//mouspush = !mouspush;
	mouseClick = true;
	MouseX = x;
	MouseY = y;
}

void Camera::Update()
{
	glm::vec3 Vaxis(0.0f, 1.0f, 0.0f);

	// Rotate the view vector by the horizontal angle around the vertical axis
	glm::vec3 View(1.0f, 0.0f, 0.0f);
	 View=Rotate(View, m_AngleH, Vaxis);

	 View=normalize(View);



	// Rotate the view vector by the vertical angle around the horizontal axis
	 glm::vec3 Haxis = cross(Vaxis, View);
	 Haxis = normalize(Haxis);
	 View=Rotate(View,m_AngleV, Haxis);


	NtargetVec = View;
	NtargetVec = normalize(NtargetVec);

	NupVec = cross(NtargetVec,Haxis);

	NupVec = normalize(NupVec);

	Ntarget = Nposition + NtargetVec;
}



void Camera::Reset(){
	Nposition = Oposition;
	Ntarget = Otarget;
	NupVec = OupVec;

	NtargetVec = normalize(Ntarget - Nposition);
	Init();

}



glm::mat4 Camera::Look(){
	ViewMat = glm::lookAt(Nposition, Nposition + NtargetVec, NupVec);
	return ViewMat;
}

void Camera::coutCamera(){
	/*std::cout << Nposition.x << " " << Nposition.y << " " << Nposition.z << endl;
	cout << NtargetVec.x << " " << NtargetVec.y << " " << NtargetVec.z << endl;
	cout << NupVec.x << " " << NupVec.y << " " << NupVec.z << endl << endl;*/
}


glm::vec3 Rotate(glm::vec3 rource, float angle, glm::vec3 Axe){
	const float SinHalfAngle = sinf(angle / 2);
	const float CosHalfAngle = cosf(angle / 2);

	const float Rx = Axe.x * SinHalfAngle;
	const float Ry = Axe.y * SinHalfAngle;
	const float Rz = Axe.z * SinHalfAngle;
	const float Rw = CosHalfAngle;

	Quaternion RotationQ(Rw, Rx, Ry, Rz);


	Quaternion ConjugateQ = RotationQ.Conjugate();


	ConjugateQ.Normalize();
	Quaternion W = (RotationQ * rource) * ConjugateQ;

	glm::vec3 result(W.x, W.y, W.z);
	return result;
}

void Camera::SetWidth(int w){
	m_windowWidth = w;
}
void Camera::SetHeithg(int h){
	m_windowHeight = h;
}

void Camera::SetPostion(glm::vec3& P){
	Oposition = P;
}

void Camera::SetLookVec(glm::vec3& LV){
	Otarget = LV + Oposition;
}

void Camera::SetUpVec(glm::vec3& up){
	OupVec = up;
}

