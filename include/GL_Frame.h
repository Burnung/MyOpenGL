#include<gl\glew.h>
#include<gl\glfw3.h>

#include"gl_util.h"
#ifndef GL_FRAME_H
#define GL_FRAME_H

class GL_CallBack;
class GL_App;

void InitGLContex(int w, int h, bool withDepth, bool withStil, const char*);

static void InitCallbacks();
void GlfwRun(GL_CallBack*);
void EndRender();
void SetInputMode(int a, int b);
int GetInputMode(int a);

class GL_CallBack{
public:
	virtual void MyKeyBoard(int key){};
	virtual void MyMouseButton(int Button, int Action, double x, double y){};
	virtual void MyMouseMove(double x, double y){};
	virtual void RenderScene(){};
};

class GL_App{
protected:
	GL_App();

	void CalcFPS();
	void RenderFPS();
	float GetRunningTime();
	//void drawText(const char *text, float x, float y, void* font);

private:
	long long m_frameTime;
	long long m_startTime;
	int m_frameCount;
	int m_fps;

};
#endif