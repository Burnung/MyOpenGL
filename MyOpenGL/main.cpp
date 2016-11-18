#include <stdlib.h>
#include<GL_Frame.h>
#include<iostream>
#include <glm/gtc/type_ptr.hpp> 
#include<glm/gtc/matrix_transform.hpp>

#include<Model.h>
#include<Camera.h>
#include<GL_Fbo.h>
#include<GL_Shader.h>
#include<GL_Blend.h>
#include<GL_Scene.h>

const int WINDOWWIDTH = 512;
const int WINDOWHEIGHT = 512;


using namespace std;

class myApp :public GL_CallBack, public GL_App{
public:
	virtual void RenderScene(){
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glm::mat4 modelMat = glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, -2));
		glm::mat4 ViewMat = m_Camera.Look();
		glm::mat4 MVP = ProMat * ViewMat *modelMat;

		//m_Fbo.BindForWrite(0);
		//m_Shader.useShder();
		m_Blend.DrawFirstPass();
		glUniformMatrix4fv(m_Blend.getFirstMVPUniform(), 1, GL_FALSE, glm::value_ptr(MVP));
		//glUniformMatrix4fv(m_MVPLocation, 1, GL_FALSE, glm::value_ptr(MVP));
		//glUniform1i(m_TexLocation, 0);
		GL_Scene::Instance().Render();
		m_Blend.DrawSecondPass();
		//glUniformMatrix4fv(m_Blend.getSecondMVPUniform(), 1, GL_FALSE, glm::value_ptr(MVP));
		CalcFPS();
		RenderFPS();
		m_Shader.DisUse();
		//m_Fbo.end();
		//m_Fbo.SaveBMP("test.bmp", 0);
	}
	void Run(){ 
		GlfwRun(this);
	}

	void Init(){
		m_Shader.loadShader("shader/Model.vert", nullptr, "shader/Model.frag");
		//m_Camera = Camera()
		ERROROUT("main.cpp is ok");
		m_Blend.Init(WINDOWWIDTH, WINDOWHEIGHT, 40);
		m_Fbo.init(1, WINDOWWIDTH, WINDOWHEIGHT);

		GL_Scene::Instance().addModel(std::string("../Models/dragon.obj"),false);
		
		//m_Model.LoadFromFile(std::string("../Models/phoenix_ugv.md2"));

		m_Shader.useShder();
		m_MVPLocation = m_Shader.getUniform("MVP");
		m_TexLocation = m_Shader.getUniform("gColorMap");
		m_Shader.DisUse();

		ProMat = glm::perspective(glm::radians(60.0f), WINDOWWIDTH /(1.0f * WINDOWHEIGHT), 0.1f, 1000.0f);
		
	}

	virtual void MyKeyBoard(int key){	
		switch (key){
		case GLFW_KEY_W:
			m_Camera.MoveForward(); break;
		case GLFW_KEY_A:
			m_Camera.MoveLeft(); break;
		case GLFW_KEY_S:
			m_Camera.MoveBack(); break;
		case GLFW_KEY_D:
			m_Camera.MoveRight(); break;
		case GLFW_KEY_ESCAPE:
		case GLFW_KEY_Q:
			EndRender(); break;
		}

	}
	virtual void MyMouseButton(int Button, int Action, double x, double y){
		if (Button != GLFW_MOUSE_BUTTON_LEFT)
			return;
		if (Action == GLFW_PRESS){
			SetInputMode(GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			m_Camera.MousePush(x, y);
			
		}
		else{
			SetInputMode(GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			m_Camera.MouseUnPress();
		}
	}

	virtual void MyMouseMove(double x, double y){
		//if (GetInputMode(GLFW_CURSOR) != GLFW_CURSOR_DISABLED)
		//	return;
		m_Camera.OnMouse(x, y);
	}

private:
	Model m_Model;
	Fbo m_Fbo;
	Camera m_Camera;
	glslShader m_Shader;
	Blend m_Blend;
	GLuint m_MVPLocation;
	GLuint m_TexLocation;
	glm::mat4 ProMat;

};


int main(){

	INITEERROROUT("log.txt");
	InitGLContex(WINDOWWIDTH, WINDOWHEIGHT, true, false, "test");
	myApp *App = new myApp;
	App->Init();
	App->Run();

	ENDERROROUT();
	//while (1);



}