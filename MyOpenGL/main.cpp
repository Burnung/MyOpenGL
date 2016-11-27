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
#include<PHO_ViewPort.h>
#include<PHO_Random.h>
#include<PHO_ViewPort.h>
#include<PHO_PathTracer.h>
#include<time.h>

const int WINDOWWIDTH = 512;
const int WINDOWHEIGHT = 512;


using namespace std;

class myApp :public GL_CallBack, public GL_App{
public:
	virtual ~myApp(){
		SAFERELEASE(m_ViewPort);
	}
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
		time_t m_start, m_end;
		//SAFERELEASE(m_ViewPort);
		m_ViewPort = new PHO_ViewPort();
		m_ViewPort->SetCameraPos(glm::vec3(0, -5, 2.5));
		//m_ViewPort->SetCameraPos(glm::vec3(350, 52, 295.6));
		m_ViewPort->SetCameraLookvec(glm::vec3(0, 0, 1.0f) - glm::vec3(0, -5, 2.5));
		//m_ViewPort->SetCameraLookvec(glm::vec3(0, -0.042612, -1));
		m_ViewPort->SetCameraUpVec(glm::vec3(0, 1.0f, 0));
		m_ViewPort->SetHeight(768);
		m_ViewPort->SetWidth(1024);
		m_ViewPort->InitCamera();
		m_ViewPort->SetFovy(90);
		PHO_Random::Instance().Init();
		PHO_PahtTracer::Instance().Init(m_ViewPort);
		SphereObj *spheres[] = {//Scene: radius, position, emission, color, material
			//Master
			new SphereObj(1000, glm::vec3(0, 0, -1000), GL_Material(glm::vec3(0, 0, 0), glm::vec3(1.0f,1.0f,1.0f), DIFF)),//Left
			new SphereObj(1000, glm::vec3(-1004.f,0,0), GL_Material(glm::vec3(0, 0, 0), glm::vec3(0.85,0.4,0.4), DIFF)),//Rght
			new SphereObj(1000, glm::vec3(1004, 0, 0), GL_Material(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.4, 0.4, 0.85), DIFF)),//Back
			new SphereObj(1000, glm::vec3(0, 1006, 0), GL_Material(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f,1.0f,1.0f), DIFF)),//Frnt
			new SphereObj(100, glm::vec3(0, 0, 110), GL_Material(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), EMMI)),
			//new SphereObj(1e5, glm::vec3(1e5 + 1, 40.8, 81.6), GL_Material(glm::vec3(0, 0, 0), glm::vec3(.75, .25, .25), DIFF)),//Left
			//new SphereObj(1e5, glm::vec3(-1e5 + 99, 40.8, 81.6), GL_Material(glm::vec3(0, 0, 0), glm::vec3(.25, .25, .75), DIFF)),//Rght
			//new SphereObj(1e5, glm::vec3(50, 40.8, 1e5), GL_Material(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(.75, .75, .75), DIFF)),//Back
			//new SphereObj(1e5, glm::vec3(50, 40.8, -1e5 + 170), GL_Material(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(.75, .75, .75), DIFF)),//Frnt
			//new SphereObj(1e5, glm::vec3(50, 1e5, 81.6), GL_Material(glm::vec3(0.0, 0.0, 0.0), glm::vec3(.75, .75, .75), DIFF)),
			//new SphereObj(1e5, glm::vec3(50, -1e5 + 81.6, 81.6), GL_Material(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(.75, .75, .75), DIFF)),//Botm
			//new SphereObj(16.5, glm::vec3(27, 16.5, 47), GL_Material(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(.99, .99, .99), SPEC)),//Top
			//new SphereObj(16.5, glm::vec3(73, 16.5, 78), GL_Material(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1, 1, 1)*0.999f, REFR)),//Mirr
			//new SphereObj(600, glm::vec3(50, 681.6 - .27, 81.6), GL_Material(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f), DIFF)),//Glas
			
		};
		Model *tmpModel = new Model();
		if (!tmpModel->LoadFromFile(std::string("../Models/dragon2.obj"), true)){
			std::cout << "cant load model" << std::endl;
			return;
		}
		tmpModel->SetMaterial(GL_Material(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.9f, 0.9f, 0.9f), SPEC));
		GL_Scene::Instance().addObject(tmpModel);
		for (auto item : spheres)
			 GL_Scene::Instance().addObject(item);
		time(&m_start);
		PHO_PahtTracer::Instance().GoTrace(100);
		PHO_PahtTracer::Instance().Save2BMP("pathTracing.bmp");
		time(&m_end);
		double diff = difftime(m_end, m_start);
		int hrs = (int)diff / 3600;
		int mins = ((int)diff / 60) - (hrs * 60);
		int secs = (int)diff - (hrs * 3600) - (mins * 60);
		std::cout << std::endl << "cost time :" << hrs << "h" << mins << "m" << secs << "s" << std::endl;
		/*m_Shader.loadShader("shader/Model.vert", nullptr, "shader/Model.frag");
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

		ProMat = glm::perspective(glm::radians(60.0f), WINDOWWIDTH /(1.0f * WINDOWHEIGHT), 0.1f, 1000.0f);*/
		
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
	PHO_ViewPort *m_ViewPort;

};


int main(){

	//INITEERROROUT("log.txt");
	InitGLContex(WINDOWWIDTH, WINDOWHEIGHT, true, false, "test");
	myApp *App = new myApp;
	App->Init();
	//App->Run();

	//ENDERROROUT();
	//while (1);



}