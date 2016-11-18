#include"GL_Scene.h"
GL_DEFINE_SINGLETON(GL_Scene);

GL_Scene::~GL_Scene(){
	clear();
}

void GL_Scene::clear(){
	for (auto item : m_Objects){
		SAFERELEASE(item);
	}
	m_Objects.clear();
}

void GL_Scene::addObject(Object* obj){
	if (obj == nullptr)
		return;
	m_Objects.push_back(obj);
}

bool GL_Scene::addModel(std::string &filename, bool kdTree){
	Model*newModel = new Model;
	if (!newModel){
		char msg[512];
		sprintf_s(msg, "error to create model : %s", filename.c_str());
		ERROROUT(msg);
		exit(1);
	}
	if (!newModel->LoadFromFile(filename, kdTree))
		return false;
	addObject(newModel);
}

bool GL_Scene::addSphereObj(glm::vec3 &center, float raduis){
	SphereObj *tmpSphere = new SphereObj;
	if (tmpSphere == nullptr){
		ERROROUT("error to creat sphere");
		return false;
	}
	tmpSphere->Init(center, raduis);
	addObject(tmpSphere);
		
}

void GL_Scene::Render(){
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	for (auto item : m_Objects){
		if (item == nullptr)
			continue;
		item->Render();
	}
}