#ifndef GL_SCENE_H
#define GL_SCENE_H
#include"gl_util.h"
#include"Model.h"
#include"Camera.h"
#include<vector>

class GL_Scene{
GL_DECLARE_SINGLETON(GL_Scene);
public:
	typedef std::vector<Object*> ObjeceVes;
	~GL_Scene();
	void addObject(Object*);
	bool addModel(std::string &filnema,bool kdTree =false);
	bool addSphereObj(glm::vec3 &Center, float raduis);
	void Render();

private:
	//Camera m_Camera;
	ObjeceVes m_Objects;
	void clear();
};


#endif