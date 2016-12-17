#ifndef GL_SCENE_H
#define GL_SCENE_H
#include"gl_util.h"
#include"Model.h"
#include"Camera.h"
#include "GL_BaseType.h"
#include<vector>
#include<map>

class GL_Scene{
GL_DECLARE_SINGLETON(GL_Scene);
public:
	typedef std::vector<Object*> ObjeceVes;
	~GL_Scene();
	void addObject(Object*);
	bool addModel(std::string &filnema,bool kdTree =false);
	bool addSphereObj(glm::vec3 &Center, float raduis, GL_Material&);
	void Render();
	glm::vec3 GoTrace(GL_Ray &ray,int depth = 0);
	void RegisterMat(GL_Material*);
	int GetMatIndex(GL_Material*);
	void PutSceneToCuda();
private:
	//Camera m_Camera;
	ObjeceVes m_Objects;
	std::vector<GL_Material*> m_MatVec;
	std::map<GL_Material*, int> m_MatToIndexMap;
	void clear();
	GL_ObjIntersection Intersect(GL_Ray &ray);
};


#endif