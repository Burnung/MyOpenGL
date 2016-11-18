#ifndef MODEL_H
#define MODEL_H
#include<vector>
#include<gl\glew.h>
//#include<QtGui\QOpenGLFunctions_4_3_Compatibility>
#include<glm\vec3.hpp>
#include<glm\vec2.hpp>
#include<assimp\Importer.hpp>

#include<assimp\scene.h>
#include <assimp\postprocess.h>
//#include <QtGui/QOpenGLShaderProgram>
#include"GL_BaseType.h"
#include"GL_kdTree.h"

class Object{
public:
	Object() :m_Postion(0.0f, 0.0f, 0.0f){};
	virtual ~Object() {};
	virtual bool LoadFromFile(std::string &filename) { return true; };
	virtual void Init(glm::vec3 &pos, float r) {  };
	virtual void Render(){};
	virtual bool InterSect(GL_Ray &ray, GL_ObjIntersection &intersection,float &dmin) { return false; };
	void setPos(glm::vec3 pos){ m_Postion = pos; };
	glm::vec3 getPos(){ return m_Postion; }
private:
	glm::vec3 m_Postion;

};

class SphereObj :public Object{
public:
	virtual ~SphereObj() {};
	virtual void Init(glm::vec3 &pos, float r);
	virtual bool InterSect(GL_Ray &ray, GL_ObjIntersection &intersection, float &dmin);
	//virtual void Render();

	float m_Raduis;  //�뾶
	
};

class Model :public  Object{
public:
	Model() :m_kdTree(nullptr){};
	virtual ~Model();
	virtual bool LoadFromFile(std::string &FileName, bool kdTree =false);
	virtual void Render();
	virtual bool InterSect(GL_Ray &ray, GL_ObjIntersection &intersection, float &dmin);

	struct ModelEntity{

		GLuint Vb, Ib;
		GLuint VAO;

		int NumIndices;

		ModelEntity();
		~ModelEntity(){};
		//void Init(const std::vector<Vertex>&, const std::vector < unsigned int>&);

		unsigned int index;
		unsigned int MaterialIndex;
	};

private:
	std::vector<GL_Material*> m_Materials;
	std::vector<ModelEntity> m_Entities;
	std::vector<Triangle*> m_Triangles;

	void Clear();
	bool InitFromScene(const aiScene *m_Scene,std::string &filename);
	bool InitEntity(int i, const aiMesh* m_mesh);
	bool InitMaterials(const aiScene *m_Scene, std::string &filename);
	bool BuildTriangles();
	GL_kdTree *m_kdTree;
	bool IskdTree;

};




#endif