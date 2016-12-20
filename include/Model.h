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
	enum MODEL_TYPE{
		MODEL_SPHERE,
		MODEL_MESH,
	};
	MODEL_TYPE m_Type;
	Object() :m_Postion(0.0f, 0.0f, 0.0f){};
	virtual ~Object() {};
	virtual void Render(){};
	virtual bool InterSect(GL_Ray &ray, GL_ObjIntersection &intersection,float &dmin) { return false; };
	void setPos(glm::vec3 pos){ m_Postion = pos; };
	glm::vec3 getPos(){ return m_Postion; }
private:
	glm::vec3 m_Postion;

};

class SphereObj :public Object{
public:
	SphereObj();
	SphereObj(float r, glm::vec3 &pos, GL_Material &mat);
	virtual ~SphereObj();
	void Init(float r, glm::vec3 &pos, GL_Material &mat);
	virtual bool InterSect(GL_Ray &ray, GL_ObjIntersection &intersection, float &dmin);
	//virtual void Render();
	GL_Material* GetMat(){ return m_Mat; }
	float m_Raduis;  //°ë¾¶

private:
	GL_Material *m_Mat;
	
};

class Model :public  Object{
public:
	Model();
	virtual ~Model();
	bool LoadFromFile(std::string &FileName, bool kdTree =false);
	virtual void Render();
	virtual bool InterSect(GL_Ray &ray, GL_ObjIntersection &intersection, float &dmin);
	void SetMaterial(GL_Material &Tmat);
	std::vector<Triangle*>& GetTris(){ return m_Triangles; }
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
	GL_Material *m_WholeMaterial;

	void Clear();
	bool InitFromScene(const aiScene *m_Scene,std::string &filename);
	bool InitEntity(int i, const aiMesh* m_mesh);
	bool InitMaterials(const aiScene *m_Scene, std::string &filename);
	bool BuildTriangles();
	GL_kdTree *m_kdTree;
	bool IskdTree;
	bool IsSameMat;

};




#endif
