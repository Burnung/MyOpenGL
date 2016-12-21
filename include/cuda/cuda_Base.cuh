
#ifndef CUDA_BASE_CUH
#define CUDA_BASE_CUH
#include <device_launch_parameters.h>
#include<glm\glm.hpp>
#include"..\gl_util.h"
#include"..\GL_BaseType.h"

#include<vector>

struct Cuda_Material;
class SphereObj;

enum AXIS{
	X_AXIS,
	Y_AXIS,
	Z_AXIS
};

/*
struct CUDA_Vertex{

glm::vec3 pos;
glm::vec3 normal;
glm::vec2 uv;


CUDA_Vertex() :pos(glm::vec3(0, 0, 0)), normal(glm::vec3(0, 0, -1)), uv(glm::vec2(0, 0)){};
CUDA_Vertex(glm::vec3 pos_, glm::vec3 normal_, glm::vec2 uv_) :pos(pos_), normal(glm::normalize(normal_)), uv(uv_)
{
}
};*/

struct CUDA_Ray{
	glm::vec3 m_Origin;
	glm::vec3 m_Dirction;
	CUDA_Ray(glm::vec3 or, glm::vec3 dir) :m_Origin(or), m_Dirction(dir){};
	CUDA_Ray(){
		m_Origin = glm::vec3(0.0f);
		m_Dirction = glm::vec3(0.0f, 0.0f, -1.0f);
	}
};

struct Cuda_Intersction{
	bool m_IsHit;
	float m_Dis;
	int m_MatID;
	Vertex m_Vertex;
	Cuda_Intersction();

	Cuda_Intersction(bool, Vertex&, Vertex&, float);
	Cuda_Intersction& operator= (Cuda_Intersction&);
};

struct CUDA_AABB{
	glm::vec3 m_MinPos, m_MaxPos;  //
	CUDA_AABB(glm::vec3 &MinPos, glm::vec3& MaxPos) : m_MinPos(MinPos), m_MaxPos(MaxPos){
	};
	CUDA_AABB(){
		m_MinPos = glm::vec3(0, 0, 0);
		m_MaxPos = glm::vec3(0, 0, 0);
	}

};

struct Cuda_Sphere{
	glm::vec3 m_Center;        //����
	float m_Radius;
	int m_MatIndex;
	Cuda_Sphere(glm::vec3 Center, float radius) :m_Center(Center), m_Radius(radius){};
	Cuda_Sphere(){
		m_Center = glm::vec3(0, 0, 0);
		m_Radius = 0;
	}
};


struct Cuda_Material{
public:
	RenderType m_RenderType;
	MaterialType m_MaterialType;
	float m_Refra;  //
	glm::vec3 m_colour;
	glm::vec3 m_emission;

	/*Cuda_Material(MaterialType MatType = DIFF, RenderType Render = TEXTURE_TYPE);
	Cuda_Material(glm::vec3 &emmision, glm::vec3 &color, MaterialType MatType = DIFF, RenderType renType = TEXTURE_TYPE);
	Cuda_Material(Cuda_Material&);
	Cuda_Material& operator= (Cuda_Material&);*/


};

struct CUDA_Triangle{
	Vertex m_p1, m_p2, m_p3;   //��˳����������
	glm::vec3 m_Normal;        //�����εķ���
	int m_MatIndex;    //��������
	glm::vec3 m_MidPoint;
	//int m_id;
};


__host__ void CUDA_InitCuda();
__host__ CUDA_AABB GetAABBFromTri(CUDA_Triangle& tri);
__host__ void ExpandBox(CUDA_AABB &ret, CUDA_AABB &tmp);
__host__ int GetMaxAxi(CUDA_AABB &aabb, float &MidPos);
__host__ Cuda_Material GetCudaMatFromMat(GL_Material*);
__host__ Cuda_Sphere* GetSphereFromObj(SphereObj*);
__host__ void GetCudaTrifromTri(CUDA_Triangle &cuda_tri, Triangle* tri);
__host__ glm::vec3 ComputTriMidPoint(CUDA_Triangle& tri);
__device__  bool IntersectWithAABB(CUDA_Ray &ray, float &dis, float min);












#endif