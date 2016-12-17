#ifndef GL_TEXTURE_H
#define GL_TEXTURE_H
#include<string>
#include<gl\glew.h>
#include<glm\vec3.hpp>
#include<glm\vec2.hpp>
#include<glm\geometric.hpp>
#include"gl_util.h"

class GL_Material;

const float GL_eps = 1e-10;

struct Vertex{

	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec2 uv;


	Vertex() :pos(glm::vec3(0, 0, 0)), normal(glm::vec3(0,0,-1)),uv(glm::vec2(0,0)){};
	Vertex(glm::vec3 pos_, glm::vec3 normal_, glm::vec2 uv_) :pos(pos_), normal(glm::normalize(normal_)), uv(uv_)
	{
	}

};
struct GL_Ray{
	glm::vec3 m_Origin;
	glm::vec3 m_Dirction;
	GL_Ray(glm::vec3 or, glm::vec3 dir) :m_Origin(or), m_Dirction(dir){};
	GL_Ray(){
		m_Origin = glm::vec3(0.0f);
		m_Dirction = glm::vec3(0.0f, 0.0f, -1.0f);
	}
};

struct GL_ObjIntersection{
	bool m_IsHit;
	float m_Dis;
	GL_Material *m_Material;
	Vertex m_Vertex;
	GL_ObjIntersection();

	GL_ObjIntersection(bool ,Vertex&, GL_Material&, float);
	GL_ObjIntersection& operator= (GL_ObjIntersection&);
};

struct AABB_Box{
	glm::vec3 m_MinPos, m_MaxPos;  //最大点和最小点
	AABB_Box(glm::vec3 &MinPos , glm::vec3& MaxPos) : m_MinPos(MinPos), m_MaxPos(MaxPos){
	};
	AABB_Box(){
		m_MinPos = glm::vec3(0, 0, 0);
		m_MaxPos = glm::vec3(0, 0, 0);
	}
	void ExpandBox(AABB_Box& T_AABB);
	bool Intersect(GL_Ray &ray, float &dis,float min);
};

struct Sphere_Box{
	glm::vec3 m_Center;        //球心
	float m_Radius;
	Sphere_Box(glm::vec3 Center, float radius) :m_Center(Center), m_Radius(radius){};
	Sphere_Box(){
		m_Center = glm::vec3(0, 0, 0);
		m_Radius = 0;
	}
	bool Intersect(GL_Ray &ray, float &dis,float min);
};

struct Triangle{
	Vertex m_p1, m_p2, m_p3;   //按顺序的三角形
	glm::vec3 m_Normal;        //三角形的法线
	GL_Material *m_PMaterial;    //材质指针
	int m_id;
	AABB_Box getAABB();
	glm::vec3 getMidPoint();
	void ComputeNormal();
	bool Intersect(GL_Ray &ray, float &Dis,float mindis,float &u,float &v);
	void ComVertex(float u, float v, Vertex &ret);//根据位置计算三角形中点的法线 uv坐标等
};

class GL_Texture {

public:
	GL_Texture(){};
	GL_Texture(GLenum, std::string&);
	~GL_Texture(){};

	bool Load(GLenum, std::string&);
	bool Load();

	void Bind(GLenum TextureUnit);


private :
	std::string m_Filename;
	int m_Width;
	int m_Height;
	GLuint m_TexId;
	GLenum m_TexType;


};

class GL_Material{
public:
	RenderType m_RenderType;
	MaterialType m_MaterialType;
	float m_Refra;  //折射率  大于等于1
	glm::vec3 m_colour;
	glm::vec3 m_emission;

	GL_Texture *m_PTexture;
	GL_Material(MaterialType MatType = DIFF, RenderType renType = TEXTURE_TYPE);
	GL_Material(glm::vec3 &emmision, glm::vec3 &color, MaterialType MatType = DIFF, RenderType renType = TEXTURE_TYPE);
	GL_Material(GL_Material&);
	GL_Material& operator= (GL_Material&);
	bool LoadTexture(GLenum, std::string&);

};

#endif