#include"GL_BaseType.h"
#include"gl_util.h"
#include<glm\glm.hpp>
GL_Texture::GL_Texture(GLenum TexType_, std::string&filename):m_TexType(TexType_),m_Filename(filename)
{
}

bool GL_Texture::Load(GLenum TexType_, std::string&filename){
	m_Filename = filename;
	m_TexType = TexType_;
	return Load();
}

bool GL_Texture::Load(){
	try{
		m_Image.read(m_Filename);
		m_Image.write(&m_Blob, "RGBA");
	}
	catch (Magick::Error &Error){
		char error[1024];
		ZERO_MEM(error);
		sprintf_s(error, sizeof(error), "Error Loading Texture %s :%s", m_Filename.c_str(), Error.what());
		ERROROUT(error);
		return false;
	}
	m_Height = m_Image.rows();
	m_Width = m_Image.columns();
	glGenTextures(1, &m_TexId);
	glBindTexture(m_TexType, m_TexId);
	glTexImage2D(m_TexType, 0, GL_RGBA, m_Width, m_Height,0, GL_RGBA, GL_UNSIGNED_BYTE, m_Blob.data());
	glTexParameterf(m_TexType, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(m_TexType, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(m_TexType, 0);
	return true;
}

void GL_Texture::Bind(GLenum TextureUnit){
	glActiveTexture(TextureUnit);
	glBindTexture(m_TexType, m_TexId);
}

GL_Material::GL_Material(RenderType RenderType, MaterialType MatType) :m_RenderType(RenderType), m_MaterialType(MatType)
, m_PTexture(nullptr),m_colour(0.5f,0.5f,0.5f),m_emission(0.2f,0.2f,0.2f){

}

GL_Material::GL_Material(GL_Material& tmpM){
	this->m_RenderType = tmpM.m_RenderType;
	this->m_MaterialType = tmpM.m_MaterialType;
	this->m_PTexture = tmpM.m_PTexture;
	this->m_colour = tmpM.m_colour;
	this->m_emission = tmpM.m_emission;
}

bool GL_Material::LoadTexture(GLenum TexType, std::string& filename){
	SAFERELEASE(m_PTexture);
	m_PTexture = new GL_Texture();
	return m_PTexture->Load(TexType, filename);
}

AABB_Box Triangle::getAABB(){
	return AABB_Box(
		glm::vec3(std::min(std::min(m_p1.pos.x, m_p2.pos.x), m_p2.pos.x),
		std::min(std::min(m_p1.pos.y, m_p2.pos.y), m_p2.pos.y),
		std::min(std::min(m_p1.pos.z, m_p2.pos.z), m_p2.pos.z)),
		glm::vec3(std::max(std::max(m_p1.pos.x, m_p2.pos.x), m_p2.pos.x),
		std::max(std::max(m_p1.pos.y, m_p2.pos.y), m_p2.pos.y),
		std::max(std::max(m_p1.pos.z, m_p2.pos.z), m_p2.pos.z))
		);

}

glm::vec3 Triangle::getMidPoint(){
	return (1.0f / 3)*(m_p1.pos + m_p2.pos + m_p3.pos);
}

bool Sphere_Box::Intersect(GL_Ray &ray, float &dis, float min){
	glm::vec3 Orig2Center = m_Center - ray.m_Origin;
	float Distb = glm::dot(ray.m_Dirction, Orig2Center);
	if (Distb < 0)   //球在反方向
		return false;
	float DidOrig2Center = glm::dot(Orig2Center, Orig2Center);
	float DisCenter2Ray = DidOrig2Center - Distb* Distb;
	if (DisCenter2Ray > m_Radius * m_Radius)
		return false;
	dis = sqrt(Distb*Distb + DisCenter2Ray - m_Radius*m_Radius);
	if (dis > min)
		return false;
	if( dis > GL_eps )
		return true;
	dis = sqrt(Distb*Distb - DisCenter2Ray + m_Radius*m_Radius);
	if (dis > GL_eps && dis < min)
		return true;
	return false;
		
	
}

void AABB_Box::ExpandBox(AABB_Box &TBox){
	this->m_MinPos =
		glm::vec3(std::min(TBox.m_MinPos.x, m_MinPos.x),
		std::min(TBox.m_MinPos.y, m_MinPos.y),
		std::min(TBox.m_MinPos.z, m_MinPos.z));

	this->m_MaxPos =
		glm::vec3(std::max(TBox.m_MaxPos.x, m_MaxPos.x),
		std::max(TBox.m_MaxPos.y, m_MaxPos.y),
		std::max(TBox.m_MaxPos.z, m_MaxPos.z));

}
bool AABB_Box::Intersect(GL_Ray &ray, float &Dis, float min){
	glm::vec3 TCenter = (m_MaxPos + m_MinPos) * 0.5f;
	glm::vec3 TDiff = m_MaxPos - m_MinPos;
	float raduis = TDiff.x > std::max(TDiff.y, TDiff.z) ? TDiff.x : std::max(TDiff.y, TDiff.z);
	Sphere_Box tmpSphereBox(TCenter, raduis*0.5f + 0.5f);  //球包围盒稍微大一点
	if (!tmpSphereBox.Intersect(ray, Dis, min))
		return false;
	float tmax, tmin;
	if (abs(ray.m_Dirction.x) < GL_eps){
		tmax = INFINITY;
		tmin = -1.f * INFINITY;
	}
	else{
		float tx1 = (m_MinPos.x - ray.m_Origin.x) / ray.m_Dirction.x;
		float tx2 = (m_MaxPos.x - ray.m_Origin.x) / ray.m_Dirction.x;
		tmax = tx1 >tx2 ? tx1 : tx2;
		tmin = tx1 < tx2 ? tx1 : tx2;
	}
	if (abs(ray.m_Dirction.y) > GL_eps){
		float ty1 = (m_MinPos.y - ray.m_Origin.y) / ray.m_Dirction.y;
		float ty2 = (m_MaxPos.y - ray.m_Origin.y) / ray.m_Dirction.y;
		tmax = std::min(tmax, std::max(ty1, ty2));
		tmin = std::max(tmin, std::min(ty1, ty2));
	}
	if (abs(ray.m_Dirction.z) > GL_eps){
		float tz1 = (m_MinPos.z - ray.m_Dirction.z) / ray.m_Dirction.z;
		float tz2 = (m_MaxPos.z - ray.m_Dirction.z) / ray.m_Dirction.z;
		tmax = std::min(tmax, std::max(tz1, tz2));
		tmin = std::max(tmin, std::min(tz1, tz2));
	}
	Dis = tmin;
	return tmax >= tmin;
}

GL_ObjIntersection::GL_ObjIntersection() :m_IsHit(false), m_Material(nullptr){
	m_Dis = INFINITY;
	m_Vertex = Vertex();

}
bool Triangle::Intersect(GL_Ray &ray, float &Dis, float mindis,float u,float v){
	glm::vec3 VecOA = m_p1.pos - ray.m_Origin;
	Dis = glm::dot(VecOA, ray.m_Dirction);
	if (Dis < 0 || Dis >mindis)
		return false;
	glm::vec3 intersection = ray.m_Origin + Dis * ray.m_Dirction;

	glm::vec3 e1 = m_p2.pos - m_p1.pos;
	glm::vec3 e2 = m_p3.pos - m_p1.pos;
	glm::vec3 v0 = intersection - m_p1.pos;
	float a, b, c, t1, t2;
	a = glm::dot(e1, e1);
	b = glm::dot(e2, e2);
	c = glm::dot(e1, e2);
	t1 = glm::dot(v0, e1);
	t2 = glm::dot(v0, e2);
	v = (t1 * c - t2 * a) / (c * c - a * b);
	u = (t1 * b - t2 * c) / (a * b - c * c);
	if (u > 0 && v > 0 && u + v < 1)
		return true;
	return false;

}

void Triangle::ComVertex(float u, float v, Vertex &ret){
	float w = 1.0f - u - v;
	ret.uv = m_p1.uv * w + m_p2.uv * u + m_p3.uv * v;
	ret.normal = m_p1.normal * w + m_p2.normal * u + m_p3.normal * v;
	ret.pos = m_p1.pos * w + m_p2.pos * u + m_p3.pos * v;
}

GL_ObjIntersection& GL_ObjIntersection::operator = (GL_ObjIntersection& tmp){
	this->m_Dis = tmp.m_Dis;
	this->m_IsHit = tmp.m_IsHit;
	this->m_Material = tmp.m_Material;
	this->m_Vertex = tmp.m_Vertex;
	return *this;
}