#include "cuda_Base.cuh"
#include <helper_cuda.h>

#include "../model.h"
#include "../GL_Scene.h"

__host__ CUDA_AABB GetAABBFromTri(CUDA_Triangle& tri){

	glm::vec3 MinPos;
	glm::vec3 MaxPos;
	glm::vec3 p1 = tri.m_p1.pos;
	glm::vec3 p2 = tri.m_p2.pos;
	glm::vec3 p3 = tri.m_p3.pos;

	MinPos.x = (p1.x < p2.x) && (p1.x < p3.x) ? p1.x : p2.x < p3.x ? p2.x : p3.x;
	MinPos.y = (p1.y < p2.y) && (p1.y < p3.y) ? p1.y : p2.y < p3.y ? p2.y : p3.y;
	MinPos.z = (p1.z < p2.z) && (p1.z < p3.z) ? p1.z : p2.z < p3.z ? p2.z : p3.z;

	MaxPos.x = (p1.x > p2.x) && (p1.x > p3.x) ? p1.x : p2.x > p3.x ? p2.x : p3.x;
	MaxPos.y = (p1.y > p2.y) && (p1.y > p3.y) ? p1.y : p2.y > p3.y ? p2.y : p3.y;
	MaxPos.z = (p1.z > p2.z) && (p1.z > p3.z) ? p1.z : p2.z > p3.z ? p2.z : p3.z;
	return CUDA_AABB(MinPos, MaxPos);
}
__host__ void ExpandBox(CUDA_AABB &ret, CUDA_AABB &tmp){
	ret.m_MinPos.x = ret.m_MinPos.x < tmp.m_MinPos.x ? ret.m_MinPos.x : tmp.m_MinPos.x;
	ret.m_MinPos.y = ret.m_MinPos.y < tmp.m_MinPos.y ? ret.m_MinPos.y : tmp.m_MinPos.y;
	ret.m_MinPos.z = ret.m_MinPos.z < tmp.m_MinPos.z ? ret.m_MinPos.z : tmp.m_MinPos.z;

	ret.m_MaxPos.x = ret.m_MaxPos.x > tmp.m_MaxPos.x ? ret.m_MaxPos.x : tmp.m_MaxPos.x;
	ret.m_MaxPos.y = ret.m_MaxPos.y > tmp.m_MaxPos.y ? ret.m_MaxPos.y : tmp.m_MaxPos.y;
	ret.m_MaxPos.z = ret.m_MaxPos.z > tmp.m_MaxPos.z ? ret.m_MaxPos.z : tmp.m_MaxPos.z;
}

__host__ int GetMaxAxi(CUDA_AABB &aabb, float &MidPos){
	glm::vec3 Diff = aabb.m_MaxPos - aabb.m_MinPos;
	glm::vec3 MidVPos = (aabb.m_MaxPos + aabb.m_MinPos) * 0.5f;
	if (Diff.x > Diff.y && Diff.x > Diff.z){
		MidPos = MidVPos.x;
		return X_AXIS;
	}
	else if (Diff.y > Diff.z){
		MidPos = MidVPos.y;
		return Y_AXIS;
	}
	else{
		MidPos = MidVPos.z;
		return Z_AXIS;
	}
	return X_AXIS;
}

__host__ Cuda_Material GetCudaMatFromMat(GL_Material*mat){
	Cuda_Material ret;
	ret.m_RenderType = mat->m_RenderType;
	ret.m_MaterialType = mat->m_MaterialType;

	ret.m_Refra = mat->m_Refra; //������  ���ڵ���1
	ret.m_colour = mat->m_colour;
	ret.m_emission = mat->m_emission;
	return ret;
}
__host__ Cuda_Sphere* GetSphereFromObj(SphereObj* Obj){

	Cuda_Sphere* ret = new Cuda_Sphere(Obj->getPos(), Obj->m_Raduis);
	ret->m_MatIndex = GL_Scene::Instance().GetMatIndex(Obj->GetMat());

	return ret;
}

__host__ void GetCudaTrifromTri(CUDA_Triangle &cuda_tri, Triangle* tri){
	cuda_tri.m_p1 = tri->m_p1;
	cuda_tri.m_p2 = tri->m_p2;
	cuda_tri.m_p3 = tri->m_p3;
	cuda_tri.m_Normal = tri->m_Normal;
	cuda_tri.m_MidPoint = ComputTriMidPoint(cuda_tri);
	cuda_tri.m_MatIndex = GL_Scene::Instance().GetMatIndex(tri->m_PMaterial);
}
__host__ glm::vec3 ComputTriMidPoint(CUDA_Triangle& tri){
	glm::vec3 ret;
	for (int i = 0; i < 3; i++)
		ret[i] = (tri.m_p1.pos[i] + +tri.m_p2.pos[i] + tri.m_p2.pos[i]);
	return ret;
}

__host__ void CUDA_InitCuda(){

	int device_count = 0;
	int device = -1;
	checkCudaErrors(cudaGetDeviceCount(&device_count));

	for (int i = 0; i < device_count; ++i)
	{
		cudaDeviceProp properties;
		checkCudaErrors(cudaGetDeviceProperties(&properties, i));

		if (properties.major > 3 || (properties.major == 3 && properties.minor >= 5))
		{
			device = i;
			std::cout << "Running on GPU " << i << " (" << properties.name << ")" << std::endl;
			break;
		}

		std::cout << "GPU " << i << " (" << properties.name << ") does not support CUDA Dynamic Parallelism" << std::endl;
	}
	if (device == -1){
		std::cerr << "cdpSimpleQuicksort requires GPU devices with compute SM 3.5 or higher.  Exiting..." << std::endl;
		exit(EXIT_WAIVED);
	}
	cudaSetDevice(device);

}

__device__  bool IntersectWithAABB(CUDA_Ray *ray, CUDA_AABB* m_AABB, float tMin){

	float tmax, tmin;
	ray->m_Dirction = glm::normalize(ray->m_Dirction);
	if (abs(ray->m_Dirction.x) < 1e-10){
		tmax = INFINITY;
		tmin = -1.f * INFINITY;
	}
	else{
		float tx1 = (m_AABB->m_MinPos.x - ray->m_Origin.x) / ray->m_Dirction.x;
		float tx2 = (m_AABB->m_MaxPos.x - ray->m_Origin.x) / ray->m_Dirction.x;
		tmax = FloatMax(tx1, tx2);
		tmin = FloatMin(tx1, tx2);
	}
	//if (abs(ray.m_Dirction.y) > GL_eps){
	float ty1 = (m_AABB->m_MinPos.y - ray->m_Origin.y) / ray->m_Dirction.y;
	float ty2 = (m_AABB->m_MaxPos.y - ray->m_Origin.y) / ray->m_Dirction.y;

	tmin = FloatMax(tmin, FloatMin(ty1, ty2));
	tmax = FloatMin(tmax, FloatMax(ty1, ty2));
	//}
	//if (abs(ray.m_Dirction.z) > GL_eps){
	float tz1 = (m_AABB->m_MinPos.z - ray->m_Origin.z) / ray->m_Dirction.z;
	float tz2 = (m_AABB->m_MaxPos.z - ray->m_Origin.z) / ray->m_Dirction.z;

	tmin = FloatMax(tmin, FloatMin(tz1, tz2));
	tmax = FloatMin(tmax, FloatMax(tz1, tz2));
	//}
	//Dis = tmin;
	return tmin <= tMin && tmax >= tmin;
}

__device__ bool IntersectWithTri(CUDA_Ray *ray, CUDA_Triangle* m_Tri, float &u,float &v,float& Dis,float tMin){
	u = 0;
	v = 0;
	double t_temp = 0;
	glm::vec3 e1 = m_Tri->m_p2.pos - m_Tri->m_p1.pos;
	glm::vec3 e2 = m_Tri->m_p3.pos - m_Tri->m_p1.pos;
	glm::vec3 pvec = glm::cross(ray->m_Dirction, e2);
	double det = glm::dot(pvec, e1);
	if (det == 0) return false;
	double invDet = 1. / det;
	glm::vec3 tvec = ray->m_Origin - m_Tri->m_p1.pos;
	u = glm::dot(tvec, pvec) * invDet;
	if (u < 0 || u > 1) return false;
	glm::vec3 qvec = glm::cross(tvec, e1);
	v = glm::dot(ray->m_Dirction, qvec) * invDet;
	if (v < 0 || u + v > 1) return false;
	t_temp = glm::dot(e2, qvec) * invDet; // Set distance along ray to intersection
	if (t_temp < tMin) {
		if (t_temp > 1e-9){    // Fairly arbritarily small value, scared to change
			Dis = t_temp;
			return true;
		}
	}
	return false;

}

__device__ void ComVertexFromTriUV(float u, float v,CUDA_Triangle* tri ,Vertex *ret){
	float w = 1.0f - u - v;
	ret->uv = tri->m_p1.uv * w + tri->m_p2.uv * u + tri->m_p3.uv * v;
	ret->normal = tri->m_p1.normal * w + tri->m_p2.normal * u + tri->m_p3.normal * v;
	ret->pos = tri->m_p1.pos * w + tri->m_p2.pos * u + tri->m_p3.pos * v;
}
__device__ float FloatMin(float m1, float m2){
	if (m1 < m2)
		return m1;
	return m2;
}
__device__ float FloatMax(float m1, float m2){
	if (m1 > m2)
		return m1;
	return m2;
}
