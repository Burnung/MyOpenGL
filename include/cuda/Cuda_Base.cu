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
	if (Diff.x > Diff.y && Diff.x > Diff.z){
		MidPos = Diff.x * 0.5;
		return X_AXIS;
	}
	else if (Diff.y > Diff.z){
		MidPos = Diff.y * 0.5;
		return Y_AXIS;
	}
	else{
		MidPos = Diff.z * 0.5;
		return Z_AXIS;
	}
	return X_AXIS;
}

__host__ Cuda_Material GetCudaMatFromMat(GL_Material*mat){
	Cuda_Material ret;
	ret.m_RenderType = mat->m_RenderType;
	ret.m_MaterialType = mat->m_MaterialType;

	ret.m_Refra = mat->m_Refra; //折射率  大于等于1
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