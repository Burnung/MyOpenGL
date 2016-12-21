#ifndef CUDA_SCENE_CUH
#define CUDA_SCENE_CUH

#include "cuda_Base.cuh"

#include<vector>
class SphereObj;
const int MAX_CUDA_KDTRE_DEPTH = 10;    //kdTree
const int MIN_CUDA_KDTRE_COUNT = 10;    //

struct CUDA_TreeNode{
	int  isLeaf;  //Ϊ1��ʾҶ�ӽڵ� Ϊ-1��ʾ�ý���δ��ʹ�� Ϊ 0 Ϊ��ͨ�ڵ�
	CUDA_AABB m_AABB;
	CUDA_Triangle *m_TriList;

	int m_Num;   //��ΪҶ�ӽڵ�ʱ��ʾ����������
	int *m_TriIndex;   //Ҷ�ӵ�����

	int m_LeftIndex;
	int m_RightIndex;
	CUDA_TreeNode(){
		isLeaf = -1;
		m_TriIndex = NULL;
		m_TriIndex = NULL;
		m_Num = 0;
	}

};

struct CUDA_KDTree{
	CUDA_TreeNode *m_TreeNode;
	CUDA_Triangle *m_TriList;  //����������
	int m_TriNum;
};
class Cuda_Scene{

	GL_DECLARE_SINGLETON(Cuda_Scene);

	int m_SphereNum;
	int m_KdTreeNum;
	int m_MatNum;

	std::vector<Cuda_Sphere*> m_Dev_Spheres;
	std::vector<CUDA_KDTree*> m_Dev_KdTree;
	std::vector<CUDA_Triangle*> m_DevTrisList;
	std::vector<std::vector<int*>> m_allTriIndex;

	Cuda_Material *m_Mat;
	Cuda_Material *m_Dev_Mat;

	Cuda_Scene(){
		m_SphereNum = 0;
		m_KdTreeNum = 0;
		m_MatNum = 0;
		m_Dev_Spheres.resize(0);
		m_Dev_KdTree.resize(0);
		m_Dev_Mat = NULL;
		m_Mat = NULL;
	}
};


__host__ void BuilKdTree(CUDA_KDTree *m_KDTree, CUDA_Triangle* Triangles, int TriNum);

__host__ void buildKdNode(CUDA_TreeNode* kdNode, CUDA_Triangle* Triangles, std::vector<CUDA_AABB>& allAABB, CUDA_AABB &ALLBound, std::vector<int>&TriIndex, int depth, int NodeIndex);

__host__ void CUDA_SetCudaSceneMat(std::vector<GL_Material*>&);

__host__ void CUDA_AddSphere(SphereObj *Sph);

__host__ void CUDA_AddKdTree(std::vector<Triangle*>& tris);

__host__ void ReleaseCudaWorld();

#endif