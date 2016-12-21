#ifndef CUDA_SCENE_CUH
#define CUDA_SCENE_CUH

#include "cuda_Base.cuh"

#include<vector>
class SphereObj;
class PHO_ViewPort;


const int MAX_CUDA_KDTRE_DEPTH = 10;    //kdTree的最大层数
const int MIN_CUDA_KDTRE_COUNT = 10;    //叶子节点三角形的最小数量

struct CUDA_TreeNode{
	int  isLeaf;  //为1表示叶子节点 为-1表示该结点未被使用 为 0 为普通节点
	CUDA_AABB m_AABB;
	CUDA_Triangle *m_TriList;
	
	int m_Num;   //当为叶子节点时表示三角形数量
	int *m_TriIndex;   //叶子的索引

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
	CUDA_Triangle *m_TriList;  //三角形索引
	int m_TriNum;
};

struct Cuda_TracerSet{

	int m_WindowWidth;
	int m_WindowHeight;
	glm::vec3 m_CamPos;
	glm::vec3 m_CamTarVec;
	glm::vec3 m_CamYVec;
	glm::vec3 m_CamXVec;

	float m_Width_recp;
	float m_Height_recp;
	float m_FovS;
	float m_Ratio;
	float m_X_Spacing;
	float m_X_Spacing_Half;
	float m_Y_Spacing;
	float m_Y_Spacing_Half;

};
class Cuda_Scene{

	GL_DECLARE_SINGLETON(Cuda_Scene);
public:
	int m_SphereNum;
	int m_KdTreeNum;
	int m_MatNum;
	bool AllIsOk;
	std::vector<Cuda_Sphere*> m_Dev_Spheres;
	std::vector<CUDA_KDTree*> m_Dev_KdTree;
	std::vector<CUDA_Triangle*> m_DevTrisList;
	std::vector<std::vector<int*>> m_allTriIndex;

	Cuda_Material *m_Dev_Mat;
	Cuda_TracerSet *m_Host_Tracer;
	Cuda_TracerSet *m_Dev_Tracer;

	~Cuda_Scene();
	void SetCudaSceneMat(std::vector<GL_Material*>&);
	void AddSphere(SphereObj *Sph);
	void AddKdTree(std::vector<Triangle*>& tris);

	void SetTracer(PHO_ViewPort* tmpView);
	void UpDateTracer(PHO_ViewPort* tmpView);

	void ReleaseWorld();
	void GoTrace(int samples);
};



__host__ void BuilKdTree(CUDA_KDTree *m_KDTree, CUDA_Triangle* Triangles, int TriNum);

__host__ void buildKdNode(CUDA_TreeNode* kdNode, CUDA_Triangle* Triangles, std::vector<CUDA_AABB>& allAABB, CUDA_AABB &ALLBound, std::vector<int>&TriIndex,int depth,int NodeIndex);


#endif