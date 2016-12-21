#include"Cuda_Scene.cuh"
#include <helper_cuda.h>
#include <helper_string.h>
#include "../model.h"
#include"../PHO_ViewPort.h"

__shared__ int* dev_TriIndex[1 << MAX_CUDA_KDTRE_DEPTH];
__shared__ CUDA_KDTree* m_devTree[100];
__shared__ Cuda_Sphere* m_devSphere[100];

PHO_DEFINE_SINGLETON_NO_CTOR(Cuda_Scene);


__global__ void SetDevTree(CUDA_KDTree *dev_Tree, int**TriIndex){
	unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;

	if (index >= (1 << MAX_CUDA_KDTRE_DEPTH))
		return;
	if (dev_Tree->m_TreeNode[index].isLeaf == 1){
		dev_Tree->m_TreeNode[index].m_TriIndex = TriIndex[index];
		dev_Tree->m_TreeNode[index].m_TriList = dev_Tree->m_TriList;
		printf("%d %d\n", index, dev_Tree->m_TreeNode[index].m_Num);
	}
}

__host__ void ReleaseHostTree(CUDA_KDTree *kdTree){
	CUDA_TreeNode *tmpNode = kdTree->m_TreeNode;
	if (tmpNode){
		for (int i = 0; i < 1 << MAX_CUDA_KDTRE_DEPTH; i++){
			if (tmpNode[i].isLeaf == 1 && tmpNode[i].m_TriIndex)
				delete[] tmpNode[i].m_TriIndex;
		}
		delete[] tmpNode;
	}
	if (kdTree->m_TriList)
		delete[] kdTree->m_TriList;
	delete kdTree;
}

__host__ void BuilKdTree(CUDA_KDTree *m_KDTree, CUDA_Triangle* Triangles, int TriNum){
	bool *IsInLeaf = new bool[TriNum];
	memset(IsInLeaf, 0, sizeof(bool)* TriNum);
	m_KDTree->m_TreeNode = new CUDA_TreeNode[1 << MAX_CUDA_KDTRE_DEPTH];

	m_KDTree->m_TriNum = TriNum;
	m_KDTree->m_TriList = Triangles;
	//开始构建kdtree
	 //得到每个三角形的包围盒
	std::vector<CUDA_AABB> m_AABB(TriNum);
	for (int i = 0; i < TriNum; i++)
		m_AABB[i] = GetAABBFromTri(Triangles[i]);
	//构建总的包围盒
	CUDA_AABB AllBound = m_AABB[0];
	for (int i = 1; i < TriNum; i++)
		ExpandBox(AllBound, m_AABB[i]);
	//建立全索引
	std::vector<int> triIndx(TriNum);
	for (int i = 0; i < TriNum; i++)
		triIndx[i] = i;

	//递归构建
	buildKdNode(m_KDTree->m_TreeNode, Triangles, m_AABB, AllBound, triIndx, 1, 0);

}
__host__ void buildKdNode(CUDA_TreeNode* kdNode, CUDA_Triangle* Triangles, std::vector<CUDA_AABB>& allAABB, CUDA_AABB &ALLBound, std::vector<int>&TriIndex, int depth, int NodeIndex){
	kdNode[NodeIndex].m_AABB = ALLBound;
	//符合条件的叶子节点
	if (depth >= MAX_CUDA_KDTRE_DEPTH || TriIndex.size() <= MIN_CUDA_KDTRE_COUNT){
		kdNode[NodeIndex].isLeaf = true;
		kdNode[NodeIndex].m_Num = TriIndex.size();
		kdNode[NodeIndex].m_TriIndex = new int[TriIndex.size()];
		memcpy(kdNode[NodeIndex].m_TriIndex, &TriIndex[0], sizeof(int)*TriIndex.size());
		kdNode[NodeIndex].m_TriList = Triangles;
		return;
	}
	float MidPos;
	int Axi = GetMaxAxi(ALLBound, MidPos);
	std::vector<int> leftIndex(0);
	std::vector<int> rightIndex(0);
	switch (Axi){
	case X_AXIS:
		for (int i = 0; i < TriIndex.size(); i++)
			Triangles[TriIndex[i]].m_MidPoint.x < MidPos ? leftIndex.push_back(TriIndex[i]) : rightIndex.push_back(TriIndex[i]);
		break;
	case Y_AXIS:
		for (int i = 0; i < TriIndex.size(); i++)
			Triangles[TriIndex[i]].m_MidPoint.y < MidPos ? leftIndex.push_back(TriIndex[i]) : rightIndex.push_back(TriIndex[i]);
		break;
	case Z_AXIS:
		for (int i = 0; i < TriIndex.size(); i++)
			Triangles[TriIndex[i]].m_MidPoint.z < MidPos ? leftIndex.push_back(TriIndex[i]) : rightIndex.push_back(TriIndex[i]);
		break;
	}
	if (leftIndex.size() == TriIndex.size() || rightIndex.size() == TriIndex.size()){
		kdNode[NodeIndex].isLeaf = true;
		kdNode[NodeIndex].m_Num = TriIndex.size();
		kdNode[NodeIndex].m_TriIndex = new int[TriIndex.size()];
		memcpy(kdNode[NodeIndex].m_TriIndex, &TriIndex[0], sizeof(int)*TriIndex.size());
		kdNode[NodeIndex].m_TriList = Triangles;
		return;
	}
	CUDA_AABB leftAABB = allAABB[leftIndex[0]];
	for (int i = 1; i < leftIndex.size(); i++)
		ExpandBox(leftAABB, allAABB[leftIndex[i]]);

	CUDA_AABB rightAABB = allAABB[rightIndex[0]];
	for (int i = 1; i < rightIndex.size(); i++)
		ExpandBox(rightAABB,allAABB[rightIndex[i]]);

	kdNode[NodeIndex].m_LeftIndex = 2 * NodeIndex + 1;
	buildKdNode(kdNode, Triangles, allAABB, leftAABB, leftIndex, depth + 1, kdNode[NodeIndex].m_LeftIndex);

	kdNode[NodeIndex].m_RightIndex = 2 * NodeIndex + 2;
	buildKdNode(kdNode, Triangles, allAABB, rightAABB, rightIndex, depth + 1, kdNode[NodeIndex].m_RightIndex);

}

Cuda_Scene::Cuda_Scene(){
	m_SphereNum = 0;
	m_KdTreeNum = 0;
	m_MatNum = 0;
	m_Dev_Spheres.resize(0);
	m_Dev_KdTree.resize(0);
	m_Host_Tracer = NULL;
	m_Dev_Tracer = NULL;
	m_Dev_Mat = NULL;
	AllIsOk = false;
}

Cuda_Scene::~Cuda_Scene(){
	ReleaseWorld();
}

void Cuda_Scene::SetCudaSceneMat(std::vector<GL_Material*>&mats){
	m_MatNum = mats.size();
	Cuda_Material *TmpMat;
	TmpMat = new Cuda_Material[m_MatNum];
	for (int i = 0; i < m_MatNum; i++)
		TmpMat[i] = GetCudaMatFromMat(mats[i]);
	checkCudaErrors(cudaMalloc((void **)&(m_Dev_Mat), mats.size() * sizeof(Cuda_Material)));
	checkCudaErrors(cudaMemcpy(m_Dev_Mat, TmpMat, mats.size()&sizeof(Cuda_Material), cudaMemcpyHostToDevice));
	delete[] TmpMat;

}
void Cuda_Scene::AddSphere(SphereObj *Sph){
	//生成 cuda_spherer
	Cuda_Sphere* tmpSphere = new Cuda_Sphere;// GetSphereFromObj(Sph);

	//将其拷贝至GPU
	Cuda_Sphere *tmp_dev_Sph;
	checkCudaErrors(cudaMalloc((void**)&tmp_dev_Sph, sizeof(Cuda_Sphere)));
	checkCudaErrors(cudaMemcpy(tmp_dev_Sph, tmpSphere, sizeof(Cuda_Sphere), cudaMemcpyHostToDevice));

	m_Dev_Spheres.push_back(tmp_dev_Sph);

	delete tmpSphere;
}

void Cuda_Scene::AddKdTree(std::vector<Triangle*>& tris){
	CUDA_Triangle *cudaTris = new CUDA_Triangle[tris.size()];
	for (int i = 0; i < tris.size(); i++){
		GetCudaTrifromTri(cudaTris[i], tris[i]);
	}

	CUDA_KDTree *hostTree = new CUDA_KDTree;
	//在cpu构造kdtree
	BuilKdTree(hostTree, cudaTris, tris.size());

	//在gup申请内存
	CUDA_Triangle *dev_cudaTris;
	CUDA_KDTree *dev_cudaTree;
	checkCudaErrors(cudaMalloc((void**)&dev_cudaTris, sizeof(CUDA_Triangle)*tris.size()));
	checkCudaErrors(cudaMemcpy(dev_cudaTris, cudaTris, sizeof(CUDA_Triangle)*tris.size(), cudaMemcpyHostToDevice));

	CUDA_TreeNode *dev_treeNode;
	checkCudaErrors(cudaMalloc((void**)&dev_treeNode, sizeof(CUDA_TreeNode)*(1 << MAX_CUDA_KDTRE_DEPTH)));
	checkCudaErrors(cudaMemcpy(dev_treeNode, hostTree->m_TreeNode, sizeof(CUDA_TreeNode)*(1 << MAX_CUDA_KDTRE_DEPTH), cudaMemcpyHostToDevice));

	//kdtree
	CUDA_KDTree *dev_tree;
	checkCudaErrors(cudaMalloc((void**)&dev_tree, sizeof(CUDA_KDTree)));
	hostTree->m_TriList = dev_cudaTris;
	CUDA_TreeNode *tmpTreeNode = hostTree->m_TreeNode;
	hostTree->m_TreeNode = dev_treeNode;
	checkCudaErrors(cudaMemcpy(dev_tree, hostTree, sizeof(CUDA_KDTree), cudaMemcpyHostToDevice));
	hostTree->m_TriList = cudaTris;
	hostTree->m_TreeNode = tmpTreeNode;

	//三角形索引
	std::vector<int*> tmpindex;
	for (int i = 0; i < (1 << MAX_CUDA_KDTRE_DEPTH); i++){
		dev_TriIndex[i] = NULL;
		if (tmpTreeNode[i].isLeaf == 1){
			checkCudaErrors(cudaMalloc((void**)&dev_TriIndex[i], sizeof(int)*tmpTreeNode[i].m_Num));
			checkCudaErrors(cudaMemcpy(dev_TriIndex[i], tmpTreeNode[i].m_TriIndex, sizeof(int)*tmpTreeNode[i].m_Num, cudaMemcpyHostToDevice));
			tmpindex.push_back(dev_TriIndex[i]);
		}
	}
	m_allTriIndex.push_back(tmpindex);

	dim3 dimBlock(32, 1, 1);
	dim3 dimGrid((1 << (MAX_CUDA_KDTRE_DEPTH)) / dimBlock.x + 1, 1, 1);

	int y = 1024 / 8;

	SetDevTree << < 8, y, 1 >> >(dev_tree, dev_TriIndex);

	ReleaseHostTree(hostTree);

	m_DevTrisList.push_back(dev_cudaTris);
	m_Dev_KdTree.push_back(dev_tree);
}

void Cuda_Scene::ReleaseWorld(){
	for (auto item : m_Dev_Spheres)
		cudaFree(item);
	for (auto item : m_allTriIndex){
		for (auto litem : item)
			cudaFree(litem);
		item.clear();
	}
	for (auto item : m_Dev_KdTree){
		cudaFree(item);
	}
	for (auto item : m_DevTrisList)
		cudaFree(item);

	delete(m_Host_Tracer);
	cudaFree(m_Dev_Mat);
	cudaFree(m_Dev_Tracer);


	m_Dev_Spheres.clear();
	m_DevTrisList.clear();
	m_Dev_Spheres.clear();
	m_allTriIndex.clear();
}
void Cuda_Scene::SetTracer(PHO_ViewPort* tmpView){

	//为dev申请内存
	checkCudaErrors(cudaMalloc((void**)&m_Dev_Tracer, sizeof(Cuda_TracerSet)));

	m_Host_Tracer->m_WindowWidth = tmpView->GetWidth();
	m_Host_Tracer->m_WindowHeight = tmpView->GetHeight();

	//设置生成光线时的一些参数
	m_Host_Tracer->m_Width_recp = 1.0f / (m_Host_Tracer->m_WindowWidth *1.0f);
	m_Host_Tracer->m_Height_recp = 1.0f / (m_Host_Tracer->m_WindowHeight *1.0f);
	m_Host_Tracer->m_Ratio = (m_Host_Tracer->m_WindowWidth *1.0f) / (m_Host_Tracer->m_WindowHeight *1.0f);

	m_Host_Tracer->m_FovS = 1.0 / tanf(tmpView->GetFovy() * 0.5);
	m_Host_Tracer->m_X_Spacing = m_Host_Tracer->m_Width_recp * (m_Host_Tracer->m_Ratio) * (m_Host_Tracer->m_FovS);
	m_Host_Tracer->m_Y_Spacing = m_Host_Tracer->m_Height_recp * (m_Host_Tracer->m_FovS);
	m_Host_Tracer->m_X_Spacing_Half = m_Host_Tracer->m_X_Spacing * 0.5f;
	m_Host_Tracer->m_Y_Spacing_Half = m_Host_Tracer->m_Y_Spacing * 0.5f;
	UpDateTracer(tmpView);

}
void Cuda_Scene::UpDateTracer(PHO_ViewPort* tmpView){
	m_Host_Tracer->m_CamPos = tmpView->GetCameraPos();
	m_Host_Tracer->m_CamTarVec = glm::normalize(tmpView->GetCameraLookVec());
	m_Host_Tracer->m_CamYVec = glm::normalize(tmpView->GetCameraUpVec());
	//计算水平和垂直方向。。 x,y,z 依次即可
	m_Host_Tracer->m_CamXVec = glm::cross(m_Host_Tracer->m_CamYVec, m_Host_Tracer->m_CamTarVec);
	m_Host_Tracer->m_CamXVec = glm::normalize(m_Host_Tracer->m_CamXVec);

	m_Host_Tracer->m_CamYVec = glm::normalize(glm::cross(m_Host_Tracer->m_CamTarVec, m_Host_Tracer->m_CamXVec));

	checkCudaErrors(cudaMemcpy(m_Dev_Tracer, m_Host_Tracer, sizeof(Cuda_TracerSet), cudaMemcpyHostToDevice));


}


__global__ void TraceAll(Cuda_TracerSet *m_Tracer, int samples){

	int nSamples = blockDim.x * blockIdx.y + blockIdx.x;
	if (nSamples >= samples)
		return;
	int x = threadIdx.x;
	int y = threadIdx.y;
	CUDA_Ray ray;


	
}


void Cuda_Scene::GoTrace(int samples){
	if (!AllIsOk){
		for (int i = 0; i < m_Dev_Spheres.size(); i++)
			m_devSphere[i] = m_Dev_Spheres[i];
		for (int i = 0; i < m_Dev_KdTree.size(); i++)
			m_devTree[i] = m_Dev_KdTree[i];
		AllIsOk = true;
	}

	dim3 dimBlock(4,samples/4+1 , 1);
	dim3 dimGrid(m_Host_Tracer->m_WindowWidth, m_Host_Tracer->m_WindowHeight);





}