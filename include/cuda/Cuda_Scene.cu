#include"Cuda_Scene.cuh"
#include <helper_cuda.h>
#include <helper_string.h>
#include "../model.h"

Cuda_Scene m_CudaScene;

__shared__ int* dev_TriIndex[1 << MAX_CUDA_KDTRE_DEPTH];


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

__host__ void CUDA_SetCudaSceneMat(std::vector<GL_Material*> &mats){
	m_CudaScene.m_MatNum = mats.size();
	m_CudaScene.m_Mat = new Cuda_Material[m_CudaScene.m_MatNum];
	for (int i = 0; i < mats.size(); i++)
		m_CudaScene.m_Mat[i] = GetCudaMatFromMat(mats[i]);

	checkCudaErrors(cudaMalloc((void **)&(m_CudaScene.m_Dev_Mat), mats.size() * sizeof(Cuda_Material)));
	checkCudaErrors(cudaMemcpy(m_CudaScene.m_Dev_Mat, m_CudaScene.m_Mat, mats.size()&sizeof(Cuda_Material), cudaMemcpyHostToDevice));

	delete[] m_CudaScene.m_Mat;
	m_CudaScene.m_Mat = NULL;
}
__host__ void CUDA_AddSphere(SphereObj *Sph){
	//生成 cuda_spherer
	Cuda_Sphere* tmpSphere = new Cuda_Sphere;// GetSphereFromObj(Sph);
	
	//将其拷贝至GPU
	Cuda_Sphere *tmp_dev_Sph;
	checkCudaErrors(cudaMalloc((void**)&tmp_dev_Sph, sizeof(Cuda_Sphere)));
	checkCudaErrors(cudaMemcpy(tmp_dev_Sph, tmpSphere, sizeof(Cuda_Sphere), cudaMemcpyHostToDevice));

	m_CudaScene.m_Dev_Spheres.push_back(tmp_dev_Sph);

	delete tmpSphere;
}

__host__ void CUDA_AddKdTree(std::vector<Triangle*>& tris){
	//生成cudaTri
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
	checkCudaErrors(cudaMemcpy(dev_cudaTris, cudaTris, sizeof(CUDA_Triangle)*tris.size(),cudaMemcpyHostToDevice));

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
	m_CudaScene.m_allTriIndex.push_back(tmpindex);

	dim3 dimBlock(32, 1, 1);
	dim3 dimGrid((1 << (MAX_CUDA_KDTRE_DEPTH )) / dimBlock.x + 1,1, 1);

	int y = 1024/8;

	SetDevTree << < 8, y, 1 >> >(dev_tree, dev_TriIndex);

	ReleaseHostTree(hostTree);

	m_CudaScene.m_DevTrisList.push_back(dev_cudaTris);
	m_CudaScene.m_Dev_KdTree.push_back(dev_tree);

	ReleaseCudaWorld();

}

__host__ void ReleaseCudaTree(CUDA_KDTree *dev_tree){
	dim3 dimBlock(16, 16, 1);
	dim3 dimGrid((1 << (MAX_CUDA_KDTRE_DEPTH / 2)) / dimBlock.x + 1, (1 << (MAX_CUDA_KDTRE_DEPTH / 2)) / dimBlock.y + 1, 1);

}

__host__ void ReleaseCudaWorld(){
	for (auto item : m_CudaScene.m_Dev_Spheres)
		cudaFree(item);
	for (auto item : m_CudaScene.m_allTriIndex){
		for (auto litem : item)
			cudaFree(litem);
		item.clear();
	}
	for (auto item : m_CudaScene.m_Dev_KdTree){
		cudaFree(item);
	}
	for (auto item : m_CudaScene.m_DevTrisList)
		cudaFree(item);

	cudaFree(m_CudaScene.m_Dev_Mat);

	m_CudaScene.m_Dev_Spheres.clear();
	m_CudaScene.m_DevTrisList.clear();
	m_CudaScene.m_Dev_Spheres.clear();
	m_CudaScene.m_allTriIndex.clear();

}