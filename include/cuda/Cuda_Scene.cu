#include"Cuda_Scene.cuh"
#include <helper_cuda.h>
#include <helper_string.h>
#include "../model.h"
#include"../PHO_ViewPort.h"

__shared__ int* dev_TriIndex[1 << MAX_CUDA_KDTRE_DEPTH];
__shared__ CUDA_KDTree* m_devTree[100];
__shared__ Cuda_Sphere* m_devSphere[100];
#define MAX_CUDA_TRACER_DEPTH 10

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


static __global__ void rngSetupStates(
	curandState *rngState)
{
	// determine global thread id
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	// Each threadblock gets different seed,
	// Threads within a threadblock get different sequence numbers
	curand_init(blockIdx.x + gridDim.x, threadIdx.x, 0, &rngState[tid]);
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
		ExpandBox(rightAABB, allAABB[rightIndex[i]]);

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
	m_Dev_Randstate = NULL;
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

	//生成并初始化随机数生成器
	checkCudaErrors(cudaMalloc((void**)&m_Dev_Randstate, sizeof(curandState)*m_Host_Tracer->m_WindowHeight * m_Host_Tracer->m_WindowWidth));

	rngSetupStates << <m_Host_Tracer->m_WindowHeight, m_Host_Tracer->m_WindowWidth >> >(m_Dev_Randstate);
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

__device__ void GetRay(int x, int y, Cuda_TracerSet* m_CudaTracer, CUDA_Ray *ray, curandState* m_randState){
	float x_jatter, y_jatter;
	int t_id = y * m_CudaTracer->m_WindowWidth + x;

	x_jatter = curand_normal(&m_randState[t_id]) * m_CudaTracer->m_X_Spacing - m_CudaTracer->m_X_Spacing_Half;
	y_jatter = curand_normal(&m_randState[t_id]) * m_CudaTracer->m_Y_Spacing - m_CudaTracer->m_Y_Spacing_Half;

	glm::vec3 XOffset = (2.0f * x * m_CudaTracer->m_Width_recp * m_CudaTracer->m_Ratio + x_jatter) * m_CudaTracer->m_CamXVec - m_CudaTracer->m_CamXVec * (m_CudaTracer->m_Ratio);
	glm::vec3 YOffset = (2.0f * y * m_CudaTracer->m_Height_recp + y_jatter) * m_CudaTracer->m_CamYVec - m_CudaTracer->m_CamYVec;
	glm::vec3 RetVec = glm::normalize(m_CudaTracer->m_FovS * m_CudaTracer->m_CamTarVec + XOffset + YOffset);

	ray->m_Dirction = RetVec;
	ray->m_Origin = m_CudaTracer->m_CamPos;
}

__device__ void HitSphere(CUDA_Ray *ray, Cuda_Intersction*SphereHitRet, Cuda_Sphere* Sphere){
	SphereHitRet->m_IsHit = false;
	float distance = 0;
	glm::vec3 n(0, 0, 0);

	glm::vec3 op = Sphere->m_Center - ray->m_Origin;
	double t, eps = 1e-4, b = glm::dot(op, ray->m_Dirction), det = b*b - glm::dot(op, op) + Sphere->m_Radius*(Sphere->m_Radius);
	if (det<0) return;
	else det = sqrt(det);
	distance = (t = b - det)>eps ? t : ((t = b + det) > eps ? t : 0);
	if (distance != 0){
		SphereHitRet->m_IsHit = true;
		SphereHitRet->m_MatID = Sphere->m_MatIndex;
		SphereHitRet->m_Dis = distance;
		glm::vec3 tpos = distance * (ray->m_Dirction) + ray->m_Origin;
		glm::vec3 tnormal = (tpos - Sphere->m_Center);

		SphereHitRet->m_Vertex.pos = tpos;
		SphereHitRet->m_Vertex.normal = tnormal;
	}
}
__global__ void HitTreeNodeTris(CUDA_Ray* ray, Cuda_Intersction* NodeInterSection, float *u, float *v, CUDA_TreeNode *nowNode,int tMin){

	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index >= nowNode->m_Num)
		return;
	float Dis = 0;
	int* nowTriIndex = nowNode->m_TriIndex;
	CUDA_Triangle* TriList = nowNode->m_TriList;
	NodeInterSection[index].m_IsHit = false;
	if (IntersectWithTri(ray, &(TriList[nowTriIndex[index]]), u[index], v[index], Dis, tMin)){
		NodeInterSection[index].m_IsHit = true;
		NodeInterSection[index].m_Dis = Dis;
		NodeInterSection[index].m_MatID = TriList[nowTriIndex[index]].m_MatIndex;
	}



}
__device__ bool HitkdTree(CUDA_Ray *ray, Cuda_Intersction*MeshHitRet, CUDA_TreeNode* treeNode, int nodeId){
	float tMin = 100000.0f;
	if (MeshHitRet->m_IsHit)
		tMin = MeshHitRet->m_Dis;
	if (IntersectWithAABB(ray, &(treeNode[nodeId].m_AABB), tMin))
		return false;

	if (!treeNode[nodeId].isLeaf){
		bool retL, retR;
		retL = HitkdTree(ray, MeshHitRet, treeNode, treeNode[nodeId].m_LeftIndex);
		retR = HitkdTree(ray, MeshHitRet, treeNode, treeNode[nodeId].m_RightIndex);
		return retL || retR;
	}
	//bool isInterTri(false);
	//getTriSize();
	CUDA_TreeNode *nowNode = &(treeNode[nodeId]);
	int* nowTriIndex = nowNode->m_TriIndex;
	CUDA_Triangle* TriList = nowNode->m_TriList;
	Cuda_Intersction* NodeInterSection = new Cuda_Intersction[nowNode->m_Num];
	float *u = new float[nowNode->m_Num];
	float *v = new float[nowNode->m_Num];
	float Dis = tMin;
	int hitTriId = -1;

	HitTreeNodeTris << <16, nowNode->m_Num / 16 + 1, 1 >> >(ray, NodeInterSection, u, v, nowNode, tMin);

	for (int i = 0; i < nowNode->m_Num; i++){
		if (NodeInterSection[i].m_IsHit && NodeInterSection[i].m_Dis < Dis){
			Dis = NodeInterSection[i].m_Dis;
			hitTriId = i;
		}
	}

	if (hitTriId == -1)
		return false;
	//tmin = m_tmin;
	//计算撞击点
	CUDA_Triangle hitTri = TriList[nowTriIndex[hitTriId]];
	ComVertexFromTriUV(u[hitTriId], v[hitTriId], &hitTri, &(MeshHitRet->m_Vertex));

	MeshHitRet->m_MatID = hitTri.m_MatIndex;
	MeshHitRet->m_Dis = Dis;
	MeshHitRet->m_IsHit = true;

	delete[] u;
	delete[] v;
	delete[] NodeInterSection;

	return true;
}

__global__ void HitAllSphere(CUDA_Ray *ray, Cuda_Intersction*SphereHitRet, int SphereNum){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= SphereNum)
		return;
	HitSphere(ray, &SphereHitRet[index], m_devSphere[index]);
}
__global__ void HitAllMesh(CUDA_Ray *ray, Cuda_Intersction*MeshHitRet, int MeshNum){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= MeshNum)
		return;
	MeshHitRet[index].m_Dis = 100000;
	MeshHitRet[index].m_IsHit = false;
	HitkdTree(ray, &MeshHitRet[index], m_devTree[index]->m_TreeNode,0);

}

__device__ Cuda_Intersction HitScene(CUDA_Ray* ray, int MeshNum, int SphereNum){
	Cuda_Intersction *Sphere_Inter = new Cuda_Intersction[SphereNum];
	Cuda_Intersction *Mesh_Inter = new Cuda_Intersction[MeshNum];
	HitAllSphere << <1, SphereNum, 1 >> >(ray, Sphere_Inter, SphereNum);
	HitAllMesh << <1, MeshNum, 1 >> >(ray, Mesh_Inter, MeshNum);

	Cuda_Intersction Ret_Inter;
	Ret_Inter.m_IsHit = false;
	Ret_Inter.m_Dis = 10000.0f;
	Cuda_Intersction *tmpInter = &Ret_Inter;
	for (int i = 0; i < MeshNum; i++){
		if (Mesh_Inter[i].m_IsHit && Mesh_Inter[i].m_Dis < tmpInter->m_Dis)
			tmpInter = &(Mesh_Inter[i]);
	}
	for (int i = 0; i < SphereNum; i++){
		if (Sphere_Inter[i].m_IsHit && Sphere_Inter[i].m_Dis < tmpInter->m_Dis){
			tmpInter = &(Sphere_Inter[i]);
		}
	}
	Ret_Inter.m_Dis = tmpInter->m_Dis;
	Ret_Inter.m_Vertex = tmpInter->m_Vertex;
	Ret_Inter.m_IsHit = tmpInter->m_IsHit;
	Ret_Inter.m_MatID = tmpInter->m_MatID;

}
__device__ glm::vec3 TraceScene(curandState* m_randState, CUDA_Ray* ray,Cuda_Material* m_DevMat, int MeshNum, int SphereNum,int n_depth){
	Cuda_Intersction myInter = HitScene(ray, MeshNum, SphereNum);
	if (!myInter.m_IsHit){
		//write Black color;
		return glm::vec3(0.0f, 0.0f, 0.0f);
	}
	//得到材质
	Cuda_Material *tMat = &(m_DevMat[myInter.m_MatID]);
	if (tMat->m_MaterialType == EMMI){
		//write emmesion color
		return tMat->m_emission;
	}
	//俄罗斯转盘 处理最大次数问题

	glm::vec3 T_Col = tMat->m_colour;
	float Col_Max = T_Col[0] > T_Col[1] && T_Col[0] > T_Col[2] ? T_Col[0] : T_Col[1] > T_Col[2] ? T_Col[1] : T_Col[2];
	if (n_depth >= MAX_CUDA_TRACER_DEPTH ){
		if (Col_Max < curand_normal(m_randState)){
			//write  color;
			return T_Col;// tMat->m_colour;
		}
		T_Col *= (1.0f / Col_Max);
	}
	//光线可能是在模型内部传递
	glm::vec3 orNormal = myInter.m_Vertex.normal;
	glm::vec3 corNormal = glm::dot(orNormal, ray->m_Dirction) < 0 ? orNormal : -1.0f * orNormal;
	//处理反射光线

	if (tMat->m_MaterialType == MaterialType::DIFF){  //为漫反射表面 随机生成光线

		float Theta = curand_normal(m_randState);
		float Tlen2 = curand_normal(m_randState);
		float Tlen = sqrtf(Tlen2);

		glm::vec3 x_axi = glm::normalize(glm::cross(glm::vec3(0, 1, 0), corNormal));
		glm::vec3 y_axi = glm::normalize(glm::cross(corNormal, x_axi));
		glm::vec3 t_dir = x_axi * cos(Theta) * Tlen + y_axi * sin(Theta) * Tlen + corNormal * sqrtf(1 - Tlen2);
		t_dir = glm::normalize(t_dir);
		CUDA_Ray newRay;
		newRay.m_Origin = myInter.m_Vertex.pos;
		newRay.m_Dirction = t_dir;
		return  T_Col * TraceScene(m_randState, &newRay,m_DevMat,MeshNum,SphereNum, n_depth + 1);
	}  //如果是镜面反射
	if (tMat->m_MaterialType == MaterialType::SPEC){
		//计算反射光线
		glm::vec3 TDir = glm::reflect(-(ray->m_Dirction), corNormal);
		CUDA_Ray new_Ray;
		new_Ray.m_Origin = myInter.m_Vertex.pos;
		new_Ray.m_Dirction = TDir;
		return  T_Col * TraceScene(m_randState, &new_Ray, m_DevMat, MeshNum, SphereNum, n_depth + 1);
	}
	//那么就是折射 既有镜面反射又有透射

	glm::vec3 refdir = glm::reflect(-ray->m_Dirction, corNormal); //反射光线
	CUDA_Ray refRay;
	refRay.m_Origin = myInter.m_Vertex.pos;
	refRay.m_Dirction = refdir;
	float Trefra = glm::dot(corNormal, orNormal) > 0 ? 1.0f / tMat->m_Refra : tMat->m_Refra;  //可能实在模型内部
	float cosTheta = glm::dot(ray->m_Dirction, corNormal);  //其实是-cos
	if (1 - cosTheta * cosTheta > 1.0f / (Trefra * Trefra)){  //发生全反射

		return tMat->m_colour * TraceScene(m_randState, &refRay, m_DevMat, MeshNum, SphereNum, n_depth + 1);
	}
	//计算折射光线
	//return glm::vec3(0, 0, 0);
	glm::vec3 refradir = glm::normalize(glm::refract(ray->m_Dirction, corNormal, 1.0f / Trefra));

	//使用菲涅耳公式计算 折射和反射的光线
	double f0 = (Trefra - 1)*(Trefra - 1) / ((Trefra + 1) * (Trefra + 1));
	double Pfre = f0 + (1 - f0) * pow(1 + cosTheta, 5);  //反射强度
	double pfra = 1.0 - Pfre;                            //折射强度
	float Tf = 0.25 + Pfre * 0.5f;     //俄罗斯转盘
	float AllRe = Pfre / Tf;
	float AllFra = pfra / (1 - Tf);
	CUDA_Ray FraRay;
	FraRay.m_Origin = myInter.m_Vertex.pos;
	FraRay.m_Dirction = refradir;
	if (n_depth <= 1){
		glm::vec3 col_re = (float)Pfre * TraceScene(m_randState, &refRay, m_DevMat, MeshNum, SphereNum, n_depth + 1);
		glm::vec3 col_fra = (float)pfra * TraceScene(m_randState, &FraRay, m_DevMat, MeshNum, SphereNum, n_depth + 1); 
		return  tMat->m_colour *(col_re + col_fra);
	}
	if (curand_normal(m_randState) < Tf)
		return AllRe * tMat->m_colour * TraceScene(m_randState, &refRay, m_DevMat, MeshNum, SphereNum, n_depth + 1);
	return AllFra * tMat->m_colour * TraceScene(m_randState, &FraRay, m_DevMat, MeshNum, SphereNum, n_depth + 1);

}

__global__ void TraceAll(Cuda_TracerSet *m_Tracer, curandState* m_randState, Cuda_Material*m_devMat, int MeshNum, int SphereNum, int samples){

	int nSamples = blockDim.x * blockIdx.y + blockIdx.x;
	if (nSamples >= samples)
		return;
	int x = threadIdx.x;
	int y = threadIdx.y;
	CUDA_Ray ray;
	GetRay(x, y, m_Tracer, &ray, m_randState);

	int t_id = y * m_Tracer->m_WindowWidth + x;
	TraceScene(&(m_randState[t_id]), &ray, m_devMat, MeshNum, SphereNum,0);




}


void Cuda_Scene::GoTrace(int samples){
	if (!AllIsOk){
		for (int i = 0; i < m_Dev_Spheres.size(); i++)
			m_devSphere[i] = m_Dev_Spheres[i];
		for (int i = 0; i < m_Dev_KdTree.size(); i++)
			m_devTree[i] = m_Dev_KdTree[i];
		AllIsOk = true;
	}

	dim3 dimBlock(4, samples / 4 + 1, 1);
	dim3 dimGrid(m_Host_Tracer->m_WindowWidth, m_Host_Tracer->m_WindowHeight, 1);

	TraceAll << <dimBlock, dimGrid >> >(m_Dev_Tracer, m_Dev_Randstate, m_Dev_Mat, m_Dev_KdTree.size(), m_Dev_Spheres.size(), samples);




}