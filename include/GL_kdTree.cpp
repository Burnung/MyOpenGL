#include"GL_kdTree.h"

GL_kdTree::GL_kdTree():IsLeaf(false),m_PLeftChild(nullptr),m_PRightChild(nullptr){
	m_Triangles = std::vector<Triangle*>();
}
GL_kdTree::~GL_kdTree(){
	if (m_PLeftChild)
		SAFERELEASE(m_PLeftChild);
	if (m_PRightChild)
		SAFERELEASE(m_PRightChild);
	m_Triangles.clear();
}

int GL_kdTree::getTriSize(){
	float dis;
//	if (!this->m_AABB.Intersect(GL_Ray(glm::vec3(0, -5.0f, 2.5f), glm::vec3(-0.223117277, 0.947524309, -0.228945851)), dis, 1000))
	//	return 0;
	if (!IsLeaf)
		return m_PLeftChild->getTriSize() + m_PRightChild->getTriSize();
	for (auto item : m_Triangles){
		if (item->m_id == 29772){
			bool tmp = this->m_AABB.Intersect(GL_Ray(glm::vec3(0, -5.0f, 2.5f), glm::vec3(-0.223117277, 0.947524309, -0.228945851)), dis,1000);
			int a = 1;
		}
	}
	return m_Triangles.size();
}
GL_kdTree* GL_kdTree::Build(std::vector<Triangle*>&triangles, int depth){

	GL_kdTree *nowkdNode = new GL_kdTree;
	if ( triangles.size() == 0)
		return nowkdNode;

	nowkdNode->m_AABB = triangles[0]->getAABB();
	nowkdNode->m_Mat = triangles[0]->m_PMaterial;
	for (auto item : triangles) 
		nowkdNode->m_AABB.ExpandBox(item->getAABB());
	if (triangles.size() <= MIN_COUNT || depth >= MAX_DEPTH){
		nowkdNode->IsLeaf = true;
		nowkdNode->m_Triangles = triangles;
		return nowkdNode;
	}
	float MidP = 0;
	//得到最长的轴
	int  nAXIS = nowkdNode->getAxis(nowkdNode->m_AABB,MidP);
	std::vector<Triangle*> leftTris;
	std::vector<Triangle*> rightTris;
	leftTris.clear();
	rightTris.clear();
	for (auto item : triangles){
		switch (nAXIS)
		{
		case X_AXIS:
			item->getMidPoint().x > MidP ? rightTris.push_back(item) : leftTris.push_back(item);
			break;
		case Y_AXIS:
			item->getMidPoint().y > MidP ? rightTris.push_back(item) : leftTris.push_back(item);
			break;
		case Z_AXIS:
			item->getMidPoint().z > MidP ? rightTris.push_back(item) : leftTris.push_back(item);
			break;
		default:
			break;
		}
	}
	if (leftTris.size() == triangles.size() || rightTris.size() == triangles.size()){
		nowkdNode->IsLeaf = true;
		nowkdNode->m_Triangles = triangles;
		return nowkdNode;
	}
	nowkdNode->m_PLeftChild = GL_kdTree::Build(leftTris, depth + 1);
	nowkdNode->m_PRightChild = GL_kdTree::Build(rightTris, depth + 1);

	return nowkdNode;

}

int GL_kdTree::getAxis(std::vector<Triangle*> triangles){
	double meanX = 0, meanY = 0, meanZ = 0;
	double DiffX = 0, DiffY = 0, DiffZ = 0;
	for (auto item : triangles)
		meanX += (item->m_p1.pos.x + item->m_p2.pos.x + item->m_p3.pos.x);
	return X_AXIS;

}
int GL_kdTree::getAxis(AABB_Box &tmpAABB,float &MidP){
	glm::vec3 Diff = tmpAABB.m_MaxPos - tmpAABB.m_MinPos;
	if (Diff.x > std::max(Diff.y, Diff.z)){
		MidP = (m_AABB.m_MaxPos.x + m_AABB.m_MinPos.x)* 0.5f;
		return X_AXIS;
	}
	else if (Diff.y > Diff.z){
		MidP = (m_AABB.m_MaxPos.y + m_AABB.m_MinPos.y) * 0.5f;
		return Y_AXIS;
	}
	else{
		MidP = (m_AABB.m_MaxPos.z + m_AABB.m_MinPos.z) * 0.5f;
		return Z_AXIS;
	}
	return X_AXIS;
}

bool GL_kdTree::InterSect(GL_Ray &ray, GL_ObjIntersection &intersection,float &tmin){

	float Dis = intersection.m_Dis;
	if (!m_AABB.Intersect(ray, Dis, tmin))
		 return false;

	if (!IsLeaf){
		bool retL, retR;
		retL = m_PLeftChild->InterSect(ray, intersection, tmin);
		retR = m_PRightChild->InterSect(ray, intersection, tmin);
		return retL || retR;
	}
	//bool isInterTri(false);
	//getTriSize();
	float m_tmin = tmin;
	Triangle *hitTri(nullptr);
	float retU = 0, retV = 0;
	float u = 0, v = 0;
	for (auto item : m_Triangles){
		if (item->Intersect(ray, Dis, tmin, u, v)){
			tmin = Dis;
			hitTri = item;
			retU = u;
			retV = v;
		}
	}
	if (hitTri == nullptr)
		return false; 
	//tmin = m_tmin;
	//计算撞击点
	Vertex HitVert;
	hitTri->ComVertex(retU, retV, HitVert);
	intersection.m_Material = hitTri->m_PMaterial;
	intersection.m_Dis = Dis;
	intersection.m_Vertex = HitVert;
	intersection.m_IsHit = true;

	return true;

}