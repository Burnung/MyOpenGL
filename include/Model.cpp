#include"Model.h"
#include"gl_util.h"
#include<iostream>


Model::Model() :m_kdTree(nullptr), m_WholeMaterial(nullptr), IsSameMat(false){
	m_Entities.clear();
	m_Materials.clear();
	m_Triangles.clear();
}

Model::~Model(){
	Clear();
}


void Model::Clear(){

	for (auto item : m_Entities){
		if (item.Ib != INVALID_OGL_VALUE)
			glDeleteBuffers(1,&(item.Ib));
		if (item.Vb != INVALID_OGL_VALUE)
			glDeleteBuffers(1, &(item.Vb));
	}
	for (auto item : m_Materials)
		SAFERELEASE(item);
	m_Materials.clear();

	for (auto item : m_Triangles)
		SAFERELEASE(item);
	m_Triangles.clear();
	SAFERELEASE(m_kdTree);
}

bool Model::LoadFromFile(std::string &filename, bool kdTree){
	//initializeOpenGLFunctions();
	Clear();
	IskdTree = kdTree;
	bool ret(false);
	Assimp::Importer m_Importer;
	const aiScene *m_Scene = m_Importer.ReadFile(filename, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs);
	if (!m_Scene){
		char error[512];
		ZERO_MEM(error);
		sprintf_s(error, sizeof(error), "file to to open file %d", filename.c_str());
		ERROROUT(error);
		return false;//exit(1);
	}
	ret = InitFromScene(m_Scene, filename);

	if (!kdTree || !ret)
		return ret;
	m_kdTree = GL_kdTree::Build(m_Triangles,0);
	if (!m_kdTree){
		ERROROUT("error to create kdTree");
		return false;
	}
	return true;
}

bool Model::InitFromScene(const aiScene* m_Scene,std::string &filename){
	m_Entities.resize(m_Scene->mNumMeshes);
	m_Materials.resize(m_Scene->mNumMaterials);
	for (int i = 0; i < m_Materials.size(); i++)
		m_Materials[i] = new GL_Material;
	for (int i = 0; i < m_Entities.size();i++){
		const aiMesh* m_aiMesh = m_Scene->mMeshes[i];
		InitEntity(i, m_aiMesh);
	}

	return InitMaterials(m_Scene, filename);

}
bool Model::InitMaterials(const aiScene *m_Scene, std::string &Filename){

	std::string::size_type SlashIndex = Filename.find_last_of("/");
	std::string Dir;

	if (SlashIndex == std::string::npos) {
		Dir = ".";
	}
	else if (SlashIndex == 0) {
		Dir = "/";
	}
	else {
		Dir = Filename.substr(0, SlashIndex);
	}

	bool Ret = true;

	for (int i = 0; i < m_Materials.size(); i++){
		const aiMaterial* pMaterial = m_Scene->mMaterials[i];
		if (pMaterial->GetTextureCount(aiTextureType_DIFFUSE) > 0) {
			aiString Path;

			if (pMaterial->GetTexture(aiTextureType_DIFFUSE, 0, &Path, NULL, NULL, NULL, NULL, NULL) == AI_SUCCESS) {
				std::string FullPath = Dir + "/" + Path.data;

				
				if (!m_Materials[i]->LoadTexture(GL_TEXTURE_2D, FullPath)) {
					char error[512];
					ZERO_MEM(error);
					sprintf_s(error, sizeof(error), "Error Loading texture %s", FullPath.c_str());
					ERROROUT(error);
					SAFERELEASE( m_Materials[i]->m_PTexture);
					m_Materials[i]->m_PTexture = nullptr;
					Ret = false;
				}
				else {
					std::cout << "Load " << FullPath.c_str() << " ok!" << std::endl;
				}
			}
		}
		if (!m_Materials[i]->m_PTexture)
			Ret = m_Materials[i]->LoadTexture(GL_TEXTURE_2D, std::string("../Models/white.png"));
		
	}
	return Ret;
}

bool Model::InitEntity(int index, const aiMesh* m_mesh){
	m_Entities[index].MaterialIndex = m_mesh->mMaterialIndex;
	std::vector<Vertex> m_Vertex(m_mesh->mNumVertices);
	std::vector<unsigned int> m_Indices(m_mesh->mNumFaces*3);
	for (int i = 0; i < m_mesh->mNumVertices; i++){
		 glm::vec3 tmpPos(m_mesh->mVertices[i].x, m_mesh->mVertices[i].y, m_mesh->mVertices[i].z);
		 glm::vec3 tmpNormal(m_mesh->mNormals[i].x, m_mesh->mNormals[i].y, m_mesh->mNormals[i].z);
		 glm::vec2 tmpUv;
		 if (m_mesh->HasTextureCoords(0)){
			 tmpUv = glm::vec2(m_mesh->mTextureCoords[0][i].x, m_mesh->mTextureCoords[0][i].y);
		 }
		 else
			 tmpUv = glm::vec2(0, 0);
		 m_Vertex[i] = Vertex(tmpPos, tmpNormal, tmpUv);
	}
	for (int i = 0; i < m_mesh->mNumFaces; i++) {
		const aiFace& Face = m_mesh->mFaces[i];
		assert(Face.mNumIndices == 3);
		m_Indices[i * 3] = Face.mIndices[0];
		m_Indices[i * 3 + 1] = Face.mIndices[1];
		m_Indices[i * 3 + 2] = Face.mIndices[2];
		if (IskdTree){
			Triangle *TmpTri = new Triangle();
			TmpTri->m_p1 = m_Vertex[Face.mIndices[0]];
			TmpTri->m_p2 = m_Vertex[Face.mIndices[1]];
			TmpTri->m_p3 = m_Vertex[Face.mIndices[2]];
			TmpTri->m_PMaterial = m_Materials[m_Entities[index].MaterialIndex];
			m_Triangles.push_back(TmpTri);
		}

	}

	//m_Entities[index].Init(m_Vertex, m_Indices);

	m_Entities[index].NumIndices = m_Indices.size();
	GLuint VB, VA;
	glGenBuffers(1, &VB);
	glGenBuffers(1, &(m_Entities[index].Ib));
	glGenVertexArrays(1, &(m_Entities[index].VAO));

	glBindBuffer(GL_ARRAY_BUFFER, m_Entities[index].Vb);
	glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex)*m_Vertex.size(), &m_Vertex[0], GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_Entities[index].Ib);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int)*m_Indices.size(), &(m_Indices[0]), GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glBindVertexArray(m_Entities[index].VAO);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);

	glBindBuffer(GL_ARRAY_BUFFER, m_Entities[index].Vb);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), 0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (const GLvoid*)12);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (const GLvoid*)24);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_Entities[index].Ib);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	return true;

}

void Model::Render(){

	for (auto item : m_Entities){
		glBindVertexArray(item.VAO);
		int Meterial_index = item.MaterialIndex;
		if (Meterial_index < m_Materials.size() && m_Materials[Meterial_index]->m_PTexture != nullptr)
			m_Materials[Meterial_index]->m_PTexture->Bind(GL_TEXTURE0);
		glDrawElements(GL_TRIANGLES, item.NumIndices, GL_UNSIGNED_INT, 0);
	}
	glBindVertexArray(0);


}

bool Model::InterSect(GL_Ray &ray, GL_ObjIntersection &intersection,float &dmin){
	if (!IskdTree)
		return false;
	return m_kdTree->InterSect(ray, intersection, dmin);
}

Model::ModelEntity::ModelEntity()
{
	Vb = INVALID_OGL_VALUE;
	Ib = INVALID_OGL_VALUE;
	VAO = INVALID_OGL_VALUE;
	NumIndices = 0;
	MaterialIndex = GL_INVALID_MATERIAL;
};

void Model::SetMaterial(GL_Material &Tmat){
	SAFERELEASE(m_WholeMaterial);
	m_WholeMaterial = new GL_Material(Tmat);
	IsSameMat = true;

	if (IskdTree){
		for (auto item : m_Triangles)
			item->m_PMaterial = m_WholeMaterial;
	}
}

SphereObj::SphereObj() : m_Mat(nullptr), m_Raduis(0,0f){
	//m_Mat = new GL_Material();
}
SphereObj::SphereObj(float r, glm::vec3 &pos, GL_Material &mat){
	m_Raduis = r;
	setPos(pos);
	m_Mat = new GL_Material(mat);
}
SphereObj::~SphereObj(){
	SAFERELEASE(m_Mat);
}

void SphereObj::Init(float r, glm::vec3 &pos, GL_Material &mat){
	m_Raduis = r;
	setPos(pos);
	SAFERELEASE(m_Mat);
	m_Mat = new GL_Material(mat);
}

bool SphereObj::InterSect(GL_Ray &ray, GL_ObjIntersection &intersection, float &dmin){

	Sphere_Box tmpBox(getPos(), m_Raduis);
	float Dis;
	if (!tmpBox.Intersect(ray, Dis, dmin))
		return false;
	intersection.m_Dis = Dis;
	intersection.m_IsHit = true;
	dmin = Dis;
	intersection.m_Material = m_Mat;
	glm::vec3 tpos = Dis * ray.m_Dirction + ray.m_Origin;
	glm::vec3 tnormal = tpos - getPos();
	Vertex tVer(tpos, tnormal,glm::vec2(0,0));
	intersection.m_Vertex = tVer;

	return true;
}