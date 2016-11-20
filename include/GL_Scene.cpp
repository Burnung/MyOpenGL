#include"GL_Scene.h"
#include"PHO_PathTracer.h"
GL_DEFINE_SINGLETON(GL_Scene);

GL_Scene::~GL_Scene(){
	clear();
}

void GL_Scene::clear(){
	for (auto item : m_Objects){
		SAFERELEASE(item);
	}
	m_Objects.clear();
}

void GL_Scene::addObject(Object* obj){
	if (obj == nullptr)
		return;
	m_Objects.push_back(obj);
}

bool GL_Scene::addModel(std::string &filename, bool kdTree){
	Model*newModel = new Model;
	if (!newModel){
		char msg[512];
		sprintf_s(msg, "error to create model : %s", filename.c_str());
		ERROROUT(msg);
		exit(1);
	}
	if (!newModel->LoadFromFile(filename, kdTree))
		return false;
	addObject(newModel);
}

bool GL_Scene::addSphereObj(glm::vec3 &center, float raduis){
	SphereObj *tmpSphere = new SphereObj;
	if (tmpSphere == nullptr){
		ERROROUT("error to creat sphere");
		return false;
	}
	tmpSphere->Init(center, raduis);
	addObject(tmpSphere);
		
}

void GL_Scene::Render(){
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	for (auto item : m_Objects){
		if (item == nullptr)
			continue;
		item->Render();
	}
}
GL_ObjIntersection GL_Scene::Intersect(GL_Ray &ray){
	GL_ObjIntersection ret_its = GL_ObjIntersection();
	GL_ObjIntersection tmp;
	for (auto item : m_Objects){
		GL_ObjIntersection tmp;
		item->InterSect(ray, tmp, ret_its.m_Dis);
		if (tmp.m_IsHit && tmp.m_Dis < ret_its.m_Dis)
			ret_its = tmp;
	}
	return ret_its;
}

glm::vec3 GL_Scene::GoTrace(GL_Ray &ray,int n_depth){

	GL_ObjIntersection myInter =  Intersect(ray);
	if (!myInter.m_IsHit)
		return glm::vec3(0, 0, 0);

	//俄罗斯转盘 处理最大次数问题
	GL_Material *tMat = myInter.m_Material;
	glm::vec3 T_Col = tMat->m_colour;
	float Col_Max = std::max(std::max(T_Col.r, T_Col.g), T_Col.b);
	if (n_depth >= PHO_PahtTracer::Instance().GetMaxDepth()){
		if (Col_Max < PHO_Random::Instance().GetDouble() > Col_Max)
			return tMat->m_colour;
		T_Col *= (1.0f / Col_Max);
	}
	//处理反射光线
	if (tMat->m_MaterialType == MaterialType::DIFF){   //为漫反射表面 随机生成光线
		float Theta = PHO_Random::Instance().GetDouble() * PI * 0.5;
		float Tlen2 = PHO_Random::Instance().GetDouble();
		float Tlen = sqrtf(Tlen2);
		glm::vec3 t_Nor = myInter.m_Vertex.normal;
		glm::vec3 x_axi = glm::normalize(glm::cross(glm::vec3(0, 1, 0), t_Nor));
		glm::vec3 y_axi = glm::normalize(glm::cross(t_Nor, x_axi));
		glm::vec3 t_dir = x_axi * cos(Theta) * Tlen + y_axi * sin(Theta) * Tlen + t_Nor * sqrtf(1 - Tlen2);
		t_dir = glm::normalize(t_dir);
		GL_Ray newRay(myInter.m_Vertex.pos, t_dir);
		return tMat->m_emission + T_Col * GoTrace(newRay, n_depth + 1);
	}
	if (tMat->m_MaterialType == MaterialType::SPEC){
		//计算反射光线
		glm::vec3 TDir = glm::reflect(-ray.m_Dirction, myInter.m_Vertex.normal);
		GL_Ray new_Ray(myInter.m_Vertex.pos, TDir);
		return tMat->m_emission + T_Col * GoTrace(new_Ray, n_depth + 1);
	}

	return glm::vec3(0, 0, 0);

}