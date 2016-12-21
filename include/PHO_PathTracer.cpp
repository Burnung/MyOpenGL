#include"PHO_PathTracer.h"
#include"PHO_ViewPort.h"
#include"GL_Scene.h"
#include"GL_Scene.h"

const int PATHTRACER_MAX_DEPTH = 9;
GL_DEFINE_SINGLETON(PHO_PahtTracer);

PHO_PahtTracer::~PHO_PahtTracer(){
	Clear();
}

void PHO_PahtTracer::Clear(){
	m_ViewPort = nullptr;
	SAFERELEASE(m_RetBMP);
}

void PHO_PahtTracer::Init(PHO_ViewPort *ViewPort){
	Clear();
	m_ViewPort = ViewPort;

	//Update 只是更新Camera信息
	m_WindowWidth = m_ViewPort->GetWidth();
	m_WindowHeight = m_ViewPort->GetHeight();

	//设置生成光线时的一些参数
	m_Width_recp = 1.0f / (m_WindowWidth *1.0f);
	m_Height_recp = 1.0f / (m_WindowHeight *1.0f);
	m_Ratio = (m_WindowWidth *1.0f) / (m_WindowHeight *1.0f);

	m_FovS = 1.0/tanf(m_ViewPort->GetFovy() * 0.5);
	m_X_Spacing = m_Width_recp * m_Ratio * m_FovS;
	m_Y_Spacing = m_Height_recp * m_FovS;
	m_X_Spacing_Half = m_X_Spacing * 0.5f;
	m_Y_Spacing_Half = m_Y_Spacing * 0.5f;

	Update();
	//malloc Ret buffer
	m_RetBMP = new BYTE[m_WindowHeight * m_WindowWidth * 3];

#ifdef PATHTRACER_USE_GPU
	GL_Scene::Instance().PutSceneToCuda();
#endif
}

void PHO_PahtTracer::Update(){

	m_CamPos = m_ViewPort->GetCameraPos();
	m_CamTarVec = glm::normalize(m_ViewPort->GetCameraLookVec());
	m_CamYVec = glm::normalize(m_ViewPort->GetCameraUpVec());
	//计算水平和垂直方向。。 x,y,z 依次即可
	m_CamXVec = glm::cross(m_CamYVec, m_CamTarVec);
	m_CamXVec = glm::normalize(m_CamXVec);

	m_CamYVec = glm::normalize(glm::cross(m_CamTarVec, m_CamXVec));
	//m_CamXVec = -1.0f * m_CamXVec;
}

GL_Ray PHO_PahtTracer::GetRay(int x,int y){
	float x_jatter, y_jatter;

	x_jatter = PHO_Random::Instance().GetDouble() * m_X_Spacing - m_X_Spacing_Half;
	y_jatter = PHO_Random::Instance().GetDouble() * m_Y_Spacing - m_Y_Spacing_Half;

	glm::vec3 XOffset = (2.0f * x * m_Width_recp * m_Ratio + x_jatter) * m_CamXVec - m_CamXVec * m_Ratio;
	glm::vec3 YOffset = (2.0f * y * m_Height_recp + y_jatter) * m_CamYVec - m_CamYVec;
	glm::vec3 RetVec = glm::normalize(m_FovS * m_CamTarVec + XOffset + YOffset);
	
	return GL_Ray(m_CamPos, RetVec);

}
void PHO_PahtTracer::WriteColor(glm::vec3 &col, int x, int y){
	if (x < 0 || y < 0)
		return;
	int Tpos = (y * m_WindowWidth + x) * 3;
	for (int i = 0; i < 3; i++){
		PHO_Clamp(0.0f, 1.0f, col[i]);
		m_RetBMP[Tpos + i] = col[i] * 255;
	}
}

void PHO_PahtTracer::ClearRetBuffer(){
	if (m_RetBMP == nullptr)
		m_RetBMP = new BYTE[m_WindowHeight * m_WindowWidth * 3];
	memset(m_RetBMP, 0, sizeof(BYTE)* m_WindowHeight * m_WindowWidth * 3);
}

void PHO_PahtTracer::GoTrace(int samples){
	ClearRetBuffer();
	if (!m_ViewPort){
		ERROROUT("Not Set ViewPort");
		exit(1);
	}
	float Inv_Sam = 1.0f / (1.0f * samples);
	Update();

#ifndef PATHTRACER_USE_GPU                                   //使用CPU 
#pragma omp parallel for schedule(dynamic, 1)                 //openPM
	for (int y = 0; y </*420*/ m_WindowHeight; y++){
		
		std::cout << "\r now is " << y / (1.0f * m_WindowHeight) * 100 << "%";
		//构造随机器种子
		unsigned short TSeed[3] = { 0, 0, y*y*y };
		//用之前一定要Init
		PHO_Random::Instance().SetSeed(TSeed);
		for (int x = 0; x < /*508*/ m_WindowWidth ; x++){
			if (x == 0 && y == m_WindowHeight -1)
				int tmpaaa = 1;
			glm::vec3 TCol(0.0f, 0.0f, 0.0f);
			//采样次数
			for (int t = 0; t < samples; t++){
				GL_Ray TRay = GetRay(x, y);
				TCol += GL_Scene::Instance().GoTrace(TRay);
			}
			TCol = TCol * Inv_Sam;
			WriteColor(TCol,x,y);
		}
	}

#else


#endif

}

void PHO_PahtTracer::Save2BMP(const char* filename){
	SaveBMP(filename, m_RetBMP, m_WindowWidth, m_WindowHeight);
}

int PHO_PahtTracer::GetMaxDepth(){
	return PATHTRACER_MAX_DEPTH;
}