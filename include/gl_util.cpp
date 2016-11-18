#include"gl_util.h"
#include<iostream>  
#include<fstream>
#include<string>

using namespace std;

GL_DEFINE_SINGLETON(ErrorLog);

void ErrorLog::Init(const char* fileName){
	of.open(fileName, std::ios::app);
	GetTime(of);
	of << " Program Star..." << std::endl;
}

void InitOutOf( const char*filneam){
	ErrorLog::Instance().Init(filneam);
}
void GetTime(std::ofstream &of){
	time_t t = time(0);
	char tmp[64];
	struct tm tmpTm;
	localtime_s(&tmpTm, &t);
	strftime(tmp, sizeof(tmp), "%Y-%m-%d-%H:%M:%S", &tmpTm);
	of << tmp;
}
void ErrorOut( const char *file, int line, const char*neirong){
	ErrorLog::Instance().getOf() << "in " << file << " line" << line << ": " << neirong << std::endl;
}
void EndErrorOut(){ 
	ErrorLog::Instance().End();
}

void GlfwErrorCallBack(int n, const char* descrption){
	char msg[512];
	ZERO_MEM(msg);
	sprintf_s(msg, sizeof(msg),"glfw Error %d - %s", n, descrption);
	ERROROUT(msg);
}

bool ReadFile(const char* pFileName, std::string& outFile)
{
	std::ifstream f(pFileName);

	bool ret = false;

	if (f.is_open()) {
		std::string line;
		while (getline(f, line)) {
			outFile.append(line);
			outFile.append("\n");
		}

		f.close();

		ret = true;
	}
	else {
		string error = string("can not open file") + string(pFileName);
		//const char *errPt = error.c_str();
		ERROROUT(error.c_str());
	}

	return ret;
}
