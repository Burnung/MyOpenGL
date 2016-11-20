#include"gl_util.h"
#include<iostream>  
#include<fstream>
#include<string>
#include<Windows.h>

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

#define WIDTHBYTES(bits)    (((bits) + 31) / 32 * 4)
void SaveBMP(const char *fileName, BYTE *buf, int width, int height){

	short res1 = 0;
	short res2 = 0;
	long pixoff = 54;
	long compression = 0;
	long cmpsize = 0;
	long colors = 0;
	long impcol = 0;
	char m1 = 'B';
	char m2 = 'M';

	DWORD widthDW = WIDTHBYTES(width * 24);

	long bmfsize = sizeof(BITMAPFILEHEADER)+sizeof(BITMAPINFOHEADER)+widthDW * height;
	long byteswritten = 0;

	BITMAPINFOHEADER header;
	header.biSize = 40;
	header.biWidth = width;
	header.biHeight = height;
	header.biPlanes = 1;
	header.biBitCount = 24;
	header.biCompression = BI_RGB;
	header.biSizeImage = 0;
	header.biXPelsPerMeter = 0;
	header.biYPelsPerMeter = 0;
	header.biClrUsed = 0;
	header.biClrImportant = 0;

	FILE* fp;
	fopen_s(&fp, fileName, "wb");
	if (fp == NULL)
	{
		return;
	}

	BYTE* topdown_pixel = (BYTE*)malloc(width * height * 3 * sizeof(BYTE));

	for (unsigned int j = 0; j < height; j++)
	for (unsigned int k = 0; k < width; k++)
	{
		memcpy(&topdown_pixel[(j*width + k) * 3], &buf[(j*width + k) * 3 + 2], sizeof(BYTE));
		memcpy(&topdown_pixel[(j*width + k) * 3 + 2], &buf[(j*width + k) * 3], sizeof(BYTE));
		memcpy(&topdown_pixel[(j*width + k) * 3 + 1], &buf[(j*width + k) * 3 + 1], sizeof(BYTE));
	}

	buf = topdown_pixel;

	// Ìî³äBITMAPFILEHEADER
	fwrite((BYTE*)&(m1), 1, 1, fp); byteswritten += 1;
	fwrite((BYTE*)&(m2), 1, 1, fp); byteswritten += 1;
	fwrite((long*)&(bmfsize), 4, 1, fp);	byteswritten += 4;
	fwrite((int*)&(res1), 2, 1, fp); byteswritten += 2;
	fwrite((int*)&(res2), 2, 1, fp); byteswritten += 2;
	fwrite((long*)&(pixoff), 4, 1, fp); byteswritten += 4;

	// Ìî³äBITMAPINFOHEADER
	fwrite((BITMAPINFOHEADER*)&header, sizeof(BITMAPINFOHEADER), 1, fp);
	byteswritten += sizeof(BITMAPINFOHEADER);

	// Ìî³äÎ»Í¼Êý¾Ý
	long row = 0;
	long rowidx;
	long row_size;
	row_size = header.biWidth * 3;
	long rc;

	for (row = 0; row < header.biHeight; row++)
	{
		rowidx = (long unsigned)row * row_size;

		// Ð´Ò»ÐÐ
		rc = fwrite((void*)(buf + rowidx), row_size, 1, fp);
		if (rc != 1)
		{
			break;
		}
		byteswritten += row_size;

		for (DWORD count = row_size; count < widthDW; count++)
		{
			char dummy = 0;
			fwrite(&dummy, 1, 1, fp);
			byteswritten++;
		}

	}

	fclose(fp);
}