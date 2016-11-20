#include"GL_Fbo.h"
#include<iostream>
#include<fstream>
//#include<Windows.h>

using namespace std;

Fbo::Fbo(int Num, int width, int height) :num(Num), texDescript(width, height)
{

};
void Fbo::init(int num, int width, int height){
	this->num = num;
	texDescript = TexState(width, height);
	init();

}
void Fbo::init(){
	//申请fbo内存并绑定
	//initializeOpenGLFunctions();

	cout << "now begin to init fbo" << endl;

	//申请渲染空间
	glGenRenderbuffers(1, &depthbuffer);
	glBindRenderbuffer(GL_RENDERBUFFER_EXT, depthbuffer);
	glRenderbufferStorage(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT16, texDescript.getWidth(), texDescript.getHeight());
	glBindRenderbuffer(GL_RENDERBUFFER_EXT, 0);

	glGenFramebuffers(1, &fboId);
	glBindFramebuffer(GL_FRAMEBUFFER_EXT, fboId);
	
	glFramebufferRenderbuffer(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER_EXT, depthbuffer);
	//glEnable(GL_DEPTH_TEST);


	//生成纹理 
	glGenTextures(num, TexId);

	for (int i = 0; i < num; i++){
		glBindTexture(GL_TEXTURE_2D, TexId[i]);
		//glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, texDescript.getFilterMode());
		//glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, texDescript.getFilterMode());
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, texDescript.getClamp());
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, texDescript.getClamp());
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, texDescript.getWidth(), texDescript.getHeight(), 0, GL_RGBA, GL_FLOAT, NULL);
		glFramebufferTexture2D(GL_FRAMEBUFFER_EXT, mybuffers[i], GL_TEXTURE_2D, TexId[i], 0);
	}
	glBindTexture(GL_TEXTURE_2D,0);

	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (status != GL_FRAMEBUFFER_COMPLETE){
		cout << "init fbo failed" << endl;
		exit(-1);
	}
	//glDrawBuffers(num, mybuffers);
	//glDrawBuffer(GL_NONE);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Fbo::destory(){
	glDeleteTextures(num, TexId);
	glDeleteRenderbuffers(1, &depthbuffer);
	glDeleteFramebuffers(1, &fboId);
}

Fbo::~Fbo(){
	//destory();
}

void Fbo::begin(){
	glBindFramebuffer(GL_FRAMEBUFFER, fboId);
	glDrawBuffers(num, mybuffers);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//glEnable(GL_DEPTH_TEST);
	//glClearDepth(1.0f);
	glViewport(0, 0, texDescript.getWidth(), texDescript.getHeight());

}

void Fbo::BindForWrite(int i){
	if (i >= 8)
		cout << "error don't hava so many Tex" << endl;
	glBindFramebuffer(GL_FRAMEBUFFER, fboId);
	glDrawBuffer(GL_COLOR_ATTACHMENT0 + i);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glViewport(0, 0, texDescript.getWidth(), texDescript.getHeight());
}
void Fbo::BindForRead(int i, GLenum TextureUnit){
	glActiveTexture(TextureUnit);
	glBindTexture(GL_TEXTURE_2D, TexId[i]);
}


void Fbo::end(){
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

GLuint Fbo::getTexture(int id){
	return TexId[id];
}

void Fbo::SaveBMP(const char *filename, int id){

	glEnable(GL_TEXTURE_2D);
	BYTE *pTexture = NULL;
	pTexture = new BYTE[texDescript.getWidth()*texDescript.getHeight() * 3];
	memset(pTexture, 0, texDescript.getWidth()*texDescript.getHeight() * 3 * sizeof(BYTE));

	glBindTexture(GL_TEXTURE_2D, TexId[id]);//TexPosId   PboTex

	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, pTexture);

	int w = texDescript.getWidth();
	int h = texDescript.getHeight();
	SaveBMP(filename, pTexture, w, h);
	if (pTexture)
	   delete[] pTexture;
	glBindTexture(GL_TEXTURE_2D, 0);//TexPosId   PboTex
	//glBindFramebuffer(GL_FRAMEBUFFER, 0);
}


void Fbo::SaveBuffToBMP(const char *fileName, int id){

	glReadBuffer(GL_COLOR_ATTACHMENT0 + id);

	BYTE* pTexture = NULL;

	int width = texDescript.getWidth();
	int height = texDescript.getHeight();

	pTexture = new BYTE[texDescript.getWidth()*texDescript.getHeight() * 3];
	memset(pTexture, 0, texDescript.getWidth()*texDescript.getHeight() * 3 * sizeof(BYTE));

	float *pData = NULL;
	pData = new float[width * height * 3];
	memset(pData, 0, width * height * 3 * sizeof(float));

	glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pTexture);

	SaveBMP(fileName, pTexture, width, height);

}

#define WIDTHBYTES(bits)    (((bits) + 31) / 32 * 4)

void Fbo::SaveBMP(const char *fileName, BYTE *buf, UINT width, UINT height){

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

	// 填充BITMAPFILEHEADER
	fwrite((BYTE*)&(m1), 1, 1, fp); byteswritten += 1;
	fwrite((BYTE*)&(m2), 1, 1, fp); byteswritten += 1;
	fwrite((long*)&(bmfsize), 4, 1, fp);	byteswritten += 4;
	fwrite((int*)&(res1), 2, 1, fp); byteswritten += 2;
	fwrite((int*)&(res2), 2, 1, fp); byteswritten += 2;
	fwrite((long*)&(pixoff), 4, 1, fp); byteswritten += 4;

	// 填充BITMAPINFOHEADER
	fwrite((BITMAPINFOHEADER*)&header, sizeof(BITMAPINFOHEADER), 1, fp);
	byteswritten += sizeof(BITMAPINFOHEADER);

	// 填充位图数据
	long row = 0;
	long rowidx;
	long row_size;
	row_size = header.biWidth * 3;
	long rc;

	for (row = 0; row < header.biHeight; row++)
	{
		rowidx = (long unsigned)row * row_size;

		// 写一行
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

void Fbo::SaveFloat(const char *fileName, int id){
	float *pFloat = NULL;
	pFloat = new float[texDescript.getWidth()*texDescript.getHeight() * 4];
	memset(pFloat, 0, texDescript.getWidth()*texDescript.getHeight() * 4 * sizeof(float));
	//glReadBuffer(buffers[id]);
	int w = texDescript.getWidth();
	int h = texDescript.getHeight();
//	glReadPixels(0, 0, w, h, GL_RGB, GL_FLOAT, pFloat);
	glBindTexture(GL_TEXTURE_2D, TexId[id]);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, pFloat);
	glBindTexture(GL_TEXTURE_2D, 0);

	ofstream ofile;
	ofile.open(fileName);
	for (int i = 0; i < texDescript.getHeight(); i++){
		for (int j = 0; j < texDescript.getWidth() * 4; j+=4)
			//if(pFloat[i * texDescript.getWidth() * 3 + j] != -1)
				ofile << pFloat[i * texDescript.getWidth() * 4 + j] << " ";
		ofile << std::endl;
	}

	if (pFloat)
		delete[] pFloat;

}


void Fbo::CoutFloat(int id){
	cout << id << ": ";
	float *pFloat = NULL;
	pFloat = new float[texDescript.getWidth()*texDescript.getHeight() * 4];
	memset(pFloat, 0, texDescript.getWidth()*texDescript.getHeight() * 4 * sizeof(float));
	//glReadBuffer(buffers[id]);
	int w = texDescript.getWidth();
	int h = texDescript.getHeight();
	//	glReadPixels(0, 0, w, h, GL_RGB, GL_FLOAT, pFloat);
	glBindTexture(GL_TEXTURE_2D, TexId[id]);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, pFloat);
	glBindTexture(GL_TEXTURE_2D, 0);

	//int index = ((h / 2)*w + w / 2) * 4;
	int index = ((959-417) * w + 493) * 4;
	cout << pFloat[index] << " " << pFloat[index + 1] << " " << pFloat[index + 2] << " "<<pFloat[index + 3] << endl;

}

/*glm::vec2 Fbo::ComputError(int id){

	glm::vec2 result(0,0);
	glm::vec2 AllSum(0,0);
	glm::vec4 *pFloat = NULL;
	pFloat = new glm::vec4[texDescript.getWidth() * texDescript.getHeight()];
	memset(pFloat, 0, texDescript.getHeight() * texDescript.getWidth()* sizeof(glm::vec4));
	glBindTexture(GL_TEXTURE_2D, TexId[id]);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, pFloat);
	glBindTexture(GL_TEXTURE_2D, 0);

	for(int r = 0;r<texDescript.getHeight();r++)
		for(int c = 0;c<texDescript.getWidth();c++){
			glm::vec4 tmp = pFloat[r*texDescript.getWidth()+c];
			if(tmp.x == 1.0){
				AllSum.x += 1.0f;
				if(tmp.y == 0.0)
					result.x += 1.0f;
			}
			else if (tmp.x == 0.0){
				AllSum += 1.0f;
				if(tmp.y == 1.0)
					result.y += 1.0f;
			}
		}

		result.x = result.x/AllSum.x;
		result.y = result.y/ AllSum.y;

		delete [] pFloat;

		return result;

}*/

int Fbo::ComputNum(int id, int lastNum){
	float *pFloat = NULL;
	pFloat = new float[texDescript.getWidth() * texDescript.getHeight() *4];
	memset(pFloat, 0, texDescript.getHeight() * texDescript.getWidth() *4 * sizeof(float));
	glBindTexture(GL_TEXTURE_2D, TexId[id]);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, pFloat);
	glBindTexture(GL_TEXTURE_2D, 0);

	int sum = 0;
	for(int i = 0; i < texDescript.getHeight(); i++)
		for(int j = 0; j<texDescript.getWidth(); j++)
			if( pFloat[(i * texDescript.getWidth() +j) * 4] ==1)
				sum++;
	delete [] pFloat;
	return sum + lastNum;
}