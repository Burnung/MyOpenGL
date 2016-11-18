#pragma once


#include<gl\glew.h>
#include<fstream>
using namespace std;
class ComputeShader{
protected:
	GLchar *ShaderName;
	GLuint ShaderProgram;

	ofstream ofile;
public :

	ComputeShader() {};
	//ComputeShader(ofstream &ofile_){ ofile = ofile_;};

	void LoadShader(const char*);
	
	void useShder();

	GLuint getUniform(const char *name);

	void DisUse();

	GLuint getProgram(){ return ShaderProgram; };


};