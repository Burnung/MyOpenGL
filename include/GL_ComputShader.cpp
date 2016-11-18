#include"GL_ComputShader.h"
#include"gl_util.h"
#include<string>
#include<iostream>

//#define GL_COMPUTE_SHADER 0x91B9
#undef SAFE_FREE
#define SAFE_FREE(p) {if(p){ free(p); } p = 0; }

#define INFO_MSG(STR)  { printf("INFO (%s): %s\n",__FUNCTION__,STR); }
#define WARN_MSG(STR)  { printf("WARNING (%s): %s\n",__FUNCTION__,STR); }
#define ERROR_MSG(STR) { printf("ERROR (%s): %s\n",__FUNCTION__,STR); }

void ComputeShader::LoadShader(const char * fileName){
	
	ofstream ofile;
	ofile.open("shader/shaderInfo.txt");

	SAFE_FREE(ShaderName);
	
	if (fileName) ShaderName = _strdup(fileName);
	
	//ShaderProgram = glCreateProgram();
	string ShaderSource;
	if(! ReadFile(ShaderName,ShaderSource))
		exit(1);
	GLuint shaderObj = glCreateShader(GL_COMPUTE_SHADER);
	const char* tmpS = ShaderSource.c_str();
	glShaderSource(shaderObj, 1, &tmpS, NULL);
	glCompileShader(shaderObj);
	
	GLint success = 0;
	glGetShaderiv(shaderObj, GL_COMPILE_STATUS, &success);
	if(! success){
		GLchar infLog[1024];
		glGetShaderInfoLog(shaderObj, 1024, NULL, infLog);
		//printf("Error compiling shader tyoe, %d: %s", GL_COMPUTE_SHADER, infLog);
		ofile<<"Error compiling shader tyoe"<<GL_COMPUTE_SHADER<<infLog<<endl;
		exit(1);
	
	}
	ShaderProgram = glCreateProgram();

	glAttachShader(ShaderProgram, shaderObj);
	
	glLinkProgram(ShaderProgram);
	
	glGetProgramiv(ShaderProgram, GL_LINK_STATUS, &success);
	
	if(success == 0){
		GLchar infLog[1024];
		glGetProgramInfoLog(ShaderProgram, sizeof(infLog), NULL, infLog);
		//printf("Error Linking shader Program : %s \n",infLog);
		ofile<<"Error Linking shader Program :"<<infLog<<endl;
		exit(1);
	}
	
	//ÕýÈ·ÐÔ
	glValidateProgram(ShaderProgram);
	glGetProgramiv(ShaderProgram, GL_VALIDATE_STATUS, &success);
	if(!success){
		GLchar infLog[1024];
		glGetProgramInfoLog(ShaderProgram, sizeof(infLog), NULL, infLog);
		//printf("Invalid shader program %s \n",infLog);
		ofile<<"Invalid shader program :"<<infLog<<endl;
		exit(1);
	
	}

}

void ComputeShader::useShder(){

	glUseProgram(ShaderProgram);

}

void ComputeShader::DisUse(){
	glUseProgram(0);
}

GLuint ComputeShader::getUniform(const char *name){
	return glGetUniformLocation(ShaderProgram, name);
}
