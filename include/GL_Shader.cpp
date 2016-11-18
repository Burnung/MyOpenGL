#include<string>
#include"GL_Shader.h"

#undef SAFE_FREE
#define SAFE_FREE(p) {if(p){ free(p); } p = 0; }

#define INFO_MSG(STR)  { printf("INFO (%s): %s\n",__FUNCTION__,STR); }
#define WARN_MSG(STR)  { printf("WARNING (%s): %s\n",__FUNCTION__,STR); }
#define ERROR_MSG(STR) { printf("ERROR (%s): %s\n",__FUNCTION__,STR); }

using namespace std;

void glslShader::loadShader(const char *vetexName_ , const char *geometryName_ ,
	                                                       const char *fragmentName_ ){
	SAFE_FREE(vertexName);
	SAFE_FREE(geometryName);
	SAFE_FREE(fragmentName);

	if (vetexName_) vertexName = _strdup(vetexName_);
	if (geometryName_) geometryName = _strdup(geometryName_);
	if (fragmentName_) fragmentName = _strdup(fragmentName_);

	 reload();

}

void glslShader::reload(){

	ShaderProgram = glCreateProgram();

	for (int i = 0; i < 3; i++){
		if (i == 0 && vertexName == 0)  continue;
		if (i == 1 && geometryName == 0)  continue;
		if (i == 2 && fragmentName == 0) continue;
		string vs, gs, fs;
		switch (i){
		case 0:
			//string vs;
			if (!ReadFile(vertexName, vs))
				exit(1);
			//cout<<"now is "<<vertexName<<endl;
			addShader(vs,GL_VERTEX_SHADER);
			break;
		case 1:  
			//string gs;
			if (!ReadFile(geometryName, gs))
				exit(0);
			addShader(gs, GL_GEOMETRY_SHADER);
			break;
		case 2:
			//string fs;
			if (!ReadFile(fragmentName, fs))
				exit(0);
			//cout<<"now is "<<fragmentName<<endl;
			addShader(fs,GL_FRAGMENT_SHADER);
		}

	}

	GLint success = 0;
	GLchar ErrorLog[1024] = { 0 };

	//链接
	glLinkProgram(ShaderProgram);
	glGetProgramiv(ShaderProgram, GL_LINK_STATUS, &success);

	if (success == 0){
		glGetProgramInfoLog(ShaderProgram, sizeof(ErrorLog), NULL, ErrorLog);
		printf("Error linking shader program : %s\n", ErrorLog);
		exit(1);
	}
	//正确性验证
	glValidateProgram(ShaderProgram);
	glGetProgramiv(ShaderProgram, GL_VALIDATE_STATUS, &success);
	if (!success){
		glGetProgramInfoLog(ShaderProgram, sizeof(ErrorLog), NULL, ErrorLog);
		printf("Invalid shader program %s \n", ErrorLog);
		exit(1);
	}

}


void glslShader::addShader(string sd,GLenum type){
	
	GLuint shaderObj = glCreateShader(type);
	if (shaderObj == 0){
		//std::cout << "Error creating shader type " << shaderType << std::endl;
		ERROROUT("Error creating shader type");
		exit(1);
	}
	GLint l;
	l = sd.length();
	const GLchar *p;
	p = sd.c_str();
	glShaderSource(shaderObj, 1, &p, &l);
	glCompileShader(shaderObj);

	GLint success;
	glGetShaderiv(shaderObj, GL_COMPILE_STATUS, &success);
	if (!success){
		GLchar infLog[1024];
		glGetShaderInfoLog(shaderObj, 1024, NULL, infLog);
		printf("Error compiling shader type,%d: %s", type, infLog);
		exit(1);
	}
	glAttachShader(ShaderProgram, shaderObj);

}

void glslShader::useShder(){
	//cout<<ShaderProgram<<endl;
	glUseProgram(ShaderProgram);
}

GLuint glslShader::getUniform(const char *name){
	return glGetUniformLocation(ShaderProgram, name);
}
void glslShader::DisUse(){
	glUseProgram(0);
}
