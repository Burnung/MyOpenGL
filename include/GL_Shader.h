#ifndef GL_SHADER_H
#define GL_SHADER_H
#include<gl\glew.h>
#include"gl_util.h"
#include<string>

class glslShader{

protected:
	GLchar *vertexName ;
	GLchar *geometryName ;
	GLchar *fragmentName ;
	GLenum shaderType ;
	GLuint ShaderProgram ;


public:

	glslShader(){vertexName = 0; geometryName = 0; fragmentName =0; ShaderProgram = 0; shaderType = -1;}


	void loadShader(const char *vetexName_=NULL,const char *geometryName_=NULL,
		              const char *fragmentName_=NULL);


	void reload();

	void addShader(std::string s,GLenum id);

	void useShder();

	GLuint getUniform(const char *name);

	void DisUse();

	GLuint getProgram(){ return ShaderProgram; };

};
#endif