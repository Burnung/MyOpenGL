#include"OpenGLWindow.h"

#include<QtCore\QcoreApplication>
#include<QtGui\QOpenGLContext>
#include <QtGui\QPainter>
#include <QOpenGLPaintDevice>
#include<QtCore\QDebug>
//#include <QtGui\qopenglfunctions_4_4_core.h>

OpenGLWindow::OpenGLWindow(QWindow *parent) :QWindow(parent),
											 m_update_pending(false),
											 m_animating(false),
											 m_context(0),
											m_device(0)
{

	setSurfaceType(QWindow::OpenGLSurface);

}
OpenGLWindow::~OpenGLWindow(){
	if (m_device)
		delete m_device;
}
void OpenGLWindow::render(QPainter *painter){

	Q_UNUSED(painter);

}
void OpenGLWindow::initialize(){

}
void OpenGLWindow::render(){
	if (!m_device)
		m_device = new QOpenGLPaintDevice;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	glClearColor(1.0f, 0.0f, 1.0f,0.0f);

	m_device->setSize(size());

	QPainter painter(m_device);
	render(&painter);
}

void OpenGLWindow::renderLater(){

	if (!m_update_pending){
		m_update_pending = true;
		QCoreApplication::postEvent(this, new QEvent(QEvent::UpdateRequest));
	}
}

bool OpenGLWindow::event(QEvent *event){

	switch(event->type()){
	case QEvent::UpdateRequest:
		m_update_pending = false;
		renderNow();
		return true;
	default:
		return QWindow::event(event);
		//break;
	}
}

void OpenGLWindow::exposeEvent(QExposeEvent *event){

	Q_UNUSED(event);
	if (isExposed())
		renderNow();
}
void OpenGLWindow::renderNow(){

	if (!isExposed())
		return;
	bool needsInitialize = false;
	if (!m_context){
		m_context = new QOpenGLContext(this);
		m_context->setFormat(requestedFormat());;
		m_context->create();
		//QOpenGLFunctions_4_2_Core *func = 0;
		//func = m_context->versionFunctions<QOpenGLFunctions_4_4_Core>();
		//if (!func){
			//qWarning() << "Could not obtain required OpenGL contex version";
			//exit(1);
		//}
		needsInitialize = true;
	}

	m_context->makeCurrent(this);

	if (needsInitialize){
		initializeOpenGLFunctions();
		initialize();
	}
	render();
	m_context->swapBuffers(this);

	if (m_animating)
		renderLater();
}
void OpenGLWindow::setAnimating(bool animating){

	m_animating = animating;
	if (animating)
		renderLater();

}
void OpenGLWindow::InitShader(QObject *p,QOpenGLShaderProgram *proID, const char *VertexFile, const char *GeometryFile, const char *FragmentFile){
	if (!proID){
		qWarning() << "this shaderPro has not been created" << endl;
		exit(1);
	}
	//*proID = new QOpenGLShaderProgram(p);
	QOpenGLShader vertexShader(QOpenGLShader::Vertex, p);
	QOpenGLShader GeometryShader(QOpenGLShader::Geometry, p);
	QOpenGLShader FragmentShader(QOpenGLShader::Fragment, p);
	for (int i = 0; i < 3; i++){
		if (i == 0 && VertexFile == 0)   continue;
		if (i == 1 && GeometryFile == 0) continue;
		if (i == 2 && FragmentFile == 0) continue;
		switch (i){
		case 0:
			qDebug() << "now is vertex" << endl;
			vertexShader.compileSourceFile(VertexFile);
			if (!vertexShader.isCompiled()){
				qWarning() << vertexShader.log();
				exit(1);
			}
			(proID)->addShader(&vertexShader);
			break;

		case 1:
			qDebug() << "now is geometry" << endl;
			GeometryShader.compileSourceFile(GeometryFile);
			if (!GeometryShader.isCompiled()){
				qWarning() << GeometryShader.log();
				exit(1);
			}
			(proID)->addShader(&GeometryShader);
			break;
		case 2:
			qDebug ()<< "now is fragment" << endl;
			FragmentShader.compileSourceFile(FragmentFile);
			if (!FragmentShader.isCompiled()){
				qWarning() << FragmentShader.log();
				exit(1);
			}
			proID->addShader(&FragmentShader);
		}
	}
	(proID)->link();
	if (!(proID)->isLinked()){
		qWarning() << (proID)->log();
		exit(1);
	}
	//proID->is
}


















