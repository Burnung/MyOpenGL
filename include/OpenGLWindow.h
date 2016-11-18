#include<QtGui\QWindow>
//#include"include\glslShader.h"
//#include<QtGui\qopenglfunctions.h>
#include<QtGui\QOpenGLFunctions_4_3_Compatibility>
#include<QtGui\QOpenGLShaderProgram>
//#include<QtGui\QOPenGLFunction_4_4_compatibility>

QT_BEGIN_NAMESPACE
class QPainter;
class QOPenGLContext;
class QOpenGLPaintDevice;
QT_END_NAMESPACE

//GL_LINK_STATUS
class OpenGLWindow :public QWindow, protected QOpenGLFunctions_4_3_Compatibility
{
	Q_OBJECT
public:
	explicit OpenGLWindow(QWindow *parent = 0);
	~OpenGLWindow();

	virtual void render(QPainter *painter);
	virtual void render();
	virtual void initialize();

	void setAnimating(bool animating);

public slots:
	void renderLater();
	void renderNow();
	
protected:
	bool event(QEvent *event) Q_DECL_OVERRIDE;
	void exposeEvent(QExposeEvent *event) Q_DECL_OVERRIDE;

	void OpenGLWindow::InitShader(QObject *p, QOpenGLShaderProgram *proID, const char *VertexFile, const char *GeometryFile, const char *FragmentFile); 
	
	//void InitShader(QOpenGLShaderProgram *proID, const char *VertexFile, const char *GeometryFile, const char *GragmentFile);

private:
	bool m_update_pending;
	bool m_animating;

	QOpenGLContext *m_context;
	QOpenGLPaintDevice *m_device;

};