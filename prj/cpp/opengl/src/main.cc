#include <iostream>
#include <glut/glut.h>

using namespace std;

#define TITLE "OpenGL Research"
#define SIZE 600, 400

void Handle_Display(void)
{

    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_TRIANGLES);
    glVertex3f(-0.5, -0.5, 0.0);
    glVertex3f(0.5, 0.0, 0.0);
    glVertex3f(0.0, 0.5, 0.0);
    glEnd();
    glFlush();
}

void run()
{
    int argc = 0;
    glutInit(&argc, nullptr);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_SINGLE | GLUT_RGBA);
    glutInitWindowSize(SIZE);
    glutCreateWindow(TITLE);
    glutDisplayFunc(Handle_Display);
    glutMainLoop();
}

int main(int argc, char *argv[])
{
    cout << TITLE << endl;
    run();
    return 0;
}