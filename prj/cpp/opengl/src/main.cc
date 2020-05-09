#include <iostream>
#ifdef __APPLE__
#include <glut/glut.h>
#else
#include <GL/glut.h>
#endif
using namespace std;

#define TITLE "OpenGL Research"
#define SIZE 600, 400

// void display(void)
// {

//     glClear(GL_COLOR_BUFFER_BIT);
//     glBegin(GL_TRIANGLES);
//     glVertex3f(-0.5, -0.5, 0.0);
//     glVertex3f(0.5, 0.0, 0.0);
//     glVertex3f(0.0, 0.5, 0.0);

//     glEnable(GL_PROGRAM_POINT_SIZE_EXT);
//     // GL_POINT_SIZE = 4.0f;

//     glEnd();
//     glFlush();
// }
static GLuint v_shader = 0;
static GLuint f_shader = 0;
static GLuint program = 0;
void prog(const char **v_source, const char **f_source)
{
    cout << "=== prog start... " << endl;
    if (program) {
        glDeleteProgram(program);
        program = 0;
    }
    v_shader = glCreateShader(GL_VERTEX_SHADER);
    f_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(v_shader, 1, v_source, NULL);
    glShaderSource(v_shader, 1, f_source, NULL);
    glCompileShader(v_shader);
    glCompileShader(f_shader);
    program = glCreateProgram();
    glAttachShader(program, v_shader);
    glAttachShader(program, f_shader);
    // glBindAttribLocation(program, 0, "vVertex");
    // glBindAttribLocation(program, 1, "vColor");
    glLinkProgram(program);
    glDeleteShader(v_shader);
    v_shader = 0;
    glDeleteShader(f_shader);
    f_shader = 0;
    glUseProgram(program);
    cout << "=== prog end... " << endl;
}

const GLchar *vertex_shader[] = {
    "void main(void) {\n",
    "    gl_Position = ftransform();\n",
    "    gl_FrontColor = gl_Color;\n",
    "}"
};

const GLchar *color_shader[] = {
    "void main() {\n",
    "    gl_FragColor = gl_Color;\n",
    "}"
};

void draw_triangle()
{
    // glBegin(GL_TRIANGLES);
    // glVertex3f(-0.5, 0.0, 0.0);
    // glVertex3f(0.5, 0.0, 0.0);
    // glVertex3f(0.0, 0.5, 0.0);

    // glEnable(GL_PROGRAM_POINT_SIZE_EXT);
    // // GL_POINT_SIZE = 4.0f;

    // glEnd();
     glBegin(GL_TRIANGLES);
        glColor3f(0.0f, 0.0f, 1.0f);
        glVertex3f(-1.0f, 0.0f, -1.0f);
        // glColor3f(0.0f, 1.0f, 0.0f);
        glVertex3f(1.0f, 0.0f, -1.0f);
        // glColor3f(1.0f, 0.0f, 0.0f);
        glVertex3d(0.0, -1.0, -1.0);
    glEnd();
}

void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    prog(vertex_shader, color_shader);
    draw_triangle();
    glutSwapBuffers();
}

// void run()
// {
//     int argc = 0;
//     glutInit(&argc, nullptr);
//     glutInitDisplayMode(GLUT_DEPTH | GLUT_SINGLE | GLUT_RGBA);
//     glutInitWindowSize(SIZE);
//     glutCreateWindow(TITLE);
//     glutDisplayFunc(Handle_Display);
//     glutMainLoop();
// }

void init()
{
}

int main(int argc, char *argv[])
{
    cout << TITLE << endl;
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutCreateWindow(TITLE);
    glutDisplayFunc(display);
    init();
    glutMainLoop();
    return 0;
}