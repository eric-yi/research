#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>

#ifdef __APPLE__
#include <glut/glut.h>
#else
#include <GL/glut.h>
#endif

using namespace std;

#define TITLE       "OpenGL Research"
#define SIZE        600, 400
#define GLSL_DIR    "/Users/yixiaobin/local/workspace/research/prj/cpp/opengl/glsl/"


const string ReadFile(const char *filepath) {
    cout << "==== Read File: " << filepath << endl;
    stringstream contents;
    ifstream file;
    file.open(filepath);
    string line;
    if (file.is_open()) {
        contents << file.rdbuf();
        file.close();
    }
    return contents.str();
}

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
void prog(const char *v_source, const char *f_source)
{
    cout << "=== prog start... " << endl;
    cout << "vertext shader source: \n" << v_source << endl;
    cout << "fragment shader source: \n" << f_source << endl;
    // if (program)
    // {
    //     glDeleteProgram(program);
    //     program = 0;
    // }
    
    int success;
    char infoLog[512];

    v_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(v_shader, 1, &v_source, NULL);
    glCompileShader(v_shader);
    glGetShaderiv(v_shader, GL_COMPILE_STATUS, &success);
    if(!success) {
        glGetShaderInfoLog(v_shader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED: \n" << infoLog << std::endl;
    }

    f_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(f_shader, 1, &f_source, NULL);
    glCompileShader(f_shader);
    glGetShaderiv(f_shader, GL_COMPILE_STATUS, &success);
    if(!success) {
        glGetShaderInfoLog(f_shader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED: \n" << infoLog << std::endl;
    }

    program = glCreateProgram();
    glAttachShader(program, v_shader);
    glAttachShader(program, f_shader);
    // glBindAttribLocation(program, 0, "vVertex");
    // glBindAttribLocation(program, 1, "vColor");
    glLinkProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if(!success) {
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n"<< infoLog << std::endl;
    }

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
    "}"};

const GLchar *color_shader[] = {
    "void main() {\n",
    "    gl_FragColor = gl_Color;\n",
    "}"};

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
    // glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(-1.0f, 0.0f, -1.0f);
    // glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(1.0f, 0.0f, -1.0f);
    // glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3d(0.0, -1.0, -1.0);
    glEnd();
}


// #define PI 3.1415926

void DrawSphere() {
    const float PI = acos(-1);
    float radius = 1.0f;
    int sectorCount = 36;
    int interleavedStride = 32;
    vector<float> interleavedVertices;
    int stackCount = 18;
    vector<float> vertices;
    std::vector<float> normals;
    std::vector<float> texCoords;
    vector<unsigned int> indices;
    vector<unsigned int> lineIndices;

    float x, y, z, xy;                              // vertex position
    float nx, ny, nz, lengthInv = 1.0f / radius;    // normal
    float s, t;                                     // texCoord

    float sectorStep = 2 * PI / sectorCount;
    float stackStep = PI / stackCount;
    float sectorAngle, stackAngle;

    for(int i = 0; i <= stackCount; ++i)
    {
        stackAngle = PI / 2 - i * stackStep;        // starting from pi/2 to -pi/2
        xy = radius * cosf(stackAngle);             // r * cos(u)
        z = radius * sinf(stackAngle);              // r * sin(u)

        // add (sectorCount+1) vertices per stack
        // the first and last vertices have same position and normal, but different tex coords
        for(int j = 0; j <= sectorCount; ++j)
        {
            sectorAngle = j * sectorStep;           // starting from 0 to 2pi

            // vertex position
            x = xy * cosf(sectorAngle);             // r * cos(u) * cos(v)
            y = xy * sinf(sectorAngle);             // r * cos(u) * sin(v)
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
    

            // normalized vertex normal
            nx = x * lengthInv;
            ny = y * lengthInv;
            nz = z * lengthInv;
            normals.push_back(nx);
            normals.push_back(ny);
            normals.push_back(nz);

            // vertex tex coord between [0, 1]
            s = (float)j / sectorCount;
            t = (float)i / stackCount;
            texCoords.push_back(s);
            texCoords.push_back(t);
        }
    }

    // indices
    //  k1--k1+1
    //  |  / |
    //  | /  |
    //  k2--k2+1
    unsigned int k1, k2;
    for(int i = 0; i < stackCount; ++i)
    {
        k1 = i * (sectorCount + 1);     // beginning of current stack
        k2 = k1 + sectorCount + 1;      // beginning of next stack

        for(int j = 0; j < sectorCount; ++j, ++k1, ++k2)
        {
            // 2 triangles per sector excluding 1st and last stacks
            if(i != 0)
            {
                indices.push_back(k1);
    indices.push_back(k2);
    indices.push_back(k1+1);
            }

            if(i != (stackCount-1))
            {
                indices.push_back(k1+1);
    indices.push_back(k2);
    indices.push_back(k2+1);
            }

            // vertical lines for all stacks
            lineIndices.push_back(k1);
            lineIndices.push_back(k2);
            if(i != 0)  // horizontal lines except 1st stack
            {
                lineIndices.push_back(k1);
                lineIndices.push_back(k1 + 1);
            }
        }
    }

     std::size_t i, j;
    std::size_t count = vertices.size();
    for(i = 0, j = 0; i < count; i += 3, j += 2)
    {
        interleavedVertices.push_back(vertices[i]);
        interleavedVertices.push_back(vertices[i+1]);
        interleavedVertices.push_back(vertices[i+2]);

        interleavedVertices.push_back(normals[i]);
        interleavedVertices.push_back(normals[i+1]);
        interleavedVertices.push_back(normals[i+2]);

        interleavedVertices.push_back(texCoords[j]);
        interleavedVertices.push_back(texCoords[j+1]);
    }


 glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    // save the initial ModelView matrix before modifying ModelView matrix
    glPushMatrix();

    // tramsform modelview matrix
    glTranslatef(0, 0, -0.5);

    // set material
    float ambient[]  = {0.5f, 0.5f, 0.5f, 1};
    float diffuse[]  = {0.7f, 0.7f, 0.7f, 1};
    float specular[] = {1.0f, 1.0f, 1.0f, 1};
    float shininess  = 128;
    glMaterialfv(GL_FRONT, GL_AMBIENT,   ambient);
    glMaterialfv(GL_FRONT, GL_DIFFUSE,   diffuse);
    glMaterialfv(GL_FRONT, GL_SPECULAR,  specular);
    glMaterialf(GL_FRONT, GL_SHININESS, shininess);

    // line color
    float lineColor[] = {0.2f, 0.2f, 0.2f, 1};

    // draw left flat sphere with lines
    // glPushMatrix();
    // glTranslatef(-2.5f, 0, 0);
    glRotatef(0.1, 1, 0, 0);   // pitch
    glRotatef(0.1, 0, 1, 0);   // heading
    glRotatef(-90, 1, 0, 0);
    glBindTexture(GL_TEXTURE_2D, 0);


    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1.0, 1.0f); // move polygon backward
     glEnableClientState(GL_VERTEX_ARRAY);
    // glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    glVertexPointer(3, GL_FLOAT, interleavedStride, &interleavedVertices[0]);
    // glNormalPointer(GL_FLOAT, interleavedStride, &interleavedVertices[3]);
    glTexCoordPointer(2, GL_FLOAT, interleavedStride, &interleavedVertices[6]);

    glDrawElements(GL_TRIANGLES, (unsigned int)indices.size(), GL_UNSIGNED_INT, indices.data());

    glDisableClientState(GL_VERTEX_ARRAY);
    // glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisable(GL_POLYGON_OFFSET_FILL);

    // draw lines with VA
      glColor4fv(lineColor);
    glMaterialfv(GL_FRONT, GL_DIFFUSE,   lineColor);

    // draw lines with VA
    //   glEnable(GL_LIGHTING);
    // glEnable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, vertices.data());

    glDrawElements(GL_LINES, (unsigned int)lineIndices.size(), GL_UNSIGNED_INT, lineIndices.data());

    glDisableClientState(GL_VERTEX_ARRAY);
    //  glDisable(GL_LIGHTING);
    // glDisable(GL_TEXTURE_2D);
    glEnable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);


    //  glPopMatrix();
    // glutSwapBuffers();

    
}

void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    const char *vert = ReadFile((string(GLSL_DIR) + "gl_research.vert").c_str()).c_str();
    cout << "vertext shader source: " << vert << endl;
    char *vert_source;
    vert_source = (char *) malloc(strlen(vert) + 1);
    memcpy(vert_source, vert, strlen(vert));
    const char *frag = ReadFile((string(GLSL_DIR) + "gl_research.frag").c_str()).c_str();
    cout << "fragment shader source: " << frag << endl;
    char *frag_source;
    frag_source = (char *) malloc(strlen(frag) + 1);
    memcpy(frag_source, frag, strlen(frag));
    // prog(vert_source, frag_source);
    // draw_triangle();
    DrawSphere();
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
    cout <<"OpenGL version supported by this platform: " << glGetString(GL_VERSION) << endl;
    cout << "Vendor: " << glGetString(GL_VENDOR) << endl;
    cout << "Renderer: " << glGetString(GL_RENDERER) << endl;
    cout << "Shader version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;
    cout << "Extensions: " << glGetString(GL_EXTENSIONS) << endl;
    glutDisplayFunc(display);
    init();
    // run();
    glutMainLoop();
    return 0;
}