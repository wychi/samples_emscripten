#include <stdlib.h>
#include "esUtil.h"

#include "opencv2/core/core.hpp"
#include "opencv2/core/opengl.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

typedef struct
{
   // Handle to a program object
   GLuint programObject;

   // Attribute locations
   GLint  positionLoc;
   GLint  texCoordLoc;

   // Sampler location
   GLint samplerLoc;

   // Texture handle
   GLuint textureId;

   GLuint vertexObject, indexObject;

} UserData;

ESContext sESContext;

///
// Initialize the shader and program object
//
int Init ( ESContext *esContext )
{
    esContext->userData = malloc(sizeof(UserData));
    UserData *userData = (UserData*) esContext->userData;

    GLbyte vShaderStr[] =  
      "attribute vec4 a_position;   \n"
      "attribute vec2 a_texCoord;   \n"
      "varying vec2 v_texCoord;     \n"
      "void main()                  \n"
      "{                            \n"
      "   gl_Position = a_position; \n"
      "   v_texCoord = a_texCoord;  \n"
      "}                            \n";
   
    GLbyte fShaderStr[] =  
      "precision mediump float;                            \n"
      "varying vec2 v_texCoord;                            \n"
      "uniform sampler2D s_texture;                        \n"
      "void main()                                         \n"
      "{                                                   \n"
      "  gl_FragColor = texture2D( s_texture, v_texCoord );\n"
      "}                                                   \n";


    // Load the shaders and get a linked program object
    userData->programObject = esLoadProgram ( (const char *)vShaderStr, (const char *)fShaderStr );
    // Get the attribute locations
    userData->positionLoc = glGetAttribLocation ( userData->programObject, "a_position" );
    userData->texCoordLoc = glGetAttribLocation ( userData->programObject, "a_texCoord" );
   
    // Get the sampler location
    userData->samplerLoc = glGetUniformLocation ( userData->programObject, "s_texture" );

    // Setup the vertex data
    GLfloat vVertices[] = { -1.0,  1.0, 0.0,  // Position 0
                             0.0,  0.0,       // TexCoord 0
                            -1.0, -1.0, 0.0,  // Position 1
                             0.0,  1.0,       // TexCoord 1
                             1.0, -1.0, 0.0,  // Position 2
                             1.0,  1.0,       // TexCoord 2
                             1.0,  1.0, 0.0,  // Position 3
                             1.0,  0.0        // TexCoord 3
                          };
    GLushort indices[] = { 0, 1, 2, 0, 2, 3 };

    glGenBuffers(1, &userData->vertexObject);
    glBindBuffer(GL_ARRAY_BUFFER, userData->vertexObject );
    glBufferData(GL_ARRAY_BUFFER, sizeof(vVertices), vVertices, GL_STATIC_DRAW );

    glGenBuffers(1, &userData->indexObject);
    glBindBuffer ( GL_ELEMENT_ARRAY_BUFFER, userData->indexObject );
    glBufferData ( GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW );

    glClearColor ( 0.0f, 0.0f, 0.0f, 1.0f );

    return GL_TRUE;
}


extern "C" int initGL(int width, int height)
{
    clock_t c = clock();

    esInitContext ( &sESContext );
    //sESContext.userData = &sUserData;

    esCreateWindow ( &sESContext, "Hello Triangle", width, height, ES_WINDOW_RGB );
    if ( !Init ( &sESContext ) )
      return 0;

    printf("initGL %f ms\n", ((float)(clock() - c)*1000.0f/CLOCKS_PER_SEC));

  return 1;
}

GLuint CreateSimpleTexture2D( void const *array, int width, int height)
{
    // Texture object handle
    GLuint textureId;
   
    // Use tightly packed data
    glPixelStorei ( GL_UNPACK_ALIGNMENT, 1 );

    // Generate a texture object
    glGenTextures ( 1, &textureId );

    // Bind the texture object
    glBindTexture ( GL_TEXTURE_2D, textureId );

    // Load the texture
    glTexImage2D ( GL_TEXTURE_2D, 0, GL_RGBA, width, height,
                   0, GL_RGBA, GL_UNSIGNED_BYTE, array );

    // Set the filtering mode
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    return textureId;
}

float statistic[] = {0,0,0,0}; // openCV(cur, avg), cur, avg
char buff[100];
extern "C" const char* getPerfStatistic() {
    sprintf(buff, "OpenCV=%f,%f, OpenGL=%f,%f", statistic[0], statistic[1], statistic[2], statistic[3]);
    return buff;
}

extern "C" int EdgeDetection(void const *array, int width, int height) {
    clock_t c = clock();

    Mat srcImage(height, width, CV_8UC4, (unsigned char*)(array));
    Mat grayMat;
    Mat grad;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    GaussianBlur( srcImage, srcImage, Size(3,3), 0, 0, BORDER_DEFAULT );
    cvtColor( srcImage, grayMat, CV_BGRA2GRAY);

    /// Generate grad_x and grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    /// Gradient X
    //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    Sobel( grayMat, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );
    /// Gradient Y
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    Sobel( grayMat, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );
    /// Total Gradient (approximate)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
    cvtColor( grad, srcImage, CV_GRAY2BGRA );

    statistic[0] = ((float)(clock() - c)*1000.0f/CLOCKS_PER_SEC);
    statistic[1] = (statistic[0] + statistic[1]) / 2;
    c = clock();

    // Mat_<Vec2f> vertex(1, 4);
    // vertex << Vec2f(-1, 1), Vec2f(-1, -1), Vec2f(1, -1), Vec2f(1, 1);

    // Mat_<Vec2f> texCoords(1, 4);
    // texCoords << Vec2f(0, 0), Vec2f(0, 1), Vec2f(1, 1), Vec2f(1, 0);

    // Mat_<int> indices(1, 6);
    // indices << 0, 1, 2, 2, 3, 0;

    // Load the texture
    ESContext *esContext = &sESContext;
    UserData *userData = (UserData*) esContext->userData;
    userData->textureId = CreateSimpleTexture2D (srcImage.data, width, height);

    // draw
    // Set the viewport
    glViewport ( 0, 0, width, height );
   
    // Clear the color buffer
    glClear ( GL_COLOR_BUFFER_BIT );

    // Use the program object
    glUseProgram ( userData->programObject );

    // Load the vertex position
    glBindBuffer (GL_ARRAY_BUFFER, userData->vertexObject );
    glVertexAttribPointer ( userData->positionLoc, 3, GL_FLOAT,
                            GL_FALSE, 5 * 4, 0 );
    // Load the texture coordinate
    glVertexAttribPointer ( userData->texCoordLoc, 2, GL_FLOAT,
                            GL_FALSE, 5 * 4, 
                            (const GLvoid *)(3 * 4) );

    glEnableVertexAttribArray ( userData->positionLoc );
    glEnableVertexAttribArray ( userData->texCoordLoc );

    // Bind the texture
    glActiveTexture ( GL_TEXTURE0 );
    glBindTexture ( GL_TEXTURE_2D, userData->textureId );

    // Set the sampler texture unit to 0
    glUniform1i ( userData->samplerLoc, 0 );

    glBindBuffer ( GL_ELEMENT_ARRAY_BUFFER, userData->indexObject );
    glDrawElements ( GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0 );

    statistic[2] = ((float)(clock() - c)*1000.0f/CLOCKS_PER_SEC);
    statistic[3] = (statistic[2] + statistic[3]) / 2;

    return 0;
}

extern "C" void FaceBeautify(void const *array, int width, int height) {
}