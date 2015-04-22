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
// Create a shader object, load the shader source, and
// compile the shader.
//
GLuint LoadShader ( GLenum type, const char *shaderSrc )
{
   GLuint shader;
   GLint compiled;
   
   // Create the shader object
   shader = glCreateShader ( type );

   if ( shader == 0 )
    return 0;

   // Load the shader source
   glShaderSource ( shader, 1, &shaderSrc, NULL );
   
   // Compile the shader
   glCompileShader ( shader );

   // Check the compile status
   glGetShaderiv ( shader, GL_COMPILE_STATUS, &compiled );

   if ( !compiled ) 
   {
      GLint infoLen = 0;

      glGetShaderiv ( shader, GL_INFO_LOG_LENGTH, &infoLen );
      
      if ( infoLen > 1 )
      {
         char* infoLog = (char*) malloc (sizeof(char) * infoLen );

         glGetShaderInfoLog ( shader, infoLen, NULL, infoLog );
         esLogMessage ( "Error compiling shader:\n%s\n", infoLog );            
         
         free ( infoLog );
      }

      glDeleteShader ( shader );
      return 0;
   }

   return shader;

}

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

   // // Load the texture
   // userData->textureId = CreateSimpleTexture2D (img.data, width, height);

      // Setup the vertex data
   GLfloat vVertices[] = { -0.5,  0.5, 0.0,  // Position 0
                            0.0,  0.0,       // TexCoord 0
                           -0.5, -0.5, 0.0,  // Position 1
                            0.0,  1.0,       // TexCoord 1
                            0.5, -0.5, 0.0,  // Position 2
                            1.0,  1.0,       // TexCoord 2
                            0.5,  0.5, 0.0,  // Position 3
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

///
// Draw a triangle using the shader pair created in Init()
//
extern "C" void Draw ()
{
    // ESContext *esContext = &sESContext;
    // UserData *userData = (UserData*) esContext->userData;
    // GLfloat vVertices[] = {  0.0f,  0.5f, 0.0f, 
    //                         -0.5f, -0.5f, 0.0f,
    //                          0.5f, -0.5f, 0.0f };
 
    // // No clientside arrays, so do this in a webgl-friendly manner
    // GLuint vertexPosObject;
    // glGenBuffers(1, &vertexPosObject);
    // glBindBuffer(GL_ARRAY_BUFFER, vertexPosObject);
    // glBufferData(GL_ARRAY_BUFFER, 9*4, vVertices, GL_STATIC_DRAW);
    
    // // Set the viewport
    // glViewport ( 0, 0, esContext->width, esContext->height );
    
    // // Clear the color buffer
    // glClear ( GL_COLOR_BUFFER_BIT );

    // // Use the program object
    // glUseProgram ( userData->programObject );

    // // Load the vertex data
    // glBindBuffer(GL_ARRAY_BUFFER, vertexPosObject);
    // glVertexAttribPointer(0 /* ? */, 3, GL_FLOAT, 0, 0, 0);
    // glEnableVertexAttribArray(0);
 
    // glDrawArrays ( GL_TRIANGLES, 0, 3 );
}

GLuint CreateSimpleTexture2D( void const *array, int width, int height)
{
    printf("array %p %dx%d \n", array, width, height);
   // Texture object handle
   GLuint textureId;
   
   // // 2x2 Image, 3 bytes per pixel (R, G, B)
   // GLubyte pixels[4 * 3] =
   // {  
   //    255,   0,   0, // Red
   //      0, 255,   0, // Green
   //      0,   0, 255, // Blue
   //    255, 255,   0  // Yellow
   // };

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

float statics[] = {0,0,0,0};

extern "C" int DrawImage(void const *array, int width, int height) {
    clock_t c = clock();

    Mat img(height, width, CV_8UC4, (unsigned char*)(array));
    GaussianBlur(img, img, Size(7,7), 1.5, 1.5);

    statics[0] = ((float)(clock() - c)*1000.0f/CLOCKS_PER_SEC);
    printf("GaussianBlur() %f ms\n", statics[0] );
    c = clock();

    cv::Point center( width/2, height/2 );
    cv::ellipse( img, center, cv::Size( width/2, height/2 ), 0, 0, 360, cv::Scalar( 255, 0, 0 ), 5, 8, 0 );

    statics[1] = ((float)(clock() - c)*1000.0f/CLOCKS_PER_SEC);
    printf("cv::ellipse() %f ms\n", statics[1]);
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
   userData->textureId = CreateSimpleTexture2D (img.data, width, height);

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

   statics[3] = ((float)(clock() - c)*1000.0f/CLOCKS_PER_SEC);
   printf("drawGL %f ms\n", statics[3]);

   return 0;
}