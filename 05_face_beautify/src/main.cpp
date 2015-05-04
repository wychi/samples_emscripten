#include <stdlib.h>
#include "esUtil.h"

#include "opencv2/core/core.hpp"
#include "opencv2/core/opengl.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"

using namespace std;
using namespace cv;

typedef struct
{
   // Handle to a program object
   GLuint programObject;

   // Attribute locations
   GLint  positionLoc;
   GLint  texCoordLoc;

   GLfloat  ppp;

   // Sampler location
   GLint samplerLoc;
   // Sampler location
   GLint samplerLoc2;
   GLint samplerLoc3;

   // Texture handle
   GLuint textureId;
   GLuint textureId2;
   GLuint textureId3;

   GLuint vertexObject, indexObject;

} UserData;

/** Global variables */
static bool isInitialized = false;
cv::String const face_cascade_name = "xml/lbpcascade_frontalface.xml";
cv::String const eyes_cascade_name = "xml/haarcascade_eye_tree_eyeglasses.xml";
cv::CascadeClassifier face_cascade;
cv::CascadeClassifier eyes_cascade;

ESContext sESContext;

///
// Initialize the shader and program object
//
int Init ( ESContext *esContext )
{
    esContext->userData = malloc(sizeof(UserData));
    UserData *userData = (UserData*) esContext->userData;

    GLbyte vShaderStr[] =  
      "attribute vec3 a_position;   \n"
      "attribute vec2 a_texCoord;   \n"
      "varying vec2 vUv;     \n"
      "void main()                  \n"
      "{                            \n"
      "   gl_Position = vec4( a_position, 1.0); \n"
      "   vUv = a_texCoord;  \n"
      "}                            \n";
   
    GLbyte fShaderStr0[] =  
      "precision mediump float;                            \n"
      "varying vec2 vUv;                            \n"
      "uniform sampler2D s_texture;                        \n"
      "void main()                                         \n"
      "{                                                   \n"
      "  gl_FragColor = texture2D( s_texture, vUv );\n"
      "}                                                   \n";

    GLbyte fShaderStr[] =  
      "precision mediump float;                            \n"
      "varying vec2 vUv;                            \n"
      "uniform sampler2D s_texture;                        \n"
      "void main()                                         \n"
      "{                                                   \n"
      "vec4 color = texture2D( s_texture, vUv );\n"
      "float alpha = 0.0;\n"
      "float l = color.y*255.0*0.836 - 14.0;\n"
      "float u = color.y*255.0*0.836 + 44.0;\n"
      "float B = color.z;\n"
      "if( (l < B && B < u) == false )\n"
      "  alpha = 0.0;\n"
      "gl_FragColor = mix(vec4(0.0, 0.0, 0.0, 0.0), color, alpha);\n"
      "}\n";    

GLbyte fShaderStr2[] =  
"precision highp float;\n"
"uniform sampler2D   s_texture;\n"
"uniform sampler2D   sTextureSamples;\n"
"uniform sampler2D   s_skin;\n"
"uniform float ppp[5];\n"
"varying vec2        vUv;\n"
"void main(void)\n"
"{\n"
"float scaleX = ppp[0];\n"
"float scaleY = ppp[1];\n"
"float scaleY2 = ppp[2];\n"
"float sigma_r2 = ppp[3];\n"
"float sigma_s2 = ppp[4];\n"
"    vec3 colorRef = texture2D(s_skin, vUv).xyz;\n"
"    vec3 color = vec3(0.0,0.0,0.0);\n"
"    float yFetch = vUv.y*scaleY2;\n"
"    float weight = 0.0;\n"
"    for(int i=0;i<25;i++){\n"
"        vec2 coords = texture2D(sTextureSamples,vec2(float(i)/30.0,yFetch)).xy;\n"
"        coords = (coords-0.5)*vec2(scaleX,scaleY);\n"
"        vec3 colorFetch = texture2D(s_skin,coords+vUv).xyz;\n"
"        vec3 colorDist = colorFetch-colorRef;\n"
"        float tmpWeight = exp(-dot(colorDist,colorDist)/sigma_r2);\n"
"        color += colorFetch*tmpWeight;\n"
"        weight += tmpWeight;\n"
"    }\n"
"    if(weight<=0.0)\n"
"        color = colorRef;\n"
"    else\n"
"        color = color/weight;\n"
"float alpha = 1.0;"
"if(colorRef.x == 0.0)\n"
"    alpha = 0.0;\n"
"gl_FragColor = mix(texture2D(s_texture, vUv), vec4(color.xyz, 1.0), alpha);\n"

"}\n";


    // Load the shaders and get a linked program object
    userData->programObject = esLoadProgram ( (const char *)vShaderStr, (const char *)fShaderStr );
    // Get the attribute locations
    userData->positionLoc = glGetAttribLocation ( userData->programObject, "a_position" );
    userData->texCoordLoc = glGetAttribLocation ( userData->programObject, "a_texCoord" );
   
    // Get the sampler location
    userData->samplerLoc = glGetUniformLocation ( userData->programObject, "s_texture" );
    userData->samplerLoc2 = glGetUniformLocation ( userData->programObject, "sTextureSamples" );
    userData->samplerLoc3 = glGetUniformLocation ( userData->programObject, "s_skin" );
    userData->ppp = glGetUniformLocation ( userData->programObject, "ppp" );

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

    glClearColor ( 0.0f, 0.0f, 1.0f, 1.0f );

    return GL_TRUE;
}


extern "C" int initGL(int width, int height)
{
    clock_t c = clock();

    esInitContext ( &sESContext );
    //sESContext.userData = &sUserData;

    printf("initGL w,h = %d,%d\n", width, height);
    esCreateWindow ( &sESContext, "Hello Triangle", width, height, ES_WINDOW_RGB );
    if ( !Init ( &sESContext ) )
      return 0;

    //-- 1. Load the cascade
    if (!isInitialized) {
        if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };
        //if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };
        isInitialized = true;
    }

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

    printf("CreateSimpleTexture2D textureId = %d\n", textureId);
    return textureId;
}

float statistic[] = {0,0,0,0}; // openCV(cur, avg), cur, avg
char buff[100];

void drawGL(const Mat& srcImage, int width, int height) {
    clock_t c = clock();

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
}

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

    drawGL(srcImage, width, height);

    return 0;
}

extern "C" void FaceBeautify(void const *array, int width, int height) {
    int MAX_KERNEL_LENGTH = 31;

    clock_t c = clock();

    Mat orgImage(height, width, CV_8UC4, (unsigned char*)(array));
    Mat srcImage;
    cvtColor(orgImage,srcImage, CV_BGRA2BGR);
    Mat dstImage = srcImage.clone();
    for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 ) {
        bilateralFilter ( srcImage, dstImage, i, i*2, i/2 );
    }
    Mat dstImage2;
    cvtColor(dstImage,dstImage2, CV_BGR2BGRA);
    
    statistic[0] = ((float)(clock() - c)*1000.0f/CLOCKS_PER_SEC);
    statistic[1] = (statistic[0] + statistic[1]) / 2;

    drawGL(dstImage2, width, height);
}

int findBiggestContour(vector<vector<Point> > contours){
    int indexOfBiggestContour = -1;
    int sizeOfBiggestContour = 0;
    for (int i = 0; i < contours.size(); i++){
        if(contours[i].size() > sizeOfBiggestContour){
            sizeOfBiggestContour = contours[i].size();
            indexOfBiggestContour = i;
        }
    }
    return indexOfBiggestContour;
}

float g_adjust_sigma_r = 20;

void drawGL2(const Mat& srcImage, int width, int height, const Mat& srcSampling, const Mat& srcSkin) {
    clock_t c = clock();

    printf("drawGL2, %d, %d\n", width, height);

    float scaleX = (255.0/width);
    float scaleY = (255.0/height);
    float scaleY2 = (64.0/height);
    float sigma_r = g_adjust_sigma_r/255.0;
    float sigma_r2 = sigma_r*sigma_r*2.0;
    float sigma_s = 5.0;
    float sigma_s2 = sigma_s*sigma_s*2.0;

    // Load the texture
    ESContext *esContext = &sESContext;
    UserData *userData = (UserData*) esContext->userData;
    userData->textureId = CreateSimpleTexture2D (srcImage.data, width, height);
    userData->textureId2 = CreateSimpleTexture2D (srcSampling.data, 64, 64);
    userData->textureId3 = CreateSimpleTexture2D (srcSkin.data, width, height);
    

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
    glUniform1i ( userData->samplerLoc, 0 );

    glActiveTexture( GL_TEXTURE1 );
    glBindTexture ( GL_TEXTURE_2D, userData->textureId2 );
    glUniform1i ( userData->samplerLoc2, 1 );

    glActiveTexture( GL_TEXTURE2 );
    glBindTexture ( GL_TEXTURE_2D, userData->textureId3 );
    glUniform1i ( userData->samplerLoc3, 2 );


    glBindBuffer ( GL_ELEMENT_ARRAY_BUFFER, userData->indexObject );

    GLfloat ppp[] = {
        scaleX, scaleY, scaleY2, sigma_r, sigma_r2, sigma_s, sigma_s2
    };
    glBindBuffer (GL_ARRAY_BUFFER, userData->ppp );
    glUniform1fv ( userData->ppp, 5, ppp );

    printf("drawGL - bindBuffer %f ms ",((float)(clock() - c)*1000.0f/CLOCKS_PER_SEC));

    glDrawElements ( GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0 );

    statistic[2] = ((float)(clock() - c)*1000.0f/CLOCKS_PER_SEC);
    statistic[3] = (statistic[2] + statistic[3]) / 2;
}

void computeSkinColor(const Mat& src, const cv::Rect& rect, Scalar& lBound, Scalar& uBound) {

    int steps = 5;
    Mat roi(src, rect);
    Scalar mean = cv::mean(roi);
    printf("RGB mean %f %f %f\n", mean.val[0], mean.val[1], mean.val[2]);

    // Mat roiHSV(hsv, faces[0]);
    // Scalar hsvmean = cv::mean(roiHSV);
    // printf("HSV mean %f %f %f\n", hsvmean.val[0], hsvmean.val[1], hsvmean.val[2]);

    int span_w = rect.width / steps;
    int span_h = rect.height / steps;
    for(int i=0;i<steps; i++) {
        for(int j=0; j<steps; j++) {
            int x = span_w * i;
            int y = span_h * j;
            Vec3b pixel_rgb = roi.at<Vec3b>(y, x);
            printf("%d, %d BGR %d %d %d\n", x, y, pixel_rgb.val[0], pixel_rgb.val[1], pixel_rgb.val[2]);
            // Vec3b pixel_hsv = roiHSV.at<Vec3b>(y, x);
            // printf("%d, %d HSV %d %d %d\n", x, y, pixel_hsv.val[0], pixel_hsv.val[1], pixel_hsv.val[2]);                
        }
    }

    lBound = Scalar(200, 140, 90);
    uBound = Scalar(255, 210, 180);
}

extern "C" void SkinSegmentation(void const *array, int width, int height, void const *buf2, float adjust_sigma_r) {

    Mat src(height, width, CV_8UC4, (unsigned char*)(array));
    Mat srcSampling(64, 64, CV_8UC4, (unsigned char*)(buf2));

    g_adjust_sigma_r = adjust_sigma_r;
    printf("SkinSegmentation w,h=%d,%d, sigma=%f\n", width, height, g_adjust_sigma_r);

    //-- Detect faces
    clock_t c = clock();

    std::vector<cv::Rect> faces;
    Mat frame_gray;
    cv::cvtColor( src, frame_gray, cv::COLOR_RGBA2GRAY );
    cv::equalizeHist( frame_gray, frame_gray );
    double scaleFactor = 1.1;
    int minWidth = width / 4;
    int minHeight = height / 4;

    face_cascade.detectMultiScale( frame_gray, faces, scaleFactor, 3, 0, cv::Size(minWidth, minHeight) );

    printf("detectMultiScale(): #faces = %u, %f ms\n", faces.size(), ((float)(clock() - c)*1000.0f/CLOCKS_PER_SEC));

    // Skin Segmentation
    Mat srcBGR;
    cv::cvtColor(src, srcBGR, CV_BGRA2BGR);
    Mat hsv;
    cv::cvtColor(srcBGR, hsv, CV_BGR2HSV);

    Scalar lBound(200, 140, 90);
    Scalar uBound(255, 210, 180);
    if( faces.size() > 0) {
        computeSkinColor(src, faces[0], lBound, uBound);
    }

    Mat bw;
    Mat img4;
    //inRange(hsv, Scalar(96-10, 104-10, 0), Scalar(96+10, 104+10, 255), bw);
    inRange(srcBGR, lBound, uBound, bw);
    cvtColor(bw,bw,CV_GRAY2BGR);

    bitwise_and(srcBGR, bw, img4); 
    //Mat dstRGB;
    Mat skin;
    //cvtColor(bw, dstRGB, CV_HSV2BGR);
    cvtColor(img4, skin, CV_BGR2BGRA);

    for( size_t i = 0; i < faces.size(); i++ )
    {
        cv::rectangle( src, faces[i], cv::Scalar( 255, 0, 0 ));
    }
    //drawGL(skin, width, height);

    // // Mat canny_output;
    // vector<vector<Point> > contours;
    // vector<Vec4i> hierarchy;

    // findContours( bw, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    // int s = findBiggestContour(contours);

    // Mat drawing = Mat::zeros( src.size(), CV_8UC1 );
    // drawContours( drawing, contours, s, Scalar(255), -1, 8, hierarchy, 0, Point() );

    statistic[0] = ((float)(clock() - c)*1000.0f/CLOCKS_PER_SEC);
    statistic[1] = (statistic[0] + statistic[1]) / 2;

    drawGL2(src, width, height, srcSampling, skin);
}


extern "C" void inRange(void const *array, int width, int height,
    int type, int l1, int l2, int l3, int u1, int u2, int u3) {

    printf("%d, %d, %d, %d, %d, %d\n", l1, l2, l3, u1, u2, u3);

    Mat src(height, width, CV_8UC4, (unsigned char*)(array));
    Mat bgr, cvt;
    cv::cvtColor(src, bgr, CV_BGRA2BGR);
    if(type == 0) {
        cvt = bgr;
    } else if(type == 1) {
        // HSV
        cv::cvtColor(bgr, cvt, CV_BGR2HSV);
    }

    Scalar lBound(l1, l2, l3);
    Scalar uBound(u1, u2, u3);

    Mat bw, dst;
    inRange(cvt, lBound, uBound, bw);
    cvtColor(bw,bw,CV_GRAY2BGR);

    bitwise_and(bgr, bw, dst);
    cv::cvtColor(dst, dst, CV_BGR2BGRA);

    drawGL(dst, width, height);
}