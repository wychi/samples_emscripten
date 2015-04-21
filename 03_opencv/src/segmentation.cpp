#include <stdio.h>

#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

void getBinMask( const Mat& comMask, Mat& binMask )
{
    if( comMask.empty() || comMask.type()!=CV_8UC1 )
        CV_Error( Error::StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)" );
    if( binMask.empty() || binMask.rows!=comMask.rows || binMask.cols!=comMask.cols )
        binMask.create( comMask.size(), CV_8UC1 );
    binMask = comMask & 1;
}

extern "C" void doSegmentation(void const *source, int originalWidth, int originalHeight, int processWidth, int processHeight) {
	printf("doSegmentation is called\n");

    cv::Mat frame_rgba_original(originalHeight, originalWidth, CV_8UC4, (unsigned char*)(source));
    cv::Mat frame_rgba;
    if (originalWidth != processWidth || originalHeight != processHeight) {
        cv::resize(frame_rgba_original, frame_rgba, cv::Size(processWidth, processHeight));
    }
    else {
        frame_rgba = frame_rgba_original;
    }

    cv::Mat frame_bgr;
    cv::cvtColor(frame_rgba, frame_bgr, cv::COLOR_RGBA2BGR);


    // copy result back
    cv::Mat res;
    cv::Mat binMask;
    getBinMask( gcapp.mask, binMask );
    gcapp.image->copyTo( res, binMask );
    cv::cvtColor(res, res, cv::COLOR_BGR2RGBA);
    memcpy(result, res.data, width*height*4);

    unsigned char const *maskPtr = NULL;
    unsigned char *destPtr = NULL;
    for (int i = 0; i < height; ++i) {
        maskPtr = gcapp.mask.ptr(i);
        destPtr = (unsigned char*)result + i * width * 4;
        for (int j = 0; j < width; ++j) {
            *destPtr++ = 255;
            *destPtr++ = 0;
            *destPtr++ = 0;
            // *destPtr++ = 255;
            // printf("%u ", *maskPtr++);
            unsigned char c = *maskPtr++;
            if (c == GC_BGD || c == GC_PR_BGD) {
                *destPtr++ = 200;
            }
            else {
                *destPtr++ = 0;   
            }
        }
        // printf("\n");
    }

}
