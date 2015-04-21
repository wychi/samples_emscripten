Sample code are copied from here

[Kaku Kuo/opencvjs_demo_segmentation](https://github.com/kakukogou/opencvjs_demo_segmentation)

[Kaku Kuo/opencvjs_demo_facedetection](https://github.com/kakukogou/opencvjs_demo_facedetection)

# Gotcha
The file path should be the same as --preload-file parameter.
```
faceDetection.cpp

cv::String const face_cascade_name = "xml/lbpcascade_frontalface.xml";
cv::String const eyes_cascade_name = "xml/haarcascade_eye_tree_eyeglasses.xml";

```

```
Makefile

...
js: faceDetection.bc
	$(CC) --bind \
	$(OBJDIR)/faceDetection.bc \

	-s EXPORTED_FUNCTIONS="['_kaku_face_detection_demo']" \
	--preload-file xml/lbpcascade_frontalface.xml \
	--preload-file xml/haarcascade_eye_tree_eyeglasses.xml \
	-o $(OBJDIR)/out.js
```