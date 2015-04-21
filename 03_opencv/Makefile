#OPENCV_INSTALL_DIR=/home/wychi/github/opencv/jsobj/install
OBJDIR=./out
WEBAPP=./webapp

ifndef OPENCV_INSTALL_DIR
    $(info OPENCV_INSTALL_DIR is undefined. try: OPENCV_INSTALL_DIR=<YOUR_PATH> emmake make)
    $(error OPENCV_INSTALL_DIR is undefined.)
endif

all: js

faceDetection.bc: ./src/segmentation.cpp
	mkdir -p $(OBJDIR)
	$(CC) ./src/faceDetection.cpp -o $(OBJDIR)/$@ \
		-I $(OPENCV_INSTALL_DIR)/include

js: faceDetection.bc
	$(CC) --bind \
	$(OBJDIR)/faceDetection.bc \
	$(OPENCV_INSTALL_DIR)/lib/libopencv_core.so \
	$(OPENCV_INSTALL_DIR)/lib/libopencv_imgproc.so \
	$(OPENCV_INSTALL_DIR)/lib/libopencv_features2d.so \
	$(OPENCV_INSTALL_DIR)/lib/libopencv_video.so \
	$(OPENCV_INSTALL_DIR)/lib/libopencv_objdetect.so \
	\
	-O3 \
	--llvm-lto 1 \
	--closure 1 \
	-s PRECISE_F32=1 \
	-s TOTAL_MEMORY=318767104 \
	-s ASSERTIONS=1 \
	-s EXPORTED_FUNCTIONS="['_kaku_face_detection_demo']" \
	--preload-file xml/lbpcascade_frontalface.xml \
	--preload-file xml/haarcascade_eye_tree_eyeglasses.xml \
	-o $(OBJDIR)/out.js

install:
	rsync -rupE $(OBJDIR)/*.js $(WEBAPP)
	rsync -rupE $(OBJDIR)/*.js.mem $(WEBAPP)
	rsync -rupE $(OBJDIR)/*.data $(WEBAPP)
