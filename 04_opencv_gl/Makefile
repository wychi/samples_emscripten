OPENCV_INSTALL_DIR=/home/wychi/github/opencv/jsobj/install
OBJDIR=./out
WEBAPP=./webapp

INCDIR=-I./src/Common
#LIBS=-lGLESv2 -lEGL -lm -lX11

COMMONSRC=./src/Common/esShader.cpp    \
          ./src/Common/esUtil.cpp

COMMONHRD=esUtil.h

all: js

main.bc: ${COMMONSRC} ${COMMONHDR} ./src/main.cpp
	mkdir -p $(OBJDIR)
	$(CC) ${COMMONSRC} ./src/main.cpp \
	-I $(OPENCV_INSTALL_DIR)/include \
	-o $(OBJDIR)/$@ \
	${INCDIR} 

js: main.bc
	$(CC) $(OBJDIR)/main.bc \
	$(OPENCV_INSTALL_DIR)/lib/libopencv_core.so \
	$(OPENCV_INSTALL_DIR)/lib/libopencv_imgproc.so \
	-O3 \
	--llvm-lto 1 \
	--closure 1 \
	-s PRECISE_F32=1 \
	-s ASM_JS=1 \
	-s EXPORTED_FUNCTIONS="['_initGL', '_Draw', '_DrawImage']" \
	-o $(OBJDIR)/out.js

install:
	rsync -rupE $(OBJDIR)/*.js $(WEBAPP)
	rsync -rupE $(OBJDIR)/*.js.mem $(WEBAPP)
	#rsync -rupE $(OBJDIR)/*.data $(WEBAPP)
