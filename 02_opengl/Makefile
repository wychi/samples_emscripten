OBJDIR=./out
WEBAPP=./webapp

INCDIR=-I./src/Common
#LIBS=-lGLESv2 -lEGL -lm -lX11

COMMONSRC=./src/Common/esShader.c    \
          ./src/Common/esTransform.c \
          ./src/Common/esShapes.c    \
          ./src/Common/esUtil.c

COMMONHRD=esUtil.h

all: js html

Hello_Triangle.bc: ${COMMONSRC} ${COMMONHDR} ./src/Hello_Triangle.c
	mkdir -p $(OBJDIR)
	$(CC) ${COMMONSRC} ./src/Hello_Triangle.c \
	-o $(OBJDIR)/$@ \
	${INCDIR} 

js: Hello_Triangle.bc
	$(CC) $(OBJDIR)/Hello_Triangle.bc \
	-O3 \
	--llvm-lto 1 \
	--closure 1 \
	-s PRECISE_F32=1 \
	-s TOTAL_MEMORY=318767104 \
	-s ASM_JS=1 \
	-o $(OBJDIR)/out.js

html: Hello_Triangle.bc
	mkdir -p $(WEBAPP)
	$(CC) $(OBJDIR)/Hello_Triangle.bc \
	-O3 \
	--llvm-lto 1 \
	--closure 1 \
	-s PRECISE_F32=1 \
	-s TOTAL_MEMORY=318767104 \
	-s ASM_JS=1 \
	-o $(WEBAPP)/index.html
