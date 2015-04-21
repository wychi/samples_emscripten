OBJDIR=./out
WEBAPP=./webapp

all: js

inter-ops.bc: ./src/inter-ops.cpp
	mkdir -p $(OBJDIR)
	$(CC) ./src/inter-ops.cpp -o $(OBJDIR)/$@

js: inter-ops.bc
	$(CC) \
	$(OBJDIR)/inter-ops.bc \
	-O3 \
	--llvm-lto 1 \
	--closure 1 \
	-s PRECISE_F32=1 \
	-s EXPORTED_FUNCTIONS="['_c_foo', '_c_add', '_c_str_in', '_c_str_out']" \
	-s ASSERTIONS=1 \
	-o $(OBJDIR)/out.js

html: inter-ops.bc
	$(CC) \
	$(OBJDIR)/inter-ops.bc \
	-O3 \
	--llvm-lto 1 \
	--closure 1 \
	-s PRECISE_F32=1 \
	-s TOTAL_MEMORY=318767104 \
	-s ASM_JS=1 \
	-s EXPORTED_FUNCTIONS="['_c_foo', '_c_add', '_c_str_in', '_c_str_out']" \
	-s ASSERTIONS=1 \
	-o $(OBJDIR)/shell.html

clean:
	rm $(OBJDIR)/*.js $(OBJDIR)/*.mem
	
install:
	mkdir -p $(WEBAPP)
	rsync -rupE $(OBJDIR)/*.js $(WEBAPP)/
	rsync -rupE $(OBJDIR)/*.js.mem $(WEBAPP)/
