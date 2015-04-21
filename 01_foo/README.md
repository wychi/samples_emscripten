# Reference
http://people.mozilla.org/~lwagner/gdc-pres/gdc-2014.html

#Gotch
To output log from native side, you have to put “\n” at the end of string.

    #include <stdio.h>
    extern "C" void c_foo() {
      printf("c_foo is called\n");
    }


Build

    mkdir -p $(OBJDIR)
    emcc \
      ./src/inter-ops.cpp \
      -Werror \
      -O3 \
      --llvm-lto 1 \
      --closure 0 \
      -s PRECISE_F32=1 \
      -s TOTAL_MEMORY=318767104 \
      -s ASSERTIONS=1 \
      -s ASM_JS=1 \
      -s EXPORTED_FUNCTIONS="['_c_foo', '_c_add', '_c_str_in', '_c_str_out']" \
      -o ./bin/out.js

If you enable closure with 1, the global object, Module, will be renamed.
So you will encounter "Module is undefined" in runtime.

Disable closure for now.

To expose functions to javascript. -s ASSERTIONS=1 can help you.

    -s EXPORTED_FUNCTIONS="['_c_foo', '_c_add', '_c_str_in', '_c_str_out']" \
    -s ASSERTIONS=1 \