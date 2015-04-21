#include <stdio.h>

const char* STR_HELLO_WORLD = "hello world!";

extern "C" void c_foo() {
	printf("c_foo is called\n");
}

extern "C" int c_add(int a, int b) {
	printf("c_add is called. args: a=%d, b=%d\n", a, b);
	return a+b;
}

extern "C" void c_str_in(const char* pStr) {
	printf("c_str_in is called. args: str=%s\n", pStr);
}

extern "C" const char* c_str_out() {
	printf("c_str_out is called.\n");
	return STR_HELLO_WORLD;
}

extern "C" void c_array_in(const char* pStr) {
	printf("c_str_in is called. args: str=%s\n", pStr);
}

int main() {
	printf("hello, world! from main()\n");
    return 0;
}
