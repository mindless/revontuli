#include <stdio.h>
#include <string.h>
int main(void){
    printf("sizeof(_Float16)=%zu\n", sizeof(_Float16));
    float vals[] = {-0.5f, -0.499f, 1.0f, 0.001f, 0.5f, 2.0f, -1.0f};
    for (unsigned i=0;i<sizeof(vals)/sizeof(vals[0]);i++){
        _Float16 h = (_Float16)vals[i];
        unsigned short b; memcpy(&b,&h,2);
        float back = (float)h;
        printf("  %-10g -> 0x%04x -> %-12g\n", vals[i], b, back);
    }
    return 0;
}
