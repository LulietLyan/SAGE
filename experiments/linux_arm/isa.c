#include <errno.h>
#include <stdio.h>
#include <sys/auxv.h>
#include <sys/prctl.h>

static void vector_length(const char *key, int request) {
    errno = 0;
    int result = prctl(request, 0UL, 0UL, 0UL, 0UL);
    printf("%s_return=%d errno=%d length_bytes=%d\n",
           key, result, errno, result < 0 ? -1 : (result & 0xffff));
}

int main(void) {
#if !defined(__aarch64__)
#error This probe is for Linux AArch64
#endif
    unsigned long cap = getauxval(AT_HWCAP);
    unsigned long cap2 = getauxval(AT_HWCAP2);
    printf("AT_HWCAP=0x%lx AT_HWCAP2=0x%lx\n", cap, cap2);
    printf("sve=%lu sve2=%lu sme=%lu sme2=%lu sme_f32f32=%lu\n",
           (cap >> 22) & 1, (cap2 >> 1) & 1, (cap2 >> 23) & 1,
           (cap2 >> 37) & 1, (cap2 >> 29) & 1);
    vector_length("PR_SVE_GET_VL", 51);
    vector_length("PR_SME_GET_VL", 64);
    return 0;
}
