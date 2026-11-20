#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

extern uint64_t sme_probe_multiply(float *, const float *, const float *);

int main(void) {
    float out[] = {-777.0f, -777.0f, -777.0f};
    float a = 2.0f, b = 3.0f;
    uint64_t first_width = sme_probe_multiply(&out[1], &a, &b);
    float first_value = out[1];
    a = -4.0f;
    b = 1.25f;
    uint64_t width = sme_probe_multiply(&out[1], &a, &b);
    int passed = first_width == width && width >= 16 && width <= 256 &&
        width % 16 == 0 && first_value == 6 && out[1] == -5 &&
        out[0] == -777 && out[2] == -777;
    printf("{\"passed\":%s,\"executed\":true,\"svl_bytes\":%" PRIu64
           ",\"first\":%.1f,\"second\":%.1f,\"guards\":%s}\n",
           passed ? "true" : "false", width, first_value, out[1],
           out[0] == -777 && out[2] == -777 ? "true" : "false");
    return passed ? 0 : 2;
}
