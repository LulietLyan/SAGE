// Private capability probe, called only by the fresh SmeProbe executable.
// Normal AAPCS entry/return: no incoming streaming mode or live ZA is assumed.
// x0 = one-float output, x1/x2 = scalar inputs; returns streaming bytes in x0.
// Saving d8-d15 is necessary because changing streaming mode clears SIMD state.
#if defined(__APPLE__)
#define PROBE_NAME _sme_probe_multiply
#else
#define PROBE_NAME sme_probe_multiply
#endif

.text
.p2align 2
.global PROBE_NAME
PROBE_NAME:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    smstart
    cntb x9
    mov x10, #1
    whilelo p0.s, xzr, x10
    ld1rw {z0.s}, p0/z, [x1]
    ld1rw {z1.s}, p0/z, [x2]
    zero {za}
    fmopa za0.s, p0/m, p0/m, z0.s, z1.s
    mov w12, #0
    st1w {za0h.s[w12, 0]}, p0, [x0]
    smstop
    mov x0, x9
    ldp d10, d11, [sp, #16]
    ldp d12, d13, [sp, #32]
    ldp d14, d15, [sp, #48]
    ldp d8, d9, [sp], #64
    ret
