// Exact .text from SAGE SmeProbe.S assembled for AArch64 Linux.
.text
.p2align 2
.global sme_probe_multiply
.type sme_probe_multiply,%function
sme_probe_multiply:
    .inst 0x6dbc27e8
    .inst 0x6d012fea
    .inst 0x6d0237ec
    .inst 0x6d033fee
    .inst 0xd503477f
    .inst 0x0420e3e9
    .inst 0xd280002a
    .inst 0x25aa1fe0
    .inst 0x8540c020
    .inst 0x8540c041
    .inst 0xc00800ff
    .inst 0x80810000
    .inst 0x5280000c
    .inst 0xe0bf0000
    .inst 0xd503467f
    .inst 0xaa0903e0
    .inst 0x6d412fea
    .inst 0x6d4237ec
    .inst 0x6d433fee
    .inst 0x6cc427e8
    .inst 0xd65f03c0
.size sme_probe_multiply,.-sme_probe_multiply
.section .note.GNU-stack,"",%progbits
