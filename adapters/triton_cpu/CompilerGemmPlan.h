#ifndef SME_PLAN_TRITON_CPU_COMPILER_GEMM_PLAN_H
#define SME_PLAN_TRITON_CPU_COMPILER_GEMM_PLAN_H

#include "sme_plan/CompilerGemmPolicy.h"
#include <memory>
#include <string>

namespace mlir {
class Pass;
}
namespace sme_plan::triton_cpu {
// policy.problem is deliberately ignored: the pass proves problem facts from
// TTIR. The caller transports verified device/calibration facts, not a tile.
std::unique_ptr<mlir::Pass> createCompilerGemmPlanPass(std::string mode,
                                                       CompilerGemmPolicyRequest policy,
                                                       std::int64_t calibrationProbeIndex = -1);
} // namespace sme_plan::triton_cpu
#endif
