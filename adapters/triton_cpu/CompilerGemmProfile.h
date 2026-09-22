#ifndef SME_PLAN_COMPILER_GEMM_PROFILE_H
#define SME_PLAN_COMPILER_GEMM_PROFILE_H
#include "sme_plan/CompilerGemmPolicy.h"
#include "llvm/Support/Error.h"
#include "llvm/ADT/StringRef.h"
namespace sme_plan::triton_cpu {
llvm::Expected<CompilerGemmPolicyRequest> readCompilerGemmProfile(llvm::StringRef json,
                                                                  int64_t threadBudget);
}
#endif
