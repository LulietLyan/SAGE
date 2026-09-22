#ifndef SME_PLAN_COMPILER_GEMM_POLICY_H
#define SME_PLAN_COMPILER_GEMM_POLICY_H

#include "sme_plan/ProblemFacts.h"
#include "sme_plan/TargetCapabilities.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace sme_plan {

enum class CompilerGemmPath { GenericCPU, SME };
enum class CompilerGemmAPack { Scalar, Vector4 };
enum class CompilerGemmBLayout { Original, LocalPanelPadSVL, GlobalPadSVL };

struct CompilerGemmSchedule {
  CompilerGemmPath path = CompilerGemmPath::SME;
  std::int64_t blockM = 0;
  std::int64_t blockN = 0;
  std::int64_t blockK = 0;
  // Zero means the full logical K, rounded up to blockK for panel allocation.
  std::int64_t stripK = 0;
  CompilerGemmAPack aPack = CompilerGemmAPack::Scalar;
  CompilerGemmBLayout bLayout = CompilerGemmBLayout::Original;
  std::int64_t threads = 1;
  std::string id() const;
};

// Independently measured row-major leading-dimension cliffs. columns is the
// FP32 row width that thrashed; padColumns is the SVL-aligned cure used in
// the paired measurement (currently the native B-pad contract). severity is
// conflictGBps/paddedGBps in (0,1]. Empty means "no measured cliff".
// This does not invent topology from a device name: only microbenchmarked
// widths enter the list. A/C cannot yet be padded; K cliffs only slow A pack.
struct CompilerGemmRowStrideConflict {
  std::int64_t columns = 0;
  std::int64_t padColumns = 0;
  double severity = 1.0;
};

// These describe an independently verified lowering contract. Hardware feature
// detection alone cannot set nativeSmeGemmVerified or authorize a transform.
// Thread counts are software teams, never an inferred count of SME units.
struct CompilerGemmDeviceFacts {
  std::string identity;
  bool nativeSmeGemmVerified = false;
  bool supportsVectorAPack = false;
  bool supportsStreamingK = false;
  bool supportsPaddedB = false;
  bool supportsGlobalPaddedB = false;
  bool supportsGenericCPU = false;
  std::int64_t microM = 0;
  std::int64_t microN = 0;
  std::int64_t microK = 1;
  std::int64_t totalThreadBudget = 1;
  // Hard resource limits supplied by the compiler/runtime, not guessed caches.
  std::uint64_t maxAPanelBytesPerTask = 0;
  std::uint64_t maxScratchBytesPerTask = 0;
  // Shared full B allocation is independent of each worker's stack scratch.
  std::uint64_t maxGlobalPanelBytes = 0;
  std::vector<CompilerGemmRowStrideConflict> rowStrideConflicts;
};

// Empirical complete-call residual normalized by an operand-load traffic proxy,
// not a physical bandwidth measurement. Points interpolate on log(panel bytes)
// and clamp at endpoints. Equal bytes alone do not authorize another strategy.
struct CompilerGemmPanelResponse {
  std::uint64_t panelBytes = 0;
  double operandBandwidthGBps = 0.0;
};

// Response coverage for one exact compiled strategy family. Full K and
// streaming K, macro tiles, A packing, and B layout must never share a response
// merely because their panel byte counts coincide. Within a streaming family,
// different strip capacities may contribute points to the same byte curve.
// Streaming candidates must lie within the observed panel-byte interval;
// full-K candidates may use endpoint clamps, with independent validation.
// This is not a logical-shape-to-schedule table and contains no timing winner.
struct CompilerGemmStrategyResponse {
  std::int64_t blockM = 0;
  std::int64_t blockN = 0;
  std::int64_t blockK = 0;
  CompilerGemmAPack aPack = CompilerGemmAPack::Scalar;
  CompilerGemmBLayout bLayout = CompilerGemmBLayout::Original;
  bool streamingK = false;
  std::int64_t independentProcesses = 0;
  std::vector<CompilerGemmPanelResponse> panelResponses;
};

// A macro-tile family actually compiled, executed, and checked in independent
// processes. This records coverage, never which schedule won a timing contest.
// It constrains the domain of the empirical panel-response model: equal panel
// byte sizes do not establish equivalent behavior for different macro tiles.
struct CompilerGemmObservedTile {
  std::int64_t blockM = 0;
  std::int64_t blockN = 0;
  std::int64_t blockK = 0;
  std::int64_t independentProcesses = 0;
};

// Aggregate rates for exactly this actual team size, measured on this device.
// copyGBps measures local B-panel copy read+write bandwidth.
// GB/s is decimal bytes/ns; GFLOP/s is FLOPs/ns. The calibration procedure must
// record its raw samples externally. No speedup is extrapolated to unmeasured
// thread counts. Scalar/vector packing and original/padded operand bandwidth
// are separate because these transformations must earn their modeled cost.
struct CompilerGemmTeamCalibration {
  std::int64_t requestedThreads = 0;
  std::int64_t actualThreads = 0;
  std::int64_t independentRounds = 0;
  bool correct = false;
  bool pathVerified = false;
  double smeGflops = 0.0;
  double genericGflops = 0.0;
  double scalarAPackGBps = 0.0;
  double vectorAPackGBps = 0.0;
  double operandBandwidthGBps = 0.0;
  double paddedOperandBandwidthGBps = 0.0;
  double copyGBps = 0.0;
  // Complete shared full-B copy region, including its own OpenMP submit/wait;
  // 8*K*N active bytes, measured separately from small local copy bandwidth.
  double globalBPackGBps = 0.0;
  double globalPaddedOperandBandwidthGBps = 0.0;
  double callOverheadNs = 0.0;
  // Marginal per-output-task submission/communication cost, measured from
  // the empty-team versus empty-task native calibration.
  double taskOverheadNs = 0.0;
  // Cross-worker synchronization/communication latency paid once per
  // output wave. This is measured independently from task dispatch; it
  // makes the tile-size trade-off explicit because smaller tiles create
  // more waves for a fixed thread team.
  double communicationNsPerWave = 0.0;
  double stripOverheadNs = 0.0;
  std::vector<CompilerGemmPanelResponse> panelResponses;
  std::vector<CompilerGemmPanelResponse> paddedPanelResponses;
  std::vector<CompilerGemmPanelResponse> globalPanelResponses;
  std::vector<CompilerGemmObservedTile> observedNativeTiles;
  std::vector<CompilerGemmObservedTile> observedGenericTiles;
  // Production SME estimation uses only the matching strategy's points.
  // The pooled rates/points above remain diagnostics and a generic CPU proxy;
  // they are never a fallback for an absent native strategy response.
  std::vector<CompilerGemmStrategyResponse> strategyResponses;
};

struct CompilerGemmCalibration {
  std::string deviceIdentity;
  std::string evidenceId;
  std::int64_t minimumIndependentRounds = 3;
  std::vector<CompilerGemmTeamCalibration> teams;
};

// A family of compiler-supported shapes, not a table of benchmark dimensions.
// M/N factors multiply the native micro-tile. The adapter may narrow these
// lists to its proven lowering contract. K strips are blockK times a factor.
struct CompilerGemmSearchSpace {
  std::vector<std::int64_t> mFactors{1, 2, 4, 8};
  std::vector<std::int64_t> nFactors{1, 2, 4, 8, 16, 32};
  std::vector<std::int64_t> blockKs{16, 32};
  std::vector<std::int64_t> stripFactors{4, 8, 16, 32, 64};
  bool includeFullK = true;
  bool includeScalarAPack = true;
  bool includeVectorAPack = true;
  bool includeOriginalB = true;
  bool includePaddedB = true;
  bool includeGlobalPaddedB = true;
  bool includeGenericCPU = true;
};

struct CompilerGemmPolicyRequest {
  ProblemFacts problem;
  TargetCapabilities capabilities;
  CompilerGemmDeviceFacts device;
  CompilerGemmCalibration calibration;
  CompilerGemmSearchSpace search;
};

struct CompilerGemmCost {
  CompilerGemmSchedule schedule;
  double predictedNs = 0.0;
  double arithmeticNs = 0.0;
  double operandReadNs = 0.0;
  double aPackNs = 0.0;
  // B packing is per task and strip; output stays in its original layout.
  double bPanelPackNs = 0.0;
  double globalBPackNs = 0.0;
  double schedulingNs = 0.0;
  double taskOccupancy = 0.0;
  std::uint64_t outputTasks = 0;
  std::uint64_t kStrips = 0;
  std::uint64_t aPanelBytesPerTask = 0;
  std::uint64_t bPanelBytesPerTask = 0;
  std::uint64_t scratchBytesPerTask = 0;
  std::uint64_t globalBPanelBytes = 0;
};

struct CompilerGemmPolicyDecision {
  bool apply = false;
  std::optional<CompilerGemmSchedule> selected;
  // Sorted predicted complete-call costs; these are NOT measured latencies.
  std::vector<CompilerGemmCost> rankedCandidates;
  std::vector<std::string> reasonCodes;
  std::string calibrationEvidenceId;
};

// Requires proven static GEMM shape, dense row-major FP32 A/B/C, and explicit device/calibration
// identity. Unsupported or unknown facts return apply=false with reasons.
// A macro-tile family must have been observed in the required number of
// independent processes for this actual team. No inference from panel bytes
// alone may authorize an unobserved BM/BN/BK family. The exact native strategy
// must also have enough independent processes and positive response points.
// Global M/N/K remain prediction inputs, not benchmark-shape lookup keys.
// This ranks candidates. Independent full-call measurements still establish
// realized speedup; it does not prove a hardware cache or SME-unit topology.
// Native candidate generation is indexed by the exact calibrated strategy
// families. Search-space factors act as filters, so large caller lists do
// not trigger a Cartesian warmup search over unobserved combinations.
CompilerGemmPolicyDecision planCompilerGemm(const CompilerGemmPolicyRequest &request);

struct CompilerGemmCalibrationProbe {
  std::int64_t index = -1;
  CompilerGemmSchedule schedule;
  bool supported = false;
  std::uint64_t aPanelBytesPerTask = 0;
  std::uint64_t bPanelBytesPerTask = 0;
  std::uint64_t scratchBytesPerTask = 0;
  std::uint64_t globalBPanelBytes = 0;
  std::vector<std::string> reasonCodes;
};

// A small structural probe family compiled from the identical canonical Triton
// operation. No calibration values or historical selections are read. Indices
// are stable, including unsupported probes: the caller must reject those with
// their reasons, never silently replace them with a different schedule.
// M/N dimensions are multiples of the verified compiler micro-tile; K values
// describe lowering alternatives, never problem-size selection thresholds.
// Actual team creation, emitted instructions, execution, and correctness must
// still be checked by the calibration runner before accepting measurements.
std::vector<CompilerGemmCalibrationProbe>
compilerGemmCalibrationSchedules(const CompilerGemmPolicyRequest &request);

} // namespace sme_plan

#endif
