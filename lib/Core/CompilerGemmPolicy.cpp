#include "sme_plan/CompilerGemmPolicy.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <tuple>

namespace sme_plan {
namespace {

using U64 = std::uint64_t;

U64 ceilDiv(U64 a, U64 b) { return a / b + (a % b != 0); }

bool multiply(std::initializer_list<U64> values, U64 &result) {
  result = 1;
  for (U64 v : values) {
    if (v && result > std::numeric_limits<U64>::max() / v)
      return false;
    result *= v;
  }
  return true;
}

bool add(U64 a, U64 b, U64 &result) {
  if (b > std::numeric_limits<U64>::max() - a)
    return false;
  result = a + b;
  return true;
}

bool positive(double value) { return std::isfinite(value) && value > 0.0; }
bool nonnegative(double value) { return std::isfinite(value) && value >= 0.0; }

// Combine measured stride penalties with the retained 2048-column heuristic.
// The fixed penalty applies even without matching calibration evidence; it
// must not be interpreted as a discovered cache period on an arbitrary device.
double conflictScale(const CompilerGemmDeviceFacts &device, U64 columns) {
  double scale = 1.0;
  for (const auto &cliff : device.rowStrideConflicts) {
    if (cliff.columns <= 0 || static_cast<U64>(cliff.columns) != columns)
      continue;
    if (!positive(cliff.severity) || cliff.severity > 1.0)
      continue;
    scale = std::min(scale, std::max(0.05, cliff.severity));
  }
  // Exact multiples, including unmeasured widths, receive the fixed penalty.
  // Other widths use only matching calibration records.
  if (columns >= 2048 && columns % 2048ull == 0ull)
    scale = std::min(scale, 0.40);
  return scale;
}

bool trueFact(const Fact<bool> &fact) { return fact.isProven() && *fact.value; }

const TensorFacts *operand(const ProblemFacts &problem, const char *role) {
  const TensorFacts *found = nullptr;
  for (const auto &tensor : problem.kernel.operands) {
    if (tensor.role != role)
      continue;
    if (found)
      return nullptr;
    found = &tensor;
  }
  return found;
}

bool isDenseF32(const TensorFacts *tensor, std::int64_t rows, std::int64_t columns) {
  if (!tensor || !tensor->dtype.isProven() || *tensor->dtype.value != "f32" ||
      tensor->logicalShape.size() != 2 || tensor->physicalStrides.size() != 2)
    return false;
  const std::int64_t shape[] = {rows, columns};
  const std::int64_t strides[] = {columns, 1};
  for (std::size_t i = 0; i < 2; ++i)
    if (!tensor->logicalShape[i].isProven() || *tensor->logicalShape[i].value != shape[i] ||
        !tensor->physicalStrides[i].isProven() || *tensor->physicalStrides[i].value != strides[i])
      return false;
  return true;
}

bool validResponses(const std::vector<CompilerGemmPanelResponse> &responses) {
  std::set<U64> sizes;
  return std::all_of(responses.begin(), responses.end(), [&](const auto &p) {
    return p.panelBytes > 0 && positive(p.operandBandwidthGBps) &&
           sizes.insert(p.panelBytes).second;
  });
}

bool validObservedTiles(const std::vector<CompilerGemmObservedTile> &tiles) {
  std::set<std::tuple<std::int64_t, std::int64_t, std::int64_t>> seen;
  return std::all_of(tiles.begin(), tiles.end(), [&](const auto &tile) {
    return tile.blockM > 0 && tile.blockN > 0 && tile.blockK > 0 && tile.independentProcesses > 0 &&
           seen.insert({tile.blockM, tile.blockN, tile.blockK}).second;
  });
}

bool observedTile(const std::vector<CompilerGemmObservedTile> &tiles,
                  const CompilerGemmSchedule &schedule, std::int64_t minimumProcesses) {
  return std::any_of(tiles.begin(), tiles.end(), [&](const auto &tile) {
    return tile.blockM == schedule.blockM && tile.blockN == schedule.blockN &&
           tile.blockK == schedule.blockK && tile.independentProcesses >= minimumProcesses;
  });
}

bool validStrategyResponses(const std::vector<CompilerGemmStrategyResponse> &strategies) {
  std::set<std::tuple<std::int64_t, std::int64_t, std::int64_t, CompilerGemmAPack,
                      CompilerGemmBLayout, bool>>
      seen;
  return std::all_of(strategies.begin(), strategies.end(), [&](const auto &s) {
    const bool packValid =
        s.aPack == CompilerGemmAPack::Scalar || s.aPack == CompilerGemmAPack::Vector4;
    const bool layoutValid = s.bLayout == CompilerGemmBLayout::Original ||
                             s.bLayout == CompilerGemmBLayout::LocalPanelPadSVL ||
                             s.bLayout == CompilerGemmBLayout::GlobalPadSVL;
    return s.blockM > 0 && s.blockN > 0 && s.blockK > 0 && s.independentProcesses > 0 &&
           packValid && layoutValid && validResponses(s.panelResponses) &&
           seen.insert({s.blockM, s.blockN, s.blockK, s.aPack, s.bLayout, s.streamingK}).second;
  });
}

const CompilerGemmStrategyResponse *strategyResponse(const CompilerGemmTeamCalibration &team,
                                                     const CompilerGemmSchedule &schedule,
                                                     std::int64_t minimumProcesses) {
  for (const auto &s : team.strategyResponses)
    if (s.blockM == schedule.blockM && s.blockN == schedule.blockN && s.blockK == schedule.blockK &&
        s.aPack == schedule.aPack && s.bLayout == schedule.bLayout &&
        s.streamingK == (schedule.stripK > 0) && s.independentProcesses >= minimumProcesses &&
        !s.panelResponses.empty())
      return &s;
  return nullptr;
}

double responseBandwidth(std::vector<CompilerGemmPanelResponse> responses, U64 bytes,
                         double fallback) {
  if (responses.empty())
    return fallback;
  std::sort(responses.begin(), responses.end(),
            [](const auto &a, const auto &b) { return a.panelBytes < b.panelBytes; });
  if (bytes <= responses.front().panelBytes)
    return responses.front().operandBandwidthGBps;
  if (bytes >= responses.back().panelBytes)
    return responses.back().operandBandwidthGBps;
  for (std::size_t i = 1; i < responses.size(); ++i) {
    if (bytes > responses[i].panelBytes)
      continue;
    const auto &lo = responses[i - 1];
    const auto &hi = responses[i];
    double alpha = std::log(static_cast<double>(bytes) / lo.panelBytes) /
                   std::log(static_cast<double>(hi.panelBytes) / lo.panelBytes);
    return lo.operandBandwidthGBps * (1.0 - alpha) + hi.operandBandwidthGBps * alpha;
  }
  return fallback;
}

bool positiveList(const std::vector<std::int64_t> &values) {
  return !values.empty() && std::all_of(values.begin(), values.end(), [](auto v) { return v > 0; });
}

bool contains(const std::vector<std::int64_t> &values, std::int64_t value) {
  return std::find(values.begin(), values.end(), value) != values.end();
}

std::optional<CompilerGemmCost> estimate(const CompilerGemmPolicyRequest &request,
                                         const CompilerGemmTeamCalibration &team,
                                         const CompilerGemmSchedule &schedule, U64 m, U64 n, U64 k,
                                         U64 lanes) {
  CompilerGemmCost cost;
  cost.schedule = schedule;
  const U64 bm = schedule.blockM, bn = schedule.blockN, bk = schedule.blockK;
  const U64 threads = schedule.threads;
  const bool sme = schedule.path == CompilerGemmPath::SME;
  const bool padded = schedule.bLayout == CompilerGemmBLayout::LocalPanelPadSVL;
  const bool global = schedule.bLayout == CompilerGemmBLayout::GlobalPadSVL;
  const auto *strategy =
      sme ? strategyResponse(team, schedule, request.calibration.minimumIndependentRounds)
          : nullptr;
  if (sme && !strategy)
    return std::nullopt;
  const U64 physicalN = n;
  U64 roundedK, stripCapacity;
  if (!multiply({ceilDiv(k, bk), bk}, roundedK))
    return std::nullopt;
  stripCapacity = schedule.stripK ? static_cast<U64>(schedule.stripK) : roundedK;
  cost.kStrips = ceilDiv(k, stripCapacity);
  if (!multiply({ceilDiv(m, bm), ceilDiv(physicalN, bn)}, cost.outputTasks))
    return std::nullopt;
  U64 waves = ceilDiv(cost.outputTasks, threads), issuedSlots;
  if (!multiply({waves, threads}, issuedSlots))
    return std::nullopt;
  cost.taskOccupancy = static_cast<double>(cost.outputTasks) / issuedSlots;
  if (sme) {
    if (!multiply({4, bm, stripCapacity}, cost.aPanelBytesPerTask) ||
        cost.aPanelBytesPerTask > request.device.maxAPanelBytesPerTask)
      return std::nullopt;
  }
  if (global) {
    U64 globalStride;
    if (!add(n, lanes, globalStride) || !multiply({4, k, globalStride}, cost.globalBPanelBytes) ||
        cost.globalBPanelBytes > request.device.maxGlobalPanelBytes ||
        !positive(team.globalBPackGBps))
      return std::nullopt;
    // A single shared full-B copy region; its separately measured complete
    // throughput already includes that region's OpenMP submission and wait.
    // Source walks ldb=N, so a measured N cliff still slows this one copy.
    const double packGBps = team.globalBPackGBps * conflictScale(request.device, physicalN);
    cost.globalBPackNs = static_cast<double>(8.0L * k * n / packGBps);
  }
  // Reserve output-tile space even if a specific lowering stores directly to
  // C. This protects the thread stack against transient compiler allocations.
  U64 cTileBytes = 0, operandPanelBytes = cost.aPanelBytesPerTask;
  if (sme) {
    if (padded) {
      U64 bStride;
      if (!add(bn, lanes, bStride) ||
          !multiply({4, stripCapacity, bStride}, cost.bPanelBytesPerTask) ||
          !add(operandPanelBytes, cost.bPanelBytesPerTask, operandPanelBytes))
        return std::nullopt;
    }
    if (!multiply({4, bm, bn}, cTileBytes) ||
        !add(operandPanelBytes, cTileBytes, cost.scratchBytesPerTask) ||
        cost.scratchBytesPerTask > request.device.maxScratchBytesPerTask)
      return std::nullopt;
    if (schedule.stripK) {
      const auto limits = std::minmax_element(
          strategy->panelResponses.begin(), strategy->panelResponses.end(),
          [](const auto &a, const auto &b) { return a.panelBytes < b.panelBytes; });
      if (operandPanelBytes < limits.first->panelBytes ||
          operandPanelBytes > limits.second->panelBytes)
        return std::nullopt;
    }
  }

  // Hardware micro-tile rounding describes executed arithmetic, while API
  // throughput must continue to use the original logical 2*M*N*K FLOPs.
  U64 arithmeticM = m, arithmeticN = physicalN;
  if (sme &&
      (!multiply({ceilDiv(m, request.device.microM), static_cast<U64>(request.device.microM)},
                 arithmeticM) ||
       !multiply(
           {ceilDiv(physicalN, request.device.microN), static_cast<U64>(request.device.microN)},
           arithmeticN)))
    return std::nullopt;
  const long double flops = 2.0L * arithmeticM * arithmeticN * k;
  const double rate = sme ? team.smeGflops : team.genericGflops;
  cost.arithmeticNs = static_cast<double>(flops / rate / cost.taskOccupancy);
  // Physical B leading dimension for this layout. Original walks ldb=N;
  // local panels walk BN+λ; global walks N+λ. Measured cliffs scale cost.
  const U64 physicalBColumns = !sme     ? physicalN
                               : padded ? bn + lanes
                               : global ? physicalN + lanes
                                        : physicalN;
  const double bCliff = conflictScale(request.device, physicalBColumns);

  if (sme) {
    const double packing =
        schedule.aPack == CompilerGemmAPack::Vector4 ? team.vectorAPackGBps : team.scalarAPackGBps;
    if (!positive(packing))
      return std::nullopt;
    // A is read with lda=K before packing. A measured K cliff slows pack
    // only; the packed panel itself is not row-major with stride K.
    const double packRate = packing * conflictScale(request.device, k);
    // Every output-N macro tile reads and writes its own packed A panel.
    long double aPackBytes = 8.0L * m * k * ceilDiv(physicalN, bn);
    cost.aPackNs = static_cast<double>(aPackBytes / packRate / cost.taskOccupancy);
    // No pooled layout or other strategy may replace missing evidence.
    double bandwidth = responseBandwidth(strategy->panelResponses, operandPanelBytes, 0.0);
    if (!positive(bandwidth))
      return std::nullopt;
    bandwidth *= bCliff;
    if (!positive(bandwidth))
      return std::nullopt;
    // Operand traffic at the micro-kernel boundary, including repeated A/B
    // loads; the independently measured response captures cache residency.
    long double readBytes = 4.0L * arithmeticM * k * ceilDiv(physicalN, request.device.microN) +
                            4.0L * physicalN * k * ceilDiv(m, request.device.microM);
    // The first strip writes C; each later strip reads and writes C.
    // C keeps logical ldc=N; B-side cliffs are what select padding.
    readBytes += 4.0L * m * physicalN * (2.0L * cost.kStrips - 1.0L);
    cost.operandReadNs = static_cast<double>(readBytes / bandwidth / cost.taskOccupancy);
  } else {
    const long double bytes =
        4.0L * (static_cast<long double>(m) * k + static_cast<long double>(k) * n +
                static_cast<long double>(m) * n);
    const double cpuBandwidth = team.operandBandwidthGBps * bCliff;
    cost.operandReadNs = static_cast<double>(bytes / cpuBandwidth / cost.taskOccupancy);
  }
  if (padded) {
    if (!positive(team.copyGBps))
      return std::nullopt;
    // Each output-M macro tile copies B again. Source walks ldb=N, so an N
    // cliff remains in the pack term; compute then uses BN+λ.
    const double copyRate = team.copyGBps * conflictScale(request.device, physicalN);
    const long double mTiles = ceilDiv(m, bm);
    const long double copied = 8.0L * mTiles * k * n;
    cost.bPanelPackNs = static_cast<double>(copied / copyRate / cost.taskOccupancy);
  }
  // taskOverheadNs is measured as the marginal cost of one submitted
  // task (emptyTasks - emptyTeam, divided by the added task count).  The
  // previous model multiplied it by waves, under-counting communication by
  // roughly the team size when a wave contains many small tiles.  Charge
  // each output task explicitly; wave-level and K-segment barriers remain
  // separate terms below.
  // Keep cross-worker synchronization separate from task dispatch.  The
  // calibrated value is an effective wave contention penalty: waves model
  // occupancy rounds, so this term captures the measured growth pressure
  // without claiming that the launcher emits a literal barrier per wave.
  cost.schedulingNs =
      team.callOverheadNs + static_cast<double>(cost.outputTasks) * team.taskOverheadNs +
      waves * team.communicationNsPerWave + waves * cost.kStrips * team.stripOverheadNs;
  // Bandwidth scaling alone is invisible when the arithmetic branch of the
  // max still dominates. Thrashing also stalls SME issue, so derate that
  // branch by the same measured cliff when the physical B stride matches.
  const double boundNs = std::max(cost.arithmeticNs / bCliff, cost.operandReadNs);
  cost.predictedNs =
      boundNs + cost.aPackNs + cost.bPanelPackNs + cost.globalBPackNs + cost.schedulingNs;
  if (!positive(cost.predictedNs))
    return std::nullopt;
  return cost;
}

} // namespace

std::string CompilerGemmSchedule::id() const {
  std::ostringstream s;
  s << (path == CompilerGemmPath::SME ? "sme" : "cpu") << ":m" << blockM << ":n" << blockN << ":k"
    << blockK << ":strip" << stripK << ":a"
    << (aPack == CompilerGemmAPack::Vector4 ? "vector4" : "scalar") << ":b"
    << (bLayout == CompilerGemmBLayout::GlobalPadSVL       ? "global-pad-svl"
        : bLayout == CompilerGemmBLayout::LocalPanelPadSVL ? "local-panel-pad-svl"
                                                           : "original")
    << ":t" << threads;
  return s.str();
}

CompilerGemmPolicyDecision planCompilerGemm(const CompilerGemmPolicyRequest &request) {
  CompilerGemmPolicyDecision decision;
  decision.calibrationEvidenceId = request.calibration.evidenceId;
  const auto &kernel = request.problem.kernel;
  const auto &device = request.device;
  const auto &calibration = request.calibration;
  const auto &search = request.search;
  auto fail = [&](const char *reason) {
    decision.reasonCodes.push_back(reason);
    return decision;
  };
  if (!kernel.operationKind.isProven() || *kernel.operationKind.value != "gemm" ||
      !kernel.m.isProven() || !kernel.n.isProven() || !kernel.k.isProven() || *kernel.m.value < 1 ||
      *kernel.n.value < 1 || *kernel.k.value < 1)
    return fail("requires-proven-positive-static-gemm-shape");
  const auto m = *kernel.m.value, n = *kernel.n.value, k = *kernel.k.value;
  if (!isDenseF32(operand(request.problem, "A"), m, k) ||
      !isDenseF32(operand(request.problem, "B"), k, n) ||
      !isDenseF32(operand(request.problem, "C"), m, n))
    return fail("requires-proven-dense-row-major-f32-operands");
  if (device.identity.empty() || calibration.deviceIdentity != device.identity ||
      calibration.evidenceId.empty() || calibration.minimumIndependentRounds < 1)
    return fail("missing-or-mismatched-calibration-identity");
  if (device.microM < 1 || device.microN < 1 || device.microK < 1 || device.totalThreadBudget < 1 ||
      !positiveList(search.mFactors) || !positiveList(search.nFactors) ||
      !positiveList(search.blockKs) ||
      (!search.stripFactors.empty() && !positiveList(search.stripFactors)))
    return fail("invalid-compiler-search-contract");
  const auto &caps = request.capabilities;
  const bool sme = device.nativeSmeGemmVerified && trueFact(caps.sme) &&
                   trueFact(caps.streamingZA) && caps.vectorLengthBits.isProven() &&
                   *caps.vectorLengthBits.value > 0 && *caps.vectorLengthBits.value % 32 == 0 &&
                   device.maxAPanelBytesPerTask > 0;
  if (!sme && !(device.supportsGenericCPU && search.includeGenericCPU))
    return fail("no-verified-compiler-compute-path");
  const U64 lanes = sme ? *caps.vectorLengthBits.value / 32 : 1;
  std::set<std::string> ids;
  std::set<std::int64_t> teamSizes;
  for (const auto &team : calibration.teams) {
    if (team.requestedThreads != team.actualThreads || team.actualThreads < 1 ||
        team.actualThreads > device.totalThreadBudget ||
        team.independentRounds < calibration.minimumIndependentRounds || !team.correct ||
        !team.pathVerified || !nonnegative(team.callOverheadNs) ||
        !nonnegative(team.taskOverheadNs) || !nonnegative(team.communicationNsPerWave) ||
        !nonnegative(team.stripOverheadNs) || !validResponses(team.panelResponses) ||
        !validResponses(team.paddedPanelResponses) || !validResponses(team.globalPanelResponses) ||
        !validObservedTiles(team.observedNativeTiles) ||
        !validObservedTiles(team.observedGenericTiles) ||
        !validStrategyResponses(team.strategyResponses)) {
      decision.reasonCodes.push_back("unverified-or-invalid-team-calibration");
      continue;
    }
    // Duplicates are ambiguous: rejecting the entire request avoids depending
    // on input order or silently picking a favorable measurement.
    if (!teamSizes.insert(team.actualThreads).second)
      return fail("duplicate-team-calibration");
    auto accept = [&](const CompilerGemmSchedule &schedule) {
      if (!ids.insert(schedule.id()).second)
        return;
      if (auto cost = estimate(request, team, schedule, m, n, k, lanes))
        decision.rankedCandidates.push_back(*cost);
    };
    // Generic CPU schedules have no strategy response curve, so retain the
    // small observed-tile walk. Native SME schedules are indexed directly by
    // the exact strategy families measured during calibration. The previous
    // implementation walked every factor/pack/layout combination and then
    // discarded almost all of them after a response lookup; indexing here
    // makes the production warmup proportional to measured families.
    if (device.supportsGenericCPU && search.includeGenericCPU && positive(team.genericGflops) &&
        positive(team.operandBandwidthGBps))
      for (const auto &tile : team.observedGenericTiles) {
        CompilerGemmSchedule candidate;
        candidate.path = CompilerGemmPath::GenericCPU;
        candidate.blockM = tile.blockM;
        candidate.blockN = tile.blockN;
        candidate.blockK = tile.blockK;
        candidate.threads = team.actualThreads;
        if (contains(search.mFactors, candidate.blockM / device.microM) &&
            candidate.blockM % device.microM == 0 &&
            contains(search.nFactors, candidate.blockN / device.microN) &&
            candidate.blockN % device.microN == 0 && contains(search.blockKs, candidate.blockK) &&
            observedTile(team.observedGenericTiles, candidate,
                         calibration.minimumIndependentRounds))
          accept(candidate);
      }
    if (sme && positive(team.smeGflops))
      for (const auto &family : team.strategyResponses) {
        if (family.blockM % device.microM != 0 || family.blockN % device.microN != 0 ||
            family.blockK % device.microK != 0 ||
            !contains(search.mFactors, family.blockM / device.microM) ||
            !contains(search.nFactors, family.blockN / device.microN) ||
            !contains(search.blockKs, family.blockK) ||
            !observedTile(team.observedNativeTiles,
                          CompilerGemmSchedule{CompilerGemmPath::SME, family.blockM, family.blockN,
                                               family.blockK, 0, family.aPack, family.bLayout,
                                               team.actualThreads},
                          calibration.minimumIndependentRounds))
          continue;
        CompilerGemmSchedule candidate;
        candidate.path = CompilerGemmPath::SME;
        candidate.blockM = family.blockM;
        candidate.blockN = family.blockN;
        candidate.blockK = family.blockK;
        candidate.aPack = family.aPack;
        candidate.bLayout = family.bLayout;
        candidate.threads = team.actualThreads;
        std::set<std::int64_t> strips;
        if (!family.streamingK) {
          if (search.includeFullK)
            strips.insert(0);
        } else if (device.supportsStreamingK)
          for (auto factor : search.stripFactors) {
            U64 strip;
            if (multiply({static_cast<U64>(family.blockK), static_cast<U64>(factor)}, strip) &&
                strip < static_cast<U64>(k) &&
                strip <= static_cast<U64>(std::numeric_limits<std::int64_t>::max()))
              strips.insert(static_cast<std::int64_t>(strip));
          }
        for (auto strip : strips) {
          candidate.stripK = strip;
          if ((candidate.aPack == CompilerGemmAPack::Scalar && !search.includeScalarAPack) ||
              (candidate.aPack == CompilerGemmAPack::Vector4 &&
               (!search.includeVectorAPack || !device.supportsVectorAPack || candidate.blockM % 4 ||
                candidate.blockK % 4)) ||
              (candidate.bLayout == CompilerGemmBLayout::Original && !search.includeOriginalB) ||
              (candidate.bLayout == CompilerGemmBLayout::LocalPanelPadSVL &&
               (!search.includePaddedB || !device.supportsPaddedB)) ||
              (candidate.bLayout == CompilerGemmBLayout::GlobalPadSVL &&
               (!search.includeGlobalPaddedB || !device.supportsGlobalPaddedB)))
            continue;
          accept(candidate);
        }
      }
  }
  std::sort(decision.rankedCandidates.begin(), decision.rankedCandidates.end(),
            [](const auto &a, const auto &b) {
              if (a.predictedNs != b.predictedNs)
                return a.predictedNs < b.predictedNs;
              // Deterministic ties prefer smaller workspaces and fewer threads.
              return std::make_tuple(a.globalBPanelBytes, a.scratchBytesPerTask, a.schedule.threads,
                                     a.aPanelBytesPerTask, a.schedule.id()) <
                     std::make_tuple(b.globalBPanelBytes, b.scratchBytesPerTask, b.schedule.threads,
                                     b.aPanelBytesPerTask, b.schedule.id());
            });
  if (decision.rankedCandidates.empty())
    return fail("no-calibrated-resource-legal-candidate");
  decision.apply = true;
  decision.selected = decision.rankedCandidates.front().schedule;
  decision.reasonCodes.push_back("calibrated-complete-call-cost-model-selected");
  decision.reasonCodes.push_back("macro-tiles-restricted-to-observed-independent-process-domain");
  decision.reasonCodes.push_back("native-response-restricted-to-exact-observed-strategy-family");
  decision.reasonCodes.push_back("strategy-response-indexed-search");
  decision.reasonCodes.push_back("communication-aware-task-and-wave-cost-model");
  decision.reasonCodes.push_back("prediction-requires-independent-execution-validation");
  return decision;
}

std::vector<CompilerGemmCalibrationProbe>
compilerGemmCalibrationSchedules(const CompilerGemmPolicyRequest &request) {
  struct Family {
    std::int64_t mFactor, nFactor, blockK, stripK;
    CompilerGemmAPack pack;
    CompilerGemmBLayout layout;
  };
  const Family families[] = {
      {4, 8, 16, 0, CompilerGemmAPack::Scalar, CompilerGemmBLayout::Original},
      {4, 32, 32, 0, CompilerGemmAPack::Vector4, CompilerGemmBLayout::Original},
      {8, 32, 32, 512, CompilerGemmAPack::Vector4, CompilerGemmBLayout::Original},
      {8, 32, 32, 1024, CompilerGemmAPack::Vector4, CompilerGemmBLayout::Original},
      {8, 32, 32, 512, CompilerGemmAPack::Vector4, CompilerGemmBLayout::LocalPanelPadSVL},
      {8, 32, 32, 1024, CompilerGemmAPack::Vector4, CompilerGemmBLayout::LocalPanelPadSVL},
      {4, 16, 32, 1024, CompilerGemmAPack::Vector4, CompilerGemmBLayout::LocalPanelPadSVL},
      {4, 32, 32, 0, CompilerGemmAPack::Vector4, CompilerGemmBLayout::GlobalPadSVL},
      {8, 32, 32, 512, CompilerGemmAPack::Vector4, CompilerGemmBLayout::GlobalPadSVL},
      {8, 32, 32, 1024, CompilerGemmAPack::Vector4, CompilerGemmBLayout::GlobalPadSVL},
  };
  const auto &device = request.device;
  const auto &kernel = request.problem.kernel;
  const auto &caps = request.capabilities;
  std::vector<std::string> commonReasons;
  const bool knownShape = kernel.operationKind.isProven() &&
                          *kernel.operationKind.value == "gemm" && kernel.m.isProven() &&
                          kernel.n.isProven() && kernel.k.isProven() && *kernel.m.value > 0 &&
                          *kernel.n.value > 0 && *kernel.k.value > 0;
  if (!knownShape)
    commonReasons.push_back("requires-proven-positive-static-gemm-shape");
  else if (!isDenseF32(operand(request.problem, "A"), *kernel.m.value, *kernel.k.value) ||
           !isDenseF32(operand(request.problem, "B"), *kernel.k.value, *kernel.n.value) ||
           !isDenseF32(operand(request.problem, "C"), *kernel.m.value, *kernel.n.value))
    commonReasons.push_back("requires-proven-dense-row-major-f32-operands");
  if (device.identity.empty() || device.microM < 1 || device.microN < 1 || device.microK < 1 ||
      device.totalThreadBudget < 1 || device.maxAPanelBytesPerTask == 0 ||
      device.maxScratchBytesPerTask == 0)
    commonReasons.push_back("invalid-compiler-device-contract");
  const bool nativePath = device.nativeSmeGemmVerified && trueFact(caps.sme) &&
                          trueFact(caps.streamingZA) && caps.vectorLengthBits.isProven() &&
                          *caps.vectorLengthBits.value > 0 &&
                          *caps.vectorLengthBits.value % 32 == 0;
  if (!nativePath)
    commonReasons.push_back("no-verified-native-sme-compiler-path");
  std::vector<CompilerGemmCalibrationProbe> probes;
  for (const auto &family : families) {
    CompilerGemmCalibrationProbe probe;
    probe.index = probes.size();
    probe.reasonCodes = commonReasons;
    auto &s = probe.schedule;
    s.blockK = family.blockK;
    s.stripK = family.stripK;
    s.aPack = family.pack;
    s.bLayout = family.layout;
    s.threads = device.totalThreadBudget;
    U64 bm = 0, bn = 0;
    if (device.microM > 0 && device.microN > 0 &&
        multiply({static_cast<U64>(device.microM), static_cast<U64>(family.mFactor)}, bm) &&
        multiply({static_cast<U64>(device.microN), static_cast<U64>(family.nFactor)}, bn) &&
        bm <= static_cast<U64>(std::numeric_limits<std::int64_t>::max()) &&
        bn <= static_cast<U64>(std::numeric_limits<std::int64_t>::max())) {
      s.blockM = bm;
      s.blockN = bn;
    } else {
      probe.reasonCodes.push_back("probe-tile-integer-overflow-or-invalid-micro-tile");
    }
    if (s.aPack == CompilerGemmAPack::Vector4 &&
        (!device.supportsVectorAPack || bm % 4 || s.blockK % 4))
      probe.reasonCodes.push_back("vector-a-pack-lowering-not-supported");
    if (s.stripK && !device.supportsStreamingK)
      probe.reasonCodes.push_back("streaming-k-lowering-not-supported");
    if (s.bLayout == CompilerGemmBLayout::LocalPanelPadSVL && !device.supportsPaddedB)
      probe.reasonCodes.push_back("local-b-panel-lowering-not-supported");
    if (s.bLayout == CompilerGemmBLayout::GlobalPadSVL && !device.supportsGlobalPaddedB)
      probe.reasonCodes.push_back("global-b-panel-runtime-not-supported");
    if (device.microK > 0 && s.blockK % device.microK)
      probe.reasonCodes.push_back("probe-block-k-incompatible-with-micro-k");
    if (probe.reasonCodes.empty()) {
      U64 strip = s.stripK, cBytes = 0, panels = 0;
      if (!strip &&
          !multiply({ceilDiv(*kernel.k.value, s.blockK), static_cast<U64>(s.blockK)}, strip))
        probe.reasonCodes.push_back("probe-rounded-k-overflow");
      if (!multiply({4, bm, strip}, probe.aPanelBytesPerTask))
        probe.reasonCodes.push_back("probe-a-panel-size-overflow");
      panels = probe.aPanelBytesPerTask;
      if (s.bLayout == CompilerGemmBLayout::LocalPanelPadSVL) {
        U64 stride = 0;
        if (!add(bn, *caps.vectorLengthBits.value / 32, stride) ||
            !multiply({4, strip, stride}, probe.bPanelBytesPerTask) ||
            !add(panels, probe.bPanelBytesPerTask, panels))
          probe.reasonCodes.push_back("probe-b-panel-size-overflow");
      }
      if (!multiply({4, bm, bn}, cBytes) || !add(panels, cBytes, probe.scratchBytesPerTask))
        probe.reasonCodes.push_back("probe-scratch-size-overflow");
      if (s.bLayout == CompilerGemmBLayout::GlobalPadSVL) {
        U64 stride = 0;
        if (!add(*kernel.n.value, *caps.vectorLengthBits.value / 32, stride) ||
            !multiply({4, static_cast<U64>(*kernel.k.value), stride}, probe.globalBPanelBytes))
          probe.reasonCodes.push_back("probe-global-panel-size-overflow");
        if (probe.globalBPanelBytes > device.maxGlobalPanelBytes)
          probe.reasonCodes.push_back("probe-global-panel-exceeds-device-limit");
      }
      if (probe.aPanelBytesPerTask > device.maxAPanelBytesPerTask)
        probe.reasonCodes.push_back("probe-a-panel-exceeds-device-limit");
      if (probe.scratchBytesPerTask > device.maxScratchBytesPerTask)
        probe.reasonCodes.push_back("probe-scratch-exceeds-device-limit");
    }
    probe.supported = probe.reasonCodes.empty();
    if (probe.supported)
      probe.reasonCodes.push_back("structural-probe-requires-execution-and-correctness-checks");
    probes.push_back(std::move(probe));
  }
  return probes;
}

} // namespace sme_plan
