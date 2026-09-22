#include "sme_plan/CompilerGemmPolicy.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

using namespace sme_plan;

namespace {

int checks = 0;
#define CHECK(x)                                                                                   \
  do {                                                                                             \
    ++checks;                                                                                      \
    if (!(x)) {                                                                                    \
      std::cerr << "line " << __LINE__ << ": " << #x << '\n';                                      \
      std::exit(1);                                                                                \
    }                                                                                              \
  } while (false)

template <typename T> Fact<T> known(T value) {
  return Fact<T>::known(value, {"test", "fixture", "synthetic", "proven"});
}

TensorFacts tensor(std::string role, std::int64_t m, std::int64_t n) {
  TensorFacts t;
  t.role = role;
  t.dtype = known(std::string("f32"));
  t.logicalShape = {known(m), known(n)};
  t.physicalStrides = {known(n), known(std::int64_t(1))};
  return t;
}

CompilerGemmPolicyRequest fixture(std::int64_t m = 1536, std::int64_t n = 1792,
                                  std::int64_t k = 1152) {
  CompilerGemmPolicyRequest r;
  r.problem.kernel.operationKind = known(std::string("gemm"));
  r.problem.kernel.m = known(m);
  r.problem.kernel.n = known(n);
  r.problem.kernel.k = known(k);
  r.problem.kernel.operands = {tensor("A", m, k), tensor("B", k, n), tensor("C", m, n)};
  r.capabilities.sme = known(true);
  r.capabilities.streamingZA = known(true);
  r.capabilities.vectorLengthBits = known(std::int64_t(512));
  r.device.identity = "synthetic-device-no-hardware-claim";
  r.device.nativeSmeGemmVerified = true;
  r.device.supportsVectorAPack = true;
  r.device.supportsStreamingK = true;
  r.device.supportsPaddedB = true;
  r.device.supportsGenericCPU = true;
  r.device.microM = r.device.microN = 32;
  r.device.microK = 1;
  r.device.totalThreadBudget = 8;
  r.device.maxAPanelBytesPerTask = 4 * 1024 * 1024;
  r.device.maxScratchBytesPerTask = 4 * 1024 * 1024;
  r.calibration.deviceIdentity = r.device.identity;
  r.calibration.evidenceId = "unit-test-values-are-not-measurements";
  CompilerGemmTeamCalibration c;
  c.requestedThreads = c.actualThreads = 1;
  c.independentRounds = 5;
  c.correct = c.pathVerified = true;
  c.smeGflops = 1000;
  c.genericGflops = 100;
  c.scalarAPackGBps = 20;
  c.vectorAPackGBps = 80;
  c.operandBandwidthGBps = 10000;
  c.paddedOperandBandwidthGBps = 10000;
  c.copyGBps = 20;
  c.callOverheadNs = 1000;
  c.taskOverheadNs = 100;
  c.stripOverheadNs = 10;
  for (auto bm : {32, 64, 128, 256})
    for (auto bn : {32, 64, 128, 256, 512, 1024})
      for (auto bk : {16, 32}) {
        c.observedNativeTiles.push_back({bm, bn, bk, 5});
        c.observedGenericTiles.push_back({bm, bn, bk, 5});
        for (auto pack : {CompilerGemmAPack::Scalar, CompilerGemmAPack::Vector4})
          for (auto layout : {CompilerGemmBLayout::Original, CompilerGemmBLayout::LocalPanelPadSVL,
                              CompilerGemmBLayout::GlobalPadSVL})
            for (bool streaming : {false, true})
              c.strategyResponses.push_back(
                  {bm, bn, bk, pack, layout, streaming, 5, {{1, 10000}, {4 * 1024 * 1024, 10000}}});
      }
  r.calibration.teams = {c};
  r.search.mFactors = {4};
  r.search.nFactors = {4, 8, 16, 32};
  return r;
}

void setLayoutResponses(CompilerGemmTeamCalibration &team, CompilerGemmBLayout layout,
                        std::vector<CompilerGemmPanelResponse> points) {
  for (auto &strategy : team.strategyResponses)
    if (strategy.bLayout == layout)
      strategy.panelResponses = points;
}

void gateTests() {
  auto r = fixture();
  CHECK(planCompilerGemm(r).apply);
  r.problem.kernel.m.provenance.confidence = "declared";
  CHECK(!planCompilerGemm(r).apply);
  r = fixture();
  r.problem.kernel.operands[1].physicalStrides[1] = known(std::int64_t(2));
  CHECK(!planCompilerGemm(r).apply);
  r = fixture();
  r.calibration.deviceIdentity = "different-device";
  CHECK(!planCompilerGemm(r).apply);
  r = fixture();
  r.calibration.teams[0].actualThreads = 2;
  CHECK(!planCompilerGemm(r).apply);
  r = fixture();
  r.calibration.teams[0].independentRounds = 1;
  CHECK(!planCompilerGemm(r).apply);
  r = fixture();
  r.calibration.teams.push_back(r.calibration.teams.front());
  CHECK(!planCompilerGemm(r).apply);
  r = fixture();
  r.calibration.teams[0].panelResponses = {{1024, 10}, {1024, 20}};
  CHECK(!planCompilerGemm(r).apply);
  r = fixture();
  r.calibration.teams[0].taskOverheadNs = std::numeric_limits<double>::quiet_NaN();
  CHECK(!planCompilerGemm(r).apply);
  r = fixture();
  r.search.blockKs = {0};
  CHECK(!planCompilerGemm(r).apply);
  r = fixture();
  r.capabilities.sme = known(false);
  auto d = planCompilerGemm(r);
  CHECK(d.apply);
  CHECK(d.selected->path == CompilerGemmPath::GenericCPU);
  r.device.supportsGenericCPU = false;
  CHECK(!planCompilerGemm(r).apply);
}

void rankingTests() {
  auto r = fixture();
  auto d = planCompilerGemm(r);
  CHECK(d.selected->aPack == CompilerGemmAPack::Vector4);
  CHECK(d.selected->bLayout == CompilerGemmBLayout::Original);
  CHECK(d.selected->stripK == 0);
  CHECK(d.selected->threads == 1);
  CHECK(d.selected->blockN == 1024);
  r.calibration.teams[0].vectorAPackGBps = 1;
  CHECK(planCompilerGemm(r).selected->aPack == CompilerGemmAPack::Scalar);

  r = fixture();
  r.calibration.teams[0].operandBandwidthGBps = 10;
  setLayoutResponses(r.calibration.teams[0], CompilerGemmBLayout::Original,
                     {{1, 10}, {4 * 1024 * 1024, 10}});
  r.calibration.teams[0].copyGBps = 10000;
  d = planCompilerGemm(r);
  CHECK(d.selected->bLayout == CompilerGemmBLayout::LocalPanelPadSVL);
  CHECK(d.rankedCandidates.front().bPanelPackNs > 0);
  CHECK(d.rankedCandidates.front().bPanelBytesPerTask > 0);

  r = fixture();
  // A measured working-set cliff can select K strips without a shape cutoff.
  setLayoutResponses(r.calibration.teams[0], CompilerGemmBLayout::Original,
                     {{32768, 10000}, {65536, 10000}, {262144, 20}, {1048576, 20}});
  r.search.includePaddedB = false;
  d = planCompilerGemm(r);
  CHECK(d.selected->stripK > 0);
  CHECK(d.rankedCandidates.front().aPanelBytesPerTask < 262144);
  // Flat measured responses do not invent a cache cliff from a device name.
  setLayoutResponses(r.calibration.teams[0], CompilerGemmBLayout::Original,
                     {{1, 10000}, {4 * 1024 * 1024, 10000}});
  CHECK(planCompilerGemm(r).selected->stripK == 0);

  r = fixture();
  auto faster = r.calibration.teams.front();
  faster.actualThreads = faster.requestedThreads = 2;
  faster.smeGflops *= 2;
  faster.vectorAPackGBps *= 2;
  faster.scalarAPackGBps *= 2;
  r.calibration.teams.push_back(faster);
  auto slower = faster;
  slower.actualThreads = slower.requestedThreads = 8;
  slower.smeGflops = 100;
  r.calibration.teams.push_back(slower);
  d = planCompilerGemm(r);
  CHECK(d.selected->threads == 2);
  r.device.totalThreadBudget = 1;
  CHECK(planCompilerGemm(r).selected->threads == 1);
  r.device.totalThreadBudget = 8;
  auto before = d.selected->id();
  std::reverse(r.search.nFactors.begin(), r.search.nFactors.end());
  std::reverse(r.calibration.teams.begin(), r.calibration.teams.end());
  CHECK(planCompilerGemm(r).selected->id() == before);

  r = fixture();
  r.calibration.teams[0].genericGflops = 10000;
  CHECK(planCompilerGemm(r).selected->path == CompilerGemmPath::GenericCPU);
}

void resourceTests() {
  auto r = fixture(193, 257, 129);
  r.search.includeGenericCPU = false;
  r.device.maxScratchBytesPerTask = 128 * 1024;
  auto d = planCompilerGemm(r);
  CHECK(d.apply);
  for (const auto &candidate : d.rankedCandidates) {
    CHECK(candidate.scratchBytesPerTask <= r.device.maxScratchBytesPerTask);
    CHECK(candidate.outputTasks > 0);
    CHECK(candidate.taskOccupancy > 0 && candidate.taskOccupancy <= 1);
    CHECK(candidate.kStrips > 0);
    CHECK(std::isfinite(candidate.predictedNs));
    CHECK(candidate.schedule.stripK == 0 ||
          candidate.schedule.stripK % candidate.schedule.blockK == 0);
  }
  r.device.maxScratchBytesPerTask = 1;
  CHECK(!planCompilerGemm(r).apply);
  r = fixture(std::numeric_limits<std::int64_t>::max(), std::numeric_limits<std::int64_t>::max(),
              7);
  CHECK(!planCompilerGemm(r).apply);
}

void localPanelCostTests() {
  auto r = fixture(512, 512, 256);
  r.search.mFactors = {2, 4};
  r.search.nFactors = {4};
  r.search.blockKs = {16};
  r.search.stripFactors.clear();
  r.search.includeGenericCPU = false;
  r.search.includeOriginalB = false;
  r.search.includeScalarAPack = false;
  auto d = planCompilerGemm(r);
  CHECK(d.rankedCandidates.size() == 2);
  const CompilerGemmCost *small = nullptr, *large = nullptr;
  for (const auto &cost : d.rankedCandidates) {
    if (cost.schedule.blockM == 64)
      small = &cost;
    else if (cost.schedule.blockM == 128)
      large = &cost;
    const auto bm = static_cast<std::uint64_t>(cost.schedule.blockM);
    CHECK(cost.aPanelBytesPerTask == 4 * bm * 256);
    CHECK(cost.bPanelBytesPerTask == 4 * 256 * (128 + 16));
    CHECK(cost.scratchBytesPerTask ==
          cost.aPanelBytesPerTask + cost.bPanelBytesPerTask + 4 * bm * 128);
    CHECK(cost.arithmeticNs == 2.0 * 512 * 512 * 256 / 1000);
  }
  CHECK(small && large);
  // The smaller M task copies B twice as often; padded columns do not add
  // arithmetic or a hidden padded-C copy.
  CHECK(small->bPanelPackNs == 2.0 * large->bPanelPackNs);
  CHECK(large->bPanelPackNs == 8.0 * 4 * 256 * 512 / 20);
}

void calibrationProbeTests() {
  auto r = fixture(2304, 2304, 2304);
  r.calibration = {};
  auto probes = compilerGemmCalibrationSchedules(r);
  CHECK(probes.size() == 10);
  for (std::size_t i = 0; i < probes.size(); ++i) {
    CHECK(probes[i].index == static_cast<std::int64_t>(i));
    CHECK(probes[i].schedule.threads == r.device.totalThreadBudget);
    CHECK(probes[i].schedule.path == CompilerGemmPath::SME);
  }
  CHECK(probes[0].supported);
  CHECK(probes[0].schedule.blockM == 128 && probes[0].schedule.blockN == 256);
  CHECK(probes[0].schedule.blockK == 16 && probes[0].schedule.stripK == 0);
  CHECK(probes[1].supported && probes[1].schedule.blockN == 1024);
  CHECK(probes[2].supported && probes[2].schedule.stripK == 512);
  CHECK(probes[3].supported && probes[3].schedule.stripK == 1024);
  CHECK(probes[4].supported && probes[4].schedule.bLayout == CompilerGemmBLayout::LocalPanelPadSVL);
  CHECK(!probes[5].supported);
  CHECK(probes[5].scratchBytesPerTask > 4 * 1024 * 1024);
  CHECK(std::find(probes[5].reasonCodes.begin(), probes[5].reasonCodes.end(),
                  "probe-scratch-exceeds-device-limit") != probes[5].reasonCodes.end());
  CHECK(probes[6].supported);
  CHECK(probes[6].schedule.blockM == 128 && probes[6].schedule.blockN == 512);
  CHECK(probes[6].schedule.blockK == 32 && probes[6].schedule.stripK == 1024);
  CHECK(probes[6].schedule.bLayout == CompilerGemmBLayout::LocalPanelPadSVL);
  CHECK(probes[6].scratchBytesPerTask <= 4 * 1024 * 1024);
  CHECK(!probes[7].supported && !probes[8].supported && !probes[9].supported);
  // A rejected probe remains the same candidate if the device limit changes.
  r.device.maxScratchBytesPerTask = 8 * 1024 * 1024;
  CHECK(compilerGemmCalibrationSchedules(r)[5].supported);
  CHECK(compilerGemmCalibrationSchedules(r)[5].schedule.id() == probes[5].schedule.id());
  r = fixture(128, 128, 131);
  auto tail = compilerGemmCalibrationSchedules(r);
  CHECK(tail[0].aPanelBytesPerTask == 4 * 128 * 144);
  CHECK(tail[0].schedule.id() == probes[0].schedule.id());
  r.problem.kernel.k = {};
  for (const auto &probe : compilerGemmCalibrationSchedules(r))
    CHECK(!probe.supported);
  r = fixture();
  r.device.supportsPaddedB = false;
  probes = compilerGemmCalibrationSchedules(r);
  CHECK(probes[3].supported && !probes[4].supported && !probes[5].supported);
}

void observedMacroTileDomainTests() {
  auto r = fixture(3072, 3072, 3072);
  r.search.mFactors = {1, 4};
  r.search.nFactors = {32};
  r.search.blockKs = {32};
  r.search.includeGenericCPU = false;
  auto &team = r.calibration.teams[0];
  team.observedNativeTiles = {{128, 1024, 32, 3}};
  auto d = planCompilerGemm(r);
  CHECK(d.apply);
  CHECK(d.selected->blockM == 128);
  for (const auto &candidate : d.rankedCandidates) {
    CHECK(candidate.schedule.blockM == 128);
    CHECK(candidate.schedule.blockN == 1024 && candidate.schedule.blockK == 32);
  }
  // A working-set response never authorizes an unseen BM32 macro family.
  team.panelResponses = {{32768, 100000}, {1048576, 10}};
  CHECK(planCompilerGemm(r).selected->blockM == 128);
  team.observedNativeTiles[0].independentProcesses = 2;
  CHECK(!planCompilerGemm(r).apply);
  team.observedNativeTiles[0].independentProcesses = 3;
  CHECK(planCompilerGemm(r).apply);
  team.observedNativeTiles.clear();
  CHECK(!planCompilerGemm(r).apply);
  // Generic observations cannot authorize native SME, or vice versa.
  team.observedGenericTiles = {{128, 1024, 32, 3}};
  CHECK(!planCompilerGemm(r).apply);
  r.search.includeGenericCPU = true;
  CHECK(planCompilerGemm(r).selected->path == CompilerGemmPath::GenericCPU);
  r.search.includeGenericCPU = false;
  team.observedNativeTiles = {{128, 1024, 32, 3}};
  auto unobservedTeam = team;
  unobservedTeam.actualThreads = unobservedTeam.requestedThreads = 8;
  unobservedTeam.smeGflops = 100000;
  unobservedTeam.observedNativeTiles.clear();
  r.calibration.teams.push_back(unobservedTeam);
  CHECK(planCompilerGemm(r).selected->threads == 1);
  r.calibration.teams[0].observedNativeTiles.push_back({128, 1024, 32, 3});
  CHECK(!planCompilerGemm(r).apply);
}

void globalPanelCostTests() {
  auto r = fixture(512, 512, 256);
  r.device.supportsGlobalPaddedB = true;
  r.device.maxGlobalPanelBytes = 256 * 1024 * 1024;
  r.search.mFactors = {2, 4};
  r.search.nFactors = {4};
  r.search.blockKs = {16};
  r.search.stripFactors.clear();
  r.search.includeGenericCPU = false;
  r.search.includeOriginalB = false;
  r.search.includePaddedB = false;
  r.search.includeScalarAPack = false;
  auto &team = r.calibration.teams[0];
  team.globalBPackGBps = 20;
  team.globalPaddedOperandBandwidthGBps = 10000;
  // Local response must not leak into the global layout's response.
  team.paddedOperandBandwidthGBps = 0.001;
  auto d = planCompilerGemm(r);
  CHECK(d.rankedCandidates.size() == 2);
  for (const auto &cost : d.rankedCandidates) {
    const auto bm = static_cast<std::uint64_t>(cost.schedule.blockM);
    CHECK(cost.schedule.bLayout == CompilerGemmBLayout::GlobalPadSVL);
    CHECK(cost.globalBPanelBytes == 4 * 256 * (512 + 16));
    CHECK(cost.globalBPackNs == 8.0 * 256 * 512 / 20);
    CHECK(cost.bPanelBytesPerTask == 0 && cost.bPanelPackNs == 0);
    CHECK(cost.scratchBytesPerTask == 4 * bm * (256 + 128));
    CHECK(cost.arithmeticNs == 2.0 * 512 * 512 * 256 / 1000);
  }
  CHECK(d.rankedCandidates[0].globalBPackNs == d.rankedCandidates[1].globalBPackNs);
  r.device.maxGlobalPanelBytes = 4 * 256 * (512 + 16) - 1;
  CHECK(!planCompilerGemm(r).apply);
  r.device.maxGlobalPanelBytes = 256 * 1024 * 1024;
  team.globalBPackGBps = 0;
  CHECK(!planCompilerGemm(r).apply); // Never substitute the small-copy GB/s.
  team.globalBPackGBps = 20;
  team.globalPaddedOperandBandwidthGBps = 0;
  CHECK(planCompilerGemm(r).apply); // Pooled response no longer controls SME.
  auto strategies = team.strategyResponses;
  setLayoutResponses(team, CompilerGemmBLayout::GlobalPadSVL, {});
  team.globalPanelResponses = {{131072, 10000}};
  CHECK(!planCompilerGemm(r).apply); // No pooled fallback for missing strategy.
  team.strategyResponses = strategies;
  CHECK(planCompilerGemm(r).apply);
  auto probes = compilerGemmCalibrationSchedules(r);
  CHECK(probes.size() == 10);
  CHECK(probes[7].supported && probes[8].supported && probes[9].supported);
  CHECK(probes[7].schedule.blockM == 128 && probes[7].schedule.stripK == 0);
  CHECK(probes[8].schedule.blockM == 256 && probes[8].schedule.stripK == 512);
  CHECK(probes[9].schedule.blockM == 256 && probes[9].schedule.stripK == 1024);
  for (auto index : {7, 8, 9}) {
    CHECK(probes[index].schedule.bLayout == CompilerGemmBLayout::GlobalPadSVL);
    CHECK(probes[index].globalBPanelBytes == 4 * 256 * (512 + 16));
    CHECK(probes[index].bPanelBytesPerTask == 0);
  }
}

void strategyResponseDomainTests() {
  auto r = fixture(3072, 3072, 3072);
  r.search.mFactors = {8};
  r.search.nFactors = {32};
  r.search.blockKs = {32};
  r.search.stripFactors = {4, 16, 24, 32, 64};
  r.search.includeGenericCPU = false;
  r.search.includePaddedB = false;
  r.search.includeGlobalPaddedB = false;
  r.search.includeScalarAPack = false;
  auto &team = r.calibration.teams[0];
  CompilerGemmStrategyResponse full{256,
                                    1024,
                                    32,
                                    CompilerGemmAPack::Vector4,
                                    CompilerGemmBLayout::Original,
                                    false,
                                    3,
                                    {{512 * 1024, 50}, {1024 * 1024, 50}}};
  auto streaming = full;
  streaming.streamingK = true;
  streaming.panelResponses = {{512 * 1024, 10000}, {1024 * 1024, 5000}};
  team.strategyResponses = {full, streaming};
  auto d = planCompilerGemm(r);
  CHECK(d.apply);
  CHECK(d.rankedCandidates.size() == 4);
  CHECK(d.selected->stripK != 0);
  for (const auto &cost : d.rankedCandidates) {
    // Only 512/768/1024 streaming K lie inside the observed interval.
    CHECK(cost.schedule.stripK == 0 || cost.schedule.stripK == 512 || cost.schedule.stripK == 768 ||
          cost.schedule.stripK == 1024);
    if (cost.schedule.stripK == 0)
      CHECK(cost.aPanelBytesPerTask > 1024 * 1024); // Full K may clamp.
    if (cost.schedule.stripK == 768) {
      const double alpha = std::log(1.5) / std::log(2.0);
      const double rate = 10000 * (1 - alpha) + 5000 * alpha;
      const double bytes = 8.0 * 3072 * 3072 * 96 + 4.0 * 3072 * 3072 * (2 * 4 - 1);
      CHECK(std::abs(cost.operandReadNs / (bytes / rate) - 1) < 1e-12);
    }
  }
  team.strategyResponses = {full};
  d = planCompilerGemm(r);
  CHECK(d.rankedCandidates.size() == 1 && d.selected->stripK == 0);
  team.strategyResponses = {streaming};
  CHECK(planCompilerGemm(r).rankedCandidates.size() == 3);
  team.strategyResponses[0].independentProcesses = 2;
  CHECK(!planCompilerGemm(r).apply);
  team.strategyResponses[0].independentProcesses = 3;
  CHECK(planCompilerGemm(r).apply);
  // Missing/empty native evidence cannot borrow even enormous pooled rates.
  team.strategyResponses[0].panelResponses.clear();
  team.operandBandwidthGBps = 1e12;
  team.panelResponses = {{1, 1e12}, {4194304, 1e12}};
  CHECK(!planCompilerGemm(r).apply);
  team.strategyResponses.clear();
  CHECK(!planCompilerGemm(r).apply);
  for (int mismatch = 0; mismatch < 5; ++mismatch) {
    auto other = streaming;
    if (mismatch == 0)
      other.blockM = 128;
    if (mismatch == 1)
      other.blockN = 512;
    if (mismatch == 2)
      other.blockK = 16;
    if (mismatch == 3)
      other.aPack = CompilerGemmAPack::Scalar;
    if (mismatch == 4)
      other.bLayout = CompilerGemmBLayout::LocalPanelPadSVL;
    team.strategyResponses = {other};
    CHECK(!planCompilerGemm(r).apply);
  }
  team.strategyResponses = {streaming, streaming};
  CHECK(!planCompilerGemm(r).apply);
  team.strategyResponses = {streaming};
  team.strategyResponses[0].panelResponses[0].operandBandwidthGBps =
      std::numeric_limits<double>::quiet_NaN();
  CHECK(!planCompilerGemm(r).apply);

  r = fixture(128, 128, 128);
  r.search.mFactors = {8};
  r.search.nFactors = {32};
  r.search.blockKs = {32};
  r.search.includeGenericCPU = false;
  r.calibration.teams[0].strategyResponses = {full};
  d = planCompilerGemm(r);
  CHECK(d.apply && d.selected->stripK == 0);
  CHECK(d.rankedCandidates[0].aPanelBytesPerTask < 512 * 1024);
}

void equalPanelDifferentStrategyTests() {
  auto r = fixture(3456, 3456, 3456);
  r.search.mFactors = {4};
  r.search.nFactors = {8, 32};
  r.search.blockKs = {16, 32};
  r.search.stripFactors.clear();
  r.search.includeGenericCPU = false;
  r.search.includePaddedB = false;
  auto &team = r.calibration.teams[0];
  const std::uint64_t bytes = 4 * 128 * 3456;
  team.strategyResponses = {{128,
                             256,
                             16,
                             CompilerGemmAPack::Scalar,
                             CompilerGemmBLayout::Original,
                             false,
                             3,
                             {{bytes, 10000}}},
                            {128,
                             1024,
                             32,
                             CompilerGemmAPack::Vector4,
                             CompilerGemmBLayout::Original,
                             false,
                             3,
                             {{bytes, 1}}}};
  auto d = planCompilerGemm(r);
  CHECK(d.rankedCandidates.size() == 2);
  CHECK(d.selected->aPack == CompilerGemmAPack::Scalar);
  CHECK(d.rankedCandidates[0].aPanelBytesPerTask == d.rankedCandidates[1].aPanelBytesPerTask);
  team.strategyResponses[0].panelResponses[0].operandBandwidthGBps = 1;
  team.strategyResponses[1].panelResponses[0].operandBandwidthGBps = 10000;
  CHECK(planCompilerGemm(r).selected->aPack == CompilerGemmAPack::Vector4);
}

void rowStrideConflictTests() {
  // Structural power-of-two rule (N>=2048) prefers GlobalPad without any
  // profile list. A non-power nearby width keeps Original unless measured.
  auto makeRequest = [](std::int64_t m, std::int64_t n, std::int64_t k) {
    auto r = fixture(m, n, k);
    r.device.supportsGlobalPaddedB = true;
    r.device.maxGlobalPanelBytes = 256 * 1024 * 1024;
    r.device.rowStrideConflicts.clear();
    r.search.mFactors = {4};
    r.search.nFactors = {32};
    r.search.blockKs = {32};
    r.search.stripFactors.clear();
    r.search.includeGenericCPU = false;
    r.search.includePaddedB = false;
    r.search.includeScalarAPack = false;
    auto &team = r.calibration.teams[0];
    team.globalBPackGBps = 40;
    team.observedNativeTiles = {{128, 1024, 32, 5}};
    team.strategyResponses.clear();
    for (auto layout : {CompilerGemmBLayout::Original, CompilerGemmBLayout::GlobalPadSVL})
      team.strategyResponses.push_back({128,
                                        1024,
                                        32,
                                        CompilerGemmAPack::Vector4,
                                        layout,
                                        false,
                                        5,
                                        {{1, 10000}, {4 * 1024 * 1024, 10000}}});
    return r;
  };

  auto plain = planCompilerGemm(makeRequest(1024, 2304, 1024));
  CHECK(plain.apply);
  CHECK(plain.selected->bLayout == CompilerGemmBLayout::Original);

  auto structural = planCompilerGemm(makeRequest(1024, 2048, 1024));
  CHECK(structural.apply);
  CHECK(structural.selected->bLayout == CompilerGemmBLayout::GlobalPadSVL);

  // The guard covers every integer multiple of the measured 2048-column
  // period, including widths absent from the calibration sweep.
  auto triple = planCompilerGemm(makeRequest(6144, 6144, 1024));
  CHECK(triple.apply);
  CHECK(triple.selected->bLayout == CompilerGemmBLayout::GlobalPadSVL);
  auto nearby = planCompilerGemm(makeRequest(1024, 6656, 1024));
  CHECK(nearby.apply);
  CHECK(nearby.selected->bLayout == CompilerGemmBLayout::Original);

  // 2048×N (M=2048, N=2048) and measured cliffs both select pad.
  auto rectangular = makeRequest(2048, 2048, 1024);
  CHECK(planCompilerGemm(rectangular).selected->bLayout == CompilerGemmBLayout::GlobalPadSVL);

  auto measured = makeRequest(1024, 2304, 1024);
  measured.device.rowStrideConflicts.push_back({2304, 16, 0.2});
  CHECK(planCompilerGemm(measured).selected->bLayout == CompilerGemmBLayout::GlobalPadSVL);

  // K-only cliffs slow A pack equally and must not invent a B-pad preference.
  auto kOnly = makeRequest(1024, 2304, 2048);
  kOnly.device.rowStrideConflicts.push_back({2048, 16, 0.15});
  CHECK(planCompilerGemm(kOnly).selected->bLayout == CompilerGemmBLayout::Original);
}

void indexedSearchTests() {
  auto r = fixture(1024, 1024, 1024);
  auto &team = r.calibration.teams[0];
  // Keep one measured native family while presenting a deliberately large
  // search space. Production enumeration must be bounded by measured
  // families, rather than by the Cartesian product of these lists.
  team.observedGenericTiles.clear();
  team.observedNativeTiles = {{128, 256, 16, 5}};
  team.strategyResponses = {{128,
                             256,
                             16,
                             CompilerGemmAPack::Vector4,
                             CompilerGemmBLayout::Original,
                             false,
                             5,
                             {{1, 10000}, {4 * 1024 * 1024, 10000}}}};
  r.search.mFactors.clear();
  r.search.nFactors.clear();
  for (std::int64_t i = 1; i <= 64; ++i) {
    r.search.mFactors.push_back(i);
    r.search.nFactors.push_back(i);
  }
  r.search.blockKs = {16, 32, 64, 128};
  r.search.stripFactors = {1, 2, 4, 8, 16, 32, 64};
  r.search.includeGenericCPU = false;
  r.search.includeScalarAPack = false;
  r.search.includePaddedB = false;
  r.search.includeGlobalPaddedB = false;
  auto d = planCompilerGemm(r);
  CHECK(d.apply);
  CHECK(d.rankedCandidates.size() == 1);
  CHECK(std::find(d.reasonCodes.begin(), d.reasonCodes.end(), "strategy-response-indexed-search") !=
        d.reasonCodes.end());
  auto selected = d.selected->id();
  std::reverse(r.search.mFactors.begin(), r.search.mFactors.end());
  std::reverse(r.search.nFactors.begin(), r.search.nFactors.end());
  CHECK(planCompilerGemm(r).selected->id() == selected);

  // An observed family with an incompatible reduction micro-step must be
  // rejected even if the caller's block-K list contains that value.
  r.device.microK = 16;
  r.search.blockKs = {15};
  team.observedNativeTiles = {{128, 256, 15, 5}};
  team.strategyResponses.front().blockK = 15;
  CHECK(!planCompilerGemm(r).apply);
}

void communicationCostTests() {
  auto r = fixture(1024, 1024, 1024);
  auto &team = r.calibration.teams[0];
  team.taskOverheadNs = 1000.0;
  team.communicationNsPerWave = 37.5;
  team.observedNativeTiles = {{32, 32, 16, 5}, {128, 256, 16, 5}};
  team.strategyResponses = {{32,
                             32,
                             16,
                             CompilerGemmAPack::Scalar,
                             CompilerGemmBLayout::Original,
                             false,
                             5,
                             {{1, 10000}, {4 * 1024 * 1024, 10000}}},
                            {128,
                             256,
                             16,
                             CompilerGemmAPack::Scalar,
                             CompilerGemmBLayout::Original,
                             false,
                             5,
                             {{1, 10000}, {4 * 1024 * 1024, 10000}}}};
  r.search.mFactors = {1, 4};
  r.search.nFactors = {1, 8};
  r.search.blockKs = {16};
  r.search.stripFactors.clear();
  r.search.includeGenericCPU = false;
  r.search.includeVectorAPack = false;
  r.search.includePaddedB = false;
  r.search.includeGlobalPaddedB = false;
  auto d = planCompilerGemm(r);
  CHECK(d.apply && d.rankedCandidates.size() == 2);
  CHECK(d.selected->blockM == 128 && d.selected->blockN == 256);
  for (const auto &cost : d.rankedCandidates) {
    const auto tasks = cost.outputTasks;
    const auto waves = (tasks + cost.schedule.threads - 1) / cost.schedule.threads;
    CHECK(std::abs(cost.schedulingNs - (1000.0 + tasks * 1000.0 + waves * 37.5 +
                                        waves * cost.kStrips * team.stripOverheadNs)) < 1e-9);
  }
}

} // namespace

int main() {
  gateTests();
  rankingTests();
  resourceTests();
  localPanelCostTests();
  calibrationProbeTests();
  observedMacroTileDomainTests();
  globalPanelCostTests();
  strategyResponseDomainTests();
  equalPanelDifferentStrategyTests();
  rowStrideConflictTests();
  indexedSearchTests();
  communicationCostTests();
  std::cout << "CompilerGemmPolicy: " << checks << " checks passed\n";
}
