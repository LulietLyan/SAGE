#include "CompilerGemmProfile.h"
#include "llvm/Support/JSON.h"
#include <cmath>
#include <dlfcn.h>
#include <sys/sysctl.h>
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include "llvm/ADT/StringExtras.h"
namespace sme_plan::triton_cpu {
namespace {
llvm::Error bad(llvm::StringRef message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}
template <class T> Fact<T> measured(T value) {
  return Fact<T>::known(
      value, {"device-calibration", "compiler-profile", "measured-and-audited", "proven"});
}
} // namespace
llvm::Expected<CompilerGemmPolicyRequest> readCompilerGemmProfile(llvm::StringRef text,
                                                                  int64_t budget) {
  if (budget < 1 || budget > 64)
    return bad("compiler GEMM needs an explicit thread budget in 1..64");
  CompilerGemmPolicyRequest r;
  r.device.totalThreadBudget = budget;
  r.device.microM = r.device.microN = 32;
  r.device.microK = 1;
  r.device.maxAPanelBytesPerTask = 4 * 1024 * 1024;
  r.device.maxScratchBytesPerTask = 4 * 1024 * 1024;
  r.device.maxGlobalPanelBytes = 256 * 1024 * 1024;
  if (text.empty())
    return r; // fixed compiler policy can bootstrap calibration
  auto parsed = llvm::json::parse(text);
  if (!parsed)
    return parsed.takeError();
  auto *o = parsed->getAsObject();
  if (!o)
    return bad("compiler calibration must be a JSON object");
  auto *d = o->getObject("device");
  auto *c = o->getObject("calibration");
  auto compilerHash = o->getString("compilerSha256");
  if (!compilerHash)
    return bad("profile must bind the current compiler binary");
  Dl_info location{};
  if (!dladdr(reinterpret_cast<void *>(&readCompilerGemmProfile), &location) || !location.dli_fname)
    return bad("cannot locate loaded compiler for calibration identity");
  auto buffer = llvm::MemoryBuffer::getFile(location.dli_fname);
  if (!buffer)
    return bad("cannot read loaded compiler binary");
  auto bytes = (*buffer)->getBuffer();
  auto digest = llvm::SHA256::hash(
      llvm::ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size()));
  if (*compilerHash != llvm::toHex(llvm::ArrayRef<uint8_t>(digest), true))
    return bad("calibration compiler binary mismatch; recalibrate this compiler");
  auto *host = d ? d->getObject("host") : nullptr;
  if (!host)
    return bad("profile must contain observed host facts");
  for (auto key : {"hw.model", "machdep.cpu.brand_string", "hw.memsize", "kern.osrelease"}) {
    auto expected = host->getString(key);
    if (!expected)
      return bad("missing host identity fact");
    std::string actual;
    if (llvm::StringRef(key) == "hw.memsize") {
      uint64_t value = 0;
      size_t size = sizeof(value);
      if (sysctlbyname(key, &value, &size, nullptr, 0))
        return bad("cannot verify host memory identity");
      actual = std::to_string(value);
    } else {
      size_t size = 0;
      if (sysctlbyname(key, nullptr, &size, nullptr, 0))
        return bad("cannot read host identity size");
      actual.resize(size);
      if (sysctlbyname(key, actual.data(), &size, nullptr, 0))
        return bad("cannot read host identity");
      while (!actual.empty() && actual.back() == '\0')
        actual.pop_back();
    }
    if (*expected != actual)
      return bad("calibration host identity mismatch");
  }
  if (!d || !c)
    return bad("compiler profile requires device and calibration objects");
  auto identity = d->getString("identity");
  auto evidence = c->getString("evidenceId");
  auto measuredIdentity = c->getString("deviceIdentity");
  if (!identity || !evidence || !measuredIdentity || *identity != *measuredIdentity)
    return bad("missing or mismatched calibration identity");
  r.device.identity = identity->str();
  r.calibration.deviceIdentity = measuredIdentity->str();
  r.calibration.evidenceId = evidence->str();
#define FLAG(field) r.device.field = d->getBoolean(#field).value_or(false)
  FLAG(nativeSmeGemmVerified);
  FLAG(supportsVectorAPack);
  FLAG(supportsStreamingK);
  FLAG(supportsPaddedB);
  FLAG(supportsGlobalPaddedB);
  FLAG(supportsGenericCPU);
#undef FLAG
  if (auto *conflicts = d->getArray("rowStrideConflicts")) {
    for (auto &value : *conflicts) {
      auto *point = value.getAsObject();
      if (!point)
        return bad("invalid row-stride conflict entry");
      auto columns = point->getInteger("columns");
      auto padColumns = point->getInteger("padColumns");
      auto severity = point->getNumber("severity");
      if (!columns || *columns < 1 || !padColumns || *padColumns < 1 || !severity ||
          !std::isfinite(*severity) || *severity <= 0.0 || *severity > 1.0)
        return bad("invalid row-stride conflict fields");
      r.device.rowStrideConflicts.push_back(
          {static_cast<std::int64_t>(*columns), static_cast<std::int64_t>(*padColumns), *severity});
    }
  }
  auto svl = d->getInteger("vectorLengthBits");
  // The checked launcher currently implements the 512-bit / 16-FP32-lane
  // padding contract.  Reject another SVL at profile load time instead of
  // selecting a plan whose metadata would later fail in the runtime.
  if (!svl || *svl != 512)
    return bad("profile needs the verified 512-bit streaming vector length");
  // Conflict measurements describe the exact physical stride used by the
  // launcher.  Accepting a profile measured with a different pad would make
  // the cost model claim a cure that generated code does not apply.
  const auto expectedPad = *svl / 32;
  for (const auto &conflict : r.device.rowStrideConflicts)
    if (conflict.padColumns != expectedPad)
      return bad("row-stride conflict padding does not match streaming vector length");
  r.capabilities.sme = measured(r.device.nativeSmeGemmVerified);
  r.capabilities.streamingZA = measured(r.device.nativeSmeGemmVerified);
  r.capabilities.vectorLengthBits = measured(*svl);
  auto *teams = c->getArray("teams");
  if (!teams)
    return bad("profile has no measured teams");
  for (auto &value : *teams) {
    auto *t = value.getAsObject();
    if (!t)
      return bad("invalid team calibration");
    CompilerGemmTeamCalibration row;
#define INTEGER(field)                                                                             \
  {                                                                                                \
    auto x = t->getInteger(#field);                                                                \
    if (!x)                                                                                        \
      return bad("missing integer " #field);                                                       \
    row.field = *x;                                                                                \
  }
    INTEGER(requestedThreads);
    INTEGER(actualThreads);
    INTEGER(independentRounds);
#undef INTEGER
    row.correct = t->getBoolean("correct").value_or(false);
    row.pathVerified = t->getBoolean("pathVerified").value_or(false);
#define RATE(field)                                                                                \
  {                                                                                                \
    auto x = t->getNumber(#field);                                                                 \
    if (!x || !std::isfinite(*x) || *x < 0)                                                        \
      return bad("missing or invalid rate " #field);                                               \
    row.field = *x;                                                                                \
  }
    RATE(smeGflops);
    RATE(genericGflops);
    RATE(scalarAPackGBps);
    RATE(vectorAPackGBps);
    RATE(operandBandwidthGBps);
    RATE(paddedOperandBandwidthGBps);
    RATE(copyGBps);
    RATE(globalBPackGBps);
    RATE(globalPaddedOperandBandwidthGBps);
    RATE(callOverheadNs);
    RATE(taskOverheadNs);
    // Profiles produced before communication calibration remain valid; an
    // absent field means no separately identified barrier cost.
    if (auto x = t->getNumber("communicationNsPerWave")) {
      if (!std::isfinite(*x) || *x < 0)
        return bad("invalid communicationNsPerWave");
      row.communicationNsPerWave = *x;
    }
    RATE(stripOverheadNs);
#undef RATE
    for (auto entry : {std::make_pair("panelResponses", &row.panelResponses),
                       std::make_pair("paddedPanelResponses", &row.paddedPanelResponses),
                       std::make_pair("globalPanelResponses", &row.globalPanelResponses)}) {
      if (auto *points = t->getArray(entry.first))
        for (auto &p : *points) {
          auto *point = p.getAsObject();
          if (!point)
            return bad("invalid response point");
          auto bytes = point->getInteger("panelBytes");
          auto bw = point->getNumber("operandBandwidthGBps");
          if (!bytes || *bytes <= 0 || !bw || !std::isfinite(*bw) || *bw <= 0)
            return bad("invalid response point fields");
          entry.second->push_back({static_cast<uint64_t>(*bytes), *bw});
        }
    }
    auto *strategies = t->getArray("strategyResponses");
    if (!strategies)
      return bad("profile lacks observed strategy response families");
    for (auto &value : *strategies) {
      auto *strategy = value.getAsObject();
      if (!strategy)
        return bad("invalid strategy response family");
      CompilerGemmStrategyResponse family;
#define STRATEGY_INT(field)                                                                        \
  {                                                                                                \
    auto x = strategy->getInteger(#field);                                                         \
    if (!x || *x < 1)                                                                              \
      return bad("invalid strategy " #field);                                                      \
    family.field = *x;                                                                             \
  }
      STRATEGY_INT(blockM);
      STRATEGY_INT(blockN);
      STRATEGY_INT(blockK);
      STRATEGY_INT(independentProcesses);
#undef STRATEGY_INT
      auto vectorPack = strategy->getBoolean("aPackVector4");
      auto streaming = strategy->getBoolean("streamingK");
      auto layout = strategy->getString("bLayout");
      if (!vectorPack || !streaming || !layout)
        return bad("strategy family lacks packing/layout/K contract");
      family.aPack = *vectorPack ? CompilerGemmAPack::Vector4 : CompilerGemmAPack::Scalar;
      family.streamingK = *streaming;
      if (*layout == "original")
        family.bLayout = CompilerGemmBLayout::Original;
      else if (*layout == "local")
        family.bLayout = CompilerGemmBLayout::LocalPanelPadSVL;
      else if (*layout == "global")
        family.bLayout = CompilerGemmBLayout::GlobalPadSVL;
      else
        return bad("unknown strategy B layout");
      auto *points = strategy->getArray("panelResponses");
      if (!points)
        return bad("strategy family lacks response points");
      for (auto &value : *points) {
        auto *point = value.getAsObject();
        if (!point)
          return bad("invalid strategy response point");
        auto bytes = point->getInteger("panelBytes");
        auto bandwidth = point->getNumber("operandBandwidthGBps");
        if (!bytes || *bytes <= 0 || !bandwidth || !std::isfinite(*bandwidth) || *bandwidth <= 0)
          return bad("invalid strategy response point fields");
        family.panelResponses.push_back({static_cast<uint64_t>(*bytes), *bandwidth});
      }
      row.strategyResponses.push_back(std::move(family));
    }
    for (auto entry : {std::make_pair("observedNativeTiles", &row.observedNativeTiles),
                       std::make_pair("observedGenericTiles", &row.observedGenericTiles)}) {
      auto *families = t->getArray(entry.first);
      if (!families)
        return bad("profile lacks observed macro-tile calibration domain");
      for (auto &value : *families) {
        auto *family = value.getAsObject();
        if (!family)
          return bad("invalid observed macro-tile");
        CompilerGemmObservedTile tile;
#define TILE_INT(field)                                                                            \
  {                                                                                                \
    auto x = family->getInteger(#field);                                                           \
    if (!x || *x < 1)                                                                              \
      return bad("invalid macro-tile " #field);                                                    \
    tile.field = *x;                                                                               \
  }
        TILE_INT(blockM);
        TILE_INT(blockN);
        TILE_INT(blockK);
        TILE_INT(independentProcesses);
#undef TILE_INT
        entry.second->push_back(tile);
      }
    }
    r.calibration.teams.push_back(std::move(row));
  }
  return r;
}
} // namespace sme_plan::triton_cpu
