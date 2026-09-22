#ifndef SME_PLAN_TARGET_CAPABILITIES_H
#define SME_PLAN_TARGET_CAPABILITIES_H

#include "sme_plan/ProblemFacts.h"

#include <cstdint>
#include <string>
#include <vector>

namespace sme_plan {

struct TargetCapabilities {
  Fact<std::string> targetTriple;
  Fact<std::string> cpuName;
  Fact<std::string> cpuFeatures;
  Fact<bool> sme;
  Fact<bool> sme2;
  Fact<bool> streamingZA;
  Fact<std::int64_t> vectorLengthBits;
  Fact<std::vector<std::string>> supportedMicrokernels;
  Fact<std::vector<std::string>> supportedDtypes;
  Fact<std::vector<std::string>> supportedTiles;
  Fact<bool> tailSupport;
  Fact<std::vector<std::string>> supportedLayoutContracts;
  Fact<bool> producerRewrite;
  Fact<std::string> adapterId;
  Fact<std::string> tritonVersion;
  Fact<std::string> llvmVersion;
  Fact<std::string> mlirVersion;
  Fact<std::string> pluginApiVersion;
};

// Calibration values are measurements, not proof that codegen exists.
struct HardwareProfile {
  std::string name;
  std::int64_t fp32LanesPerSVL = 0;
  std::int64_t maxZATiles = 0;
  std::int64_t zRegisters = 0;
  std::int64_t l1DataBytes = 0;
  std::int64_t l2Bytes = 0;
  double effectiveL2Fraction = 0.0;
  double sustainableSMEGflops = 0.0;
  double memoryBandwidthGBps = 0.0;
  double packEffectiveGBps = 0.0;
  double programOverheadNs = 0.0;
  double smeStateOverheadNs = 0.0;
};

} // namespace sme_plan

#endif
