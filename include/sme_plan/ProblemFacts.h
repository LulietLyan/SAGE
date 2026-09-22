#ifndef SME_PLAN_PROBLEM_FACTS_H
#define SME_PLAN_PROBLEM_FACTS_H

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace sme_plan {

enum class FactState { Static, RuntimeSymbol, Unknown };

struct Provenance {
  std::string source;
  std::string stage;
  std::string locator;
  std::string confidence;
};

template <typename T> struct Fact {
  FactState state = FactState::Unknown;
  std::optional<T> value;
  std::string symbol;
  Provenance provenance;

  static Fact known(T value, Provenance provenance) {
    Fact fact;
    fact.state = FactState::Static;
    fact.value = std::move(value);
    fact.provenance = std::move(provenance);
    return fact;
  }

  static Fact runtime(std::string symbol, Provenance provenance) {
    Fact fact;
    fact.state = FactState::RuntimeSymbol;
    fact.symbol = std::move(symbol);
    fact.provenance = std::move(provenance);
    return fact;
  }

  static Fact unknown(Provenance provenance) {
    Fact fact;
    fact.provenance = std::move(provenance);
    return fact;
  }

  bool isKnown() const { return state == FactState::Static && value.has_value(); }

  bool isProven() const { return isKnown() && provenance.confidence == "proven"; }
};

struct TensorFacts {
  std::string role;
  Fact<std::string> dtype;
  std::vector<Fact<std::int64_t>> logicalShape;
  std::vector<Fact<std::int64_t>> physicalShape;
  std::vector<Fact<std::int64_t>> physicalStrides;
  Fact<std::int64_t> offset;
  Fact<std::int64_t> alignment;
  Fact<std::vector<std::int64_t>> permutation;
  Fact<std::string> aliases;
  std::vector<Fact<std::int64_t>> padding;
};

struct KernelFacts {
  std::string operationId;
  Fact<std::string> operationKind;
  Fact<std::int64_t> m;
  Fact<std::int64_t> n;
  Fact<std::int64_t> k;
  std::vector<TensorFacts> operands;
  std::vector<Fact<std::int64_t>> localTile;
  Fact<std::string> programMapping;
  Fact<std::string> accumulator;
  Fact<std::string> epilogue;
  Fact<std::string> mask;
  Fact<std::string> tail;
};

struct LaunchFacts {
  std::vector<Fact<std::int64_t>> gridExtents;
  Fact<std::string> programMapping;
  std::vector<std::pair<std::string, Fact<std::int64_t>>> runtimeShapes;
  Fact<std::string> workspace;
  Fact<std::string> callBoundary;
};

struct ProducerFacts {
  Fact<bool> present;
  Fact<std::string> location;
  Fact<std::string> layout;
  Fact<std::string> ownership;
  Fact<std::string> reuseGroup;
  Fact<std::string> lifetime;
};

struct MergeEvidence {
  std::string field;
  std::string chosenSource;
  std::vector<std::string> inputs;
  std::string rule;
};

struct ProblemFacts {
  KernelFacts kernel;
  LaunchFacts launch;
  ProducerFacts producer;
  std::vector<MergeEvidence> mergeEvidence;
};

} // namespace sme_plan

#endif
