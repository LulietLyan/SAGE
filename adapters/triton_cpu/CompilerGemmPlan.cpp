#include "CompilerGemmPlan.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <utility>

using namespace mlir;

namespace sme_plan::triton_cpu {
namespace {

// An affine expression for one tensor element. i/j are symbolic tensor indices,
// s is the K-loop induction value; p, Dq, Rq denote pid, pid/q, pid%q. We never
// infer global dimensions from local tensor types. Shape types only name the
// local iteration axes whose address/mask expressions are subsequently proven.
struct Expr {
  int64_t constant = 0;
  std::map<std::string, int64_t> terms;
  static Expr number(int64_t x) {
    Expr e;
    e.constant = x;
    return e;
  }
  static Expr variable(std::string s) {
    Expr e;
    e.terms[std::move(s)] = 1;
    return e;
  }
  Expr scale(int64_t n) const {
    Expr e;
    e.constant = constant * n;
    for (const auto &kv : terms)
      if (kv.second * n)
        e.terms[kv.first] = kv.second * n;
    return e;
  }
  Expr add(const Expr &b) const {
    Expr e = *this;
    e.constant += b.constant;
    for (const auto &kv : b.terms) {
      e.terms[kv.first] += kv.second;
      if (!e.terms[kv.first])
        e.terms.erase(kv.first);
    }
    return e;
  }
  bool operator==(const Expr &e) const { return constant == e.constant && terms == e.terms; }
};
using Coords = std::vector<Expr>;

std::optional<int64_t> integer(Value value) {
  auto c = value.getDefiningOp<arith::ConstantOp>();
  if (!c)
    return std::nullopt;
  if (auto i = dyn_cast<IntegerAttr>(c.getValue()))
    return i.getInt();
  if (auto d = dyn_cast<DenseIntElementsAttr>(c.getValue())) {
    if (d.isSplat())
      return d.getSplatValue<APInt>().getSExtValue();
  }
  return std::nullopt;
}
bool zero(Value v) { return matchPattern(v, m_Zero()) || matchPattern(v, m_AnyZeroFloat()); }
std::optional<int64_t> intAttr(Operation *op, StringRef key) {
  if (auto a = op->getAttrOfType<IntegerAttr>(key))
    return a.getInt();
  return std::nullopt;
}

struct Address {
  unsigned argument = 0;
  Expr offset;
};

struct Symbolic {
  triton::FuncOp function;
  scf::ForOp loop;

  std::optional<Coords> inputCoords(Operation *op, const Coords &out) const {
    StringRef name = op->getName().getStringRef();
    if (name == "tt.splat")
      return Coords{};
    if (name == "tt.expand_dims") {
      auto axis = intAttr(op, "axis");
      if (!axis || *axis < 0 || static_cast<size_t>(*axis) >= out.size())
        return std::nullopt;
      Coords in = out;
      in.erase(in.begin() + *axis);
      return in;
    }
    if (name == "tt.broadcast") {
      auto t = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
      if (!t || static_cast<size_t>(t.getRank()) != out.size())
        return std::nullopt;
      Coords in = out;
      for (int64_t i = 0; i < t.getRank(); ++i)
        if (t.getDimSize(i) == 1)
          in[i] = Expr{};
      return in;
    }
    return std::nullopt;
  }

  std::optional<Expr> expression(Value v, const Coords &coords, unsigned depth = 0) {
    if (depth > 96)
      return std::nullopt;
    if (v == loop.getInductionVar())
      return Expr::variable("s");
    if (auto c = integer(v))
      return Expr::number(*c);
    Operation *op = v.getDefiningOp();
    if (!op)
      return std::nullopt;
    StringRef name = op->getName().getStringRef();
    if (name == "tt.get_program_id")
      return Expr::variable("p");
    if (name == "tt.make_range") {
      auto start = intAttr(op, "start");
      if (coords.size() != 1 || !start)
        return std::nullopt;
      return coords[0].add(Expr::number(*start));
    }
    if (auto in = inputCoords(op, coords))
      return expression(op->getOperand(0), *in, depth + 1);
    if (op->getNumOperands() != 2)
      return std::nullopt;
    auto a = expression(op->getOperand(0), coords, depth + 1);
    auto b = expression(op->getOperand(1), coords, depth + 1);
    if (!a || !b)
      return std::nullopt;
    if (name == "arith.addi")
      return a->add(*b);
    if (name == "arith.subi")
      return a->add(b->scale(-1));
    if (name == "arith.muli") {
      if (a->terms.empty())
        return b->scale(a->constant);
      if (b->terms.empty())
        return a->scale(b->constant);
      return std::nullopt;
    }
    if ((name == "arith.divsi" || name == "arith.divui" || name == "arith.remsi" ||
         name == "arith.remui") &&
        *a == Expr::variable("p") && b->terms.empty() && b->constant > 0) {
      bool div = name == "arith.divsi" || name == "arith.divui";
      if (b->constant == 1)
        return div ? Expr::variable("p") : Expr{};
      return Expr::variable(std::string(div ? "D" : "R") + std::to_string(b->constant));
    }
    return std::nullopt;
  }

  std::optional<Address> address(Value v, const Coords &coords, unsigned depth = 0) {
    if (depth > 96)
      return std::nullopt;
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      if (arg.getOwner() == &function.getBody().front() && arg.getArgNumber() < 3)
        return Address{arg.getArgNumber(), {}};
      // Support the frontend's optional pointer-strength-reduction form, but
      // prove the recurrence rather than treating arbitrary block args as A/B.
      if (arg.getOwner() == loop.getBody() && arg.getArgNumber() > 0) {
        unsigned index = arg.getArgNumber() - 1;
        auto step = integer(loop.getStep());
        auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
        if (!step || *step <= 0 || !yield || index >= yield.getNumOperands())
          return std::nullopt;
        Operation *next = yield.getOperand(index).getDefiningOp();
        if (!next || next->getName().getStringRef() != "tt.addptr" || next->getOperand(0) != arg)
          return std::nullopt;
        auto delta = expression(next->getOperand(1), coords, depth + 1);
        auto init = address(loop.getInitArgs()[index], coords, depth + 1);
        if (!delta || !init || !delta->terms.empty() || delta->constant % *step)
          return std::nullopt;
        init->offset = init->offset.add(Expr::variable("s").scale(delta->constant / *step));
        return init;
      }
      return std::nullopt;
    }
    Operation *op = v.getDefiningOp();
    if (!op)
      return std::nullopt;
    if (auto in = inputCoords(op, coords))
      return address(op->getOperand(0), *in, depth + 1);
    if (op->getName().getStringRef() != "tt.addptr" || op->getNumOperands() != 2)
      return std::nullopt;
    auto base = address(op->getOperand(0), coords, depth + 1);
    auto off = expression(op->getOperand(1), coords, depth + 1);
    if (!base || !off)
      return std::nullopt;
    base->offset = base->offset.add(*off);
    return base;
  }

  // Each entry means expr < 0. Conjunction order and broadcast representation
  // do not matter; every original mask conjunct must match one required bound.
  std::optional<std::vector<Expr>> mask(Value v, const Coords &coords, unsigned depth = 0) {
    if (depth > 96)
      return std::nullopt;
    if (auto c = integer(v)) {
      if (*c == -1 || *c == 1)
        return std::vector<Expr>{};
      return std::nullopt;
    }
    Operation *op = v.getDefiningOp();
    if (!op)
      return std::nullopt;
    if (auto in = inputCoords(op, coords))
      return mask(op->getOperand(0), *in, depth + 1);
    if (op->getName().getStringRef() == "arith.andi") {
      auto a = mask(op->getOperand(0), coords, depth + 1),
           b = mask(op->getOperand(1), coords, depth + 1);
      if (!a || !b)
        return std::nullopt;
      a->insert(a->end(), b->begin(), b->end());
      return a;
    }
    auto cmp = dyn_cast<arith::CmpIOp>(op);
    if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::slt)
      return std::nullopt;
    auto a = expression(cmp.getLhs(), coords, depth + 1),
         b = expression(cmp.getRhs(), coords, depth + 1);
    if (!a || !b)
      return std::nullopt;
    return std::vector<Expr>{a->add(b->scale(-1))};
  }
};

struct ProvenGemm {
  triton::FuncOp function;
  int64_t m = 0, n = 0, k = 0, bm = 0, bn = 0, bk = 0;
  int64_t oldGrid = 0;
};

std::optional<int64_t> bound(const std::vector<Expr> &mask, const Expr &coordinate) {
  std::optional<int64_t> result;
  for (const auto &e : mask)
    if (e.terms == coordinate.terms) {
      int64_t value = coordinate.constant - e.constant;
      if (result || value <= 0)
        return std::nullopt;
      result = value;
    }
  return result;
}
bool exactMask(const std::vector<Expr> &mask, Expr a, int64_t na, Expr b, int64_t nb) {
  if (mask.size() != 2)
    return false;
  a = a.add(Expr::number(-na));
  b = b.add(Expr::number(-nb));
  return (mask[0] == a && mask[1] == b) || (mask[1] == a && mask[0] == b);
}

std::optional<ProvenGemm> proveGemm(ModuleOp mod, std::string &reason) {
  auto reject = [&](StringRef code) -> std::optional<ProvenGemm> {
    reason = code.str();
    return std::nullopt;
  };
  SmallVector<triton::FuncOp> funcs;
  mod.walk([&](triton::FuncOp f) { funcs.push_back(f); });
  if (funcs.size() != 1)
    return reject("requires_one_triton_function");
  auto f = funcs.front();
  if (f.getBody().empty() || !llvm::hasSingleElement(f.getBody()) || f.getNumArguments() != 3 ||
      f.getFunctionType().getNumResults() != 0)
    return reject("requires_three_fp32_pointer_arguments");
  for (auto arg : f.getArguments()) {
    auto ptr = dyn_cast<triton::PointerType>(arg.getType());
    if (!ptr || !ptr.getPointeeType().isF32())
      return reject("requires_three_fp32_pointer_arguments");
  }
  SmallVector<triton::DotOp> dots;
  SmallVector<triton::LoadOp> loads;
  SmallVector<triton::StoreOp> stores;
  SmallVector<scf::ForOp> loops;
  SmallVector<triton::GetProgramIdOp> pids;
  bool extra = false;
  const std::set<std::string> allowed{
      "tt.func",        "tt.return",    "tt.get_program_id", "tt.make_range",  "tt.splat",
      "tt.expand_dims", "tt.broadcast", "tt.addptr",         "tt.load",        "tt.store",
      "tt.dot",         "scf.for",      "scf.yield",         "arith.constant", "arith.addi",
      "arith.subi",     "arith.muli",   "arith.divsi",       "arith.divui",    "arith.remsi",
      "arith.remui",    "arith.cmpi",   "arith.andi"};
  f.walk([&](Operation *op) {
    if (!allowed.count(op->getName().getStringRef().str()))
      extra = true;
    if (auto x = dyn_cast<triton::DotOp>(op))
      dots.push_back(x);
    if (auto x = dyn_cast<triton::LoadOp>(op))
      loads.push_back(x);
    if (auto x = dyn_cast<triton::StoreOp>(op))
      stores.push_back(x);
    if (auto x = dyn_cast<scf::ForOp>(op))
      loops.push_back(x);
    if (auto x = dyn_cast<triton::GetProgramIdOp>(op))
      pids.push_back(x);
  });
  if (extra || dots.size() != 1 || loads.size() != 2 || stores.size() != 1 || loops.size() != 1 ||
      pids.size() != 1)
    return reject("requires_complete_single_loop_load_dot_store_without_extra_operations");
  if (pids[0].getAxis() != triton::ProgramIDDim::X)
    return reject("requires_one_dimensional_x_grid");
  auto dot = dots[0];
  auto loop = loops[0];
  auto store = stores[0];
  auto at = dyn_cast<RankedTensorType>(dot.getA().getType());
  auto bt = dyn_cast<RankedTensorType>(dot.getB().getType());
  auto ct = dyn_cast<RankedTensorType>(dot.getC().getType());
  if (!at || !bt || !ct || at.getRank() != 2 || bt.getRank() != 2 || ct.getRank() != 2 ||
      !at.getElementType().isF32() || !bt.getElementType().isF32() ||
      !ct.getElementType().isF32() || at.getDimSize(0) != ct.getDimSize(0) ||
      bt.getDimSize(1) != ct.getDimSize(1) || at.getDimSize(1) != bt.getDimSize(0) ||
      dot.getInputPrecision() != triton::InputPrecision::IEEE)
    return reject("requires_rank_two_ieee_fp32_dot");
  ProvenGemm g{f, 0, 0, 0, ct.getDimSize(0), ct.getDimSize(1), at.getDimSize(1), 0};
  auto lo = integer(loop.getLowerBound()), hi = integer(loop.getUpperBound()),
       step = integer(loop.getStep());
  if (!lo || !hi || !step || *lo != 0 || *hi <= 0 || *step != g.bk || g.bm <= 0 || g.bn <= 0 ||
      g.bk <= 0)
    return reject("requires_static_zero_based_k_loop");
  auto acc = dyn_cast<BlockArgument>(dot.getC());
  auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
  if (!acc || acc.getOwner() != loop.getBody() || acc.getArgNumber() == 0 || !yield)
    return reject("requires_loop_carried_zero_accumulator");
  unsigned ai = acc.getArgNumber() - 1;
  if (ai >= loop.getNumResults() || !zero(loop.getInitArgs()[ai]) ||
      yield.getOperand(ai) != dot.getResult() || store.getValue() != loop.getResult(ai) ||
      dot->getParentOp() != loop || store->getParentOp() != f)
    return reject("requires_unmodified_accumulator_and_store");
  auto al = dot.getA().getDefiningOp<triton::LoadOp>();
  auto bl = dot.getB().getDefiningOp<triton::LoadOp>();
  if (!al || !bl || al == bl || al->getParentOp() != loop || bl->getParentOp() != loop ||
      !al.getOther() || !bl.getOther() || !zero(al.getOther()) || !zero(bl.getOther()) ||
      !al.getMask() || !bl.getMask() || !store.getMask())
    return reject("requires_zero_filled_masked_operand_loads");
  // Volatile loads and non-default memory semantics must not be discarded.
  if (al.getIsVolatile() || bl.getIsVolatile())
    return reject("volatile_load_is_not_gemm_contract");
  Symbolic sym{f, loop};
  Coords ij{Expr::variable("i"), Expr::variable("j")};
  auto aa = sym.address(al.getPtr(), ij), ba = sym.address(bl.getPtr(), ij),
       ca = sym.address(store.getPtr(), ij);
  auto am = sym.mask(al.getMask(), ij), bm = sym.mask(bl.getMask(), ij),
       cm = sym.mask(store.getMask(), ij);
  if (!aa || !ba || !ca || !am || !bm || !cm || aa->argument != 0 || ba->argument != 1 ||
      ca->argument != 2)
    return reject("unproven_pointer_or_mask_provenance");
  // Recover numTilesN from the actual pid quotient/remainder expressions.
  std::optional<int64_t> nt;
  f.walk([&](Operation *op) {
    auto name = op->getName().getStringRef();
    if ((name == "arith.divsi" || name == "arith.divui" || name == "arith.remsi" ||
         name == "arith.remui") &&
        op->getOperand(0) == pids[0].getResult()) {
      auto c = integer(op->getOperand(1));
      if (!c || *c <= 0 || (nt && *nt != *c))
        extra = true;
      else
        nt = *c;
    }
  });
  if (extra)
    return reject("inconsistent_pid_quotient_and_remainder");
  int64_t nn = nt.value_or(1);
  Expr pm = nn == 1 ? Expr::variable("p") : Expr::variable("D" + std::to_string(nn));
  Expr pn = nn == 1 ? Expr{} : Expr::variable("R" + std::to_string(nn));
  Expr row = pm.scale(g.bm).add(ij[0]), col = pn.scale(g.bn).add(ij[1]);
  Expr ka = Expr::variable("s").add(ij[1]), kb = Expr::variable("s").add(ij[0]);
  auto m = bound(*am, row), n = bound(*bm, col), ak = bound(*am, ka), bk = bound(*bm, kb);
  if (!m || !n || !ak || !bk || *ak != *bk)
    return reject("unproven_global_m_n_k_from_load_masks");
  g.m = *m;
  g.n = *n;
  g.k = *ak;
  // Padded loop extents are legal only if every extra lane is zero-filled by
  // these proven logical-K masks; require the smallest full number of batches.
  if (*hi != g.k && *hi != ((g.k + g.bk - 1) / g.bk) * g.bk &&
      *hi != std::max<int64_t>(2 * g.bk, ((g.k + g.bk - 1) / g.bk) * g.bk))
    return reject("k_loop_extent_not_logical_or_rounded_k");
  if (g.m > 32768 || g.n > 32768 || g.k > 32768 || nn != (g.n + g.bn - 1) / g.bn)
    return reject("dimension_or_grid_outside_proven_i32_contract");
  if (!(aa->offset == row.scale(g.k).add(ka)) || !(ba->offset == kb.scale(g.n).add(col)) ||
      !(ca->offset == row.scale(g.n).add(col)))
    return reject("requires_proven_contiguous_a_b_c_address_formulas");
  if (!exactMask(*am, row, g.m, ka, g.k) || !exactMask(*bm, kb, g.k, col, g.n) ||
      !exactMask(*cm, row, g.m, col, g.n))
    return reject("mask_semantics_differ_from_logical_gemm");
  // Reject unrecognized loop-carried values even when not feeding the dot.
  for (unsigned i = 0; i < loop.getNumResults(); ++i)
    if (i != ai) {
      Value v = loop.getRegionIterArgs()[i];
      auto ty = dyn_cast<RankedTensorType>(v.getType());
      if (!ty || !isa<triton::PointerType>(ty.getElementType()) || !sym.address(v, ij))
        return reject("extra_loop_carried_state");
    }
  g.oldGrid = ((g.m + g.bm - 1) / g.bm) * nn;
  reason = "proven_full_masked_contiguous_fp32_gemm";
  return g;
}

ProblemFacts problemFacts(ProvenGemm g) {
  ProblemFacts p;
  Provenance provenance{"early_ttir_symbolic_gemm_proof", "ttir-before-scalarize",
                        g.function.getName().str(), "proven"};
  auto integerFact = [&](int64_t v) { return Fact<int64_t>::known(v, provenance); };
  auto stringFact = [&](std::string v) {
    return Fact<std::string>::known(std::move(v), provenance);
  };
  p.kernel.operationId = g.function.getName().str();
  p.kernel.operationKind = stringFact("gemm");
  p.kernel.m = integerFact(g.m);
  p.kernel.n = integerFact(g.n);
  p.kernel.k = integerFact(g.k);
  p.kernel.localTile = {integerFact(g.bm), integerFact(g.bn), integerFact(g.bk)};
  p.kernel.programMapping = stringFact("row_major_1d");
  p.kernel.accumulator = stringFact("zero_fp32");
  p.kernel.epilogue = stringFact("store_only");
  p.kernel.mask = stringFact("logical_m_n_k_zero_fill");
  p.kernel.tail = stringFact("masked");
  auto operand = [&](std::string role, int64_t rows, int64_t cols) {
    TensorFacts t;
    t.role = std::move(role);
    t.dtype = stringFact("f32");
    t.logicalShape = {integerFact(rows), integerFact(cols)};
    t.physicalShape = t.logicalShape;
    t.physicalStrides = {integerFact(cols), integerFact(1)};
    t.offset = integerFact(0);
    return t;
  };
  p.kernel.operands = {operand("A", g.m, g.k), operand("B", g.k, g.n), operand("C", g.m, g.n)};
  p.launch.gridExtents = {integerFact(g.oldGrid), integerFact(1), integerFact(1)};
  p.launch.programMapping = stringFact("row_major_1d");
  p.launch.callBoundary = stringFact("complete_synchronous_call");
  return p;
}

// This is compiler-owned canonical TTIR, not an external GEMM implementation.
// The verified function keeps its original name, signature, attrs and source
// location. Only its body is replaced, and it continues through ordinary
// TritonCPU tensor/pointer/dot lowering to the same native SME code generator.
std::string replacementIR(const ProvenGemm &g, const CompilerGemmSchedule &s) {
  const int64_t bm = s.blockM, bn = s.blockN, bk = s.blockK;
  const int64_t kp = std::max<int64_t>(2 * bk, ((g.k + bk - 1) / bk) * bk);
  const int64_t nn = (g.n + bn - 1) / bn;
  auto t1 = [](int64_t x, const char *e) { return "tensor<" + std::to_string(x) + "x" + e + ">"; };
  auto t2 = [](int64_t x, int64_t y, const char *e) {
    return "tensor<" + std::to_string(x) + "x" + std::to_string(y) + "x" + e + ">";
  };
  const auto mi = t1(bm, "i32"), ni = t1(bn, "i32"), ki = t1(bk, "i32"), kbool = t1(bk, "i1");
  const auto m1i = t2(bm, 1, "i32"), k1i = t2(bk, 1, "i32"), oneKi = t2(1, bk, "i32"),
             oneNi = t2(1, bn, "i32");
  const auto mki = t2(bm, bk, "i32"), kni = t2(bk, bn, "i32"), mni = t2(bm, bn, "i32");
  const auto m1p = t2(bm, 1, "!tt.ptr<f32>"), k1p = t2(bk, 1, "!tt.ptr<f32>");
  const auto mkp = t2(bm, bk, "!tt.ptr<f32>"), knp = t2(bk, bn, "!tt.ptr<f32>"),
             mnp = t2(bm, bn, "!tt.ptr<f32>");
  const auto mkf = t2(bm, bk, "f32"), knf = t2(bk, bn, "f32"), mnf = t2(bm, bn, "f32");
  const auto m1b = t2(bm, 1, "i1"), oneNb = t2(1, bn, "i1"), oneKb = t2(1, bk, "i1"),
             k1b = t2(bk, 1, "i1");
  const auto mkb = t2(bm, bk, "i1"), knb = t2(bk, bn, "i1"), mnb = t2(bm, bn, "i1");
  std::ostringstream o;
  o << "module { tt.func public @__sme_plan_generated(%A: !tt.ptr<f32>, %B: !tt.ptr<f32>, %C: "
       "!tt.ptr<f32>) {\n";
  for (const auto &c : std::vector<std::pair<std::string, int64_t>>{
           {"zero", 0}, {"bm", bm}, {"bn", bn}, {"bk", bk}, {"kp", kp}, {"nt", nn}})
    o << "%" << c.first << " = arith.constant " << c.second << " : i32\n";
  auto dense = [&](const char *name, int64_t value, const std::string &type) {
    o << "%" << name << " = arith.constant dense<" << value << "> : " << type << "\n";
  };
  dense("lda", g.k, m1i);
  dense("ldb", g.n, k1i);
  dense("ldc", g.n, m1i);
  dense("am_step", bk, mki);
  dense("bm_step", bk * g.n, kni);
  dense("M", g.m, m1i);
  dense("N", g.n, oneNi);
  dense("K", g.k, ki);
  o << "%za = arith.constant dense<0.000000e+00> : " << mkf << "\n"
    << "%zb = arith.constant dense<0.000000e+00> : " << knf << "\n"
    << "%zc = arith.constant dense<0.000000e+00> : " << mnf << "\n"
    << "%pid = tt.get_program_id x : i32\n%pm = arith.divsi %pid, %nt : i32\n%pn = arith.remsi "
       "%pid, %nt : i32\n"
    << "%rm0 = arith.muli %pm, %bm : i32\n%rn0 = arith.muli %pn, %bn : i32\n"
    << "%ri = tt.make_range {start=0:i32,end=" << bm << ":i32} : " << mi << "\n"
    << "%ci = tt.make_range {start=0:i32,end=" << bn << ":i32} : " << ni << "\n"
    << "%ki = tt.make_range {start=0:i32,end=" << bk << ":i32} : " << ki << "\n"
    << "%r0 = tt.splat %rm0 : i32 -> " << mi << "\n%r = arith.addi %r0, %ri : " << mi << "\n"
    << "%c0 = tt.splat %rn0 : i32 -> " << ni << "\n%c = arith.addi %c0, %ci : " << ni << "\n"
    << "%r2 = tt.expand_dims %r {axis=1:i32} : " << mi << " -> " << m1i << "\n"
    << "%c2 = tt.expand_dims %c {axis=0:i32} : " << ni << " -> " << oneNi << "\n"
    << "%krow = tt.expand_dims %ki {axis=0:i32} : " << ki << " -> " << oneKi << "\n"
    << "%kcol = tt.expand_dims %ki {axis=1:i32} : " << ki << " -> " << k1i << "\n"
    << "%ao = arith.muli %r2, %lda : " << m1i << "\n%abase = tt.splat %A : !tt.ptr<f32> -> " << m1p
    << "\n"
    << "%arow = tt.addptr %abase, %ao : " << m1p << ", " << m1i << "\n"
    << "%arows = tt.broadcast %arow : " << m1p << " -> " << mkp << "\n"
    << "%ak = tt.broadcast %krow : " << oneKi << " -> " << mki << "\n"
    << "%ap = tt.addptr %arows, %ak : " << mkp << ", " << mki << "\n"
    << "%bo = arith.muli %kcol, %ldb : " << k1i << "\n%bbase = tt.splat %B : !tt.ptr<f32> -> "
    << k1p << "\n"
    << "%brow = tt.addptr %bbase, %bo : " << k1p << ", " << k1i << "\n"
    << "%brows = tt.broadcast %brow : " << k1p << " -> " << knp << "\n"
    << "%bnv = tt.broadcast %c2 : " << oneNi << " -> " << kni << "\n"
    << "%bp = tt.addptr %brows, %bnv : " << knp << ", " << kni << "\n"
    << "%rmask = arith.cmpi slt, %r2, %M : " << m1i << "\n"
    << "%cmask = arith.cmpi slt, %c2, %N : " << oneNi << "\n"
    << "%am = tt.broadcast %rmask : " << m1b << " -> " << mkb << "\n"
    << "%bmask = tt.broadcast %cmask : " << oneNb << " -> " << knb << "\n"
    << "%result:3 = scf.for %s = %zero to %kp step %bk iter_args(%pa=%ap,%pb=%bp,%acc=%zc) -> ("
    << mkp << ", " << knp << ", " << mnf << ") : i32 {\n"
    << "%ss = tt.splat %s : i32 -> " << ki << "\n%kk = arith.addi %ss, %ki : " << ki << "\n"
    << "%km = arith.cmpi slt, %kk, %K : " << ki << "\n"
    << "%kam = tt.expand_dims %km {axis=0:i32} : " << kbool << " -> " << oneKb << "\n"
    << "%kbm = tt.expand_dims %km {axis=1:i32} : " << kbool << " -> " << k1b << "\n"
    << "%kama = tt.broadcast %kam : " << oneKb << " -> " << mkb << "\n"
    << "%kbmb = tt.broadcast %kbm : " << k1b << " -> " << knb << "\n"
    << "%ma = arith.andi %am, %kama : " << mkb << "\n%mb = arith.andi %kbmb, %bmask : " << knb
    << "\n"
    << "%va = tt.load %pa, %ma, %za : " << mkp << "\n%vb = tt.load %pb, %mb, %zb : " << knp << "\n"
    << "%next = tt.dot %va, %vb, %acc, inputPrecision = ieee : " << mkf << " * " << knf << " -> "
    << mnf << "\n"
    << "%na = tt.addptr %pa, %am_step : " << mkp << ", " << mki << "\n"
    << "%nb = tt.addptr %pb, %bm_step : " << knp << ", " << kni << "\n"
    << "scf.yield %na, %nb, %next : " << mkp << ", " << knp << ", " << mnf << "\n}\n"
    << "%co = arith.muli %r2, %ldc : " << m1i << "\n%cb = tt.splat %C : !tt.ptr<f32> -> " << m1p
    << "\n"
    << "%cr = tt.addptr %cb, %co : " << m1p << ", " << m1i << "\n"
    << "%crs = tt.broadcast %cr : " << m1p << " -> " << mnp << "\n"
    << "%cn = tt.broadcast %c2 : " << oneNi << " -> " << mni << "\n"
    << "%cp = tt.addptr %crs, %cn : " << mnp << ", " << mni << "\n"
    << "%mc = tt.broadcast %rmask : " << m1b << " -> " << mnb << "\n"
    << "%nc = tt.broadcast %cmask : " << oneNb << " -> " << mnb << "\n"
    << "%cm = arith.andi %mc, %nc : " << mnb << "\n"
    << "tt.store %cp, %result#2, %cm : " << mnp << "\ntt.return\n}}\n";
  return o.str();
}

class CompilerGemmPlanPass : public PassWrapper<CompilerGemmPlanPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CompilerGemmPlanPass)
  CompilerGemmPlanPass() = default;
  CompilerGemmPlanPass(std::string mode, CompilerGemmPolicyRequest policy, int64_t probeIndex)
      : mode(std::move(mode)), policy(std::move(policy)), calibrationProbeIndex(probeIndex) {}
  StringRef getArgument() const final { return "sme-plan-compiler-gemm"; }
  StringRef getDescription() const final {
    return "Prove complete TTIR GEMM and select compiler-owned tiling and layout";
  }
  void runOnOperation() final {
    auto module = getOperation();
    Builder builder(module.getContext());
    std::string reason;
    auto proven = proveGemm(module, reason);
    module->setAttr("sme_plan.mode", builder.getStringAttr(mode));
    module->setAttr("sme_plan.proof", builder.getStringAttr(reason));
    if (!proven) {
      module->setAttr("sme_plan.verified", builder.getI64IntegerAttr(0));
      if (mode != "cpu" && mode != "generic") {
        module.emitError("compiler-owned GEMM rejected: ") << reason;
        signalPassFailure();
      }
      return;
    }
    ProvenGemm &g = *proven;
    std::string original;
    {
      llvm::raw_string_ostream os(original);
      g.function.print(os);
    }
    module->setAttr("sme_plan.original_ttir", builder.getStringAttr(original));
    auto setInt = [&](StringRef key, int64_t value) {
      auto attr = builder.getI64IntegerAttr(value);
      module->setAttr(key, attr);
      g.function->setAttr(key, attr);
    };
    setInt("sme_plan.verified", 1);
    setInt("sme_plan.m", g.m);
    setInt("sme_plan.n", g.n);
    setInt("sme_plan.k", g.k);
    setInt("sme_plan.lda", g.k);
    setInt("sme_plan.ldb", g.n);
    setInt("sme_plan.ldc", g.n);
    setInt("sme_plan.original_grid_x", g.oldGrid);
    setInt("sme_plan.original_grid_y", 1);
    setInt("sme_plan.original_grid_z", 1);
    setInt("sme_plan.original_block_m", g.bm);
    setInt("sme_plan.original_block_n", g.bn);
    setInt("sme_plan.original_block_k", g.bk);
    CompilerGemmSchedule schedule;
    if (mode == "cpu" || mode == "generic") {
      schedule.path = CompilerGemmPath::GenericCPU;
      schedule.blockM = g.bm;
      schedule.blockN = g.bn;
      schedule.blockK = g.bk;
      schedule.threads = policy.device.totalThreadBudget;
    } else if (mode == "sme-fixed") {
      schedule.blockM = 128;
      schedule.blockN = 256;
      schedule.blockK = 16;
      schedule.threads = policy.device.totalThreadBudget;
    } else if (mode == "calibration-probe") {
      policy.problem = problemFacts(g);
      auto probes = compilerGemmCalibrationSchedules(policy);
      if (calibrationProbeIndex < 0 ||
          static_cast<size_t>(calibrationProbeIndex) >= probes.size()) {
        module.emitError("invalid compiler calibration probe index");
        signalPassFailure();
        return;
      }
      const auto &probe = probes[calibrationProbeIndex];
      if (!probe.supported || probe.index != calibrationProbeIndex) {
        std::string codes;
        for (const auto &r : probe.reasonCodes)
          codes += r + ";";
        module.emitError("unsupported compiler calibration probe: ") << codes;
        signalPassFailure();
        return;
      }
      schedule = probe.schedule;
      setInt("sme_plan.calibration_probe", calibrationProbeIndex);
      setInt("sme_plan.calibration_probe_count", probes.size());
      setInt("sme_plan.predicted_scratch_bytes", probe.scratchBytesPerTask);
    } else if (mode == "ours" || mode.rfind("ours-rank-", 0) == 0 || mode == "ours-worst") {
      policy.problem = problemFacts(g);
      auto decision = planCompilerGemm(policy);
      if (!decision.apply || !decision.selected) {
        std::string codes;
        for (const auto &r : decision.reasonCodes)
          codes += r + ";";
        module.emitError("compiler GEMM policy has no admissible candidate: ") << codes;
        signalPassFailure();
        return;
      }
      size_t rank = 0;
      if (mode == "ours-worst")
        rank = decision.rankedCandidates.size() - 1;
      else if (mode.rfind("ours-rank-", 0) == 0) {
        try {
          const auto requested = std::stoull(mode.substr(10));
          if (requested < 1 || requested > decision.rankedCandidates.size()) {
            module.emitError("requested GEMM rank is outside the ranked candidate list");
            signalPassFailure();
            return;
          }
          rank = requested - 1;
        } catch (...) {
          module.emitError("invalid GEMM rank mode");
          signalPassFailure();
          return;
        }
      }
      schedule = decision.rankedCandidates[rank].schedule;
      module->setAttr("sme_plan.calibration_evidence",
                      builder.getStringAttr(decision.calibrationEvidenceId));
      setInt("sme_plan.candidate_count", decision.rankedCandidates.size());
      setInt("sme_plan.selected_rank", rank + 1);
      setInt("sme_plan.predicted_ns",
             static_cast<int64_t>(decision.rankedCandidates[rank].predictedNs));
    } else {
      module.emitError("unknown compiler GEMM mode: ") << mode;
      signalPassFailure();
      return;
    }
    if (schedule.blockM <= 0 || schedule.blockN <= 0 || schedule.blockK <= 0 ||
        schedule.threads <= 0 || schedule.threads > policy.device.totalThreadBudget ||
        schedule.stripK < 0 || (schedule.stripK && schedule.stripK % schedule.blockK)) {
      module.emitError("invalid compiler GEMM schedule");
      signalPassFailure();
      return;
    }
    setInt("sme_plan.selected_sme", schedule.path == CompilerGemmPath::SME);
    setInt("sme_plan.path", schedule.path == CompilerGemmPath::SME);
    setInt("sme_plan.block_m", schedule.blockM);
    setInt("sme_plan.block_n", schedule.blockN);
    setInt("sme_plan.block_k", schedule.blockK);
    setInt("sme_plan.threads", schedule.threads);
    setInt("sme_plan.grid_x", ((g.m + schedule.blockM - 1) / schedule.blockM) *
                                  ((g.n + schedule.blockN - 1) / schedule.blockN));
    setInt("sme_plan.grid_y", 1);
    setInt("sme_plan.grid_z", 1);
    setInt("sme_plan.k_strip", schedule.stripK);
    setInt("sme_plan.a_pack_vector4", schedule.aPack == CompilerGemmAPack::Vector4);
    int64_t pad = 0;
    if (schedule.bLayout != CompilerGemmBLayout::Original) {
      if (!policy.capabilities.vectorLengthBits.isProven() ||
          *policy.capabilities.vectorLengthBits.value <= 0) {
        module.emitError("B panel padding requires proven streaming vector length");
        signalPassFailure();
        return;
      }
      pad = *policy.capabilities.vectorLengthBits.value / 32;
    }
    // The proof above describes the application's original B with ldb=N.
    // These mutually exclusive storage transformations are implemented by the
    // generated launcher and/or late lowering; C always keeps logical stride N.
    setInt("sme_plan.b_panel_pad",
           schedule.bLayout == CompilerGemmBLayout::LocalPanelPadSVL ? pad : 0);
    setInt("sme_plan.b_global_pad",
           schedule.bLayout == CompilerGemmBLayout::GlobalPadSVL ? pad : 0);
    setInt("sme_plan.reuse_m", 4);
    if (mode == "cpu" || mode == "generic")
      return;
    auto replacement = parseSourceString<ModuleOp>(replacementIR(g, schedule), module.getContext());
    if (!replacement || failed(verify(*replacement))) {
      module.emitError("compiler-generated GEMM TTIR failed verification");
      signalPassFailure();
      return;
    }
    std::string replacementReason;
    auto generatedProof = proveGemm(*replacement, replacementReason);
    if (!generatedProof || generatedProof->m != g.m || generatedProof->n != g.n ||
        generatedProof->k != g.k || generatedProof->bm != schedule.blockM ||
        generatedProof->bn != schedule.blockN || generatedProof->bk != schedule.blockK) {
      module.emitError("compiler-generated GEMM failed semantic re-verification: ")
          << replacementReason;
      signalPassFailure();
      return;
    }
    auto newFunction = *replacement->getOps<triton::FuncOp>().begin();
    g.function.getBody().takeBody(newFunction.getBody());
    g.function.walk([&](Operation *op) { op->setLoc(g.function.getLoc()); });
    setInt("sme_plan.rewritten", 1);
    if (failed(verify(module)))
      signalPassFailure();
  }

private:
  std::string mode = "generic";
  CompilerGemmPolicyRequest policy;
  int64_t calibrationProbeIndex = -1;
};
} // namespace

std::unique_ptr<mlir::Pass> createCompilerGemmPlanPass(std::string mode,
                                                       CompilerGemmPolicyRequest policy,
                                                       std::int64_t calibrationProbeIndex) {
  return std::make_unique<CompilerGemmPlanPass>(std::move(mode), std::move(policy),
                                                calibrationProbeIndex);
}
} // namespace sme_plan::triton_cpu
