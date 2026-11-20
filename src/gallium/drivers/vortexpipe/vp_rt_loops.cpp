/* SPDX-License-Identifier: MIT */
#include "vp_nir_to_llvm.h"

#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Transforms/Utils/LCSSA.h"
#include "llvm/Transforms/Utils/LoopPeel.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"

using namespace llvm;

extern "C" bool
vp_rt_flags_have_mask(LLVMModuleRef module, LLVMValueRef flags, uint32_t mask)
{
   KnownBits known = computeKnownBits(unwrap(flags), unwrap(module)->getDataLayout());
   return (known.One.getZExtValue() & mask) == mask;
}

/* nir_lower_shader_calls selects the resume body on loop entry and the
 * ordinary shader body on subsequent iterations. The native continuation
 * jumps directly into the appropriate path. */
static bool
has_entry_selector(const Loop *loop)
{
   auto *branch = dyn_cast<BranchInst>(loop->getHeader()->getTerminator());
   if (!branch || !branch->isConditional())
      return false;
   auto *flag = dyn_cast<PHINode>(branch->getCondition());
   if (!flag || flag->getParent() != loop->getHeader() ||
       flag->getNumIncomingValues() != 2)
      return false;
   auto *a = dyn_cast<ConstantInt>(flag->getIncomingValue(0));
   auto *b = dyn_cast<ConstantInt>(flag->getIncomingValue(1));
   return a && b && a != b &&
          loop->contains(flag->getIncomingBlock(0)) !=
          loop->contains(flag->getIncomingBlock(1));
}

static bool
has_separate_bodies(Loop *loop, const DominatorTree &dom)
{
   auto *branch = cast<BranchInst>(loop->getHeader()->getTerminator());
   for (BasicBlock *block : loop->blocks()) {
      if (block->hasAddressTaken())
         return false;
      bool first = dom.dominates(branch->getSuccessor(0), block);
      bool second = dom.dominates(branch->getSuccessor(1), block);
      for (Instruction &inst : *block) {
         if (auto *call = dyn_cast<CallBase>(&inst))
            if (call->cannotDuplicate())
               return false;
         /* Only one copy of either body can execute. Computation, clocks
          * and subgroup operations shared by both paths must stay joined. */
         if (first == second && !isa<PHINode, BranchInst>(&inst))
            return false;
      }
   }
   return true;
}

extern "C" void
vp_lower_rt_resume_loops(LLVMModuleRef module)
{
   Function *kernel = unwrap(module)->getFunction("kernel_main");
   if (!kernel || kernel->isDeclaration())
      return;
   LoopAnalysisManager loops;
   FunctionAnalysisManager functions;
   CGSCCAnalysisManager sccs;
   ModuleAnalysisManager modules;
   PassBuilder passes;
   passes.registerModuleAnalyses(modules);
   passes.registerCGSCCAnalyses(sccs);
   passes.registerFunctionAnalyses(functions);
   passes.registerLoopAnalyses(loops);
   passes.crossRegisterProxies(loops, functions, sccs, modules);

   auto initial = functions.getResult<LoopAnalysis>(*kernel).getLoopsInPreorder();
   if (llvm::none_of(initial, has_entry_selector))
      return;
   FunctionPassManager canonicalize;
   canonicalize.addPass(LoopSimplifyPass());
   canonicalize.addPass(LCSSAPass());
   canonicalize.run(*kernel, functions);

   auto &info = functions.getResult<LoopAnalysis>(*kernel);
   auto &dom = functions.getResult<DominatorTreeAnalysis>(*kernel);
   auto &evolution = functions.getResult<ScalarEvolutionAnalysis>(*kernel);
   auto &assumptions = functions.getResult<AssumptionAnalysis>(*kernel);
   auto ordered = info.getLoopsInPreorder();
   for (Loop *loop : llvm::reverse(ordered)) {
      if (has_entry_selector(loop) && canPeel(loop) &&
          !getLoopConvergenceHeart(loop) && has_separate_bodies(loop, dom)) {
         ValueToValueMapTy map;
         peelLoop(loop, 1, &info, &evolution, dom, &assumptions, true, map);
      }
   }
}
