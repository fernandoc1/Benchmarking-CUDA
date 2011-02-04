/*
 * BlockUnificationPass.cpp
 *
 *  Created on: Aug 9, 2010
 *      Author: coutinho
 */

#include <ocelot/analysis/interface/SyncEliminationPass.h>

namespace analysis {

SyncEliminationPass::SyncEliminationPass() : KernelPass(StaticSingleAssignment, "SyncElimination")
{
}

SyncEliminationPass::~SyncEliminationPass()
{
}

void SyncEliminationPass::runOnKernel(ir::Kernel& k)
{
	DivergenceAnalysis divAnalysis;

	divAnalysis.runOnKernel(k);

	DataflowGraph::iterator block = ++k.dfg()->begin();
	DataflowGraph::iterator blockEnd = --k.dfg()->end();

	for (; block != blockEnd; block++) {
		if (!divAnalysis.isDivBlock(block)) {
			DataflowGraph::Instruction inst = *(--block->_instructions.end());
			if (typeid(ir::PTXInstruction) == typeid(*(inst.i))) {
				ir::PTXInstruction *ptxInst = static_cast<ir::PTXInstruction*> (inst.i);
				if (ptxInst->opcode == ir::PTXInstruction::Opcode::Bra) {
					ptxInst->uni = true;
				}
			}
		}
	}
}

}
