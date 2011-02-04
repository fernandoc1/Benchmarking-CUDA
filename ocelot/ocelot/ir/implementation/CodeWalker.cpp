#include <iostream>
#include <hydrazine/interface/Version.h>
#include "ocelot/ir/interface/Module.h"
#include "ocelot/ir/interface/ControlFlowGraph.h"
#include "ocelot/ir/interface/PTXInstruction.h"
#include "ocelot/ir/interface/CodeWalker.h"


/// Modeled after Module.writeIR()
void ir::CodeWalker::visit(ir::Module* m)
{
	worker->process(m);
	
	// visit kernels
	ir::Module::KernelMap::const_iterator kernel = m->kernels().begin();
	for ( ;	kernel != m->kernels().end(); ++kernel) {
		visit(kernel->second);
	}

	worker->postProcess(m);
}

void ir::CodeWalker::visit(ir::Kernel* k)
{
	worker->process(k);
	
	// visit kernel code
	visit(k->cfg());
}

void ir::CodeWalker::visit(ir::ControlFlowGraph* cfg)
{
	worker->process(cfg);

	ControlFlowGraph::BlockPointerVector blocks = cfg->executable_sequence();
	ControlFlowGraph::BlockPointerVector::iterator block = blocks.begin();
	for(; block != blocks.end(); ++block) {
		visit(*block);
	}

	worker->postProcess(cfg);
}

void ir::CodeWalker::visit(ir::ControlFlowGraph::iterator bb)
{
	worker->process(bb);
	
	ir::ControlFlowGraph::BasicBlock::InstructionList::iterator it = bb->instructions.begin();
	for (; it != bb->instructions.end(); it++) {
		visit(*it);
	}
}

void ir::CodeWalker::visit(ir::Instruction* inst) {
	worker->process(inst);
}
