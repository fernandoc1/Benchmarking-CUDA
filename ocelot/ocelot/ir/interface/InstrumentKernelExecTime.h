#ifndef INSTRUMENTKERNELEXECTIME_H_
#define INSTRUMENTKERNELEXECTIME_H_

#include <string> 
#include <ocelot/ir/interface/PTXOperand.h>
#include "ocelot/ir/interface/Worker.h"

namespace ir {
	
	// This class instruments the kernel, to get the total number of cycles
	// executed.
	class InstrumentKernelExecTime : public Worker {
	private:
		// This is the first basic block in the kernel, that is, it is the
		// first basic block that will be traversed during the kernel execution.
		// This basic block is always empty. We need a reference to this
		// block in order to be able to insert instrumentation in the next
		// block, where we will initialize all the counters.
		ir::ControlFlowGraph::iterator entryBlock;
		
		// We need a reference to the previously visited block, when visiting
		// the second basic block (where initialization code will be inserted)
		// and the exit block (we insert the printing code in the first block
		// before the exit block). 
		ir::ControlFlowGraph::iterator previousBlock;
		
		// We need a reference to the exitBlock in order to know that a basic
		// block, during the visitation, is indeed the exitBlock. We must keep
		// this reference because it is obtained from the CFG, and we do not
		// have access to the CFG during the traversal of basic blocks.
		ir::ControlFlowGraph::iterator exitBlock;

	public:
		InstrumentKernelExecTime()
			: entryBlock(NULL)
			, previousBlock(NULL)
			, exitBlock(NULL)
		{}
		
		virtual ~InstrumentKernelExecTime() {}
		
		virtual void process(ir::Module* m);

		virtual void postProcess(ir::Module* m) {}
		virtual void process(ir::Kernel* k) {}

		virtual void process(ir::ControlFlowGraph* cfg);

		virtual void postProcess(ir::ControlFlowGraph* cfg) {}

		virtual void process(ir::ControlFlowGraph::iterator bb);
		
		virtual void process(ir::Instruction* inst) {}
	};

}

#endif /*INSTRUMENTKERNELEXECTIME_H_*/
