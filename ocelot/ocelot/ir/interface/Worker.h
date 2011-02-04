#ifndef WORKER_H_
#define WORKER_H_

#include <ocelot/ir/interface/Module.h>
#include <ocelot/ir/interface/ControlFlowGraph.h>

namespace ir {
	
	// This class defines a general interface to objects that do some
	// action on the control flow graph. The worker interface is used in
	// conjunction with the walker interface. The walker traverses the
	// CFG, while the worker does some action on each part of the CFG that
	// is visited.
	class Worker
	{		
	public:
		virtual void process(ir::Module* m) = 0;
		virtual void postProcess(ir::Module* m) = 0;
		virtual void process(ir::Kernel* k) = 0;
		virtual void process(ir::ControlFlowGraph* cfg) = 0;
		virtual void postProcess(ir::ControlFlowGraph* cfg) = 0;
		virtual void process(ir::ControlFlowGraph::iterator bb) = 0;
		virtual void process(ir::Instruction* inst) = 0;
	};
	
}

#endif /*WORKER_H_*/
