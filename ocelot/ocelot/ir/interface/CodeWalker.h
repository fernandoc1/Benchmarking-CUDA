#ifndef CODEWALKER_H_
#define CODEWALKER_H_

#include "ocelot/ir/interface/Module.h"
#include "ocelot/ir/interface/Worker.h"

namespace ir {

    // The CodeWalkwer interface is used in conjunction with a Worker
    // interface. The code walker traverses the control flow graph, and
    // for each part of the CFG that it visits, it calls the appropriate
    // method of the worker.
	class CodeWalker
	{
	private:	
		Worker* worker;
		
	public:
		CodeWalker() 
			: worker(NULL)
		{};
		
		CodeWalker(Worker* w)
			: worker(w)
		{};
		
		virtual ~CodeWalker() {
			// do not delete worker, because it can be reused
		};
		
		virtual void visit(ir::Kernel* k);
		virtual void visit(ir::Module* m);
		virtual void visit(ir::ControlFlowGraph* cfg);
		virtual void visit(ir::ControlFlowGraph::iterator bb);
		virtual void visit(ir::Instruction* inst);
	};

}

#endif /*CODEWALKER_H_*/
