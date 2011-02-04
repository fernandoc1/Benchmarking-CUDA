#ifndef PRINTERWORKER_H_
#define PRINTERWORKER_H_

#include <ocelot/ir/interface/Worker.h>

namespace ir {

    /// This worker prints the kernel code in PTX format. It will produce a
    /// "ready-to-compile" file.
	class PrinterWorker : public Worker {
	private:
		/// This is the file where the code will be printed.
		std::ostream&               output;
		
		/// This map keeps record of all the global variables in the PTX file
		/// to avoid printing these variables more than once, when the register
		/// declarations are printed.
		ir::Module::GlobalMap*      globalVariables;
		
	public:
		PrinterWorker()
			: output(std::cout)
		{}

		PrinterWorker(std::ostream& out)
			: output(out)
		{}

		virtual ~PrinterWorker() {};
		
		virtual void process(ir::Module* m);

		virtual void postProcess(ir::Module* m) {}

		virtual void process(ir::Kernel* k) {
			k->write(output);
		}

		virtual void process(ir::ControlFlowGraph* cfg) {};
		virtual void postProcess(ir::ControlFlowGraph* cfg) {};
		virtual void process(ir::ControlFlowGraph::iterator bb) {};
		virtual void process(ir::Instruction* inst) {};
	};

}

#endif /*PRINTERWORKER_H_*/
