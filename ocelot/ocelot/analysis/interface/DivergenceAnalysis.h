/*
 * DivergenceAnalysis.h
 *
 *  Created on: Jun 1, 2010
 *      Author: Diogo Sampaio
 */

#ifndef DIVERGINGENCEANALYSIS_H_
#define DIVERGINGENCEANALYSIS_H_

#include <ocelot/ir/interface/Module.h>
#include <ocelot/ir/interface/PTXKernel.h>
#include <ocelot/analysis/interface/BranchInfo.h>
#include <ocelot/analysis/interface/Pass.h>
#include <ocelot/analysis/interface/DataflowGraph.h>
#include <ocelot/graphs/interface/DivergenceGraph.h>

#define ANALYSISINFO

namespace analysis {
/*!\brief Makes a divergence analysis of a kernel */
class DivergenceAnalysis
{
	public:
		typedef std::set<BranchInfo> branch_set;

	private:
		ir::PTXKernel *_kernel;
		/*!\brief Holds the variables marks of divergent blocks */
		graph_utils::DivergenceGraph _divergGraph;
		/*!\brief Holds the amount of time used to realize the divergence analysis */
		struct timeval _analysisTime;
		/*!\brief A set with all branch instruction of the kernel */
		branch_set _branches;
		/*!\brief A set with all branch instruction considered divergent of the kernel */
		branch_set _divergentBranches;
		/*!\brief A set with all branch instruction considered not divergent of the kernel */
		branch_set _notDivergentBranches;
		bool _doCFGanalysis;

		/*!\brief Make the inital dataflow analysis */
		void _analyzeDataFlow();
		/*!\brief Makes the controlflow analysis, dependent of the results of the dataflow analysis */
		void _analyzeControlFlow();
		/*!\brief Taints the destiny variable of a phi instruction with a predicate */
		void _addPredicate(const DataflowGraph::PhiInstruction &phi, const graph_utils::DivergenceGraph::node_type &predicate);

	public:
		DivergenceAnalysis();
		/* inherit from KernelPass */
		virtual void initialize( const ir::Module& m ) {};
		/* inherit from KernelPass */
		virtual void runOnKernel( ir::Kernel& k );
		/* inherit from KernelPass */
		virtual void finalize() {};

		/*\brief re-run the analysis on a already passed kernel */
		void run();

		/*!\brief Tests if a block ends with a divergent branch instruction */
		bool isDivBlock( DataflowGraph::const_iterator &block ) const;
		/*!\brief Tests if a block ends with a divergent branch instruction */
		bool isDivBlock( DataflowGraph::iterator &block ) const;
		/*!\brief Tests if a block ends with a divergent branch instruction */
		bool isDivBlock( const DataflowGraph::Block *block ) const;

		/*!\brief Tests if a block ends with a branch instruction */
		bool isPossibleDivBlock( DataflowGraph::const_iterator &block ) const;
		/*!\brief Tests if a block ends with a branch instruction */
		bool isPossibleDivBlock( const DataflowGraph::Block *block ) const;

		/*!\brief Tests if a branch instruction is divergent */
		bool isDivBranch(const DataflowGraph::InstructionVector::const_iterator &instruction ) const;
		/*!\brief Tests if a instruction uses divergent variables */
		bool isDivInstruction( const DataflowGraph::Instruction &instruction ) const;
		/*!\brief Tests if a instruction is a branch instruction with possibility of divergence */
		bool isPossibleDivBranch(const DataflowGraph::InstructionVector::const_iterator &instruction ) const;

		bool doControlFlowAnalysis() const { return _doCFGanalysis; };
		void setControlFlowAnalysis(const bool doControlFlowAnalysis) {_doCFGanalysis = doControlFlowAnalysis;};

		/*!\brief Returns the first block from the DFG */
		DataflowGraph::const_iterator beginBlock() const;
		/*!\brief Returns the last block from the DFG */
		DataflowGraph::const_iterator endBlock() const;

		/*!\brief Returns a set of all branches of the kernel */
		const branch_set &getBranches() const {return _branches;};
		/*!\brief Returns a set of all possible divergent branches of the kernel */
		branch_set &getDivergentBranches() {return _divergentBranches;};
		/*!\brief Returns a set of all not divergent branches of the kernel */
		branch_set &getNonDivergentBranches() {return _notDivergentBranches;};

		/*!\brief Returns the dataflow/divergence graph build by the analysis */
		const graph_utils::DivergenceGraph &getDirtGraph() const { return _divergGraph;};
		/*!\brief Returns the amount of time used to realize the divergence analysis */
		const struct timeval &getAnalysisTime() const { return _analysisTime;};
		/*!\brief Returns the kernel dataflow graph */
		const DataflowGraph *getDFG() const {return _kernel->dfg();};
};
}

namespace std{
bool operator<(const analysis::BranchInfo x, const analysis::BranchInfo y);
bool operator<=(const analysis::BranchInfo x, const analysis::BranchInfo y);
bool operator>(const analysis::BranchInfo x, const analysis::BranchInfo y);
bool operator>=(const analysis::BranchInfo x, const analysis::BranchInfo y);
}

#endif /* DIVERGINGENCEANALYSIS_H_ */
