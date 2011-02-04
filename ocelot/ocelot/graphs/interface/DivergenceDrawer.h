#ifndef DIVERGENCEDRAWER_H_
#define DIVERGENCEDRAWER_H_

#include <ocelot/analysis/interface/DivergenceAnalysis.h>
#include <ocelot/graphs/interface/BlockDivergenceInfo.h>

namespace graph_utils{
class DivergenceDrawer
{
	private:
		/*!\brief Divergence profiled ptx execution data */
		BlocksDivergenceInfo _divInfo;
		/*!\brief kernel name analyzed */
		string _kernelName;
		/*!\brief ptx path, for writing dot graph */
		string _path;
		/*!\brief The static divergence analysis of the kernel */
		analysis::DivergenceAnalysis *divAnalysis;
		/*!\brief Prints all graphs */
		bool _all;
		/*!\brief Make divergence static analysis comparison to profiling results */
		bool _profiling;
		/*!\brief Draw divergence variables graph */
		bool _divergence;
		/*!\brief Draw variables graph */
		bool _vars;
		/*!\brief Draw controlflow graph */
		bool _cfg;
		/*!\brief Draw (data+control)flow graph  - (D+C)FG */
		bool _dfg;

		inline void _sanityTest() const;
		inline string _edges ( const analysis::DataflowGraph::Block &block, const bool isFullGraph = false ) const;

	public:
		void drawVariablesGraph() const;
		void drawDivergenceGraph() const;
		void drawControlFlowGraph() const;
		void drawFullGraph() const;
		void draw() const;
		DivergenceDrawer( const string &kernelName, const string &path, analysis::DivergenceAnalysis *divergenceAnalysis,
				const bool &allGraphs, const bool &compareProfiling = false, const bool &dirtGraph = false,
				const bool &varsGraph = false, const bool &dfgGraph = false, const bool &cfgGraph = false);

		DivergenceDrawer( const string &kernelName, const string &path, analysis::DivergenceAnalysis *divergenceAnalysis,
						const bool &allGraphs = true);
		string getAnalysisStatistics() const;
};

std::ostream& operator<<(std::ostream& out, const DivergenceDrawer& dd);

}
#endif /* DIVERGENCEDRAWER_H_ */
