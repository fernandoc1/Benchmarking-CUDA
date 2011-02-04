#include <sys/time.h>
#include <cxxabi.h>
#include <hydrazine/implementation/ArgumentParser.h>
#include <ocelot/ir/interface/Module.h>
#include <ocelot/ir/interface/PTXKernel.h>
#include <ocelot/analysis/interface/DivergenceAnalysis.h>
#include <ocelot/graphs/interface/DivergenceDrawer.h>

/* This application generates control, data / divergence flow graphs on dot language for a input ptx file, and prints divergence analysis statistics based on profiling results */
int main ( int argc, char** argv )
{
	hydrazine::ArgumentParser parser(argc, argv);
	parser.description("Tests diverging branches analysis");

	std::string ptxFilename, currentPath;
	bool dirtGraph = false;
	bool varsGraph = false;
	bool dfgGraph = false;
	bool cfgGraph = false;
	bool allGraphs = false;
	bool compareProfiling = false;
	bool stats = false;
	bool cfgPropagation = true;

	parser.parse("-i", "--input", ptxFilename, "", "Input PTX file.");

	parser.parse("-p", "--prof", compareProfiling, false, "Compare analysis results with a branch profiling file for the same kernel. File must be named as \"prof.<kernel name>.txt\"");

	parser.parse("-s", "--statistics", stats, false, "Prints profiling x analysis statistics");

	parser.parse("-a", "--all", allGraphs, false, "Prints all graphs");

	parser.parse("-d", "--data", varsGraph, false, "Prints data propagation graph");
	parser.parse("-v", "--div", dirtGraph, false, "Prints divergent data propagation graph");
	parser.parse("-c", "--cfg", cfgGraph, false, "Prints control flow graph");
	parser.parse("-f", "--dfg", dfgGraph, false, "Prints complete data & control flow graph with divergences");
	parser.parse("", "--nodfg", cfgPropagation, true, "Don't make the control dependency analysis");

	parser.parse();

	if(ptxFilename == ""){
		cerr << "No input ptx file" << std::endl;
		parser.help();
		exit(EXIT_FAILURE);
	}

	ir::Module module(ptxFilename);
	ir::Module::KernelMap::const_iterator k_it = module.kernels().begin();

	for ( ; k_it != module.kernels().end(); ++k_it )
	{
		ir::PTXKernel* kernel = static_cast<ir::PTXKernel*> (k_it->second);

		if(kernel == NULL)
		{
			std::cerr << "Invalid kernel" << std::endl;
			exit(EXIT_FAILURE);
		}

		int status = -1;
		std::string kernelName = abi::__cxa_demangle(kernel->name.c_str(), 0, 0, &status);
		if ( status < 0 )
		{
			std::cerr << "Error: could not demangle kernel name " << kernel->name << std::endl;
			exit(EXIT_FAILURE);
		}

		kernelName = kernelName.substr(0, kernelName.find("("));

		/* If the kernel name is a container, remove everthing from begining to the :: and after '<' */
//		if(kernelName.find('<') != std::string::npos){
//			kernelName = kernelName.substr(0, kernelName.find('<'));
//		}
//		if(kernelName.find(':') != std::string::npos){
//			kernelName = kernelName.substr(kernelName.find_last_of(':') + 1, kernelName.size() - 1);
//		}
//		if(kernelName.find(' ') != std::string::npos){
//			kernelName = kernelName.substr(kernelName.find_last_of(' ') + 1, kernelName.size() - 1);
//		}

		analysis::DivergenceAnalysis divergenceAnalysis;

		if(!cfgPropagation){
			divergenceAnalysis.setControlFlowAnalysis(false);
		}
		divergenceAnalysis.runOnKernel(*k_it->second);

		if(ptxFilename.find('/') != std::string::npos){
			currentPath = ptxFilename.substr(0, 1 + ptxFilename.find_last_of('/'));
		} else {
			currentPath = "";
		}

		graph_utils::DivergenceDrawer drawer(kernelName, currentPath, &divergenceAnalysis, allGraphs, compareProfiling, dirtGraph, varsGraph, dfgGraph, cfgGraph);

		if(allGraphs || dirtGraph || varsGraph || dfgGraph || cfgGraph){
			drawer.draw();
		}

		if(stats){
			string stat = drawer.getAnalysisStatistics();
			if(stat != ""){
				cout << kernelName << ';' << divergenceAnalysis.getAnalysisTime().tv_sec * 1000000 + divergenceAnalysis.getAnalysisTime().tv_usec << stat << endl;
			}
		}
	}

	return EXIT_SUCCESS;
}
