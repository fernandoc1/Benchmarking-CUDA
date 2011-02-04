#include <set>
#include <iostream>
#include <hydrazine/interface/Version.h>
#include <ocelot/ir/interface/Parameter.h>
#include <ocelot/ir/interface/PTXKernel.h>
#include <ocelot/ir/interface/ControlFlowGraph.h>
#include "ocelot/ir/interface/PrinterWorker.h"


/// Modeled after Module.writeIR().
void ir::PrinterWorker::process(ir::Module* m)
{
	output << "/*\n* AUTO GENERATED OCELOT PTX FILE\n";
	output << "* Ocelot Version : " << hydrazine::Version().toString() << std::endl;
	output << "* From file : " << m->path() << std::endl;
	output << "*/\n\n";
		
	// print module version
	output << ".version 1.4" << std::endl;
	
	// Print targets. They are stored in the "targets" vector in each 
	// statement. ATTENTION: the vector is empty in statements before the
	// .target directive. So we pick the last statement of the basic block.
	const ir::PTXStatement& stmt = m->statements().back();
	std::vector<std::string>::const_iterator target = stmt.targets.begin();
	if (target != stmt.targets.end()) {
		output << ".target " << *target;
		++target;
	}
	for (; target != stmt.targets.end(); ++target) {
		output << ", " << *target;
	}
	output << "\n\n";
	
	output << "/* Globals */\n";
	ir::Module::GlobalMap::iterator global = m->globals().begin();
	for ( ; global != m->globals().end(); ++global) {
		output << global->second.statement.toString() << "\n";
	}
	output << "\n";

	output << "/* Textures */\n";
	ir::Module::TextureMap::const_iterator texture = m->textures().begin();
	for ( ; texture != m->textures().end(); ++texture) {
		output << texture->second.toString() << "\n";
	}
	output << "\n";
	
	// get global variables to not reprint them as labels
	globalVariables = &(m->globals());
}
