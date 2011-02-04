/*
 * BlockUnificationPass.h
 *
 *  Created on: Aug 9, 2010
 *      Author: coutinho
 */

#ifndef SYNCELIMINATIONPASS_H_
#define SYNCELIMINATIONPASS_H_

#include <ocelot/analysis/interface/DivergenceAnalysis.h>
#include <ocelot/analysis/interface/Pass.h>
namespace analysis {
class SyncEliminationPass : public KernelPass
{
private:

public:
	SyncEliminationPass();
	~SyncEliminationPass();

	/*! \brief Initialize the pass using a specific module */
	virtual void initialize( const ir::Module& m ){};
	/*! \brief Run the pass on a specific kernel in the module */
	virtual void runOnKernel( ir::Kernel& k );
	/*! \brief Finalize the pass */
	virtual void finalize(){};
};

}

#endif /* BLOCKUNIFICATIONPASS_H_ */
