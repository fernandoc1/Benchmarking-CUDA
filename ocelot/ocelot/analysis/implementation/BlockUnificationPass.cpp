/*
 * BlockUnificationPass.cpp
 *
 *  Created on: Aug 9, 2010
 *      Author: coutinho
 */

#include <set>

#include <ocelot/analysis/interface/BlockUnificationPass.h>
#include <ocelot/ir/interface/PTXKernel.h>
#include <ocelot/ir/interface/ControlFlowGraph.h>
#include <ocelot/ir/interface/PostdominatorTree.h>
#include <ocelot/analysis/interface/BlockExtractor.h>
#include <ocelot/analysis/interface/DivergenceAnalysis.h>

using namespace analysis;

BlockUnificationPass::BlockUnificationPass()
	: KernelPass(StaticSingleAssignment, "BlockUnification")
{
}

BlockUnificationPass::~BlockUnificationPass()
{
}

void BlockUnificationPass::initialize( const ir::Module& m )
{

}

void BlockUnificationPass::runOnKernel( ir::Kernel& k )
{
	InstructionConverter instConv;
	ir::PTXInstruction::ComputeCapability deviceCapability = ir::PTXInstruction::Cap_2_0;
	DataflowGraph::iterator unificationBranch;
	DataflowGraph::iterator unificationTarget1;
	DataflowGraph::iterator unificationTarget2;
	BlockMatcher::MatrixPath bestPath;
	float largestGain = 0.0;

	// analyze kernel for divergence
	DivergenceAnalysis divAnalysis;
	divAnalysis.runOnKernel(k);

	do {
		largestGain = 0.0;

		DataflowGraph::iterator block = k.dfg()->begin();
		for (; block != k.dfg()->end(); ++block) {
			ir::ControlFlowGraph::const_iterator irBlock = block->block();
			DataflowGraph::const_iterator constBlock = block;

			if (irBlock->endsWithConditionalBranch() &&
				divAnalysis.isDivBlock(constBlock)
			) {
				// get the fallthrough block
				DataflowGraph::iterator fallthroughBlock = block->fallthrough();

				// get the branch block
				DataflowGraph::iterator branchBlock = fallthroughBlock;
				DataflowGraph::BlockPointerSet branchTargets = block->targets();
				DataflowGraph::BlockPointerSet::const_iterator it =
						branchTargets.begin();
				for (; it != branchTargets.end(); ++it) {
					if (*it != fallthroughBlock) {
						branchBlock = *it;
						break;
					}
				}
				assertM(branchBlock != fallthroughBlock,
						"Block unification pass error: could not find fallthrough");

				ir::PostdominatorTree* pdomTree = k.pdom_tree();
				ir::ControlFlowGraph::const_iterator postDomBlk =
						pdomTree->getPostDominator(block->block());
				bool haveBranch2FallthroughPath = thereIsPathFromB1toB2(
						branchBlock->block(), fallthroughBlock->block(),
						postDomBlk, new std::set<ir::ControlFlowGraph::BasicBlock*>);
				bool haveFallthrough2BranchPath = thereIsPathFromB1toB2(
						fallthroughBlock->block(), branchBlock->block(),
						postDomBlk, new std::set<ir::ControlFlowGraph::BasicBlock*>);
				if (!haveBranch2FallthroughPath && !haveFallthrough2BranchPath) {
					// Calculate branch targets' unification gain
					BlockMatcher::MatrixPath path;
					float gain = BlockMatcher::calculateUnificationGain(
							k.dfg(), *fallthroughBlock, *branchBlock, path,
							instConv, deviceCapability);

					if (gain > largestGain) {
						largestGain = gain;
						unificationBranch = block;
						unificationTarget1 = fallthroughBlock;
						unificationTarget2 = branchBlock;
						bestPath = path;
					}
				}
			}
		}

		if (largestGain > 10.0) {
			// Unify the basic block pair with biggest gain (if there's one)
			cout << ">>>>> unifying blocks: " << unificationTarget1->block()->label
								<< " and " << unificationTarget2->block()->label << std::endl;
			weaveBlocks(unificationBranch, unificationTarget1, unificationTarget2, bestPath, k.dfg());
		}

		// refresh divergence analysis data
		divAnalysis.run();
	} while (false); //(largestGain > 0.0);
}

bool BlockUnificationPass::thereIsPathFromB1toB2(ir::ControlFlowGraph::const_iterator t,
		ir::ControlFlowGraph::const_iterator e,
		ir::ControlFlowGraph::const_iterator c,
		std::set<ir::ControlFlowGraph::BasicBlock*>* visited) const {
	const ir::ControlFlowGraph::BlockPointerVector& sucessors = t->successors;
	ir::ControlFlowGraph::BlockPointerVector::const_iterator succ = sucessors.begin();
	for (; succ != sucessors.end(); succ++) {
		if (visited->find(&(**succ)) == visited->end()) {
			if (*succ == e) {
				return true;
			} else if (*succ != c) {
				visited->insert(&(**succ));
				if (thereIsPathFromB1toB2(*succ, e, c, visited)) {
					return true;
				}
			}
		}
	}
	return false;
}

void BlockUnificationPass::finalize()
{

}

void BlockUnificationPass::replaceRegisters(DataflowGraph* dfg,
		BlockExtractor& extractor) {
	const VariableAliases& varMap = extractor.getNewRegisterNames();
	for (analysis::DataflowGraph::iterator block = dfg->begin(); block
			!= dfg->end(); ++block) {
		// Go over the list of phi-functions, replacing the uses.
		// TODO: if the block contains only one predecessor, but its phi
		// functions have more than one parameter, then replace the phi
		// function by a single parameter phi.
		for (DataflowGraph::PhiInstructionVector::iterator phi =
				block->phis().begin(); phi != block->phis().end(); ++phi) {
			for (unsigned u = 0; u < phi->s.size(); ++u) {
				DataflowGraph::Register& phiRegId = phi->s[u];
				VariableAliases::const_iterator regAliasIt =
						varMap.find(phiRegId.id);
				if (regAliasIt != varMap.end()) {
					std::cout << "(PHI) Replacing " << phiRegId.id
							<< " by " << regAliasIt->second << std::endl;
					phiRegId.id = regAliasIt->second;
				}
			}
		}
		// Go over the ordinary instructions, performing the substitution:
		// TODO: check if this code is necessary:
		/*
		for (analysis::DataflowGraph::InstructionVector::const_iterator inst1 =
				block->instructions().begin();
				inst1 != block->instructions().end(); ++inst1) {
			ir::PTXInstruction* ptxInst1 =
					static_cast<ir::PTXInstruction*> (inst1->i);
			extractor.checkOperandReplacements(*ptxInst1);
		}
		*/
	}
}


void BlockUnificationPass::weaveBlocks(DataflowGraph::iterator branchBlock, DataflowGraph::iterator target1, DataflowGraph::iterator target2, BlockMatcher::MatrixPath& extractionPath, DataflowGraph* dfg)
{
	DataflowGraph::iterator oldFallthroughBlock = branchBlock;
	DataflowGraph::iterator oldBranchBlock = branchBlock;

	// get branch predicate
	ir::ControlFlowGraph::const_iterator irBlock = branchBlock->block();
	ir::Instruction* branchInst = irBlock->getTerminator();
	ir::PTXInstruction* branchInstPtx = static_cast<ir::PTXInstruction*>(branchInst);
	ir::PTXOperand* branchPredicate = &(branchInstPtx->pg);

	std::string labelPrefix = "$BBweave_" + target1->block()->label + "_" + target2->block()->label;
	int blockNum = 0;

	// while not consumed path, generate basic blocks
	BlockExtractor extractor(dfg, target1, target2, extractionPath, *branchPredicate);
	while (extractor.hasNext()) {
		if (extractor.nextStep() == BlockMatcher::Match ||
				extractor.nextStep() == BlockMatcher::Substitution) {
			// block label
			std::stringstream blockLabel;
			blockLabel << labelPrefix << "_uni_" << blockNum++;

			// create block
			DataflowGraph::iterator newUnifiedBlock = dfg->insert(oldFallthroughBlock, target1, blockLabel.str());
			extractor.extractUnifiedBlock(newUnifiedBlock);

			// link blocks
			dfg->addEdge(newUnifiedBlock, target2, ir::ControlFlowGraph::Edge::Branch);
			if (oldFallthroughBlock == oldBranchBlock) {
				// oldFallthroughBlock and oldBranchBlock all point to branchBlock.

				// remove oldBranchBlock -> target2 because there's
				// already a edge from branchBlock to newUnifiedBlock
				dfg->removeEdge(oldBranchBlock, target2);
			} else {
				dfg->removeEdge(oldFallthroughBlock, newUnifiedBlock);
				dfg->redirect(oldBranchBlock, target2, newUnifiedBlock);
				dfg->addEdge(oldFallthroughBlock, newUnifiedBlock, ir::ControlFlowGraph::Edge::Branch);

				// add goto in oldFallthroughBlock to newUnifiedBlock
				ir::PTXInstruction gotoPtx(ir::PTXInstruction::Bra);
				ir::PTXOperand gotoLabelOperand(blockLabel.str(), ir::PTXOperand::Label, ir::PTXOperand::s32);
				gotoPtx.setDestination(gotoLabelOperand);
				gotoPtx.uni = true;
				ir::Instruction& gotoInst = gotoPtx;
				dfg->insert(oldFallthroughBlock, gotoInst);
			}

			oldFallthroughBlock = newUnifiedBlock;
			oldBranchBlock = newUnifiedBlock;
		} else {

			// create fallthrough block
			std::stringstream fallthroughLabel;
			fallthroughLabel << labelPrefix << "_ft_" << blockNum++;
			DataflowGraph::iterator newFallthoughBlock = dfg->insert(oldFallthroughBlock, target1, fallthroughLabel.str());

			// create branch block
			std::stringstream branchLabel;
			branchLabel << labelPrefix << "_bra_" << blockNum++;
			DataflowGraph::iterator newBranchBlock = dfg->insert(oldBranchBlock, target2, branchLabel.str());

			// fill blocks
			extractor.extractDivergentBlocks(newFallthoughBlock, newBranchBlock);

			if (oldFallthroughBlock == branchBlock) {
				const ir::PTXOperand braLabelOperand(branchLabel.str(), ir::PTXOperand::Label, ir::PTXOperand::s32);
				branchInstPtx->setDestination(braLabelOperand);
			} else {
				ir::PTXInstruction braPtx(ir::PTXInstruction::Bra);
				ir::PTXOperand braLabelOperand(branchLabel.str(), ir::PTXOperand::Label, ir::PTXOperand::s32);
				braPtx.setDestination(braLabelOperand);
				braPtx.setPredicate(*branchPredicate);
				ir::Instruction& bra = braPtx;
				dfg->insert(oldFallthroughBlock, bra);
			}

			// all needed edges were already created in block creation,
			// no more edge manipulation needed
			oldFallthroughBlock = newFallthoughBlock;
			oldBranchBlock = newBranchBlock;
		}
	}

	// remove branch instruction on BranchBlock if needed
	if (branchBlock->targets().size() == 0) {
		// branchBlock does not have branch at end, remove that instruction
		unsigned int branchInstPos = branchBlock->instructions().size() - 1;
		dfg->erase(branchBlock, branchInstPos);
	}

	if (oldFallthroughBlock == oldBranchBlock) {
		// If target1 has no fall-through, then
		// switch branch and fall-through edges.

		if ( !(target1->block()->has_fallthrough_edge()) ) {
			dfg->setEdgeType(oldFallthroughBlock, target1, ir::ControlFlowGraph::BasicBlock::Edge::Branch);
			dfg->setEdgeType(oldFallthroughBlock, target2, ir::ControlFlowGraph::BasicBlock::Edge::FallThrough);
		}
	}
	// weaved block finishes with a divergent section

	// copy target2 targets to a ending divergent fallthrough block
	dfg->copyOutgoingBranchEdges(target1, oldFallthroughBlock);
	// remove target1
	dfg->erase(target1);

	// copy target2 targets to a ending unified block or divergent branch block
	dfg->copyOutgoingBranchEdges(target2, oldBranchBlock);
	// remove target2
	dfg->erase(target2);

	// remove empty blocks or blocks with just a goto
	// TODO: removing this blocks is not essential for correctness, yet it
	// might increase performance.

	// replaces all uses of old registers to use new ones
	// in code after unified basic blocks
	std::cerr << "\n\n\nCalling register replacement\n\n\n";
	replaceRegisters(dfg, extractor);

	// recalculate live in and live out
	//dfg->compute();
	//dfg->toSsa();
}
