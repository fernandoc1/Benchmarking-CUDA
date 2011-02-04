#include <cstdlib>
#include <ocelot/ir/interface/ControlFlowGraph.h>
#include "ocelot/ir/interface/PTXInstrCreator.h"
#include "ocelot/ir/interface/PrecBranchProfiler.h"


void ir::PrecBranchProfiler::process(ir::Module* m) {
	
	/*// Create the PTXstatement with prof_data_pt declaration.
	ir::PTXStatement profDataPtDeclaration(ir::PTXStatement::Global);
	profDataPtDeclaration.type = PTXOperand::u64;
	profDataPtDeclaration.name = "prof_data_pt";
	
	// Create the prof_data_pt global variable.
	ir::Global profDataPtGlobal(profDataPtDeclaration);
	profDataPtGlobal.local = false;
	profDataPtGlobal.pointer = NULL;
	
	// Add _prof_data_pt to global variables.
	m->globals["prof_data_pt"] = profDataPtGlobal;*/
}

void ir::PrecBranchProfiler::postProcess(ir::Module* m) {
	// get prof_data_sz declaration
	ir::Module::GlobalMap::iterator g_it = m->globals().find("prof_data_sz");
	if (g_it == m->globals().end()) {
		std::cerr << "prof_data_sz not found in module file" << std::endl;
		exit(EXIT_FAILURE);
	}

	// modify initial value of prof_data_sz to the number of instrumented basic blocks
	ir::PTXStatement& profDataSzDeclaration = g_it->second.statement;
	profDataSzDeclaration.array.values[0].u32 = profDataSize;
}

void ir::PrecBranchProfiler::process(ir::Kernel* k) {
	kernelName = k->name;
	instrumentedBlocksInfo.clear();
}

void ir::PrecBranchProfiler::process(ir::ControlFlowGraph* cfg) {
	// We need a reference to cfg to inset new basic blocks.
	kernelCFG = cfg;
	
	// We save these references because they will help us to find out the
	// actuall blocks that must be instrumented.
	//
	// First, we must instrument the second block after the entry block. This
	// seems to be a convention used in the Ocelot compiler. This block will
	// receive the code that will initialize the counters.
	entryBlock = cfg->get_entry_block();
}

void ir::PrecBranchProfiler::postProcess(ir::ControlFlowGraph* cfg) {
	instrumentBasicBlocks();
	generateInitialization(initializationBlock);

	if (instrumentedBlocksInfo.size() > profDataSize) {
		profDataSize = instrumentedBlocksInfo.size();
	}
}

void ir::PrecBranchProfiler::generateBasicBlockIdWrites(ir::ControlFlowGraph::iterator bb) {
	
	for (unsigned int i=0; i < instrumentedBlocksInfo.size(); ++i) {
		ir::PTXOperand storeAddr = resultsArray;
		storeAddr.addressMode    = ir::PTXOperand::Indirect;
		storeAddr.offset         = 8*(3*i);
		
		ir::PTXOperand basicBlockIdConstant(ir::PTXOperand::u64, (unsigned long long)instrumentedBlocksInfo[i].id);
		
		// mov.u64 %dummy basicBlockId[i]
		PTXInstrCreator::createTwoOperandInst(bb, ir::PTXInstruction::Mov, dummy, basicBlockIdConstant);
		
		// st.global.u64 [%resultsArray+sizeof(long long)*3*i], %dummy;
		ir::PTXInstruction* newInst = new PTXInstruction(ir::PTXInstruction::St);
		newInst->setAddressSpace(ir::PTXInstruction::Global);
		newInst->setDestination(storeAddr);
		newInst->setOperandA(dummy);
		PTXInstrCreator::verifyInstruction(newInst);
		bb->addInstructionEnd(newInst);
	}
}

void ir::PrecBranchProfiler::generateInitialization(ir::ControlFlowGraph::iterator bb) {
	ir::PTXOperand zero16(ir::PTXOperand::u16, (long long int)0);
	ir::PTXOperand tidX(PTXOperand::tidX);
	ir::PTXOperand tidY(PTXOperand::tidY);
	ir::PTXOperand tidZ(PTXOperand::tidZ);
	ir::PTXOperand ctaidX(PTXOperand::ctaIdX);
	ir::PTXOperand ctaidY(PTXOperand::ctaIdY);
	ir::PTXOperand ctaidZ(PTXOperand::ctaIdZ);
	ir::PTXOperand threadId("%threadId", PTXOperand::Register, PTXOperand::u16);
	ir::PTXOperand firstThread("%firstThread", PTXOperand::Register, PTXOperand::pred);
	
	// insert code to load   	
	// ld.global.u64 %resultsArray, [prof_data_pt];
	ir::PTXInstruction* newInst = new PTXInstruction(ir::PTXInstruction::Ld);
	newInst->setAddressSpace(ir::PTXInstruction::Global);
	newInst->setDestination(resultsArray);
	newInst->setOperandA(profArrayPt);
	PTXInstrCreator::verifyInstruction(newInst);
	bb->addInstructionBegin(newInst);
	
	// add the folowing code at the first block:
	// 		if (tid.x == 0) {
	//  		resultsArray[3*sizeof(long long)*0] = FIRST_BLOCK_ID
	//			resultsArray[3*sizeof(long long)*1] = SECOND_BLOCK_ID 
	//			...
	//  	}
	
	// split the first block 
	kernelCFG->split_block(bb, 1, ControlFlowGraph::BasicBlock::Edge::FallThrough);
	
	// mov.u16 %threadId, %tid.x;
	PTXInstrCreator::createTwoOperandInst(bb, ir::PTXInstruction::Mov, threadId, tidX);
	
	// setp.eq.u16 %firstThread, %threadId, 0;
	newInst = PTXInstrCreator::createThreeOperandInst(bb, ir::PTXInstruction::SetP, firstThread, threadId, zero16);
	newInst->setComparisonOperator(ir::PTXInstruction::Eq);
	
	// mov.u16 %threadId, %tid.y;
	PTXInstrCreator::createTwoOperandInst(bb, ir::PTXInstruction::Mov, threadId, tidY);
	
	// setp.eq.and.u16 %firstThread, %threadId, 0, %firstThread;
	newInst = PTXInstrCreator::createFourOperandInst(bb, ir::PTXInstruction::SetP, firstThread, threadId, zero16, firstThread);
	newInst->setComparisonOperator(ir::PTXInstruction::Eq);
	newInst->setBooleanOperator(ir::PTXInstruction::BoolAnd);
	
	// mov.u16 %threadId, %tid.y;
	PTXInstrCreator::createTwoOperandInst(bb, ir::PTXInstruction::Mov, threadId, tidZ);
	
	// setp.eq.and.u16 %firstThread, %threadId, 0, %firstThread;
	newInst = PTXInstrCreator::createFourOperandInst(bb, ir::PTXInstruction::SetP, firstThread, threadId, zero16, firstThread);
	newInst->setComparisonOperator(ir::PTXInstruction::Eq);
	newInst->setBooleanOperator(ir::PTXInstruction::BoolAnd);
	
	// mov.u16 %threadId, %tid.y;
	PTXInstrCreator::createTwoOperandInst(bb, ir::PTXInstruction::Mov, threadId, ctaidX);
	
	// setp.eq.and.u16 %firstThread, ctaid.x, 0, %firstThread;
	newInst = PTXInstrCreator::createFourOperandInst(bb, ir::PTXInstruction::SetP, firstThread, threadId, zero16, firstThread);
	newInst->setComparisonOperator(ir::PTXInstruction::Eq);
	newInst->setBooleanOperator(ir::PTXInstruction::BoolAnd);
	
	// mov.u16 %threadId, %tid.y;
	PTXInstrCreator::createTwoOperandInst(bb, ir::PTXInstruction::Mov, threadId, ctaidY);
	
	// setp.eq.and.u16 %firstThread, ctaid.y, 0, %firstThread;
	newInst = PTXInstrCreator::createFourOperandInst(bb, ir::PTXInstruction::SetP, firstThread, threadId, zero16, firstThread);
	newInst->setComparisonOperator(ir::PTXInstruction::Eq);
	newInst->setBooleanOperator(ir::PTXInstruction::BoolAnd);
	
	// mov.u16 %threadId, %tid.y;
	PTXInstrCreator::createTwoOperandInst(bb, ir::PTXInstruction::Mov, threadId, ctaidZ);
	
	// setp.eq.and.u16 %firstThread, ctaid.z, 0, %firstThread;
	newInst = PTXInstrCreator::createFourOperandInst(bb, ir::PTXInstruction::SetP, firstThread, threadId, zero16, firstThread);
	newInst->setComparisonOperator(ir::PTXInstruction::Eq);
	newInst->setBooleanOperator(ir::PTXInstruction::BoolAnd);
	
	std::string kernelStartLabel = kernelName + "_START";
	ir::PTXOperand kernelStartLabelOperand(kernelStartLabel, PTXOperand::Label, PTXOperand::u64);
	
	// @!%firstThread bra $<kernel name>_START;
	ir::PTXOperand notFirstThread = firstThread;
	notFirstThread.condition = ir::PTXOperand::InvPred;
	
	newInst = new PTXInstruction(ir::PTXInstruction::Bra);
	newInst->setPredicate(notFirstThread);
	newInst->setBranchUni(false);
	newInst->setDestination(kernelStartLabelOperand);
	bb->addInstructionEnd(newInst);
	
	// go to next block in executable sequence
	ir::ControlFlowGraph::BasicBlock::EdgeList::iterator fallthroughEdge = bb->get_fallthrough_edge();
	ir::ControlFlowGraph::iterator writeBlock = fallthroughEdge->tail;
	
	// write block will contain instructions to write the block ids in 
	// resultsArray (that will be executed by only by the first thread in the
	// entire grid) and the next block will contain the beggining of the 
	// original kernel code  
	kernelCFG->split_block(writeBlock, 0, ControlFlowGraph::BasicBlock::Edge::FallThrough);
	writeBlock->setLabel(kernelName + "_inst_init");
	generateBasicBlockIdWrites(writeBlock);
	
	// the next block is the original first block
	fallthroughEdge = writeBlock->get_fallthrough_edge();
	ir::ControlFlowGraph::iterator kernelStartBlock = fallthroughEdge->tail;
	kernelStartBlock->setLabel(kernelStartLabel);
}

void ir::PrecBranchProfiler::instrumentBasicBlocks() {
	ir::PTXOperand zero32(ir::PTXOperand::u32, (long long unsigned int)0);
	ir::PTXOperand one32(ir::PTXOperand::u32, (long long unsigned int)1);
	ir::PTXOperand one64(ir::PTXOperand::u64, (long long unsigned int)1);
	ir::PTXOperand laneId(PTXOperand::laneId);
	ir::PTXOperand iWasSelected("%iWasSelected", PTXOperand::Register, PTXOperand::pred);
	ir::PTXOperand threadInSet("%threadInSet", PTXOperand::Register, PTXOperand::pred);
	ir::PTXOperand uni("%uni", PTXOperand::Register, PTXOperand::pred);
	ir::PTXOperand div("%div", PTXOperand::Register, PTXOperand::pred);
	ir::PTXOperand selectedThreadId("%selectedThreadId", PTXOperand::Register, PTXOperand::u32);
	ir::PTXOperand laneIdCopy("%laneIdCopy", PTXOperand::Register, PTXOperand::u32);
	
	ir::PTXOperand threadNotInSet = threadInSet;
	threadNotInSet.condition      = ir::PTXOperand::InvPred;
	
	ir::PTXOperand storeAddr = resultsArray;
	storeAddr.addressMode    = ir::PTXOperand::Indirect;
	
	for (unsigned int bbIdx=0; bbIdx<instrumentedBlocksInfo.size(); bbIdx++) {
		ir::ControlFlowGraph::iterator bb = instrumentedBlocksInfo[bbIdx].bb;
		const ir::PTXInstruction* terminator = (PTXInstruction*) bb->getTerminator();
		
		if (terminator && terminator->isPredicated()) {
			// use createInstructionEnd() generate code to do the following:
			// 
			// selectedThreadId = 0;
			// do {
			//     iWasSelected   = (selectedThreadId == laneid);
			//     threadInSet    = vote.any.pred(iWasSelected);
			//     selectedThreadId++;
			// } while (!threadInSet);
			// @iWasSelected atomicAdd(timesReached, 1);
			// uni           = vote.uni.pred(branchCondition);
			// divergent     = !uni;
			// divergent     = divergent && iWasSelected;
			// @divergent    atomicAdd(divergencies, 1);
			
			// mov.u32 %selectedThreadId 0;
			PTXInstrCreator::createTwoOperandInst(bb, ir::PTXInstruction::Mov, selectedThreadId, zero32);
	
			// mov.u32 %laneIdCopy %laneid
			PTXInstrCreator::createTwoOperandInst(bb, ir::PTXInstruction::Mov, laneIdCopy, laneId);

			// split the block to create thread selection loop
			kernelCFG->split_block(bb, bb->instructions.size() - 1, ControlFlowGraph::BasicBlock::Edge::FallThrough);
			ir::ControlFlowGraph::BasicBlock::EdgeList::iterator fallthroughEdge = bb->get_fallthrough_edge();
			bb = fallthroughEdge->tail;
			
			// setp.eq.u32 %iWasSelected %laneIdCopy %selectedThreadId;
			ir::PTXInstruction* newInst = PTXInstrCreator::createThreeOperandInst(bb, ir::PTXInstruction::SetP, iWasSelected, laneIdCopy, selectedThreadId);
			newInst->setComparisonOperator(ir::PTXInstruction::Eq);
	
			// vote.uni.pred %threadInSet %iWasSelected
			newInst = new PTXInstruction(PTXInstruction::Vote);
			newInst->setVoteMode(PTXInstruction::Any);
			newInst->setDestination(threadInSet);
			newInst->setOperandA(iWasSelected);
			bb->addInstructionEnd(newInst);
			
			// add.u32 %selectedThreadId, %selectedThreadId, 1;
			PTXInstrCreator::createThreeOperandInst(bb, ir::PTXInstruction::Add, selectedThreadId, selectedThreadId, one32);
				
			// @!%threadInSet bra $<selectWirterThread>;
			newInst = new PTXInstruction(ir::PTXInstruction::Bra);
			newInst->setPredicate(threadNotInSet);
			newInst->setBranchUni(false);
			ir::PTXOperand threadSelectLoopLabelOpr(bb->label, PTXOperand::Label, PTXOperand::u64);
			newInst->setDestination(threadSelectLoopLabelOpr);
			bb->addInstructionEnd(newInst);
	
			// split the block again to add couter increment code
			kernelCFG->split_block(bb, bb->instructions.size() - 1, ControlFlowGraph::BasicBlock::Edge::FallThrough);
			fallthroughEdge = bb->get_fallthrough_edge();
			bb              = fallthroughEdge->tail;
	
			// @iWasSelected atom.global.add.u64 %lixo, [%resultsArray+8*(3*basicBlockId+1)], 1;
			newInst = new PTXInstruction(PTXInstruction::Atom);
			newInst->setPredicate(iWasSelected);
			newInst->setAddressSpace(ir::PTXInstruction::Global);
			newInst->setAtomicOperation(PTXInstruction::AtomicAdd);
			newInst->setDestination(dummy);
			storeAddr.offset = 8*(3*bbIdx + 1);
			newInst->setOperandA(storeAddr);
			newInst->setOperandB(one64);
			bb->addInstructionEnd(newInst);
			
			terminator = (PTXInstruction*) bb->getTerminator();
			const ir::PTXOperand* branchPred = terminator->getPredicate();
			
			// vote.uni.pred %uni %condition
			newInst = new PTXInstruction(PTXInstruction::Vote);
			newInst->setVoteMode(PTXInstruction::Uni);
			newInst->setDestination(uni);
			newInst->setOperandA(*branchPred);
			bb->addInstructionEnd(newInst);
			
			// not.pred %div %uni
			PTXInstrCreator::createTwoOperandInst(bb, ir::PTXInstruction::Not, div, uni);
			
			// and.pred %div %iWasSelected %div
			PTXInstrCreator::createThreeOperandInst(bb, ir::PTXInstruction::And, div, iWasSelected, div);
			
			// @%div atom.global.add.u64 %lixo, [8*(3*basicBlockId+2)], 1;
			newInst = new PTXInstruction(PTXInstruction::Atom);
			newInst->setPredicate(div);
			newInst->setAddressSpace(ir::PTXInstruction::Global);
			newInst->setAtomicOperation(PTXInstruction::AtomicAdd);
			newInst->setDestination(dummy);
			storeAddr.offset = 8*(3*bbIdx + 2);
			newInst->setOperandA(storeAddr);
			newInst->setOperandB(one64);
			bb->addInstructionEnd(newInst);
		} 
	}
}

/*
 * We must add instructions, at the end of the source program, to transfer to
 * the CPU memory the results of the instrumentation. In this particular case,
 * where we are only interested in measuring the total number of execution
 * cycles taken by the PTX program, the 'process' method will visit every basic
 * block in the target program, but will modify only two kinds of blocks:
 * 1) It will add code at the begining of the first basic block after the
 * program entry point. 
 * 2) It will add code at the last basic blocks before the program exit block.
 * This code will transfer to the CPU the results of the instrumentation.
 * TODO: test the instrumentation in programs with multiple terminators.
 */
void ir::PrecBranchProfiler::process(ir::ControlFlowGraph::iterator bb) {
	
	if (bb == entryBlock) {
		ir::ControlFlowGraph::edge_iterator fallthroughEdge = bb->get_fallthrough_edge();
		initializationBlock = fallthroughEdge->tail;
	}
	
	// if block has a conditional branch at its end
	if (bb->endsWithConditionalBranch()) {
		BasicBlockInfo bbInfo;
		bbInfo.id = bb->id;
		bbInfo.bb = bb;
		instrumentedBlocksInfo.push_back(bbInfo);
	}
}

