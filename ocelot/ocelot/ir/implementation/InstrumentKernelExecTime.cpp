#include <ocelot/ir/interface/ControlFlowGraph.h>
#include "ocelot/ir/interface/PTXInstrCreator.h"
#include "ocelot/ir/interface/InstrumentKernelExecTime.h"

using namespace std;

void ir::InstrumentKernelExecTime::process(ir::Module* m) {
	
	// Create the PTXstatement with _prof_data_pt declaration.
	ir::PTXStatement profDataPtDeclaration(ir::PTXStatement::Global);
	profDataPtDeclaration.type = PTXOperand::u64;
	profDataPtDeclaration.name = "prof_data_pt";
	// initialize it???
	
	// Create the _prof_data_pt global variable.
	Global profDataPtGlobal(profDataPtDeclaration);
	profDataPtGlobal.local = false;
	profDataPtGlobal.pointer = NULL;
	
	// Add _prof_data_pt to global variables.
	m->globals()["prof_data_pt"] = profDataPtGlobal;
}

void ir::InstrumentKernelExecTime::process(ir::ControlFlowGraph* cfg) {
	// We save these references because they will help us to find out the
	// actuall blocks that must be instrumented.
	//
	// First, we must instrument the second block after the entry block. This
	// seems to be a convention used in the Ocelot compiler. This block will
	// receive the code that will initialize the counters.
	entryBlock = cfg->get_entry_block();

	// Second, we must instrument the last block before the exit block. This
	// block will receive the code that will transfer from the GPU memory to the
	// CPU memory the results of the instrumentation.
	exitBlock  = cfg->get_exit_block();
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
void ir::InstrumentKernelExecTime::process(ir::ControlFlowGraph::iterator bb) {
	
	if (previousBlock == entryBlock) {
		ir::PTXOperand clock(PTXOperand::clock);
		ir::PTXOperand start("%start", PTXOperand::Register, PTXOperand::u32);
		
		// add the folowing instructions at the *beginning* of the first block:
		// mov.u32 %start, %clock;	
		ir::PTXInstruction* newInst = new ir::PTXInstruction(ir::PTXInstruction::Mov);
		newInst->setDestination(start);
		newInst->setOperandA(clock);
		PTXInstrCreator::verifyInstruction(newInst);
		bb->addInstructionBegin(newInst);
	}

	if (bb == exitBlock) {
		// add several instructions at the end of the last block before the
		// program exit.
		ir::PTXOperand zero(ir::PTXOperand::s64, (long long int)0);
		ir::PTXOperand constEight(ir::PTXOperand::u32, (unsigned long long int)8);
		ir::PTXOperand u32Max(ir::PTXOperand::s64, (long long int)0x100000000);
		ir::PTXOperand clock(PTXOperand::clock);
		ir::PTXOperand ctaIdX(PTXOperand::ctaIdX);
		ir::PTXOperand ctaIdY(PTXOperand::ctaIdY);
		ir::PTXOperand ctaIdZ(PTXOperand::ctaIdZ);
		ir::PTXOperand nctaIdX(PTXOperand::nctaIdX);
		ir::PTXOperand nctaIdY(PTXOperand::nctaIdY);
		ir::PTXOperand ntidX(PTXOperand::ntidX);
		ir::PTXOperand ntidY(PTXOperand::ntidY);
		ir::PTXOperand ntidZ(PTXOperand::ntidZ);
		ir::PTXOperand tidX(PTXOperand::tidX);
		ir::PTXOperand tidY(PTXOperand::tidY);
		ir::PTXOperand tidZ(PTXOperand::tidZ);
		ir::PTXOperand start("%start", PTXOperand::Register, PTXOperand::u32);
		ir::PTXOperand end("%end", PTXOperand::Register, PTXOperand::u32);
		ir::PTXOperand prd0("%prd0", PTXOperand::Register, PTXOperand::s64);
		ir::PTXOperand prd1("%prd1", PTXOperand::Register, PTXOperand::s64);
		ir::PTXOperand prh0("%prh0", PTXOperand::Register, PTXOperand::u16);
		ir::PTXOperand prh1("%prh1", PTXOperand::Register, PTXOperand::u16);
		ir::PTXOperand overflow("%overflow", PTXOperand::Register, PTXOperand::pred);
		ir::PTXOperand elapsedTime("%elapsedTime", PTXOperand::Register, PTXOperand::u64);
		ir::PTXOperand blockId("%blockId", PTXOperand::Register, PTXOperand::u32);
		ir::PTXOperand blockIdX("%bx", PTXOperand::Register, PTXOperand::u32);
		ir::PTXOperand blockIdXY("%bxy", PTXOperand::Register, PTXOperand::u32);
		ir::PTXOperand blockIdZ("%bz", PTXOperand::Register, PTXOperand::u32);
		ir::PTXOperand blockSz("%blockSz", PTXOperand::Register, PTXOperand::u32);
		ir::PTXOperand nctaXY("%nctaxy", PTXOperand::Register, PTXOperand::u32);
		ir::PTXOperand ntidXY("%ntidxy", PTXOperand::Register, PTXOperand::u32);
		ir::PTXOperand ntidZu32("%ntidz", PTXOperand::Register, PTXOperand::u32);
		ir::PTXOperand threadId("%threadId", PTXOperand::Register, PTXOperand::u32);
		ir::PTXOperand threadIdX("%tx", PTXOperand::Register, PTXOperand::u32);
		ir::PTXOperand threadIdXY("%txy", PTXOperand::Register, PTXOperand::u32);
		ir::PTXOperand threadIdZ("%tz", PTXOperand::Register, PTXOperand::u32);
		ir::PTXOperand profArrayPt("prof_data_pt", PTXOperand::Address, PTXOperand::u64);
		ir::PTXOperand storeAddr("%storeAddr", PTXOperand::Register, PTXOperand::u64);

		// mov.u32 	%end, %clock;
		PTXInstrCreator::createTwoOperandInst(previousBlock, ir::PTXInstruction::Mov, end, clock);
		
		// cvt.s64.u32	%prd0, %start;
		PTXInstrCreator::createTwoOperandInst(previousBlock, ir::PTXInstruction::Cvt, prd0, start);
		
		// cvt.s64.u32	%prd1, %end;
		PTXInstrCreator::createTwoOperandInst(previousBlock, ir::PTXInstruction::Cvt, prd1, end);
		
		// sub.s64 %prd0, %prd1, %prd0;
		PTXInstrCreator::createThreeOperandInst(previousBlock, ir::PTXInstruction::Sub, prd0, prd1, prd0);
		
		// setp.le.s64 %overflow, %prd0, 0;
		ir::PTXInstruction* newInst = new PTXInstruction(ir::PTXInstruction::SetP);
		newInst->setComparisonOperator(ir::PTXInstruction::Le);
		newInst->setDestination(overflow);
		newInst->setOperandA(prd0);
		newInst->setOperandB(zero);
		PTXInstrCreator::verifyInstruction(newInst);
		previousBlock->addInstructionEnd(newInst);
		
		// @%overflow add.s64 %prd0, %prd0, 0x100000000;
		newInst = PTXInstrCreator::createThreeOperandInst(previousBlock, ir::PTXInstruction::Add, prd0, prd0, u32Max);
		newInst->setPredicate(overflow);
		
		// cvt.u64.s64 %elapsedTime, %prd0;
		PTXInstrCreator::createTwoOperandInst(previousBlock, ir::PTXInstruction::Cvt, elapsedTime, prd0);
		
		// cvt.u32.u16 %bx, %ctaid.x;
		PTXInstrCreator::createTwoOperandInst(previousBlock, ir::PTXInstruction::Cvt, blockIdX, ctaIdX);
		
		//  cvt.u32.u16	%bz, %ctaid.z;
		PTXInstrCreator::createTwoOperandInst(previousBlock, ir::PTXInstruction::Cvt, blockIdZ, ctaIdZ);
		
		// mov.u16 %prh0, %nctaid.x;
		PTXInstrCreator::createTwoOperandInst(previousBlock, ir::PTXInstruction::Mov, prh0, nctaIdX);
		
		// mov.u16 %prh1, %nctaid.y;
		PTXInstrCreator::createTwoOperandInst(previousBlock, ir::PTXInstruction::Mov, prh1, nctaIdY);
		
		// mul.wide.u16	%nctaxy, %prh1, %prh0;
		PTXInstrCreator::createThreeOperandModInst(previousBlock, ir::PTXInstruction::Mul, ir::PTXInstruction::wide, nctaXY, prh1, prh0);
		
		// mov.u16 %prh1, %ctaid.y;
		PTXInstrCreator::createTwoOperandInst(previousBlock, ir::PTXInstruction::Mov, prh1, ctaIdY);
		
		// mad.wide.u16	%bxy, %prh1, %prh0, %bx;
		PTXInstrCreator::createFourOperandModInst(previousBlock, ir::PTXInstruction::Mad, ir::PTXInstruction::wide, blockIdXY, prh1, prh0, blockIdX);

		// mad.lo.u32 %blockId, %bz, %nctaxy, %bxy;
		PTXInstrCreator::createFourOperandModInst(previousBlock, ir::PTXInstruction::Mad, ir::PTXInstruction::lo, blockId, blockIdZ, nctaXY, blockIdXY);
		
		// cvt.u32.u16 %tx, %tid.x;
		PTXInstrCreator::createTwoOperandInst(previousBlock, ir::PTXInstruction::Cvt, threadIdX, tidX);
		
		// cvt.u32.u16 %tz, %tid.z;
		PTXInstrCreator::createTwoOperandInst(previousBlock, ir::PTXInstruction::Cvt, threadIdZ, tidZ);
		
		// mov.u16 %prh0, %ntid.x;
		PTXInstrCreator::createTwoOperandInst(previousBlock, ir::PTXInstruction::Mov, prh0, ntidX);
		
		// mov.u16 %prh1, %ntid.y;
		PTXInstrCreator::createTwoOperandInst(previousBlock, ir::PTXInstruction::Mov, prh1, ntidY);
		
		// mul.wide.u16	%ntidxy, %prh1, %prh0;
		PTXInstrCreator::createThreeOperandModInst(previousBlock, ir::PTXInstruction::Mul, ir::PTXInstruction::wide, ntidXY, prh1, prh0);
		
		// mov.u16 %prh1, %tid.y;
		PTXInstrCreator::createTwoOperandInst(previousBlock, ir::PTXInstruction::Mov, prh1, tidY);
		
		// mad.wide.u16	%txy, %prh1, %prh0, %tx;
		PTXInstrCreator::createFourOperandModInst(previousBlock, ir::PTXInstruction::Mad, ir::PTXInstruction::wide, threadIdXY, prh1, prh0, threadIdX);
		
		// mad.lo.u32 %threadId, %tz, %ntidxy, %txy;
		PTXInstrCreator::createFourOperandModInst(previousBlock, ir::PTXInstruction::Mad, ir::PTXInstruction::lo, threadId, threadIdZ, ntidXY, threadIdXY);
		
		// cvt.u32.u16	%ntidz, %ntid.z;
		PTXInstrCreator::createTwoOperandInst(previousBlock, ir::PTXInstruction::Cvt, ntidZu32, ntidZ);
		
		// mul.lo.u32 %blockSz, %ntidz, %ntidxy;
		PTXInstrCreator::createThreeOperandModInst(previousBlock, ir::PTXInstruction::Mul, ir::PTXInstruction::lo, blockSz, ntidZu32, ntidXY);
		
		// mad.lo.u32 %threadId, %blockId, %blockSz, %threadId;
		PTXInstrCreator::createFourOperandModInst(previousBlock, ir::PTXInstruction::Mad, ir::PTXInstruction::lo, threadId, blockId, blockSz, threadId);
		
		// ld.global.u64 %storeAddr, [prof_data_pt];
		//PTXInstrCreator::createTwoOperandInst(previousBlock, ir::PTXInstruction::Mov, storeAddr, profArrayPt);
		newInst = new PTXInstruction(ir::PTXInstruction::Ld);
		newInst->setAddressSpace(ir::PTXInstruction::Global);
		newInst->setDestination(storeAddr);
		newInst->setOperandA(profArrayPt);
		PTXInstrCreator::verifyInstruction(newInst);
		previousBlock->addInstructionEnd(newInst);
		
		// mad.wide.u32 %storeAddr, %threadId, 8, %storeAddr;
		PTXInstrCreator::createFourOperandModInst(previousBlock, ir::PTXInstruction::Mad, ir::PTXInstruction::wide, storeAddr, threadId, constEight, storeAddr);
		
		// st.global.u64 [%storeAddr], %elapsedTime;
		newInst = new PTXInstruction(ir::PTXInstruction::St);
		newInst->setAddressSpace(ir::PTXInstruction::Global);
		newInst->setDestination(storeAddr);
		newInst->setOperandA(elapsedTime);
		PTXInstrCreator::verifyInstruction(newInst);
		previousBlock->addInstructionEnd(newInst);
	}
	
	previousBlock = bb;
}
