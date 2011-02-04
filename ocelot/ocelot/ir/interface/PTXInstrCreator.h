#ifndef PTX_INSTR_CREATOR_H_
#define PTX_INSTR_CREATOR_H_

#include <ocelot/ir/interface/PTXInstruction.h>

/*!
 * This is a utility class that provides several methods to create
 * instructions. These methods are simply syntactic sugar that
 * removes from the programmer the burden of setting the many
 * instruction parameters during the creation of new instructions.
 */
class PTXInstrCreator
{
public:

	/*! This function verifies if inst is a well formed instruction. */
	static void verifyInstruction(ir::PTXInstruction* inst) {
		std::string error = inst->valid();
		if (error != "") {
			std::cerr << "Error generating instruction: " << error << std::endl;
		}
	}

	/*! This method creates an instruction with two operands, such as
	 * mov.u32 %start, %clock
	 */
	static ir::PTXInstruction* createTwoOperandInst(
		ir::ControlFlowGraph::iterator bb,
		ir::PTXInstruction::Opcode opcode,
		ir::PTXOperand& dst,
		ir::PTXOperand& src
	) {
		ir::PTXInstruction* inst = new ir::PTXInstruction(opcode);
		inst->setDestination(dst);
		inst->setOperandA(src);
		verifyInstruction(inst);
		
		bb->addInstructionEnd(inst);
		return inst;
	}

	/*! This method creates an instruction with three operands, such as
	 * sub.s64 %prd0, %prd1, %prd0;
	 */
	static ir::PTXInstruction* createThreeOperandInst(
		ir::ControlFlowGraph::iterator bb,
		ir::PTXInstruction::Opcode opcode,
		ir::PTXOperand& dst,
		ir::PTXOperand& src1,
		ir::PTXOperand& src2
	) {
		ir::PTXInstruction* inst = new ir::PTXInstruction(opcode);
		inst->setDestination(dst);
		inst->setOperandA(src1);
		inst->setOperandB(src2);
		verifyInstruction(inst);
		
		bb->addInstructionEnd(inst);
		return inst;
	}
	
	/*! This method creates an floating point instruction with four operands 
	 *  that needs modifiers such as mad.lo.s32 %r3, %r0, %r1, %r2;
	 */
	static ir::PTXInstruction* createFourOperandInst(
		ir::ControlFlowGraph::iterator bb,
		ir::PTXInstruction::Opcode opcode,
		ir::PTXOperand& dst,
		ir::PTXOperand& src1,
		ir::PTXOperand& src2,
		ir::PTXOperand& src3
	) {
		ir::PTXInstruction* inst = new ir::PTXInstruction(opcode);
		inst->setDestination(dst);
		inst->setOperandA(src1);
		inst->setOperandB(src2);
		inst->setOperandC(src3);
		verifyInstruction(inst);
		
		bb->addInstructionEnd(inst);
		return inst;
	}
	
	/*! This method creates an instruction with three operands that needs
	 *  modifiers, such as mul.lo.s32 %r2, %r0, %r1;
	 */
	static ir::PTXInstruction* createThreeOperandModInst(
		ir::ControlFlowGraph::iterator bb,
		ir::PTXInstruction::Opcode opcode,
		ir::PTXInstruction::Modifier modifier,
		ir::PTXOperand& dst,
		ir::PTXOperand& src1,
		ir::PTXOperand& src2
	) {
		ir::PTXInstruction* inst = new ir::PTXInstruction(opcode);
		inst->setModifier(modifier);
		inst->setDestination(dst);
		inst->setOperandA(src1);
		inst->setOperandB(src2);
		verifyInstruction(inst);
		
		bb->addInstructionEnd(inst);		
		return inst;
	}
	
	/*! This method creates an floating point instruction with four operands 
	 *  that needs modifiers such as mad.lo.s32 %r3, %r0, %r1, %r2;
	 */
	static ir::PTXInstruction* createFourOperandModInst(
		ir::ControlFlowGraph::iterator bb,
		ir::PTXInstruction::Opcode opcode,
		ir::PTXInstruction::Modifier modifier,
		ir::PTXOperand& dst,
		ir::PTXOperand& src1,
		ir::PTXOperand& src2,
		ir::PTXOperand& src3
	) {
		ir::PTXInstruction* inst = new ir::PTXInstruction(opcode);
		inst->setModifier(modifier);
		inst->setDestination(dst);
		inst->setOperandA(src1);
		inst->setOperandB(src2);
		inst->setOperandC(src3);
		verifyInstruction(inst);

		bb->addInstructionEnd(inst);		
		return inst;
	}
	
	
};

#endif
