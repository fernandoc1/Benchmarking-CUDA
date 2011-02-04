/*
 * DivergenceGraphTest.cpp
 *
 *  Created on: Jul 1, 2010
 *      Author: undead
 */

/*
 * *0	: 1 3 7	:0 dirty
 * 1 	: 3 4		:1 dirty
 * 2	:	3 6 7	:2 clean
 * 3	:	8			:3 dirty
 * 4	:	 			:4 dirty
 * 5	: 3			:5 clean
 * 6	: 7 8 9	:6 clean
 * 7	: 8 9 1	:7 dirty
 * 8	: 9			:8 dirty
 * 9	: 3 4		:9 dirty
 * */

#include <ocelot/graphs/interface/DivergenceGraph.h>
#include <iostream>

int main(){
	using namespace graph_utils;
	using namespace std;

	bool result = true;
	DivergenceGraph dirtGraph;
	dirtGraph.insertEdge(ir::PTXOperand::tidX, 0);
	dirtGraph.insertEdge(0, 1);
	dirtGraph.insertEdge(0, 3);
	dirtGraph.insertEdge(0, 7);
	dirtGraph.insertEdge(1, 3);
	dirtGraph.insertEdge(1, 4);
	dirtGraph.insertEdge(2, 3);
	dirtGraph.insertEdge(2, 6);
	dirtGraph.insertEdge(2, 7);
	dirtGraph.insertEdge(3, 8);
	dirtGraph.insertEdge(5, 3);
	dirtGraph.insertEdge(6, 7);
	dirtGraph.insertEdge(6, 8);
	dirtGraph.insertEdge(6, 9);
	dirtGraph.insertEdge(ir::PTXOperand::laneId, 6);
	dirtGraph.insertEdge(7, 8);
	dirtGraph.insertEdge(7, 9);
	dirtGraph.insertEdge(7, 1);
	dirtGraph.insertEdge(8, 9);
	dirtGraph.insertEdge(9, 3);
	dirtGraph.insertEdge(9, 4);
	dirtGraph.insertEdge(ir::PTXOperand::laneId, 2);
	dirtGraph.insertEdge(ir::PTXOperand::laneId, 5);
	dirtGraph.eraseSpecialSource(ir::PTXOperand::laneId);

	dirtGraph.computeDivergence();

	for( int a = 0; (a < 10) && (result); a++ ){
		switch( a ){
			case 2:
			case 5:
			case 6:
				result = !dirtGraph.isDivNode(a);
			break;
			default:
				result = dirtGraph.isDivNode(a);
			break;
		}
		if( !result )
			cout << "failed with variable " << a << endl;
	}

	dirtGraph.print(cout);

	cout << endl;

	if( result ){
		cout << "success !!!" << endl;
		return 0;
	}

	cout << "error!!!" << endl;

	return 1;
}
