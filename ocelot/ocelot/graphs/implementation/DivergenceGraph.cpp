/*
 * DivergenceGraph.cpp
 *
 *  Created on: May 19, 2010
 *      Author: Diogo Sampaio
 */

#include <ocelot/graphs/interface/DivergenceGraph.h>
#include <boost/assign/list_of.hpp>

namespace graph_utils {

/*!\brief Clears the divergence graph */
void DivergenceGraph::clear(){
	DirectionalGraph::clear();
	_divergentNodes.clear();
	_specials.clear();
	_divergenceSources.clear();
	_upToDate = true;
}

/*!\brief Insert a special register source, possible source of divergence */
void DivergenceGraph::insertSpecialSource( const ir::PTXOperand::SpecialRegister &tid ){
	_upToDate = false;
	if( _specials.find(tid) == _specials.end() ){
		node_set a;
		_specials[tid] = a;
	}
}

/*!\brief Removes a special register source */
void DivergenceGraph::eraseSpecialSource( const ir::PTXOperand::SpecialRegister &tid ){
	_upToDate = false;
	_specials.erase(tid);
}

/*!\brief Define a node as being divergent, not depending on it's predecessors */
void DivergenceGraph::setAsDiv( const node_type &node ){
	if( nodes.find(node) != nodes.end() ){
		_upToDate = false;
		_divergenceSources.insert(node);
//		nodes.erase(node);
	}
}

/*!\brief Unset a node as being divergent, not depending on it's predecessors */
void DivergenceGraph::unsetAsDiv( const node_type &node ){
	if( _divergenceSources.find(node) != _divergenceSources.end() ){
		_upToDate = false;
//		nodes.insert(node);
		_divergenceSources.erase(node);
	}
}

/*!\brief Removes a node from the divergence graph */
bool DivergenceGraph::eraseNode( const node_type &nodeId ){
	_upToDate = false;
	_divergentNodes.erase(nodeId);
	return DirectionalGraph::eraseNode(nodeId);
}

/*!\brief Removes a node from the divergence graph */
bool DivergenceGraph::eraseNode( const node_iterator &node ){
	if( nodes.find(*node) == nodes.end() ){
		return false;
	}

	_upToDate = false;
	_divergentNodes.erase(*node);

	return DirectionalGraph::eraseNode(*node);
}

/*!\brief Inserts a edge[arrow] between two nodes of the graph / Can create nodes if they don't exist */
int DivergenceGraph::insertEdge( const node_type &fromNode, const node_type &toNode, const bool createNewNodes ){
	_upToDate = false;
	return DirectionalGraph::insertEdge(fromNode, toNode, createNewNodes);
}

/*!\brief Inserts a edge[arrow] between two nodes of the graph / Can create nodes if they don't exist */
int DivergenceGraph::insertEdge( const ir::PTXOperand::SpecialRegister &origin, const node_type &toNode, const bool createNewNodes ){
	if( createNewNodes ){
		insertSpecialSource(origin);
	}else if( _specials.find(origin) == _specials.end() ){
		return 1;
	}

	_specials[origin].insert(toNode);
	return 0;
}

/*!\brief Inserts a edge[arrow] between two nodes of the graph / Can remove nodes if they are isolated */
int DivergenceGraph::eraseEdge( const node_type &fromNode, const node_type &toNode, const bool removeIsolatedNodes ){
	_upToDate = false;
	if( removeIsolatedNodes ){
		size_t actualNodesCount = nodes.size();
		int result = DirectionalGraph::eraseEdge(fromNode, toNode, removeIsolatedNodes);

		if( actualNodesCount == nodes.size() ){
			return result;
		}

		if( nodes.find(fromNode) == nodes.end() ){
			_divergentNodes.erase(fromNode);
		}else{
			_divergentNodes.erase(toNode);
		}

		return result;
	}

	return DirectionalGraph::eraseEdge(fromNode, toNode, removeIsolatedNodes);
}

/*!\brief Gests a list[set] of the divergent nodes */
const DirectionalGraph::node_set DivergenceGraph::getDivNodes() const{
	return _divergentNodes;
}

/*!\brief Tests if a node is divergent */
bool DivergenceGraph::isDivNode( const node_type &node ) const{
	return _divergentNodes.find(node) != _divergentNodes.end();
}

/*!\brief Tests if a node is a divergence source */
bool DivergenceGraph::isDivSource( const node_type &node ) const{
	return _divergenceSources.find(node) != _divergenceSources.end();
}

/*!\brief Tests if a special register is source of divergence */
bool DivergenceGraph::isDivSource( const ir::PTXOperand::SpecialRegister srt ) const{
	return ((srt == ir::PTXOperand::SpecialRegister::laneId) || (srt <= ir::PTXOperand::SpecialRegister::tidZ));
}

/*!\brief Tests if a special register is present on the graph */
bool DivergenceGraph::hasSpecial( const ir::PTXOperand::SpecialRegister &special ) const{
	return _specials.find(special) != _specials.end();
}

/*!\brief Gives the number of divergent nodes */
size_t DivergenceGraph::divNodesCount() const{
	return _divergentNodes.size();
}

/*!\brief Gives an iterator at the first divergent node of the graph */
inline DirectionalGraph::node_iterator DivergenceGraph::beginDivNodes() const{
	return _divergentNodes.begin();
}

/*!\brief Gives the divergent node iterator limit of the graph */
inline DirectionalGraph::node_iterator DivergenceGraph::endDivNodes() const{
	return _divergentNodes.end();
}

/*!\brief Computes divergence spread
 * 1) Clear preview divergent nodes list
 * 2) Set all nodes that are directly dependent of a divergent source {tidX, tidY, tidZ and laneid } as new divergent nodes
 * 3) Set all nodes that are explicitly defined as divergence sources as new divergent nodes
 * 4) For each new divergent nodes
 * 4.1) Set all non divergent nodes that depend directly on the divergent node as new divergent nodes
 * 4.1.1) Go to step 4 after step 4.3 until there are new divergent nodes
 * 4.2) Insert the node to the divergent nodes list
 * 4.3) Remove the node from the new divergent list
 */
void DivergenceGraph::computeDivergence(){
	if( _upToDate ){
		return;
	}
	/* 1) Clear preview divergent nodes list */
	_divergentNodes.clear();
	node_set newDirtyNodes;

	/* 2) Set all nodes that are directly dependent of a divergent source {tidX, tidY, tidZ and laneid } as divergent */
	{
		map<ir::PTXOperand::SpecialRegister, node_set>::iterator dirt = _specials.begin();
		map<ir::PTXOperand::SpecialRegister, node_set>::iterator endDirt = _specials.end();

		for( ; dirt != endDirt; dirt++ ){
			if( isDivSource(dirt->first) ){
				const_node_iterator node = dirt->second.begin();
				const_node_iterator endNode = dirt->second.end();

				for( ; node != endNode; node++ ){
					newDirtyNodes.insert(*node);
				}
			}
		}
	}
	{
		/* 3) Set all nodes that are explicitly defined as divergence sources as new divergent nodes */

		node_iterator dirt = _divergenceSources.begin();
		node_iterator last = _divergenceSources.end();

		for( ; dirt != last; dirt++ ){
			newDirtyNodes.insert(*dirt);
		}
	}

	/* 4) For each new divergent nodes */
	while( newDirtyNodes.size() != 0 ){
		node_type originNode = *newDirtyNodes.begin();
		node_set newReachedNodes = getOutNodesSet(originNode);
		node_iterator current = newReachedNodes.begin();
		node_iterator last = newReachedNodes.end();

		/* 4.1) Set all non divergent nodes that depend directly on the divergent node as new divergent nodes */
		for( ; current != last; current++ ){
			if( !isDivNode(*current) ){
				/* 4.1.1) Go to step 4 after step 4.3 until there are new divergent nodes */
				newDirtyNodes.insert(*current);
			}
		}

		/* 4.2) Insert the node to the divergent nodes list */
		_divergentNodes.insert(originNode);
		/* 4.3) Remove the node from the new divergent list */
		newDirtyNodes.erase(originNode);
	}

	_upToDate = true;
}

/*!\brief Gives a string as name for a special register */
const string DivergenceGraph::getSpecialName( const ir::PTXOperand::SpecialRegister& in ) const{
	using ir::PTXOperand;
	const static map<PTXOperand::SpecialRegister, string> specialNames =
	    boost::assign::map_list_of(PTXOperand::tidX, "tidX")(PTXOperand::tidY, "tidY")(PTXOperand::tidZ, "tidZ")(PTXOperand::ntidX, "ntidX")(PTXOperand::ntidY, "ntidY")(
	        PTXOperand::ntidZ, "ntidZ")(PTXOperand::laneId, "laneId")(PTXOperand::warpId, "warpId")(PTXOperand::warpSize, "warpSize")(PTXOperand::ctaIdX, "ctaIdX")(
	        PTXOperand::ctaIdY, "ctaIdY")(PTXOperand::ctaIdZ, "ctaIdZ")(PTXOperand::nctaIdX, "nctaIdX")(PTXOperand::nctaIdY, "nctaIdY")(PTXOperand::nctaIdZ, "nctaIdZ")(
	        PTXOperand::smId, "smId")(PTXOperand::nsmId, "nsmId")(PTXOperand::gridId, "gridId")(PTXOperand::clock, "clock")(PTXOperand::pm0, "pm0")(PTXOperand::pm1, "pm1")(
	        PTXOperand::pm2, "pm2")(PTXOperand::pm3, "pm3")(PTXOperand::SpecialRegister_invalid, "SpecialRegister_invalid");

	return specialNames.find(in)->second;
}

/*!\brief Prints the divergence graph in dot language */
std::ostream& DivergenceGraph::print( std::ostream& out ) const{
	using ir::PTXOperand;
	out << "digraph DirtyVariablesGraph{" << endl;

	/* Print dirt sources */
	map<PTXOperand::SpecialRegister, node_set>::const_iterator dirt = _specials.begin();
	map<PTXOperand::SpecialRegister, node_set>::const_iterator endDirt = _specials.end();

	out << "//Dirt sources:" << endl;
	for( ; dirt != endDirt; dirt++ ){
		if( dirt->second.size() ){
			out << getSpecialName(dirt->first) << "[style=filled, fillcolor = \"" << (isDivSource(dirt->first)?"tomato":"lightblue") << "\"]" << endl;
		}
	}

	/* Print nodes */
	out << "//Nodes:" << endl;
	const_node_iterator node = getBeginNode();
	const_node_iterator endNode = getEndNode();

	for( ; node != endNode; node++ ){
		out << *node << " [style=filled, fillcolor = \"" << (isDivNode(*node)?"lightyellow":"white") << "\"]" << endl;
	}

	out << endl;

	/* Print edges coming out of dirt sources */
	dirt = _specials.begin();
	endDirt = _specials.end();

	out << "//Dirt out edges:" << endl;
	for( ; dirt != endDirt; dirt++ ){
		if( dirt->second.size() ){
			node = dirt->second.begin();
			endNode = dirt->second.end();

			for( ; node != endNode; node++ ){
				out << getSpecialName(dirt->first) << "->" << *node << "[color = \"" << (isDivSource(dirt->first)?"red":"blue") << "\"]" << endl;
			}
		}
	}

	/* Print arrows between nodes */
	node = getBeginNode();
	endNode = getEndNode();

	out << "//Nodes edges:" << endl;
	for( ; node != endNode; node++ ){
		const node_set outArrows = getOutNodesSet(*node);
		const_node_iterator nodeOut = outArrows.begin();
		const_node_iterator endNodeOut = outArrows.end();

		for( ; nodeOut != endNodeOut; nodeOut++ ){
			out << *node << "->" << *nodeOut << endl;
		}
	}

	out << '}';

	return out;
}

std::ostream& operator<<( std::ostream& out, const DivergenceGraph& graph ){
	return graph.print(out);
}

}
