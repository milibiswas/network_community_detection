# -*- coding: utf-8 -*-
"""
    Name        : __randomwalk
    Description : This file implements Random Walk Algorithm (Graph)
    
	====================================
    Created on Thu Mar  7 09:46:06 2019
    @author: Mili Biswas
    ====================================
	
	: Detail Description & Parameters
	
	 None

"""

import DataLoader as dl
import networkx as nx
import random

class __randomwalk(object):
    def __init__(self,Graph):
        self.graph = Graph     # This is networkx graph to be given upon constructor calling
        
    def getRandomWlak(self,node,numberOfHop=20):
        path=[node]  #Initialize the random path as empty list
        for count in range(numberOfHop):
            nbrs=self.getNeighbour(node)
            node=nbrs[self.returnRandom(len(nbrs)-1)]
            path.append(node)
        return path       # output is the sequence of nodes traversed or in other words a Path
        
    def returnRandom(self,n):
        return random.randint(0,n)
    
    def getNeighbour(self,node):
        nbrs=self.graph.neighbors(node)
        return list(nbrs)
        
        
        
if __name__=="__main__":
    graphObj=dl.dataloader()
    rnd = __randomwalk(graphObj.get_graph())
    print(rnd.getRandomWlak(200))
    print("======== End ========")
    
    nx.draw(graphObj.get_graph())
    




