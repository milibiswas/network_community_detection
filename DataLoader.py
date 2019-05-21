# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 20:46:59 2019

@author: Mili Biswas

@ DataLoader

"""

import networkx as nx


class dataloader(object):
    def __init__(self,filePath="./edges.csv"):
        self.filePath=filePath
        self.G=nx.Graph()
        self.load_graph_data()
        
    def load_graph_data(self):
        input_file = open(self.filePath,"r")
        lines = input_file.readlines() 
        for each in lines:
            e=each.strip("\n").strip(" ").split(",")
            self.G.add_edge(int(e[0]),int(e[1]))
        #print(self.G.number_of_nodes())
    def get_graph(self):
        return self.G

