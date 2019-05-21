# coding=utf8
from __future__ import division
import networkx as nx
from DataLoader import dataloader
from random_walk import __randomwalk
from matplotlib import pyplot as plt
import copy
import numpy as np


def __edgeBetweennessCentrality(G):
    betweenness = dict.fromkeys(G, 0.0)
    betweenness.update(dict.fromkeys(G.edges(), 0.0))
    nodes = G
    for s in nodes:
        SList, Pred, dict_s = __shortestPath(G, s)
        betweenness = __store_edges(betweenness, SList, Pred, dict_s, s)
    
    for n in G:
        del betweenness[n]
    betweenness = __normalized(betweenness, len(G),True)
    
    return betweenness


def __shortestPath(G, s):
    '''
        s => node from input
        G => input graph
        This function calculates the shortest path from a node to all other nodes in the graph
        '''
    SList = []
    Pred = {}                      # predecessors set in dictionary
    for v in G:
        Pred[v] = []
    dict_s = dict.fromkeys(G, 0.0) # default dictionary creation
    D = {}
    dict_s[s] = 1.0                # initialize the node value
    D[s] = 0
    Q = [s]
    while Q:
        v = Q.pop(0)
        SList.append(v)
        Dv = D[v]
        sigmav = dict_s[v]
        for w in G[v]:
            if w not in D:
                Q.append(w)
                D[w] = Dv + 1
            if D[w] == Dv + 1:
                dict_s[w] += sigmav
                Pred[w].append(v)
    return SList, Pred, dict_s


def __store_edges(betweenness, S, P, sigma, s):
    
    '''
        This function collects the edges
        
        '''
    
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        const = (1 + delta[w]) / sigma[w]
        for v in P[w]:
            c = sigma[v] * const
            if (v, w) not in betweenness:
                betweenness[(w, v)] += c
            else:
                betweenness[(v, w)] += c
            delta[v] += c
        if w != s:
            betweenness[w] += delta[w]
    return betweenness


def __normalized(betweenness, n, normalized):
    
    '''
        This function normalizes the betweenness values.
        '''
    
    if n <= 1:
        factor = None
    else:
        factor = 1 / (n * (n - 1))
    if factor is not None:
        for v in betweenness:
            betweenness[v] *= factor
    return betweenness

def degree_centrality(G):
    return G.degree()

def degree_centrality_norm(G,normType=None):
    list_of_degree=list(G.degree())
    list_of_degree.sort(key=lambda x:x[1],reverse=True)
    max_degree=list_of_degree[0][1]
    no_of_nodes=len(list_of_degree)
    no_of_edges=G.number_of_edges()
    
    if normType=="mpd":
        if no_of_nodes>1:
            return [(x,y/(no_of_nodes-1)) for x,y in list_of_degree ]
        else:
            return list_of_degree
    elif normType=="md":
        return [(x,y/(max_degree)) for x,y in list_of_degree ]
    elif normType=="ds":
        return [(x,y/(2*no_of_edges)) for x,y in list_of_degree ]
    else:
        print("Default normalized degree centrality : maximum possible degree")
        if no_of_nodes>1:
            return [(x,y/(no_of_nodes-1)) for x,y in list_of_degree ]
        else:
            return list_of_degree


def __subSampling(G,no_nodes,degree_threshold):
    '''
        This function returns graph or subgraph components
        '''
    list_nodes_tobe_kept=[]
    list_tobe_checked=[]
    
    node_degrees=degree_centrality(G)  # This can be removed as we call it already. just pass the value.
    
    
    for i,j in node_degrees:
        if j>=degree_threshold:
            list_nodes_tobe_kept.append(i)
        else:
            list_tobe_checked.append(i)

    #This random list is for creating the reandom sample nodes degrees in random
    random_list=np.random.randint(low=5,high=degree_threshold+1,size=abs(no_nodes-len(list_nodes_tobe_kept)))

    
    cnt=len(list_nodes_tobe_kept)
    for degree in random_list:
        if cnt<=no_nodes:
            list_nodes_tobe_kept.append(list_tobe_checked[degree])
            cnt += 1
        else:
            break
    return list_nodes_tobe_kept


def __createGraph(G,sampleNodeList):
    nodeList=sampleNodeList
    listEdges=[]
    for node in nodeList:
        listEdges.extend([(node,x) for x in G.neighbors(node) if x in nodeList])
    
    s=set([ (x,y) if (x<=y) else (y,x) for x,y in listEdges])
    
    G1=nx.Graph()
    for x,y in s:
        G1.add_edge(x,y)
    
    return G1


def edge_degree_stats(G):
    '''
        This function provides the frequency distribution of the nodes based on degrees.
        '''
    
    nodes_degree=degree_centrality(G)
    
    dict_states={}
    
    for x,y in nodes_degree:
        if y in dict_states:
            dict_states[y]=dict_states.get(y)+1
        else:
            dict_states[y]=1
    return dict_states


def edge_remove(G):
    d=__edgeBetweennessCentrality(G)
    lst_of_tuple=list(d.items())
    lst_of_tuple.sort(key=lambda x:x[1], reverse=True)
    return lst_of_tuple[0][0]


def girvan_newman(Gx):
    temp_graph=copy.deepcopy(Gx)
    c=list(nx.connected_component_subgraphs(temp_graph))
    l=len(c)
    cnt=0  # this is for number of iteration girvan-newman will iterate
    while (cnt<=15):
        temp_graph.remove_edge(*edge_remove(temp_graph))
        c=list(nx.connected_component_subgraphs(temp_graph))
        l=len(c)
        cnt=cnt+1
    return c

################################################  Main Program Execution ###########################################


if __name__=="__main__":
    print("======================================================")
    print("                    Program output                    ")
    print("======================================================")
    
    ###########################
    #  Graph initialization
    ###########################
    
    d=dataloader()
    d.load_graph_data()
    G=d.get_graph()
    
    G=__createGraph(G,__subSampling(G,500,3000))
    #G=nx.karate_club_graph()
    
    
    
    ##############################################################
    # Degree centrality calculation along with normalization
    ##############################################################
    
    degree_centrality=list(degree_centrality(G))
    degree_centrality.sort(key=lambda x:x[1],reverse=True)
    
    print("===========================================")
    print("Top 10 users with highest degree centrality")
    print("===========================================")
    
    for i,j in degree_centrality[:][:10]:
        print("Node : {} and degree is : {}".format(i,j))

    print("======================================================")
    print("Top 10 users with highest Normalized degree centrality")
    print("======================================================")
    lst=degree_centrality_norm(G,"md")[0:10]
    for i,j in lst:
        print("Node : {} and normalized degree is : {}".format(i,j))
    
    ##################################################################
    # This section is for calculating Girvan-Newman community
    # detection
    ##################################################################

    c=girvan_newman(G)

    print("======================================================")
    print("Communities after applying Girvan-Newman")
    print("======================================================")
    for i in c:
        print(list(i))


    ##############################
    #   visualization part
    ##############################


    g1=G
    pos=nx.spring_layout(g1)
    lst_colours=range(3)

    plt.figure(figsize=(20,25))
    nx.draw(g1, pos, edge_color='k', node_color='w' ,with_labels=True, font_weight='light', width= 0.2)
    for i,node in enumerate(c):
        nsd=list(node.nodes())
        nx.draw_networkx_nodes(g1, pos, nodelist=nsd,node_color=[(i+1)*20 for i in range(len(node))])
    plt.title("Fig - Community")
    plt.savefig("community.png")
    plt.show()
    
    plt.figure(figsize=(20,25))
    nx.draw(g1, pos, edge_color='k',  with_labels=True, font_weight='light', width= 0.2)
    nsd=[x for x,_ in lst]
    nx.draw_networkx_nodes(g1, pos, nodelist=nsd, node_color="b")
    plt.title("Fig - Degree Centrality")
    plt.savefig("DegreeCentrality.png")
    plt.show()
    
    #####################################
    # Random Walk with 200 hops
    #####################################
    
    rnd = __randomwalk(G)
    highest_degree_node,_=lst[0]
    
    print("=================================================")
    print("   Message pass through First Random Walk        ")
    print("=================================================")
    first_random_walk=rnd.getRandomWlak(highest_degree_node,200)
    print(first_random_walk)
    edgelist_1=[]
    for i,node in enumerate(first_random_walk[:-1]):
        edgelist_1.append((node,first_random_walk[i+1]))
    
    print("=================================================")
    print("  Message pass through Second Random Walk        ")
    print("=================================================")
    second_random_walk=rnd.getRandomWlak(highest_degree_node,200)
    print(second_random_walk)
    edgelist_2=[]
    for i,node in enumerate(second_random_walk[:-1]):
        edgelist_2.append((node,second_random_walk[i+1]))
    print("===================End of Code===================")

    #################################
    #   visualization - Random Walk
    #################################

    plt.figure(figsize=(20,25))
    nx.draw(g1, pos, edge_color='k', node_color='w' ,with_labels=True, font_weight='light', width= 0.5)
    nx.draw_networkx_edges(g1, pos, edgelist=edgelist_1,edge_color="r")
    nx.draw_networkx_nodes(g1, pos, nodelist=first_random_walk,node_color="b")
    plt.title("Random Walk - 1")
    plt.savefig("RandomWalk1.png")
    plt.show()

    plt.figure(figsize=(20,25))
    nx.draw(g1, pos, edge_color='k', node_color='w' ,with_labels=True, font_weight='light', width= 0.5)
    nx.draw_networkx_edges(g1, pos, edgelist=edgelist_2,edge_color="r")
    nx.draw_networkx_nodes(g1, pos, nodelist=second_random_walk,node_color="b")
    plt.title("Random Walk - 2")
    plt.savefig("RandomWalk2.png")
    plt.show()
