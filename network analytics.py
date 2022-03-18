# -*- coding: utf-8 -*-
"""
Name: _BOTTA_RAJU___________ Batch ID: _05102021__________
Topic: Network Analytics

@author: LENOVO
"""

'''
1.	Problem Statement: -
There are two datasets consisting of information for the connecting routes and flight halt. Create network analytics models on both the datasets separately and measure degree centrality, degree of closeness centrality, and degree of in-between centrality.
'''
# Sol: 
# Network analysis for Flight Halt

import pandas as pd
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt

G = pd.read_csv("/content/flight_hault.csv")
G = G.iloc[:, 0:10]

g = nx.Graph()

G.head()
g = nx.from_pandas_edgelist(G, source = 'Goroka', target = 'Papua New Guinea')

print(nx.info(g))

# out: Graph with 8138 nodes and 7956 edges
________________________________________

# Degree Centrality
b = nx.degree_centrality(g)  
print(b) 

pos = nx.spring_layout(g, k = 0.15)  
nx.draw_networkx(g, pos, node_size = 25, node_color = 'blue')

 
# closeness centrality
closeness = nx.closeness_centrality(g)
print(closeness)


## Betweeness Centrality 
b = nx.betweenness_centrality(g) # Betweeness_Centrality
print(b)


## Eigen-Vector Centrality
evg = nx.eigenvector_centrality(g) # Eigen vector centrality
print(evg)

# cluster coefficient
cluster_coeff = nx.clustering(g)
print(cluster_coeff)

# Average clustering
cc = nx.average_clustering(g) 
print(cc)

# =============== ************ =====================

# Network analysis for connecting routes
________________________________________

G = pd.read_csv("/content/connecting_routes.csv")
G = G.iloc[:, 1:10]

g = nx.Graph()

g = nx.from_pandas_edgelist(G, source = 'AER', target = 'KZN')
print(nx.info(g))

# Degree Centrality

b = nx.degree_centrality(g)  
print(b) 

pos = nx.spring_layout(g, k = 0.15)
nx.draw_networkx(g, pos, node_size = 25, node_color = 'blue')


# closeness centrality
closeness = nx.closeness_centrality(g)
print(closeness)

## Betweeness Centrality 
b = nx.betweenness_centrality(g) # Betweeness_Centrality
print(b)

## Eigen-Vector Centrality
evg = nx.eigenvector_centrality(g) # Eigen vector centrality
print(evg)

# cluster coefficient
cluster_coeff = nx.clustering(g)
print(cluster_coeff)

# Average clustering
cc = nx.average_clustering(g) 
print(cc)

out: 0.4870933566129556
________________________________________


'''
2.	Problem statement
There are three datasets given (Facebook, Instagram, and LinkedIn). Construct and visualize the following networks:
●	circular network for Facebook
●	star network for Instagram
●	star network for LinkedIn        '''

 
# Circular Network for Facebook..:

# import matplotlib.pyplot library
import matplotlib.pyplot as plt

# import networkx library
import networkx as nx

G = pd.read_csv("/content/facebook.csv")
G = G.iloc[:, 0:10]

g = nx.Graph()
G.columns

g = nx.from_pandas_edgelist(G, source = '1', target = '9')

print(nx.info(g))

# draw a graph with red
# node and value edge color

# create a cubical empty graph
G = nx.cubical_graph()
 
# plotting the graph
plt.subplot(122)

nx.draw(G, pos = nx.circular_layout(G), node_color = 'r', edge_color = 'b')

 # --------------------------------------------
 
# Star  Network for Instagram:


G = pd.read_csv("/content/instagram.csv")
G = G.iloc[:, 0:9]

g = nx.Graph()
G.columns
 
g = nx.from_pandas_edgelist(G, source = '1', target = '8')

print(nx.info(g))


# import required module
import networkx as nx

# create object
G = nx.star_graph(8)

# illustrate graph
nx.draw(G, node_color = 'green',
    node_size = 100)

# ---------------------------------------------------

# Star Network for LinkedIn :

G = pd.read_csv("/content/linkedin.csv")
G = G.iloc[:, 0:10]

g = nx.Graph()
G.columns


g = nx.from_pandas_edgelist(G, source = '1', target = '10')

print(nx.info(g))

# import required module
import networkx as nx

# create object
G = nx.star_graph(10)

# illustrate graph
nx.draw(G, node_color = 'blue', node_size = 100)


 























