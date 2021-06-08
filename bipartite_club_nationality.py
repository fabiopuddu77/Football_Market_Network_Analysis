from random import random

import pandas as pd
import itertools
import plotly.graph_objects as go
import networkx as nx
from collections import Counter
from matplotlib import pyplot as plt

url = "Datasets/leghe_euro_1020_INOUT.csv"
frame = pd.read_csv(url, error_bad_lines=False, sep=',')
frame = frame[frame.Movement == "Out"]
frame = frame.replace('FC Internazionale','Inter').replace('Parma Calcio 1913','Parma FC')
lista_club = ['Inter', 'Juventus FC', 'Cagliari Calcio','AC Milan']
lista_naz = ['Brasile', 'Argentina', 'Inghilterra', 'Francia', 'Spagna','Uruguay']
frame = frame[frame['Club'].isin(lista_club)]
frame = frame[frame['Nationality'].isin(lista_naz)]
frame['new_col'] = frame["Club"] + "," + frame["Nationality"]

def get_edges(data, column):
    series = data[column].dropna().apply(lambda x: x.split(","))
    cross = series.apply(lambda x: list(itertools.combinations(x, 2)))
    lists = [item for sublist in cross for item in sublist]
    source = [i[0] for i in lists]
    target = [i[1] for i in lists]
    edges = pd.DataFrame({"source": source, "target": target})
    edges["weight"] = 1
    # return edges
    return edges.groupby(by=["source", "target"], as_index=False)["weight"].sum()


df_edges_old = get_edges(data=frame, column="new_col")
df = df_edges_old[df_edges_old['weight'] > 0]

# df = df_edges[~df_edges[['source', 'target']].apply(frozenset, axis=1).duplicated()]

B = nx.from_pandas_edgelist(df, source="source", target="target", edge_attr=["weight"], create_using=nx.Graph)

B.add_nodes_from(frame['Club'], bipartite=0)
B.add_nodes_from(frame['Nationality'], bipartite=1)

d = dict(Counter(frame.Club))
a = dict(Counter(frame.Nationality))

clubs = list(df.source.unique())
clubsinv = list(df.target.unique())

plt.figure(figsize=(40, 30))

# # 3. Draw the parts we want
top = nx.bipartite.sets(B)[0]
pos = nx.bipartite_layout(B, top)

club_size = [B.degree(club)*30 for club in clubs]

k = list(d.keys())
v = list(d.values())

lista = list(a.keys())
pesi = list(a.values())
# Draw EVERYONE
nx.draw_networkx_nodes(B, pos, nodelist=k, node_color='yellow', node_size=[team*3 for team in v])
nx.draw_networkx_nodes(B, pos, nodelist=lista, node_color='red', node_size=[peso*3 for peso in pesi])
# nx.draw_networkx_nodes(B, pos, nodelist=clubsinv, node_color='red', node_size=200)

# nx.draw(B, nodelist=d.keys(), node_size=[v * 100 for v in d.values()])

# Draw POPULAR clubs
popular_clubsinv = [clubinv for clubinv in clubsinv if B.degree(clubinv) > 20]

edges,weights = zip(*nx.get_edge_attributes(B,'weight').items())

all_weights = []
# 4 a. Iterate through the graph nodes to gather all the weights
for (node1, node2, data) in B.edges(data=True):
    all_weights.append(data['weight'])  # we'll use this when determining edge thickness

# 4 b. Get unique weights
unique_weights = list(set(all_weights))

# 4 c. Plot the edges - one by one!
for weight in unique_weights:

    # 4 d. Form a filtered list with just the weight you want to draw
    weighted_edges = [(node1, node2) for (node1, node2, edge_attr) in B.edges(data=True) if
                      edge_attr['weight'] == weight]
    width = weight * len(clubs)*3 / sum(all_weights)
    if weight > 20:
        nx.draw_networkx_edges(B, pos, edgelist=weighted_edges,
                               width=weight * len(clubs)*5 / sum(all_weights), edge_color= "lightblue")
    # elif weight > 15:
    #     nx.draw_networkx_edges(B, pos, edgelist=weighted_edges,
    #                            width=width, edge_color="blue")

    nx.draw_networkx_edges(B, pos, edgelist=weighted_edges,
                           width=width, edge_color="lightblue")

node_labels = dict(zip(clubs,popular_clubsinv))
nx.draw_networkx_labels(B, pos, font_size=10)
plt.axis('off')

plt.title("Bipartite Network Team Nationality 2010 2020")

plt.show()
