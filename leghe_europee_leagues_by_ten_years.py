import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

from numpy import sqrt
import glob
import PIL
import matplotlib
import pandas as pd
import numpy as np
import networkx as nx
import random
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import colors, cm
import itertools
#%%

df = pd.read_csv("Datasets/leghe_euro_9020.csv",error_bad_lines=False, sep=',')
df["CountryInvolved"].replace({"Italia": "Serie A", "Portogallo": "Liga Nos",
                               "Inghilterra": "Premier League", "Spagna": "La Liga",
                               "Francia": "Ligue 1", "Germania": "Bundesliga"}, inplace=True)

df["League"].replace({"serie-a": "Serie A", "liga-nos": "Liga Nos",
                               "premier-league": "Premier League", "laliga": "La Liga",
                               "ligue-1": "Ligue 1", "1-bundesliga": "Bundesliga",
                               "first-division-bis-91-92-": "Premier League"}, inplace=True)

df = df[df.Movement == "Out"]
lista_unica = ["Serie A","Liga Nos","Premier League","La Liga","Ligue 1","Bundesliga"]
df = df[df["CountryInvolved"].isin(lista_unica)]
df['Leghe'] = df["CountryInvolved"] +","+ df["League"]

df = df.replace((list(range(1990, 2000))), '1990-99')\
    .replace((list(range(2000, 2010))), '2000-09').replace((list(range(2010, 2021))), '2010-20')

anni = ['1990-99', '2000-09', '2010-20']

# for f in df.columns:
#     print(df[f].value_counts())
#     print('***********************************')

for anno in anni:
    frame = df[df.Season == anno]

    def get_edges(data, column):

      series = data[column].dropna().apply(lambda x: x.split(","))
      cross = series.apply(lambda x: list(itertools.combinations(x, 2)))
      lists = [item for sublist in cross for item in sublist]
      source = [i[0] for i in lists]
      target = [i[1] for i in lists]
      edges = pd.DataFrame({"source": source, "target": target})
      edges["weight"] = 1
      #return edges
      return edges.groupby(by=["source", "target"], as_index=False)["weight"].sum()


    df_edges_old = get_edges(data=frame, column="Leghe")
    df_edges = df_edges_old[df_edges_old['weight'] > 0]


    g = nx.from_pandas_edgelist(df_edges, edge_attr=["weight"],create_using=nx.Graph)

    layout = nx.spring_layout(g)

    frame = df_edges

    clubs = list(frame.source.unique())
    clubsinv = list(frame.target.unique())

    dict(zip(clubs, clubsinv))

    fig = plt.figure(figsize=(15, 10),facecolor='black')

    #fig.set_facecolor("#00000F")
    # 1. Create the graph

    # 2. Create a layout for our nodes
    #pos = nx.kamada_kawai_layout(g)
    #pos = nx.spring_layout(g)
    #layout = nx.circular_layout(g)

    #pos = nx.multipartite_layout(g)
    # # 3. Draw the parts we want

    club_size = [g.degree(club) for club in clubs]
    nx.draw_networkx_nodes(g,
                           layout,
                           nodelist=clubs,
                           node_size=club_size, # a LIST of sizes, based on g.degree
                           node_color=club_size)

    # Draw EVERYONE
    #nx.draw_networkx_nodes(g, layout, nodelist=clubs, node_color='lightblue', node_size=200)

    # Draw POPULAR clubs
    popular_clubsinv = [clubinv for clubinv in clubsinv if g.degree(clubinv) > 20]
    #nx.draw_networkx_nodes(g, layout, nodelist=popular_clubsinv, node_color='#F4D03F', node_size=300)

    edges,weights = zip(*nx.get_edge_attributes(g,'weight').items())

    #nx.draw(g, pos, node_color='b', edgelist=edges, edge_color=weights, width=1.0, edge_cmap=plt.cm.Blues)

    all_weights = []
    # 4 a. Iterate through the graph nodes to gather all the weights
    for (node1, node2, data) in g.edges(data=True):
        all_weights.append(data['weight'])  # we'll use this when determining edge thickness

    # 4 b. Get unique weights
    unique_weights = list(set(all_weights))

    # 4 c. Plot the edges - one by one!
    for weight in unique_weights:
        # 4 d. Form a filtered list with just the weight you want to draw
        weighted_edges = [(node1, node2) for (node1, node2, edge_attr) in g.edges(data=True) if
                          edge_attr['weight'] == weight]

        width = weight * len(clubs) * 20/ sum(all_weights)
        print(weight)
        if weight > 200:
            nx.draw_networkx_edges(g, layout, edgelist=weighted_edges,
                                   width=width, edge_color= "red")
        elif weight > 100:
            nx.draw_networkx_edges(g, layout, edgelist=weighted_edges,
                                   width=width, edge_color= "orange")
        else:
            nx.draw_networkx_edges(g, layout, edgelist=weighted_edges,
                                   width=width, edge_color= "yellow")


    # nx.draw_networkx_edges(g, layout, width=weighted_edges, alpha=0.5,
    #                        edge_color=[g[u][v]['Costo'] for u,v in g.edges])


    node_labels = dict(zip(clubs,clubsinv))
    nx.draw_networkx_labels(g, layout, font_size=16, font_color = 'white')

    # 4. Turn off the axis because I know you don't want it
    plt.axis('off')
    plt.title("Leghe Europee " + (str(anno)))

    # 5. Tell matplotlib to show it
    plt.show()

