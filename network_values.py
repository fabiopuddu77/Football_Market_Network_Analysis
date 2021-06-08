
import PIL
import matplotlib
import pandas as pd
import numpy as np
import networkx as nx
import random
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import colors, cm

#%%

df = pd.read_csv("Datasets/leghe_euro_values.csv",error_bad_lines=False, sep=',')
df = df[df.Movement == "Out"]
# df['Costo'] = df['Costo'].str.replace(' mila €','000').str.replace(' mln €','0000').str.replace(r'\D', '').replace(r'^\s*$', np.nan, regex=True)
#
# df = df.replace('FC Internazionale','Inter').replace('Parma Calcio 1913','Parma FC')
# #
# #27038 values
#
# df.dropna(subset = ["Costo"], inplace=True)
# df['Costo'] = df['Costo'].astype(int)
# df.to_csv(r'Datasets Serie A/serieavalues.csv', index = False)
#9438 values

df['squadre'] = df["Club"] +","+ df["ClubInvolved"]
df["League"].replace({"serie-a": 0, "liga-nos": 1,
                               "premier-league": 2, "laliga": 3,
                               "ligue-1": 4, "1-bundesliga": 5,
                               "first-division-bis-91-92-": 2}, inplace=True)

# df = df[df.Club != "Real Madrid CF"]
# df = df[df.ClubInvolved != "Real Madrid CF"]

new = pd.DataFrame(zip(df.Club,df.ClubInvolved,df.Costo,df.League))
new.rename(columns={0:'Club', 1: 'ClubInvolved', 2: 'Costo', 3:'League'}, inplace=True)
df_edges = new.groupby(by=['Club', 'ClubInvolved', 'League'], as_index=False)['Costo'].sum()
#df = df_edges[~df_edges[['Club', 'ClubInvolved']].apply(frozenset, axis=1).duplicated()]
df_edges = df_edges[df_edges.Costo > 70000000]
# lista_unica = list(df_edges.Club.unique())
# df_edges = df_edges[df_edges['ClubInvolved'].isin(lista_unica)]


g = nx.from_pandas_edgelist(df_edges, source="Club", target="ClubInvolved", edge_attr=["Costo"],create_using=nx.Graph)

d = dict(zip(df_edges.Club, df_edges.League))

for lg in d:
    g.add_node(lg, league = d[lg])

df = df_edges

clubs = list(df.Club.unique())
clubsinv = list(df.ClubInvolved.unique())

plt.figure(figsize=(40, 30))

# 1. Create the graph

# 2. Create a layout for our nodes
layout = nx.spring_layout(g)
#
# # 3. Draw the parts we want

club_size = [g.degree(club)*30 for club in clubs]

k = list(d.keys())
v = list(d.values())

nx.draw_networkx_nodes(g,
                       layout,
                       nodelist=clubs,
                       node_size=150, # a LIST of sizes, based on g.degree
                       node_color=v)

# Draw EVERYONE
#nx.draw_networkx_nodes(g, layout, nodelist=clubs, node_color='lightblue', node_size=200)

# Draw POPULAR clubs
popular_clubsinv = [clubinv for clubinv in clubsinv if g.degree(clubinv) > 20]
#nx.draw_networkx_nodes(g, layout, nodelist=popular_clubsinv, node_color='#F4D03F', node_size=300)

# Set Edge Color based on weight
# values = range(1838) #this is based on the number of edges in the graph, use print len(g.edges()) to determine this
# jet = plt.get_cmap('YlOrRd')
# cNorm = colors.Normalize(vmin=0, vmax=values[-1])
# scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
# colorList = []
#
#
# for i in range(1838):
#     colorVal = scalarMap.to_rgba(values[i])
#     colorList.append(colorVal)

for u,v,d in g.edges(data=True):
    d['weight'] = random.random()

edges,weights = zip(*nx.get_edge_attributes(g,'weight').items())

pos = nx.spring_layout(g)

all_weights = []
# 4 a. Iterate through the graph nodes to gather all the weights
for (node1, node2, data) in g.edges(data=True):
    all_weights.append(data['Costo'])  # we'll use this when determining edge thickness

# 4 b. Get unique weights
unique_weights = list(set(all_weights))


# 4 c. Plot the edges - one by one!
for weight in unique_weights:

    # 4 d. Form a filtered list with just the weight you want to draw
    weighted_edges = [(node1, node2) for (node1, node2, edge_attr) in g.edges(data=True) if
                      edge_attr['Costo'] == weight]

    width = weight * len(clubs) / sum(all_weights)

    print(weight)
    if weight < 150000000:
        nx.draw_networkx_edges(g, layout, edgelist=weighted_edges,
                               width=width, edge_color= "lightblue")
    elif weight < 200000000:
        nx.draw_networkx_edges(g, layout, edgelist=weighted_edges,
                               width=width, edge_color="b")
    else:
        nx.draw_networkx_edges(g, layout, edgelist=weighted_edges,
                               width=width, edge_color="black")




# nx.draw_networkx_edges(g, layout, width=weighted_edges, alpha=0.5,
#                        edge_color=[g[u][v]['Costo'] for u,v in g.edges])


node_labels = dict(zip(clubs,popular_clubsinv))
nx.draw_networkx_labels(g, layout, font_size=8)

# 4. Turn off the axis because I know you don't want it
plt.axis('off')

plt.title("Network Transfermarkt Prize Money")

# 5. Tell matplotlib to show it
plt.show()

