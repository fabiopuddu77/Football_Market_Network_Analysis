
import itertools
import plotly.graph_objects as go
import matplotlib
import pandas as pd
import networkx as nx
import numpy as np
matplotlib.use('TkAgg')

url = "Datasets/leghe_euro_1020_INOUT.csv"
frame = pd.read_csv(url, error_bad_lines=False, sep=',')
frame["CountryInvolved"].replace({"Italia": "Serie A", "Portogallo": "Liga Nos",
                               "Inghilterra": "Premier League", "Spagna": "La Liga",
                               "Francia": "Ligue 1", "Germania": "Bundesliga"}, inplace=True)

frame["League"].replace({"serie-a": "Serie A", "liga-nos": "Liga Nos",
                               "premier-league": "Premier League", "laliga": "La Liga",
                               "ligue-1": "Ligue 1", "1-bundesliga": "Bundesliga",
                               "first-division-bis-91-92-": "Premier League"}, inplace=True)

frame['Costo'] = frame['Costo'].str.replace(' mila €','000')\
    .str.replace(' mln €','0000').str.replace(r'\D', '').replace(r'^\s*$', np.nan, regex=True)
frame.dropna(subset = ["Costo"], inplace=True)
frame['Costo'] = frame['Costo'].astype(int)
IN = frame[frame.Movement == "In"]
IN.name = 'to league'

def ruoli_eta(ds):
    ruoli =  ds.replace(['DC','TS','TD','Difesa'],'D').replace(['P','AD','AS','SP','Attacco'],'A').replace(['M','CC','TQ','Centrocampo','CD','CS'],'C')
    ruoli =  ruoli.replace(['POR'],'P')

    ruoli = ruoli[ruoli.Età != '-']
    ruoli = ruoli[ruoli.Età != '115']
    ruoli = ruoli[ruoli.Età != '-1776']
    ruoli['Pos'].value_counts()
    ruoli['Età'].value_counts()

    ruoli =  ruoli.replace(list(map(str,range(13,18))),'13-17').replace(list(map(str,range(18,21))),'18-20')\
    .replace(list(map(str,range(21,23))),'21-22').replace(list(map(str,range(23,26))),'23-25')\
    .replace(list(map(str,range(26,29))),'26-28').replace(list(map(str,range(29,32))),'29-31')\
    .replace(list(map(str,range(32,36))),'32-35').replace(list(map(str,range(36,116))),'36-42')

    ruoli['ruoloeta'] = ruoli["Pos"] +"|"+ ruoli["Età"]
    ruoli['lega_player'] = ruoli["ruoloeta"] +","+ ruoli["League"]
    ruoli.dropna(subset=["Costo"], inplace=True)
    return ruoli

def get_edges(data, column):

  series = data[column].dropna().apply(lambda x: x.split(","))
  cross = series.apply(lambda x: list(itertools.combinations(x, 2)))
  lists = [item for sublist in cross for item in sublist]
  source = [i[0] for i in lists]
  target = [i[1] for i in lists]
  edges = pd.DataFrame({"source": source, "target": target})
  edges["weight"] = data['Costo']/200000
  #return edges
  return edges.groupby(by=["source", "target"], as_index=False)["weight"].sum()


df_edges_old = get_edges(data=ruoli_eta(IN), column="lega_player")
df = df_edges_old[df_edges_old['weight'] > 0]

idx = df.groupby(['source'])['weight'].transform(max) == df['weight']
df = df[idx]

#df = df_edges[~df_edges[['source', 'target']].apply(frozenset, axis=1).duplicated()]

g = nx.from_pandas_edgelist(df, source="source", target="target", edge_attr=["weight"],create_using=nx.Graph)


a = dict(g.degree)
for char in a:
    if a[char] > 0:
        g.add_node(char, size = a[char])


def make_edge(x, y, text, width):

    return go.Scatter(x=x,
                      y=y,
                      line=dict(width=width,
                                color='black'),
                      hoverinfo='text',
                      text=([text]),
                      opacity=1,
                      mode='lines')

def make_edge2(x, y, text, width):

    return go.Scatter(x=x,
                      y=y,
                      line=dict(width=width,
                                color='red'),
                      hoverinfo='text',
                      text=([text]),
                      opacity=1,
                      mode='lines')
def make_edge3(x, y, text, width):

    return go.Scatter(x=x,
                      y=y,
                      line=dict(width=width,
                                color='orange'),
                      hoverinfo='text',
                      text=([text]),
                      opacity=1,
                      mode='lines')

#pos = nx.kamada_kawai_layout(g)
#pos = nx.spring_layout(g)
pos = nx.circular_layout(g)
#pos =nx.fruchterman_reingold_layout(g)
#pos = nx.spiral_layout(g)
#pos = nx.multipartite_layout(g)
#
# Position nodes in layers of straight lines.

# For each edge, make an edge_trace, append to list

edge_trace = []
for edge in g.edges():
    char_1 = edge[0]
    char_2 = edge[1]
    x0, y0 = pos[char_1]
    x1, y1 = pos[char_2]
    text = char_1 + '--' + char_2 + ': ' + str(g.edges()[edge]['weight'])

    trace = make_edge2([x0, x1, None], [y0, y1, None], text,
                       g.edges()[edge]['weight']/300)
    edge_trace.append(trace)

node_trace = go.Scatter(x=[],y=[],
    mode='markers+text',
    hoverinfo='text',
    textfont=dict(size=14),
    fillcolor='yellow',
    text= [],
    marker=dict(
        showscale=False,
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |'YlGnBu'
        colorscale= 'Reds',
        reversescale=False,
        color=[],
        size=[],
        line_width=0.5))

all_weights = []
# 4 a. Iterate through the graph nodes to gather all the weights
for (node1, node2, data) in g.edges(data=True):
    all_weights.append(data['weight'])

for node in g.nodes():
    x, y = pos[node]
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])
    clubs = list(df.source.unique())
    if g.nodes()[node]['size'] < 10:
        node_trace['marker']['size']+= tuple([g.nodes()[node]['size'] * len(clubs) *400/ sum(all_weights)])
        node_trace['text'] += tuple(['<b>' + node + '</b>'])
    elif g.nodes()[node]['size'] >= 10:
        node_trace['marker']['size'] += tuple([g.nodes()[node]['size'] * len(clubs) *400 / sum(all_weights)])
        node_trace['text'] += tuple(['<b>' + node + '</b>'])

node_adjacencies = []
node_text = []
node_betweenness = []

for node, adjacencies in enumerate(g.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))
    node_text.append('# of connections: '+ str(len(adjacencies[1])))

for value in list(nx.betweenness_centrality(g).values()):
    node_betweenness.append(value * 250)

for node, name in enumerate(g.nodes()):
    node_text.append(name)

node_trace.marker.color = node_adjacencies
#node_trace.text = node_text

# layout= Layout(title= "Coauthorship network of scientists working on network theory and experiment"+\
#               "<br> Data source: <a href='https://networkdata.ics.uci.edu/data.php?id=11'> [1]</a>",
#     font= dict(size=12),
#     showlegend=False,
#     autosize=False,
#     width=width,
#     height=height,
#     xaxis=layout.XAxis(axis),
#     yaxis=layout.YAxis(axis),
#     margin=layout.Margin(
#         l=40,
#         r=40,
#         b=85,
#         t=100,
#     ),
#     hovermode='closest',
#     annotations=[
#            dict(
#            showarrow=False,
#             text='This igraph.Graph has the Kamada-Kawai layout',
#             xref='paper',
#             yref='paper',
#             x=0,
#             y=-0.1,
#             xanchor='left',
#             yanchor='bottom',
#             font=dict(
#             size=14
#             )
#             )
#         ]
#     )



fig = go.Figure(
         layout=go.Layout(
            title='Network Analysis Age and Position '+ IN.name +" by Values",
            titlefont_size=20,
            paper_bgcolor='rgba(0,0,0,0)', # transparent background
            plot_bgcolor='white', # transparent 2nd background
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
            )
# Add all edge traces
for trace in edge_trace:
    fig.add_trace(trace)
# Add node trace
fig.add_trace(node_trace)

fig.show()
fig.write_html('ruolo_eta_values_IN.html')


