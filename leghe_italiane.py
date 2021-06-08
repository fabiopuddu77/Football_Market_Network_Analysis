
import pandas as pd
import itertools
import plotly.graph_objects as go
import networkx as nx
from collections import Counter
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra

url = "Datasets/leghe_it.csv"
frame = pd.read_csv(url, error_bad_lines=False, sep=',')
frame = frame[frame.Movement == "Out"]
frame = frame.replace('FC Internazionale','Inter').replace('Parma Calcio 1913','Parma FC')
lista_unica = list(frame.Club.unique())
frame = frame[frame['ClubInvolved'].isin(lista_unica)]
frame['new_col'] = frame["Club"] +","+ frame["ClubInvolved"]

# frame = frame[frame.Club != "Calcio Catania"]
# a = pd.DataFrame(frame['new_col'].value_counts()).reset_index()
# a.columns = ['squadre', 'counts']
# b = a[a['counts'] > 20]

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


df_edges_old = get_edges(data=frame, column="new_col")
df = df_edges_old[df_edges_old['weight'] > 0]

#df = df_edges[~df_edges[['source', 'target']].apply(frozenset, axis=1).duplicated()]

g = nx.from_pandas_edgelist(df, source="source", target="target", edge_attr=["weight"],create_using=nx.Graph)

#a = dict(g.degree)
a = Counter(df.source)
for char in a:
    if a[char] > 0:
        g.add_node(char, size = a[char])


def make_edge(x, y, text, width):

    return go.Scatter(x=x,
                      y=y,
                      line=dict(width=width,
                                color='grey'),
                      hoverinfo='text',
                      text=([text]),
                      opacity=1,
                      mode='lines')

def make_edge2(x, y, text, width):

    return go.Scatter(x=x,
                      y=y,
                      line=dict(width=width,
                                color='yellow'),
                      hoverinfo='text',
                      text=([text]),
                      opacity=1,
                      mode='lines')
def make_edge3(x, y, text, width):

    return go.Scatter(x=x,
                      y=y,
                      line=dict(width=width,
                                color='red'),
                      hoverinfo='text',
                      text=([text]),
                      opacity=1,
                      mode='lines')
def make_edge4(x, y, text, width):

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
#pos = nx.circular_layout(g)
pos = nx.spiral_layout(g)
#pos = nx.multipartite_layout(g)
#
# Position nodes in layers of straight lines.

# For each edge, make an edge_trace, append to list

edge_trace = []
for edge in g.edges():
    # if g.edges()[edge]['weight'] > 1:
        if g.edges()[edge]['weight'] < 5:
            char_1 = edge[0]
            char_2 = edge[1]
            x0, y0 = pos[char_1]
            x1, y1 = pos[char_2]
            text = char_1 + '--' + char_2 + ': ' + str(g.edges()[edge]['weight'])

            trace = make_edge([x0, x1, None], [y0, y1, None],text,
                             g.edges()[edge]['weight']/60)

            edge_trace.append(trace)

        elif g.edges()[edge]['weight'] < 18:
            char_1 = edge[0]
            char_2 = edge[1]
            x0, y0 = pos[char_1]
            x1, y1 = pos[char_2]
            text = char_1 + '--' + char_2 + ': ' + str(g.edges()[edge]['weight'])

            trace = make_edge4([x0, x1, None], [y0, y1, None],text,
                             g.edges()[edge]['weight']/60)

            edge_trace.append(trace)

        elif g.edges()[edge]['weight'] < 25:
            char_1 = edge[0]
            char_2 = edge[1]
            x0, y0 = pos[char_1]
            x1, y1 = pos[char_2]
            text = char_1 + '--' + char_2 + ': ' + str(g.edges()[edge]['weight'])

            trace = make_edge2([x0, x1, None], [y0, y1, None], text,
                               g.edges()[edge]['weight'] / 30)

            edge_trace.append(trace)

        elif g.edges()[edge]['weight'] >= 25:
            char_1 = edge[0]
            char_2 = edge[1]
            x0, y0 = pos[char_1]
            x1, y1 = pos[char_2]
            text = char_1 + '--' + char_2 + ': ' + str(g.edges()[edge]['weight'])

            trace = make_edge3([x0, x1, None], [y0, y1, None], text,
                               g.edges()[edge]['weight']/18)

            edge_trace.append(trace)

node_trace = go.Scatter(x=[],y=[],
    mode='markers',
    hoverinfo='text',
    text= [],
    marker=dict(
        showscale=True,
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |'YlGnBu'
        colorscale='Hot',
        reversescale=True,
        color=[],
        size=[],
        colorbar=dict(
            thickness=15,
            title='Degree Centrality',
            xanchor='left',
            titleside='right'
        ),
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
    if g.nodes()[node]['size'] <50:
        node_trace['marker']['size']+= tuple([g.nodes()[node]['size'] * len(clubs) / sum(all_weights)])
        node_trace['text'] += tuple(['<b>' + node + '</b>'])
    elif g.nodes()[node]['size'] < 100:
        node_trace['marker']['size'] += tuple([g.nodes()[node]['size'] * len(clubs) *5/ sum(all_weights)])
        node_trace['text'] += tuple(['<b>' + node + '</b>'])
    elif g.nodes()[node]['size'] >= 100:
        node_trace['marker']['size'] += tuple([g.nodes()[node]['size'] * len(clubs) * 10 / sum(all_weights)])
        node_trace['text'] += tuple(['<b>' + node + '</b>'])

node_adjacencies = []
node_text = []
node_betweenness = []

for node, adjacencies in enumerate(g.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))
    node_text.append('# of connections: '+ str(len(adjacencies[1])))

for value in list(nx.betweenness_centrality(g).values()):
    node_betweenness.append(value)
    node_text.append('Betweenness: ')

node_trace.marker.color = node_adjacencies
#node_trace.marker.color = node_betweenness

fig = go.Figure(
             layout=go.Layout(
                title='Network Graph Football Italy 2010-2020',
                titlefont_size=20,
                paper_bgcolor='rgba(0,0,0,0)', # transparent background
                plot_bgcolor='black', # transparent 2nd background
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

#fig.write_html('leghe_it_2010_20.html')

# # import dash
# # import dash_core_components as dcc
# # import dash_html_components as html
# #
# # app = dash.Dash()
# # app.layout = html.Div([
# #     dcc.Graph(figure=fig)
# # ])
#
#
