import operator
import pandas as pd
import itertools
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
import seaborn as sns

#%%

df = pd.read_csv("Datasets/leghe_euro_values.csv",error_bad_lines=False, sep=',')
df = df[df.Movement == "Out"]

# df['Costo'] = df['Costo'].str.replace(' mila €','000').str.replace(' mln €','0000').str.replace(r'\D', '').replace(r'^\s*$', np.nan, regex=True)
#
df = df.replace('FC Internazionale','Inter').replace('Parma Calcio 1913','Parma FC')
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
# df = df[df.Club != "FC Barcellona"]
# df = df[df.ClubInvolved != "FC Barcellona"]

new = pd.DataFrame(zip(df.Club,df.ClubInvolved,df.Costo,df.League))
new.rename(columns={0:'Club', 1: 'ClubInvolved', 2: 'Costo', 3:'League'}, inplace=True)
df_edges = new.groupby(by=['Club', 'ClubInvolved', 'League'], as_index=False)['Costo'].sum()
#df = df_edges[~df_edges[['Club', 'ClubInvolved']].apply(frozenset, axis=1).duplicated()]
df_edges = df_edges[df_edges.Costo > 70000000]
# lista_unica = list(df_edges.Club.unique())
# df_edges = df_edges[df_edges['ClubInvolved'].isin(lista_unica)]


g = nx.from_pandas_edgelist(df_edges, source="Club", target="ClubInvolved", edge_attr=["Costo"],create_using=nx.Graph)

def plot_total(metric,title):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,10))
    fig.suptitle(title, fontsize=16)
    sns.set(style="dark")

    # La centralità Eigenvector calcola la centralità di un nodo basato sulla centralità dei suoi vicini.
    eig = sorted(metric(g).items(), key=operator.itemgetter(1), reverse=True)
    eig = pd.DataFrame(eig, columns=['teams', 'value']).head(4)

    ax.set_title(metric, size=10, color='Black')
    ax.tick_params(labelsize=7)
    # fig.patch.set_visible(False)
    sns.barplot(x='teams', y="value", data=eig, palette='YlOrRd', ax=ax)

    plt.setp(ax, ylabel=(''), xlabel=(''))
    plt.show()

if __name__ == '__main__':
    #short(Shortest)

    #plot_year(nx.closeness_centrality, title= 'Closeness Centrality')
    #plot_total(nx.pagerank, title= 'Page Rank')
    #plot_total(nx.betweenness_centrality, title= 'Betweenness Centrality')
    #plot_year(nx.edge_betweenness_centrality, title= 'Edge Betweenness Centrality')
    #plot_total(nx.eigenvector_centrality, title= 'Eigen Vector Centrality')
    plot_total(nx.degree_centrality, title= 'Degree Centrality')
    #plot_total(nx.algorithms.transitivity)
    #plot_year(nx.algorithms.katz_centrality_numpy, title= 'Katz Centrality')

    #plot_year(nx.algorithms.shortest_path, title= 'Katz Centrality')

