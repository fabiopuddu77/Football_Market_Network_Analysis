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

url = "Datasets/leghe_euro_9020.csv"
frame = pd.read_csv(url, error_bad_lines=False, sep=',')
frame = frame[frame.Movement == "Out"]
frame = frame.replace('FC Internazionale','Inter').replace('Parma Calcio 1913','Parma FC')
frame['new_col'] = frame["Club"] +","+ frame["ClubInvolved"]

frame = frame.replace((list(range(1990, 2000))), '1990-99')\
    .replace((list(range(2000, 2010))), '2000-09').replace((list(range(2010, 2021))), '2010-20')

anni = ['1990-99', '2000-09', '2010-20']

def annata(i):
    df = frame[frame['Season'] == i]
    return df

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


def plot_year(metric,title):

    def a(num):
        for x in num:
            yield x

    anni = ['1990-99', '2000-09', '2010-20']
    n2=(0,1,2)

    x1=a(anni)
    x2=a(n2)


    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,10))
    fig.suptitle(title, fontsize=16)

    sns.set(style="dark")

    for i,j in zip(x1,x2):

        df_plot = get_edges(data=annata(i), column="new_col")
        df_plot = df_plot[df_plot['weight'] > 0]

        g = nx.from_pandas_edgelist(df_plot, source="source", target="target", edge_attr=["weight"],create_using=nx.Graph)

        #La centralità Eigenvector calcola la centralità di un nodo basato sulla centralità dei suoi vicini.
        eig=sorted(metric(g).items(),key=operator.itemgetter(1), reverse=True )
        eig=pd.DataFrame(eig, columns=['teams', 'value']).head(3)

        ax[j].set_title(i ,size = 5, color = 'Black')
        ax[j].tick_params(labelsize=5)
        #fig.patch.set_visible(False)
        sns.barplot(x = 'teams',y="value", data=eig, palette='YlOrRd', ax=ax[j])

        plt.setp(ax[j], ylabel=(''), xlabel=(''))
        plt.tight_layout()



        # labelsize = 1
        # rcParams['xtick.labelsize'] = labelsize
        # rcParams['ytick.labelsize'] = labelsize
    plt.show()

def plot_total(metric):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,10))
    sns.set(style="dark")
    df_plot = get_edges(data=frame, column="new_col")
    df_plot = df_plot[df_plot['weight'] > 0]

    g = nx.from_pandas_edgelist(df_plot, source="source", target="target", edge_attr=["weight"], create_using=nx.Graph)

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
    #plot_year(nx.pagerank, title= 'Page Rank')
    #plot_year(nx.betweenness_centrality, title= 'Betweenness Centrality')
    #plot_year(nx.edge_betweenness_centrality, title= 'Edge Betweenness Centrality')
    #plot_year(nx.eigenvector_centrality, title= 'Eigen Vector Centrality')
    plot_year(nx.degree_centrality, title= 'Degree Centrality')
    #plot_total(nx.algorithms.transitivity)
    #plot_year(nx.algorithms.katz_centrality_numpy, title= 'Katz Centrality')


