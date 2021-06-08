
import pandas as pd
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


url = "Datasets/leghe_it.csv"
frame = pd.read_csv(url, error_bad_lines=False, sep=',')
frame = frame[frame.Movement == "Out"]
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

start = 'Olbia Calcio 1905'
end = 'Juventus FC'
Shortest = (list(nx.all_shortest_paths(g, start, end)))

def short(prova):
    pesi = []
    for lista in prova:
        a = 0
        for j in range(0, len(lista)-1):
            a += g.edges[lista[j],lista[j+1]]['weight']
        pesi.append(a/(len(lista)-1))

    print('I percorsi migliori in termini di probabilità per il procuratore '
    'per partire dalla squadra {} \nper arrivare '
    'alla squadra {} sono: \n'.format(start, end))
    perc = []
    values = sorted(pesi, reverse=True)[:3]
    for i in range(0,3):
        a = sorted(pesi, reverse=True)[i]

        #max_value = max(pesi)
        max_index = pesi.index(a)
        print(prova[max_index])

        perc.append(tuple(prova[max_index]))
    diz = {'Percorsi' : perc, 'Valori': values}
    data = pd.DataFrame(diz)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,10))
    #sns.set(style="dark")
    ax.set_title('Probability',size=13, color='Black')
    ax.tick_params(labelsize=8)
    # fig.patch.set_visible(False)
    sns.barplot(x='Percorsi', y="Valori", data=data, palette='YlOrRd', ax=ax)

    plt.setp(ax, ylabel=(''), xlabel=(''))
    plt.show()

    return prova[max_index]


short(Shortest)