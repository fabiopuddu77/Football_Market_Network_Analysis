
import pandas as pd
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

url = "Datasets/leghe_it.csv"
frame = pd.read_csv(url, error_bad_lines=False, sep=',')
frame = frame[frame.Movement == "Out"]
df = frame

frame['Costo'] = frame['Costo'].str.replace(' mila €','000')\
    .str.replace(' mln €','0000').str.replace(r'\D', '').replace(r'^\s*$', np.nan, regex=True)
frame.dropna(subset = ["Costo"], inplace=True)

new = pd.DataFrame(zip(frame.Club,frame.ClubInvolved,frame.Costo))
new.rename(columns={0:'Club', 1: 'ClubInvolved', 2: 'Costo'}, inplace=True)
new['Costo'] = new['Costo'].astype(int)
df_edges = new.groupby(by=['Club', 'ClubInvolved'], as_index=False)['Costo'].sum()
#df = df_edges[~df_edges[['Club', 'ClubInvolved']].apply(frozenset, axis=1).duplicated()]
df_edges = df_edges[df_edges.Costo > 0]
lista_unica = list(df_edges.Club.unique())
df_edges = df_edges[df_edges['ClubInvolved'].isin(lista_unica)]

g = nx.from_pandas_edgelist(df_edges, source="Club", target="ClubInvolved", edge_attr=["Costo"],create_using=nx.Graph)

start = 'Olbia Calcio 1905'
end = 'Juventus FC'
Shortest = (list(nx.all_shortest_paths(g, start, end)))

def short(prova):
    pesi = []
    for lista in Shortest:
        a = 0
        for j in range(0, len(lista)-1):
            a += g.edges[lista[j],lista[j+1]]['Costo']
        pesi.append(a/(len(lista)-1))
    print('I percorsi migliori in termini monetari per il procuratore '
    'per partire dalla squadra {} \n per arrivare '
    'alla squadra {} sono: \n'.format(start, end))
    perc = []
    values = sorted(pesi, reverse=True)[:3]
    for i in range(0, 3):
        a = sorted(pesi, reverse=True)[i]

        # max_value = max(pesi)
        max_index = pesi.index(a)
        print(prova[max_index])

        perc.append(tuple(prova[max_index]))
    diz = {'Percorsi': perc, 'Valori': values}
    data = pd.DataFrame(diz)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
    #sns.set(style="dark")
    ax.set_title('Money Values', size=13, color='Black')
    ax.tick_params(labelsize=8)
    # fig.patch.set_visible(False)
    sns.barplot(x='Percorsi', y="Valori", data=data, palette='YlOrRd', ax=ax)

    plt.setp(ax, ylabel=(''), xlabel=(''))
    plt.show()

    return prova[max_index]

short(Shortest)