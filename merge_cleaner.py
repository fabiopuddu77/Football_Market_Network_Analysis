
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import pandas as pd
import networkx as nx
import numpy as np
import itertools
import glob

import random

anni = [i for i in range(2010,2021)]
#leghe = ["1-bundesliga","laliga","liga-nos",'ligue-1','premier-league','serie-a']
legheA = ['premier-league']
cartella = 'PREMIER LEAGUE 1990-2020'
cartelle = ['BUNDESLIGA 1990-2020','SERIE A 1990-2020','LIGA NOS 1990-2020','LIGA 1990-2020','LIGUE 1 1990-2020','PREMIER LEAGUE 1990-2020']
leghe_euro = ['bundesliga', 'laliga', 'liga-nos','ligue1','premierleague', 'serie_a']
li = []

for anno in anni:
    for lega in legheA:
        url = "Dataset/" + str(cartella) + "/" + str(anno) + "/" + str(lega) +".csv"
        df = pd.read_csv(url, error_bad_lines=False, sep=',')
        li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)
frame = frame[frame.Window == "e"]
frame.isnull().sum()

# frame = frame[frame.Movement == "Out"]

for lega in leghe_euro:
    url = "Dataset/" + str(lega) + "_2010_21.csv"
    df = pd.read_csv(url, error_bad_lines=False, sep=',')
    li.append(df)
frame = pd.concat(li, axis=0, ignore_index=True)

# frame['new_col'] = frame["Club"] +","+ frame["ClubInvolved"]

# frame.dropna(subset = ["Club"], inplace=True)
#
# frame.dropna(subset = ["ClubInvolved"], inplace=True)
#
# frame.dropna(subset = ["Nationality"], inplace=True)
# frame.dropna(subset = ["Costo"], inplace=True)
### UNIONE DATASET SERIE MINORI

# legheB = ['serie_b_IT2','serie_c_IT3A','serie_c_IT3B']
# l2 = []
# for anno in anni:
#     for lega in legheB:
#         url = "Datasets_seriemin/" + str(anno) + "/" + str(lega) + '_'+ str(anno)+ ".csv"
#         df2 = pd.read_csv(url, error_bad_lines=False, sep=',')
#         li.append(df2)
#
#
# leghe2 = 'serie_c_IT3C'
# anni = [i for i in range(2014,2021)]
# for anno in anni:
#     url = "Datasets_seriemin/" + str(anno) + "/" + str(leghe2) + '_'+ str(anno)+ ".csv"
#     df3 = pd.read_csv(url, error_bad_lines=False, sep=',')
#     li.append(df3)
#
# frame_it = pd.concat(li, axis=0, ignore_index=True)

# frame_it['new_col'] = frame_it["Club"] +","+ frame_it["ClubInvolved"]
#
# Squadre = ['Borussia Dortmund','FC Schalke 04','FC Bayern Monaco',
#            'Bayer 04 Leverkusen','SV Werder Brema','Eintracht Francoforte',
#           'Villarreal CF', 'Valencia CF', 'Deportivo de La Coruña','Real Madrid CF', 'Sevilla FC','FC Barcellona',
#            'Atlético de Madrid','Benfica Lisbona','FC Porto','Sporting CP','Olympique Lione',
#            'FC Paris Saint-Germain','AS Monaco','Manchester United', 'Chelsea FC','FC Arsenal',
#            'Manchester City', 'Tottenham Hotspur','FC Liverpool','Cagliari Calcio','Juventus FC', 'AC Milan', 'Udinese Calcio', 'SS Lazio',
#            'Genoa CFC','Parma FC','FC Internazionale', 'AS Roma', 'SSC Napoli','UC Sampdoria', 'Torino FC','Inter', 'Atalanta','Parma Calcio 1913']
#
# frame = frame.replace('FC Internazionale','Inter').replace('Parma Calcio 1913','Parma FC')
# #df["column1"].replace({"a": "x", "b": "y"}, inplace=True)
#
# frame['Costo'] = frame['Costo'].str.replace(' mila €','000')\
#     .str.replace(' mln €','0000').str.replace(r'\D', '').replace(r'^\s*$', np.nan, regex=True)
# frame = frame[frame["Club"].isin(Squadre)]
#
# frame.dropna(subset = ["Costo"], inplace=True)
# frame['Costo'] = frame['Costo'].astype(int)
# #df.to_csv(r'Datasets Serie A/serieavalues.csv', index = False)
# #9438 values
#
# df = df[~df[['Club', 'ClubInvolved']].apply(frozenset, axis=1).duplicated()]
#
# new = pd.DataFrame(zip(frame.Club,frame.ClubInvolved,frame.Costo))
# new.rename(columns={0:'Club', 1: 'ClubInvolved', 2: 'Costo'}, inplace=True)
# frame = frame[frame.Movement == "Out"]

