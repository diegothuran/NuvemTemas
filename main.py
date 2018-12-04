from clusterization.cluster import Cluster
import collections
import pandas as pd
import nltk
from clusterization.top_words import Top_Words

cluster = Cluster()

clusterizado, frases = cluster.clusterizar()
num_cluster = len(set(clusterizado.labels_))

# Printar Clusters
clustering = collections.defaultdict(list)
label = []
for idx, label in enumerate(clusterizado.labels_):
    clustering[label].append(frases[idx])
dic_cluster = dict(clustering)

top = Top_Words(dic_cluster,num_cluster)
result = top.top_words()

print(result)
