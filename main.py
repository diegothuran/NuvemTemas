from clusterization.cluster import Cluster
import collections
import pandas as pd

cluster = Cluster()

clusterizado,frases = cluster.clusterizar()

#Printar Clusters
clustering = collections.defaultdict(list)
label = []
for idx, label in enumerate(clusterizado.labels_):
    clustering[label].append(frases[idx])
#print(dict(clustering))

print(clustering)

