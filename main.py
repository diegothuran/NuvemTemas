from clusterization.cluster import Cluster
import collections
import pandas as pd
from summarization.crispin import Texto


def join_strings(list_of_strings):
    """
        Método para transformar tokens em uma única sentença
    :param list_of_strings: Lista com os tokens
    :return: sentença formada pela união dos tokens
    """
    return " ".join(list_of_strings)


cluster = Cluster()

clusterizado, frases = cluster.clusterizar()

#Printar Clusters
clustering = collections.defaultdict(list)
label = []
for idx, label in enumerate(clusterizado.labels_):
    clustering[label].append(frases[idx])
#print(dict(clustering))

print(Texto(join_strings(clustering[1])).resumir())

