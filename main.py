from clusterization.cluster import Cluster
import collections
import pandas as pd
import nltk
from clusterization.top_words import Top_Words
from wordcloud import WordCloud
import matplotlib.pyplot as plt

cluster = Cluster()

clusterizado, frases = cluster.clusterizar()
num_cluster = len(set(clusterizado.labels_))

# Printar Clusters
clustering = collections.defaultdict(list)
label = []
for idx, label in enumerate(clusterizado.labels_):
    clustering[label].append(frases[idx])
dic_cluster = dict(clustering)

top = Top_Words(dic_cluster, num_cluster)
result = top.top_words()

print(result)
wc = WordCloud(background_color="white",width=1000,height=1000, max_words=10,relative_scaling=0.5,
               normalize_plurals=False).generate_from_frequencies(result['clusters'][0])
plt.imshow(wc)
plt.show()
