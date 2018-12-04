import nltk
from processing.processing import PreProcessing
import json


class Top_Words():
    def __init__(self, clusters, num_cluster):
        self.clusters = clusters
        self.num_cluster = num_cluster
        self.processo = PreProcessing()

    def top_words(self):
        list_top = []
        list_final = []
        for ind in range(0, self.num_cluster):
            top_words = []

            for frase in self.clusters[ind]:
                frase = self.processo.removerStopWords(self.processo.removerAcentos(frase))
                top_words.extend(nltk.word_tokenize(frase))

            freq = nltk.FreqDist(top_words)
            list_top.append(dict(freq.most_common(10)))
        list_final.append(('clusters', list_top))
        return dict(list_final)
