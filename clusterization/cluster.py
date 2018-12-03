from sklearn.cluster import KMeans
from processing.processing import PreProcessing
import csv
from sklearn.feature_extraction.text import TfidfVectorizer


class Cluster():
    def __init__(self):
        self.processo = PreProcessing()
        self.tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000, min_df=0.2, use_idf=True,
                                                ngram_range=(1, 3))

    # COLETANDO DADOS PARA CLUSTERIZAÇÂO
    def getDados(self):
        frases = []
        original = []
        with open('clusterization/lista_total.csv', 'r', encoding="utf8") as file:
            reader = csv.reader(file)
            for row in reader:
                frases.append(
                    self.processo.Stemming(self.processo.removerAcentos(self.processo.removerStopWords(row[0]))))
                original.append(row[0])
        return frases, original

    # VETORIZANDO BASE DE DADOS USANDO TFIDF
    def dadosVectorizer(self, frases):
        frases_vectorizer = self.tfidf_vectorizer.fit_transform(frases)
        return frases_vectorizer

    # CALCULAR QUANTIDADE DE CLUSTER PARA K_MEANS
    def calculoWcss(self, frases_vectorizer):
        wcss = []
        qtd_cluster = 0
        soma = 0
        min = 999999999999999999

        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
            kmeans.fit(frases_vectorizer)
            wcss.append(kmeans.inertia_)

        for ind in range(0, len(wcss)):
            soma = soma + wcss[ind]

        media = soma / len(wcss)

        for ind in range(0, len(wcss)):
            if (min > (abs(media - wcss[ind]))):
                min = abs(media - wcss[ind])
                qtd_cluster = ind + 1

        print("Quantidade Cluster: " + str(qtd_cluster))

        # GRÀFICO DO WCSS
        '''plt.plot(range(1, 11), wcss)
        plt.xlabel('Número de clusters')
        plt.ylabel('WCSS')

        plt.show()'''

        return qtd_cluster

    # EXECUTA CLUSTERIZAÇÂO K-MEANS
    def clusterizar(self):
        print("---------- CARREGAR DADOS ----------")
        frases, original = self.getDados()
        print("------------------------------------")

        print("---------- Vetorizar DADOS ----------")
        frases_vectorizer = self.dadosVectorizer(frases)
        print("-------------------------------------")

        print("---------- CALCULO CLUSTER ----------")
        qtd_cluster = self.calculoWcss(frases_vectorizer)
        print("-------------------------------------")

        print("---------- CLUSTER K-MEANS ----------")
        km = KMeans(n_clusters=qtd_cluster, init='k-means++')
        km.fit_predict(frases_vectorizer)
        print("-------------------------------------")

        return km, original
