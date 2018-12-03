import nltk
import re
import processing.Utils as Utils
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class PreProcessing():
    def __init__(self):
        self.__acentos = Utils.ACENTOS
        self.__s_acentos = Utils.S_ACENTOS
        self.stop_words = set(stopwords.words("portuguese"))
        self.more_stopwords = Utils.MORE_STOPWORDS

    def PreprocessamentoSemStopWords(self,instancia):
        # remove links dos tweets
        # remove stopwords
        instancia = re.sub(r"http\S+", "", instancia).lower().replace(',', '').replace('.', '').replace(';','').replace('-','')
        stopwords = set(nltk.corpus.stopwords.words('portuguese'))
        palavras = [i for i in instancia.split() if not i in stopwords]
        return (" ".join(palavras))

    def removerStopWords(self,texto):
        texto = ' '.join([word for word in word_tokenize(texto) if word not in self.stop_words])
        texto = ' '.join([word for word in word_tokenize(texto) if word not in self.more_stopwords])
        texto.replace(',',' ').replace('.',' ').replace(';',' ').replace('!',' ').replace('?',' ').replace('"','')\
                    .replace('*',' ').replace('#',' ').replace('%',' ').replace('  ',' ').lower()
        return texto


    def Stemming(self,instancia):
        stemmer = nltk.stem.RSLPStemmer()
        palavras = []
        for w in instancia.split():
            palavras.append(stemmer.stem(w))
        return (" ".join(palavras))


    def Preprocessamento(self,instancia):
        # remove links, pontos, virgulas,ponto e virgulas dos tweets
        # coloca tudo em minusculo
        instancia = re.sub(r"http\S+", "", instancia).lower().replace(',', '').replace('.', '').replace(';', '').replace(
            '-', '').replace(':', '')
        return (instancia)

    def removerAcentos(self, texto):
        for i in range(0, len(self.__acentos)):
            texto = texto.replace(self.__acentos[i], self.__s_acentos[i])
        return texto
