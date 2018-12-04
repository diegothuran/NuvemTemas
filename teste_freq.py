import nltk
import json

palavras = []
texto = ["eu achei um porqueira uma merda fidida", "ótimo produto para o meu cachorro"]
for t in texto:
    palavras.extend(nltk.word_tokenize(t))

freq = nltk.FreqDist(palavras)
print(freq.most_common(10))
teste = freq.most_common(10)
dici = dict(teste)
print(dici)
text = json.dump(freq.most_common(10))
print(text)