import numpy as np
import pandas as pd
import nltk
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from flask import Flask, request, jsonify, after_this_request

#iniciando a API
app = Flask(__name__)

#função para fracionar frases em palavras
def Tokenize (frase):
    frase = frase.lower()
    frase = nltk.word_tokenize(frase, 'portuguese')
    return frase

#função para reduzir a pavavra ao seu radical
def Stemming (frase):
    stemmer = RSLPStemmer()
    sentence = []
    for palavra in frase:
        sentence.append(stemmer.stem(palavra.lower()))
    return sentence

#função para remover as stopwords
def RemoveStopWords (frase):
    stopwords = nltk.corpus.stopwords.words('portuguese')
    sentence = []
    for palavra in frase:
        if palavra not in stopwords:
            sentence.append(palavra)
    return sentence

#função para chamar todas as funções anteriores 
def tratamentoTexto (frases):
    lista = []
    for frase in frases:
        
        frase = Tokenize(frase)
        frase = RemoveStopWords(frase)
        frase = Stemming(frase)
        frase = " ".join(frase) # formar a frase novamente
        lista.append (frase)
    return lista
    
def tratamentoFraseUnica (frase):

    frase = Tokenize(frase)
    frase = Stemming(frase)
    frase = RemoveStopWords(frase)
    frase = " ".join(frase) # formar a frase novamente
    return frase


def AnaliseTweet(tweet):

    file = ("https://raw.githubusercontent.com/GabrielOliveiraBR/CSV_Tweets/main/DadosJuntos.csv")

    data = pd.read_csv(file)

    data = data.dropna() # remove linhas vazias
    classificacao = data.racismo # armazena na variável categoria apenas os itens da coluna categoria do dataset
    frases = data.text # armazena na variável frases apenas os itens da coluna frases do dataset
    textos = pd.Series(tratamentoTexto(frases)) #transformando a lista em uma Series
    df = pd.DataFrame(data=dict(text=textos, classificacao=classificacao)) #criar DF com o texto já tratado
    #print(df.shape)

    X_train, X_test, y_train, y_test = train_test_split(df.text, df.classificacao) #criando os DFs de treinamento e teste


    # Criação do modelo
    modelo = make_pipeline(TfidfVectorizer(), MultinomialNB())
    # Treinamento
    modelo.fit(X_train, y_train)
    # Predição das categorias dos textos de teste
    y_pred = modelo.predict(X_test)


    frase = tratamentoFraseUnica(tweet)
    #predicao = modelo.predict(frase)
    predicao = modelo.predict([frase])
    racis = str(predicao[0])
    return racis

#recebe o texto da extensão
@app.route('/analise', methods=['GET'])
def ChamarAnalise():
    @after_this_request
    def add_header(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    
    
    #separa apenas o texto da url
    txt = request.args.get('texto')
    ras = AnaliseTweet(txt)
    jsonResp = {'racista': ras} #coloca o resultado em um json
    
    return jsonify(jsonResp) #retorna o resultado para extensão 

if __name__ == '__main__':
    app.run(host='localhost', port=8989, debug=True)
