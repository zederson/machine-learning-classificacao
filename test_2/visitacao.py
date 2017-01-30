from dados import carregar_acessos
from sklearn.naive_bayes import MultinomialNB

x,y = carregar_acessos()

treino_dados = x[:90]
treino_marcacoes = y[:90]

test_dados = x[-9:]
test_marcacoes = y[-9:]



modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)

tests = [[1,0,1], [0,1,0]]
result = modelo.predict(test_dados)

diferencas = result - test_marcacoes
acertos = [d for d in diferencas if d == 0]
taxa = 100.0 * len(acertos) / len(test_dados)

print(taxa)
