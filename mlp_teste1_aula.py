from mlp_aula import MLP

treino_entradas = [[0,0], [0,1], [1,0], [1,1]]
treino_saidas = [[0], [1], [1], [0]]

rede_neural = MLP(2, 2, 1)
rede_neural.treinar(treino_entradas, treino_saidas)

for exemplo in treino_entradas:
    print(exemplo, round(rede_neural.predizer(exemplo)[0]))