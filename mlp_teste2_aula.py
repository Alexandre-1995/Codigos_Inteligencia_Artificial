import pandas as pd
from mlp_aula import MLP

ds = pd.read_csv("wine.data", sep=",", header=None)
ds[14] = [0] * len(ds)
ds[15] = [0] * len(ds)
ds[16] = [0] * len(ds)
ds.loc[ds[0]==1, 14] = 1
ds.loc[ds[0]==2, 15] = 1
ds.loc[ds[0]==3, 16] = 1

ds_treino = ds.sample(120)
ds_teste = ds.drop(ds_treino.index)

treino_entradas = ds_treino.drop(columns=[0,14,15,16]).values.tolist()
treino_saidas = ds_treino.drop(columns=range(14)).values.tolist()

teste_entradas = ds_teste.drop(columns=[0,14,15,16]).values.tolist()
teste_saidas = ds_teste.drop(columns=range(14)).values.tolist()

# ( (x - vmin) / (vmax - vmin) ) * (max - min) + min
for i_atributo in range(13):
    vmin = min(ds_treino[i_atributo + 1])
    vmax = max(ds_treino[i_atributo + 1])
    for i in range(len(treino_entradas)):
        v = (treino_entradas[i][i_atributo] - vmin) / (vmax - vmin)
        treino_entradas[i][i_atributo] = v
    for i in range(len(teste_entradas)):
        v = (teste_entradas[i][i_atributo] - vmin) / (vmax - vmin)
        teste_entradas[i][i_atributo] = v


rede_neural = MLP(13, 5, 3)
rede_neural.treinar(treino_entradas, treino_saidas)

acertos = 0
erros = 0
for i in range(len(teste_entradas)):
    exemplo = teste_entradas[i]
    pred = rede_neural.predizer(exemplo)
    if ((round(pred[0]) == teste_saidas[i][0])
            and (round(pred[1]) == teste_saidas[i][1])
            and (round(pred[2]) == teste_saidas[i][2])):
        acertos += 1
    else:
        erros += 1
        
print("Acertos =", acertos, "Erros = ", erros)