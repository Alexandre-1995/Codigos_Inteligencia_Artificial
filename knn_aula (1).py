import pandas as pd

class KNN():
    def __init__(self, ds):
        self.ds = ds
        self.n_colunas = len(ds.columns)
        
    def __distancia(self, exemplo1, exemplo2):
        soma = 0
        for i in range(len(exemplo1)):
            soma += (exemplo1[i] - exemplo2[i]) ** 2
        return soma ** 0.5
        
    def predizer(self, exemplo, k = 3):
        dist = []
        for i in range(len(self.ds)):
            tmp = self.__distancia(exemplo,
                                   self.ds.iloc[i].values[:(self.n_colunas-1)])
            dist.append(tmp)
            
        self.ds[self.n_colunas] = dist
        ds_ord = self.ds.sort_values(by=[self.n_colunas])
        ds_ord = ds_ord[:k]
        return ds_ord[self.n_colunas - 1].value_counts().index[0]



ds = pd.read_csv("iris.data", sep=",", header=None)
ds_treino = ds.sample(100)
ds_teste = ds.drop(ds_treino.index)
ds_treino.reset_index(drop=True, inplace=True)
ds_teste.reset_index(drop=True, inplace=True)

classificador = KNN(ds_treino)

acertos=0
erros=0
for i in range(len(ds_teste)):
    exemplo = ds_teste.iloc[i]
    rotulo_verd = exemplo[4]
    del exemplo[4]
    rotulo_pred = classificador.predizer(exemplo)
    if rotulo_verd == rotulo_pred:
        acertos += 1
    else:
        erros += 1

print("Acertos =", acertos, " Erros =", erros)