import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from math import inf

class NB():
    def __init__(self, ds):
        self.colunas = list(ds.columns)
        self.coluna_classe = self.colunas[-1]
        del self.colunas[-1]
        self.classes = ds[self.coluna_classe].unique()
        self.p_classe = {}
        self.p_atributo = {}
    
        for c in self.classes:
            condicao = (ds[self.coluna_classe] == c)
            ds_classe = ds[condicao]
            n_c = len(ds_classe)
            p_c = n_c / len(ds)
            self.p_classe.update({ c : p_c })
    
            for atributo in self.colunas:
                valores_atributo = ds[atributo].unique()
                
                for valor in valores_atributo:
                    p_a = self.__get_count_atributo(ds_classe, atributo, valor) / n_c
                    self.p_atributo.update({ (c, atributo, valor) : p_a })
                
    def __get_count_atributo(self, ds_classe, atributo, valor):
        condicao = (ds_classe[atributo] == valor)
        ds_atributo = ds_classe[condicao]
        return len(ds_atributo)
    
    def predizer(self, exemplo):
        c_pred = -1
        c_max = -inf
        
        for c in self.classes:
            p_atual = self.p_classe[c]
            
            for atributo in self.colunas:
                valor = exemplo[atributo]
                p_atual *= self.p_atributo[ (c, atributo, valor) ]
            if p_atual > c_max:
                c_max = p_atual
                c_pred = c
            
            
        return c_pred


ds = pd.read_csv("iris.data", sep=",", header=None)
ds_treino = ds.sample(100)
ds_teste = ds.drop(ds_treino.index)
ds_treino.reset_index(drop=True, inplace=True)
ds_teste.reset_index(drop=True, inplace=True)

kbins = KBinsDiscretizer(n_bins=3,
                         encode='ordinal',
                         strategy='uniform')

ds_temp = ds_treino.values[:, :-1]
kbins.fit(ds_temp)
ds_temp = kbins.transform(ds_temp)
ds_temp = pd.DataFrame(ds_temp)
ds_temp[4] = ds_treino[4]
ds_treino = ds_temp

ds_temp = ds_teste.values[:, :-1]
ds_temp = kbins.transform(ds_temp)
ds_temp = pd.DataFrame(ds_temp)
ds_temp[4] = ds_teste[4]
ds_teste = ds_temp


classificador = NB(ds_treino)

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