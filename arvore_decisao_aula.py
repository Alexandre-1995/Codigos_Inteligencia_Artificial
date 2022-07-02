import math
from math import inf
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

class NoArvore():
    def __init__(self):
        self.atributo = None
        self.subarvores = {}
        
class ArvoreDecisao():
    def __init__(self, ds):
        self.raiz = self.__construir_arvore(ds)
        
    def __todos_rotulos_iguais(self, ds):
        colunas = list(ds.columns)
        coluna_classe = colunas[-1]
        valores_unicos = ds[coluna_classe].unique()
        return (len(valores_unicos) == 1)
    
    def __todos_atributos_usados(self, ds):
        colunas = list(ds.columns)
        return (len(colunas) == 1)
    
    def __obter_rotulo(self, ds):
        colunas = list(ds.columns)
        coluna_classe = colunas[-1]
        rotulo = ds[coluna_classe].value_counts().index[0]
        return rotulo
    
    def __obter_atributos(self, ds):
        colunas = list(ds.columns)
        del colunas[-1]
        return colunas
    
    def __entropia(self, ds):
        colunas = list(ds.columns)
        coluna_classe = colunas[-1]
        v = ds[coluna_classe].value_counts()
        qtd = len(ds)
        
        h = 0
        for valor in v:
            p = valor / qtd
            h += -1 * p * math.log2(p)
            
        return h
    
    
    def __obter_valores(self, ds, atributo):
        valores_unicos = ds[atributo].unique()
        return valores_unicos
    
    def __calcular_gi(self, ds, atributo):
        h = self.__entropia(ds)
        
        valores_atributo = self.__obter_valores(ds, atributo)
        qtd = len(ds)
        
        gi = h
        for valor in valores_atributo:
            condicao = (ds[atributo] == valor)
            ds_valor = ds[condicao]
            h_valor = self.__entropia(ds_valor)
            gi -= (len(ds_valor) / qtd) * h_valor
            
        return gi
    
    def __construir_arvore(self, ds):
        novo_no = NoArvore()
        
        if (self.__todos_rotulos_iguais(ds) 
            or self.__todos_atributos_usados(ds)):
            novo_no.rotulo = self.__obter_rotulo(ds)
            return novo_no
        
        atributos = self.__obter_atributos(ds)
        
        atributo_escolhido = None
        atributo_gi = -inf
        for atributo in atributos:
            gi = self.__calcular_gi(ds, atributo)
            if gi > atributo_gi:
                atributo_gi = gi
                atributo_escolhido = atributo
                
        novo_no.atributo = atributo_escolhido
        
        valores_atributo = self.__obter_valores(ds, atributo_escolhido)
        for valor in valores_atributo:
            condicao = (ds[atributo_escolhido] == valor)
            ds_valor = ds[condicao]
            ds_valor = ds_valor.drop(atributo_escolhido,
                                     axis=1,
                                     inplace=False)
            subarvore = self.__construir_arvore(ds_valor)
            novo_no.subarvores.update({ valor : subarvore })
            
        return novo_no
        
    def __predizer_r(self, exemplo, no):
        if no.atributo == None:
            return no.rotulo
        
        valor = exemplo[no.atributo]
        if valor in no.subarvores:
            return self.__predizer_r(exemplo, no.subarvores[valor])
        else:
            return None        
    
    def predizer(self, exemplo):
        return self.__predizer_r(exemplo, self.raiz)
   
    

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


classificador = ArvoreDecisao(ds_treino)

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