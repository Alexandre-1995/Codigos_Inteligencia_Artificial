import numpy as np

class Neuronio():
    def __init__(self, n_entradas):
        self.b = 0
        v = 1.0 / np.sqrt(n_entradas)
        self.w = [np.random.uniform(-v, v) for _ in range(n_entradas)]
        
    def __funcao_ativacao(self, u):
        y = 1.0 / (1.0 + np.exp(-u))
        return y
        
    def obter_saida(self, entradas):
        u = self.b
        for i in range(len(entradas)):
            u += entradas[i] * self.w[i]
        y = self.__funcao_ativacao(u)
        return y


class MLP():
    def __init__(self, n_entradas, n_oculta, n_saida):
        self.camada_oculta = [Neuronio(n_entradas) for _ in range(n_oculta)]
        self.camada_saida = [Neuronio(n_oculta) for _ in range(n_saida)]
        
    def predizer(self, exemplo):
        y_h = [n.obter_saida(exemplo) for n in self.camada_oculta]
        y_s = [n.obter_saida(y_h) for n in self.camada_saida]
        return y_s
    
    def treinar(self, treino_entradas, treino_saidas, eta = 0.1, max_erro = 0.001):
        erro_medio = 1
        while erro_medio > max_erro:
            erro_medio = 0
            for i_exemplo in range(len(treino_entradas)):
                # exemplo
                entrada = treino_entradas[i_exemplo]
                saida = treino_saidas[i_exemplo]
                
                # obter saída
                y_h = [n.obter_saida(entrada) for n in self.camada_oculta]
                y_s = [n.obter_saida(y_h) for n in self.camada_saida]
                
                # cálculo do erro
                erro_neuronios = [0] * len(self.camada_saida)
                erro_exemplo = 0
                for k in range(len(self.camada_saida)):
                    erro_neuronios[k] = saida[k] - y_s[k]
                    erro_exemplo += (erro_neuronios[k] ** 2)
                erro_exemplo = 0.5 * erro_exemplo
                erro_medio += erro_exemplo
                
                # calcular deltas
                delta_s = [0] * len(self.camada_saida)
                for k in range(len(self.camada_saida)):
                    delta_s[k] = erro_neuronios[k] * y_s[k] * (1.0 - y_s[k])
                    
                delta_h = [0] * len(self.camada_oculta)
                for j in range(len(self.camada_oculta)):
                    delta = y_h[j] * (1.0 - y_h[j])
                    soma = 0
                    for k in range(len(self.camada_saida)):
                        soma += delta_s[k] * self.camada_saida[k].w[j]
                    delta = delta * soma
                    delta_h[j] = delta
                
                # ajuste dos pesos
                for k in range(len(self.camada_saida)):
                    self.camada_saida[k].b += eta * delta_s[k]
                    for j in range(len(self.camada_saida[k].w)):
                        self.camada_saida[k].w[j] += eta * delta_s[k] * y_h[j]
                        
                for j in range(len(self.camada_oculta)):
                    self.camada_oculta[j].b += eta * delta_h[j]
                    for i in range(len(self.camada_oculta[j].w)):
                        self.camada_oculta[j].w[i] += eta * delta_h[j] * entrada[i]
                
            erro_medio = erro_medio / len(treino_entradas)
            print(erro_medio)
        
        
        
        