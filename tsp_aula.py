import numpy as np
import random
import matplotlib.pyplot as plt

class Mapa():
    def __init__(self):
        self.cidades = {}

    def adiciona_cidade(self, posicao):
        num_cidade = len(self.cidades)+1
        self.cidades.update({num_cidade : posicao})

    def calcula_distancia(self, c1, c2):
        pos1 = self.cidades[c1]
        pos2 = self.cidades[c2]
        dist = (pos1[0] - pos2[0])**2
        dist += (pos1[1] - pos2[1])**2
        dist = dist ** 0.5
        return dist

class Cromossomo():
    def __init__(self, mapa):
        self.mapa = mapa
        self.genes = np.random.permutation(np.arange(1, len(mapa.cidades) + 1))
        
    def avaliar(self):
        dist = 0
        for i in range(0, len(self.genes) - 1):
            dist += self.mapa.calcula_distancia(self.genes[i], 
                                                self.genes[i + 1])
        dist += self.mapa.calcula_distancia(self.genes[0], 
                                            self.genes[len(self.genes)-1])
        self.fitness = -1 * dist
       
def avaliar_todos(populacao):
    for cromossomo in populacao:
        cromossomo.avaliar()

def order_crossover(p1, p2, pt_corte1, pt_corte2):
    novo_cromossomo = Cromossomo(p1.mapa)
    novo_cromossomo.genes = np.repeat(-1, len(p1.genes))
    
    novo_cromossomo.genes[pt_corte1:(pt_corte2+1)] = p1.genes[pt_corte1:(pt_corte2+1)]
    
    p2_indices = list(range(pt_corte2 + 1, len(p1.genes)))
    p2_indices = p2_indices + list(range(0, pt_corte2 + 1))
    
    n_idx = pt_corte2+1
    for i in p2_indices:
        gene = p2.genes[i]
        if gene not in novo_cromossomo.genes:
            if n_idx >= len(novo_cromossomo.genes):
                n_idx = 0
            novo_cromossomo.genes[n_idx] = gene
            n_idx += 1
            
    return novo_cromossomo

def crossover(p1, p2):
    pt_corte1 = random.randint(0, len(p1.genes) - 1)
    pt_corte2 = random.randint(pt_corte1, len(p1.genes) - 1)
    
    n1 = order_crossover(p1, p2, pt_corte1, pt_corte2)
    n2 = order_crossover(p2, p1, pt_corte1, pt_corte2)
    
    return n1, n2

def mutacao(cromossomo):
    if random.random() < 0.5:
        i_1 = random.randint(0, len(cromossomo.genes) - 1)
        i_2 = random.randint(0, len(cromossomo.genes) - 1)
        temp = cromossomo.genes[i_1]
        cromossomo.genes[i_1] = cromossomo.genes[i_2]
        cromossomo.genes[i_2] = temp

def operadores_variacao(selecionados):
    novos_cromossomos = []
    for i in range(len(selecionados) // 2):
        p1 = selecionados[2 * i]
        p2 = selecionados[2 * i + 1]
        n1, n2 = crossover(p1, p2)
        mutacao(n1)
        mutacao(n2)
        novos_cromossomos.append(n1)
        novos_cromossomos.append(n2)
    return novos_cromossomos
        

def selecionar(populacao, n_elitismo):
    lista_selecionados = []
    for _ in range(len(populacao) - n_elitismo):
        c1 = np.random.choice(populacao)
        c2 = np.random.choice(populacao)
        selecionado = c1 if c1.fitness > c2.fitness else c2
        lista_selecionados.append(selecionado)
    return lista_selecionados

def melhor_solucao(populacao):
    populacao.sort(key = lambda x : -x.fitness)
    return populacao[0]

def ga(mapa, n_populacao, n_geracoes):
    vetor_fitness = []
    
    populacao = [Cromossomo(mapa) for _ in range(n_populacao)]
    avaliar_todos(populacao)
    vetor_fitness.append(melhor_solucao(populacao).fitness)
    
    for _ in range(n_geracoes):
        selecionados = selecionar(populacao, 2)
        novos_cromossomos = operadores_variacao(selecionados)
        avaliar_todos(novos_cromossomos)
        
        populacao.sort(key = lambda x : -x.fitness)
        e1 = populacao[0]
        e2 = populacao[1]
        
        populacao = novos_cromossomos
        populacao.append(e1)
        populacao.append(e2)
        
        vetor_fitness.append(melhor_solucao(populacao).fitness)
        
    plt.plot(vetor_fitness)

    return melhor_solucao(populacao)


mapa = Mapa()
mapa.adiciona_cidade((1,8))
mapa.adiciona_cidade((0,0))
mapa.adiciona_cidade((15,11))
mapa.adiciona_cidade((20,4))
mapa.adiciona_cidade((10,5))
mapa.adiciona_cidade((11,4))
mapa.adiciona_cidade((20,64))
mapa.adiciona_cidade((17,1))
mapa.adiciona_cidade((90,56))

solucao = ga(mapa, 50, 100)
print(solucao.genes)

