import random
import numpy as np
import matplotlib.pyplot as plt


class Labirinto:
    def __init__(self, m, inicio, objetivo):
        self.m = m
        self.inicio = inicio
        self.objetivo = objetivo
        self.movimentos = ["CIMA", "BAIXO", "ESQUERDA", "DIREITA"]
        
    def mover(self, posicao, movimento):
        mov = {"NONE":[0,0], "CIMA":[-1,0], "BAIXO":[1,0], "ESQUERDA":[0,-1], "DIREITA":[0,1]}
        
        nova_posicao = [0,0]
        nova_posicao[0] = posicao[0] + mov[movimento][0]
        nova_posicao[1] = posicao[1] + mov[movimento][1]
        
        if (nova_posicao[0] < 0
            or nova_posicao[0] >= len(self.m)
            or nova_posicao[1] < 0
            or nova_posicao[1] >= len(self.m[0])):
                return None # Retorna None se o movimento for inválido
        if self.m[nova_posicao[0]][nova_posicao[1]] == 0:
            return nova_posicao
        else:
            return None # Retorna None se o movimento for inválido


class Cromossomo():
    def __init__(self, labirinto):
        self.labirinto = labirinto
        self.genes = np.random.choice(labirinto.movimentos, 
                                      size=30)
        
    def avaliar(self, imprimir=False):
        atual = self.labirinto.inicio
        
        qtd_mov = 0
        qtd_mov_inv = 0
        for gene in self.genes:
            if atual == self.labirinto.objetivo:
                break
            qtd_mov += 1
            prox = self.labirinto.mover(atual, gene)
            if prox is not None:
                atual = prox
            else:
                qtd_mov_inv += 1
                
        dist = abs(atual[0] - self.labirinto.objetivo[0])
        dist += abs(atual[1] - self.labirinto.objetivo[1])
        
        if imprimir:
            print("qtd_mov=", qtd_mov)
            print("qtd_mov_inv=", qtd_mov_inv)
        
        #self.fitness = -1 * dist
        self.fitness = -1 * (dist + qtd_mov + qtd_mov_inv)


def avaliar_todos(populacao):
    for cromossomo in populacao:
        cromossomo.avaliar()

def crossover(p1, p2):
    n1 = Cromossomo(p1.labirinto)
    n2 = Cromossomo(p1.labirinto)
    
    pt_corte = random.randint(0, len(p1.genes) - 1)
    
    n1.genes[:pt_corte] = p1.genes[:pt_corte]
    n1.genes[pt_corte:] = p2.genes[pt_corte:]
    
    n2.genes[:pt_corte] = p2.genes[:pt_corte]
    n2.genes[pt_corte:] = p1.genes[pt_corte:]
    
    return n1, n2


def mutacao(cromossomo):
    if random.random() < 0.5:
        idx = random.randint(0, len(cromossomo.genes) - 1)
        cromossomo.genes[idx] = np.random.choice(cromossomo.labirinto.movimentos)

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

def ga(labirinto, n_populacao, n_geracoes):
    vetor_fitness = []
    
    populacao = [Cromossomo(labirinto) for _ in range(n_populacao)]
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


def main():

    map1 = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]
    
    maze1 = Labirinto(map1, [0,0], [9,9])

    map2 = [[0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]]

    maze2 = Labirinto(map2, [0,0], [3,5])

    solucao = ga(labirinto=maze1, 
                 n_populacao = 50, 
                 n_geracoes = 100)
    
    print(solucao.genes, solucao.fitness)
    solucao.avaliar(imprimir=True)

main()
