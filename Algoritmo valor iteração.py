# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 21:39:12 2021

@author: alexa
"""
import numpy as np

r=-100
R = np.array([[r, -1, 10],[-1, -1, -1],[-1, -1, -1]])
Us = np.array([[0, 0, 0],[0, 0, 0],[0, 0, 0]])
S=[(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
pifinal=np.array([['adddddd', 'adddddd', 'adddddd'],['adddddd', 'adddddd', 'adddddd'],['adddddd', 'adddddd', 'adddddd']])


eps=0.3
gamma=0.99

def valueIteration(R, Us, eps):
    delta = eps*(1 - gamma)/gamma + 1
    U = Us
    while delta > eps*(1 - gamma)/gamma:
        U = Us
        delta = 0
        for s in S:
            Us[s] = R[s] + gamma*maxAct(U,s)
            delta = max(delta, abs(Us[s] - U[s]))
    return U

def maxAct(U,s):
    cima=s[0]-1
    if cima <= 0:
        cima=0
    baixo=s[0]+1
    if baixo  >= len(Us):
        baixo=len(Us)-1
    direita = s[1]+1
    if direita >= len(Us[0]):
        direita=len(Us[0])-1
    esquerda=s[1]-1
    if esquerda <= 0 :
        esquerda=0
        
    U1=0.8*U[cima, s[1]] + 0.1*U[s[0], direita] + 0.1*U[s[0], esquerda]
    U2=0.8*U[baixo, s[1]] + 0.1*U[s[0], direita] + 0.1*U[s[0], esquerda]
    U3=0.8*U[s[0], direita] + 0.1*U[cima, s[1]] + 0.1*U[baixo, s[1]]
    U4=0.8*U[s[0], esquerda] + 0.1*U[cima, s[1]] + 0.1*U[baixo, s[1]]
    
    return(max(U1, U2, U3, U4))

def politica(UFINAL,S):
    for s in S:
        cima=s[0]-1
        if cima <= 0:
            cima=0       
        baixo=s[0]+1
        if baixo  >= len(Us):
            baixo=len(Us)-1 
        direita = s[1]+1
        if direita >= len(Us[0]):
            direita=len(Us[0])-1
        
        esquerda=s[1]-1
        if esquerda <= 0 :
            esquerda=0
    
        U_cima=UFINAL[cima, s[1]] 
        U_baixo=UFINAL[baixo, s[1]]
        U_direita=UFINAL[s[0], direita]
        U_esquerda=UFINAL[s[0], esquerda]

        pi={U_cima:'cima', U_baixo:'baixo', U_direita:'direita', U_esquerda:'esquerda'}
        
        pifinal[s]=pi[max(pi.keys())]
    
    return pifinal

print(R)
UFINAL= valueIteration(R, Us, eps)
print(UFINAL)
piÓtimo = politica(UFINAL,S)
print(piÓtimo)