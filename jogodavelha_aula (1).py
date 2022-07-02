# GUI e Board baseados em (Autor: Maurits van der Schee):
# https://www.leaseweb.com/labs/2013/12/python-tictactoe-tk-minimax-ai/

from tkinter import Tk, Button
from tkinter.font import Font
from copy import deepcopy
from random import randint
from abc import ABC, abstractmethod
from time import time
from math import inf

class Board:
    def __init__(self):
        self.human = "X"
        self.computer = "O"
        self.empty = "."

        self.cells = {}
        for x in range(3):
            for y in range(3):
                self.cells[x,y] = self.empty

    def resultado(self, x, y, value):
        new_board = Board()
        new_board.__dict__ = deepcopy(self.__dict__)
        new_board.cells[x,y] = value
        return new_board

    def utilidade(self):
        for x in range(3):
            v1 = self.cells[x,0]
            v2 = self.cells[x,1]
            v3 = self.cells[x,2]
            if (v1 == v2 and v1 == v3):
                if (v1 == self.human):
                    return -1
                elif (v1 == self.computer):
                    return +1

        for y in range(3):
            v1 = self.cells[0, y]
            v2 = self.cells[1, y]
            v3 = self.cells[2, y]
            if (v1 == v2 and v1 == v3):
                if (v1 == self.human):
                    return -1
                elif (v1 == self.computer):
                    return +1

        v1 = self.cells[0, 0]
        v2 = self.cells[1, 1]
        v3 = self.cells[2, 2]
        if (v1 == v2 and v1 == v3):
            if (v1 == self.human):
                return -1
            elif (v1 == self.computer):
                return +1

        v1 = self.cells[0, 2]
        v2 = self.cells[1, 1]
        v3 = self.cells[2, 0]
        if (v1 == v2 and v1 == v3):
            if (v1 == self.human):
                return -1
            elif (v1 == self.computer):
                return +1
        
        return 0

    def terminal(self):
        if (self.utilidade() != 0):
            return True
        for (x,y) in self.cells:
            if self.cells[x,y] == self.empty:
                return False
        return True

    def movimentos(self):
        movimentos = []
        for x in range(3):
            for y in range(3):
                if self.cells[x,y] == self.empty:
                    movimentos.append((x,y))
        return movimentos

    
class JogadorAutomatico(ABC):
    @abstractmethod
    def jogar(self, board):
        return -1, -1

class JogadorAleatorio(JogadorAutomatico):
    def jogar(self, board):
        while (True):
            x = randint(0, 2)
            y = randint(0, 2)
            if board.cells[x,y] == board.empty:
                return x,y
        return -1, -1

class JogadorMinimax(JogadorAutomatico):
    def jogar(self, board):        
        max_v = -inf
        decisao = None
        
        acoes = board.movimentos()
        for acao in acoes:
            novo_board = board.resultado(acao[0], acao[1], 
                                         board.computer)
            v = self.get_min_minimax(novo_board)
            if v > max_v:
                decisao = acao
                max_v = v
        
        return decisao
    
        
    def get_min_minimax(self, board):
        if board.terminal():
            return board.utilidade()
        
        min_v = +inf
        acoes = board.movimentos()
        for acao in acoes:
            novo_board = board.resultado(acao[0], acao[1], 
                                         board.human)
            v = self.get_max_minimax(novo_board)
            if v < min_v:
                min_v = v
                
        return min_v
    
        
    def get_max_minimax(self, board):
        if board.terminal():
            return board.utilidade()
        
        max_v = -inf
        acoes = board.movimentos()
        for acao in acoes:
            novo_board = board.resultado(acao[0], acao[1], 
                                         board.computer)
            v = self.get_min_minimax(novo_board)
            if v > max_v:
                max_v = v
                
        return max_v
    
        

class JogadorMinimaxAlfaBeta(JogadorAutomatico):
    def jogar(self, board):
        max_v = -inf
        decisao = None
        
        acoes = board.movimentos()
        for acao in acoes:
            novo_board = board.resultado(acao[0], acao[1], 
                                         board.computer)
            v = self.get_min_alfa_beta(novo_board,
                                       max_v, +inf)
            if v > max_v:
                decisao = acao
                max_v = v
        
        return decisao
    
        
    def get_min_alfa_beta(self, board, alfa, beta):
        if board.terminal():
            return board.utilidade()
        
        min_v = +inf
        acoes = board.movimentos()
        for acao in acoes:
            novo_board = board.resultado(acao[0], acao[1], 
                                         board.human)                                         
            v = self.get_max_alfa_beta(novo_board, alfa, beta)
            if v < min_v:
                min_v = v
            if min_v <= alfa:
                return min_v
            if min_v < beta:
                beta = min_v
                
        return min_v
    
        
    def get_max_alfa_beta(self, board, alfa, beta):
        if board.terminal():
            return board.utilidade()
        
        max_v = -inf
        acoes = board.movimentos()
        for acao in acoes:
            novo_board = board.resultado(acao[0], acao[1], 
                                         board.computer)
            v = self.get_min_alfa_beta(novo_board, alfa, beta)
            if v > max_v:
                max_v = v
            if max_v >= beta:
                return max_v
            if max_v > alfa:
                alfa = max_v
                
        return max_v
    
    
		
class GUI:
 
    def __init__(self, jogador):
        self.jogador = jogador
        self.app = Tk()
        self.app.title('TicTacToe')
        self.app.resizable(width=False, height=False)
        self.font = Font(family="Helvetica", size=32)
        self.buttons = {}
        for x in range(3):
            for y in range(3):
                handler = lambda x=x, y=y: self.resultado(x, y)
                button = Button(self.app, command=handler, font=self.font, width=2, height=1)
                button.grid(row=y, column=x)
                self.buttons[x,y] = button
        handler = lambda: self.reset()
        button = Button(self.app, text='reset', command=handler)
        button.grid(row=4, column=0, columnspan=3, sticky="WE")
        
        # Instance of the Board
        self.board = Board()
        
        self.update()

    def reset(self):
        self.board = Board()
        self.update()
 
    def resultado(self,x,y):
        # jogada do humano
        self.board = self.board.resultado(x, y, self.board.human)
        self.update()

        # jogada do computador
        if (not self.board.terminal()):            
            inicio = time()
            xy = self.jogador.jogar(self.board)
            print("Tempo para decisao:", time() - inicio)
            self.board = self.board.resultado(*xy, self.board.computer)
            self.update()

    def update(self):
        for (x,y) in self.board.cells:
            text = self.board.cells[x,y]
            self.buttons[x,y]['text'] = text
            self.buttons[x,y]['disabledforeground'] = 'black'
            if text==self.board.empty:
                self.buttons[x,y]['state'] = 'normal'
            else:
                self.buttons[x,y]['state'] = 'disabled'
        
        if self.board.terminal():
            for (x,y) in self.board.cells:
                self.buttons[x,y]['disabledforeground'] = 'red'
                self.buttons[x,y]['state'] = 'disabled'
            for (x,y) in self.board.cells:
                self.buttons[x,y].update()
 
    

    def mainloop(self):
        self.app.mainloop()


#GUI(JogadorAleatorio()).mainloop()		
#GUI(JogadorMinimax()).mainloop()
GUI(JogadorMinimaxAlfaBeta()).mainloop()