"""
An example implementation of the abstract Node class for use in MCTS
If you run this file then you can play against the computer.
A tic-tac-toe board is represented as a tuple of 9 values, each either None,
True, or False, respectively meaning 'empty', 'X', and 'O'.
The board is indexed by row:
0 1 2
3 4 5
6 7 8
For example, this game board
O - X
O X -
X - -
corrresponds to this tuple:
(False, None, True, False, True, None, True, None, None)
"""
import copy
from collections import namedtuple
from random import choice, getrandbits
from MCTS import MCTS, Node
import numpy as np
import math
from scipy.stats import norm
#from alphazero.NNet import NNetWrapper
#from MCTSaz import MCTSaz
from game.TicTacToeGame import TicTacToeGame
import pickle

_TTTB = namedtuple("TicTacToeBoard", "tup turn winner terminal")

states=dict()
states_cache_bvoi=dict()
previous_tree = None
alphazero_agent=None


def load_tictactoe_alphazero():
    alphazero_agent = NNetWrapper()
    alphazero_agent.load_checkpoint('./pretrained_models','best-25eps-25sim-10epch.pth.tar')








# Inheriting from a namedtuple is convenient because it makes the class
# immutable and predefines __init__, __repr__, __hash__, __eq__, and others
class TicTacToeBoard(Node):

    def __init__(self, max, tup, turn, winner, terminal, meanvalue, depth):
        standard_derv=(10-depth)*0.3/10
        self.is_max=max
        self.tup=tup
        self.marked=0
        self.turn=turn
        self.winner=winner
        self.terminal=terminal
        self.meanvalue=meanvalue
        self.depth=depth
        self.hash = getrandbits(128)
        self.probability_density = lambda x : norm.pdf(x, loc=meanvalue, scale=standard_derv, size=None)
        self.buckets=[]
        num_of_buckets = 13
        j = 0
        for i in np.linspace(0,1,num_of_buckets + 2):
            self.buckets.append((norm.ppf(i,loc=meanvalue, scale=standard_derv), j/num_of_buckets))
            j = j + 1
        self.buckets=self.buckets[1:-1]
        def dist(x):
            if self.buckets[0]>x:
                return self.buckets[0]
            for i in range(1, len(self.buckets)):

                if x<self.buckets[i]:
                    d1=x-self.buckets[i-1]
                    d2=self.buckets[i]-x
                    if d1>d2:
                        return self.buckets[i-1]
                    else:
                        return self.buckets[i]
            return self.buckets[-1]

        #self.distribution=lambda x : dist(norm.rvs(loc=meanvalue,scale=standard_derv, size=None))






    def find_children(board):
        if board.terminal:  # If the game is finished then no moves can be made
            return set()
        # Otherwise, you can make a move in each of the empty spots
        return {
            board.make_move(i) for i, value in enumerate(board.tup) if value == 0
        }

    def find_children_bvoi(board, distribution_mode="NN"):
        if board.terminal:  # If the game is finished then no moves can be made
            return set()
        # Otherwise, you can make a move in each of the empty spots
        return {
            board.make_move_bvoi(i) for i, value in enumerate(board.tup) if value == 0
        }



    def find_random_child(board):
        if board.terminal:
            return None  # If the game is finished then no moves can be made
        empty_spots = [i for i, value in enumerate(board.tup) if value == 0]
        return board.make_move(choice(empty_spots))

    def find_random_child_bvoi(board, distribution_mode = "NN"):
        if board.terminal:
            return None  # If the game is finished then no moves can be made
        empty_spots = [i for i, value in enumerate(board.tup) if value == 0]
        return board.make_move_bvoi(choice(empty_spots))

    def reward(board):
        if not board.terminal:
            raise RuntimeError(f"reward called on nonterminal board {board}")
        if board.winner is board.turn:
            # It's your turn and you've already won. Should be impossible.
            raise RuntimeError(f"reward called on unreachable board {board}")
        if board.turn is (not board.winner):
            return 0  # Your opponent has just won. Bad.
        if board.winner is None:
            return 0.5  # Board is a tie
        # The winner is neither True, False, nor None
        raise RuntimeError(f"board has unknown winner type {board.winner}")

    def is_terminal(board):
        return board.terminal

    def make_move(board, index):
        tup = board.tup[:index] + ((1,) if board.turn else (-1,)) + board.tup[index + 1 :]
        state=states.get(tup)
        if state is not None:
            return state
        turn = not board.turn
        winner = _find_winner(tup)
        is_terminal = (winner is not None) or not any(v == 0 for v in tup)
        ret = TicTacToeBoard(not board.is_max, tup, turn, winner, is_terminal, 0, board.depth + 1)
        states[tup] = ret
        return ret


    def make_move_bvoi(board, index):
        tup = board.tup[:index] + ((1,) if board.turn else (-1,)) + board.tup[index + 1 :]
        state=states_cache_bvoi.get(tup)
        if state is not None:
            return state
        turn = not board.turn
        winner = _find_winner(tup)
        is_terminal = (winner is not None) or not any(v == 0 for v in tup)
        to_simulate = TicTacToeBoard(not board.is_max, tup, turn, winner, is_terminal, 0 , board.depth + 1)
        mcts = MCTS(to_simulate)
        sum = 0
        num_sims = 25

        for i in range(num_sims):
            sum += mcts.simulate(to_simulate, invert_reward = False)

        if not board.is_max:
            meanvalue = sum / num_sims
        else:
            meanvalue = (num_sims - sum) / num_sims

        ret = TicTacToeBoard(not board.is_max, tup, turn, winner, is_terminal, meanvalue, board.depth + 1)

        states[tup] = ret
        return ret


    def make_move_bvoi_NN(board, index):
        tup = board.tup[:index] + ((1,) if board.turn else (-1,)) + board.tup[index + 1 :]
        state=states_cache_bvoi.get(tup)
        if state is not None:
            return state
        turn = not board.turn
        winner = _find_winner(tup)
        is_terminal = (winner is not None) or not any(v == 0 for v in tup)
        if not board.is_max:
            tup2=tup
        else:
            tup2=[-1*i for i in tup]

        X = np.asarray(tup2).astype('float32')
        X = np.reshape(X, (3, 3,))

        meanvalue = alphazero_agent.predict(X)[1][0]# 0 is pi and 1 is v
        if not not board.is_max:
            meanvalue = -meanvalue
        ret = TicTacToeBoard(not board.is_max, tup, turn, winner, is_terminal, meanvalue , board.depth + 1)
        states[tup] = ret
        return ret


    def to_pretty_string(board):
        to_char = lambda v: ("X" if v == 1 else ("O" if v == -1 else " "))
        rows = [
            [to_char(board.tup[3 * row + col]) for col in range(3)] for row in range(3)
        ]
        return (
            "\n  1 2 3\n"
            + "\n".join(str(i + 1) + " " + " ".join(row) for i, row in enumerate(rows))
            + "\n"
        )


def play_game(mode="uct", mode2 ="uct"):
    board = new_tic_tac_toe_board()
    tree = MCTS(board, mode=mode)
    tree2 = MCTS(board, mode=mode2)
    game=TicTacToeGame()
    #rival=MCTSaz(game,alphazero_agent)
    #prob, v = alphazero_agent.predict(np.asarray(board.tup).astype('float32').reshape((3, 3)))

    print(board.to_pretty_string())
    while True:
        #row_col = input("enter row,col: ")
        #row, col = map(int, row_col.split(","))
        #index = 3 * (row - 1) + (col - 1)
        #if board.tup[index] != 0:
        #    print("Invalid move")
        #    row_col = input("enter row,col: ")
        #    row, col = map(int, row_col.split(","))
        #    index = 3 * (row - 1) + (col - 1)
        #board = board.make_move(index)
        #print(board.to_pretty_string())
        #if board.terminal:
        #    break
        # You can train as you go, or only at the beginning.
        # Here, we train as we go, doing fifty rollouts each turn.
        for i in range(50):
            tree.do_rollout(board)
        board = tree.choose(board)
        print(board.to_pretty_string())
        if board.terminal:
            break
        for i in range(300):
            tree2.do_rollout(board)
        board = tree2.choose(board)
        print(board.to_pretty_string())
        
        if board.terminal:
            break
    return board


def _winning_combos():
    for start in range(0, 9, 3):  # three in a row
        yield (start, start + 1, start + 2)
    for start in range(3):  # three in a column
        yield (start, start + 3, start + 6)
    yield (0, 4, 8)  # down-right diagonal
    yield (2, 4, 6)  # down-left diagonal


def _find_winner(tup):
    "Returns None if no winner, True if X wins, False if O wins"
    for i1, i2, i3 in _winning_combos():
        v1, v2, v3 = tup[i1], tup[i2], tup[i3]
        if -1 == v1 == v2 == v3:
            return False
        if 1 == v1 == v2 == v3:
            return True
    return None


def new_tic_tac_toe_board():
    return TicTacToeBoard(True, (0,) * 9, True, None, False,0, 0)

def simulate(node, dict):
    path = []
    invert_reward = False
    while True:
        path.append((node))
        if node.is_terminal():
            reward = node.reward()
            reward = 1 - reward if invert_reward else reward
            break
        node = node.find_random_child()
                #print(node.is_terminal())
                #print(node.get_legal_moves(1 if node.is_max else -1))
        invert_reward = not invert_reward
    path2 = reversed(path)
    for node in path2:
        n = node.tup
        if dict.get(n) is None:
            dict[n] = (reward, 1)
        else:
            dict[n] = ((dict[n][0]*dict[n][1] + reward)/(dict[n][1] + 1), dict[n][1] + 1)


def simulate_until_no_tomorrow(load = False):

    if load:
        f = open("weak_heuristic", "rb")
        result_dict = pickle.load(f)
        f.close()
    else:
        result_dict = {}

    init = new_tic_tac_toe_board()
    for _ in range(10):
        simulate(init, result_dict)
    f = open("weak_heuristic", "wb")
    pickle.dump(result_dict, f)
    f.close()
if __name__ == "__main__":

    #alphazero_agent = NNetWrapper()
    #alphazero_agent.load_checkpoint('./pretrained_models/alternative', 'best-25eps-25sim-10epch.pth.tar')

    import time

    start = time.time()
    sum = 100
    for i in range(0,25):
        fail=0
        board=play_game(mode="bvoi-greedy", mode2= "uct")
        
        if board.reward() != 0.5:
            if not board.is_max:
                print("Kawabanga")
            else:
                print("Pikapika")
            print(board.to_pretty_string())
            fail = 1
        sum=sum-fail
    print("mikapika")
    for i in range(0,25):
        fail=0
        board=play_game(mode="uct", mode2= "bvoi-greedy")
        if board.reward() != 0.5:
          if board.is_max:
            print("Kawabanga")
          else:
              print("Pikapika")
          print(board.to_pretty_string())
          fail = 1
        sum=sum-fail
    
    print("We got ", sum)
    print("tie out of 5")
    end = time.time()
    print("Time:", start-end)