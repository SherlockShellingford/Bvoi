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
from NormalSearchAlgorithm import NormalSearchAlgorithm
_TTTB = namedtuple("TicTacToeBoard", "tup turn winner terminal")

states=dict()
states2=dict()
states_cache_bvoi=dict()
previous_tree = None
alphazero_agent = None
global weak_heuristic_dict
weak_heuristic_dict = None



def load_tictactoe_alphazero():
    alphazero_agent = NNetWrapper()
    alphazero_agent.load_checkpoint('./pretrained_models','best-25eps-25sim-10epch.pth.tar')


def sample(to_simulate,  num_sims = 25):
    sum = 0
    for i in range(num_sims):
        sum += mcts.simulate(to_simulate, invert_reward=False)

    meanvalue = sum / num_sims
    return meanvalue





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

    def find_children_bvoi(board, distribution_mode="sample"):
        if board.terminal:  # If the game is finished then no moves can be made
            return set()
        # Otherwise, you can make a move in each of the empty spots
        return {
            board.make_move_bvoi(i, distribution_mode = distribution_mode) for i, value in enumerate(board.tup) if value == 0
        }



    def find_random_child(board):
        if board.terminal:
            return None  # If the game is finished then no moves can be made
        empty_spots = [i for i, value in enumerate(board.tup) if value == 0]
        return board.make_move(choice(empty_spots))

    def find_random_child_bvoi(board, distribution_mode = "sample"):
        if board.terminal:
            return None  # If the game is finished then no moves can be made
        empty_spots = [i for i, value in enumerate(board.tup) if value == 0]
        return board.make_move_bvoi(choice(empty_spots), distribution_mode = distribution_mode)

    def reward(board):
        if not board.terminal:
            raise RuntimeError(f"reward called on nonterminal board {board}")
        if board.winner is board.turn:
            # It's your turn and you've already won. Should be impossible.
            print(board.tup)
            print(board.winner)
            print(board.turn)
            print(board.is_max)
            print(board.terminal)
            raise RuntimeError(f"reward called on unreachable board {board}")
        if board.winner is None:
            return 0.5  # Board is a tie

        if board.turn is (not board.winner):
            return 0  # Your opponent has just won. Bad.

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



    def make_move_bvoi(board, index, distribution_mode = "weak_heuristic"):
        tup = board.tup[:index] + ((1,) if board.turn else (-1,)) + board.tup[index + 1 :]
        state=states_cache_bvoi.get(tup)
        if state is not None:
            return state
        turn = not board.turn
        winner = _find_winner(tup)
        is_terminal = (winner is not None) or not any(v == 0 for v in tup)
        to_simulate = TicTacToeBoard(not board.is_max, tup, turn, winner, is_terminal, 0 , board.depth + 1)
        mcts = MCTS(to_simulate)

        if distribution_mode == "sample":
            meanvalue = sample(to_simulate)
        elif distribution_mode == "weak heuristic":
            global weak_heuristic_dict
            if weak_heuristic_dict.get(to_simulate.tup) is not None:
                meanvalue = weak_heuristic_dict[to_simulate.tup][0]
            else:
                flipped_to_simulate = flip_board(to_simulate)
                if weak_heuristic_dict.get(flipped_to_simulate.tup) is not None:
                    meanvalue = 1 - weak_heuristic_dict[flipped_to_simulate.tup][0]
                else:
                    meanvalue = sample(to_simulate)
        elif distribution_mode == 'none':
            meanvalue = 0
        ret = TicTacToeBoard(not board.is_max, tup, turn, winner, is_terminal, meanvalue, board.depth + 1)
        if distribution_mode != 'none':
            states_cache_bvoi[tup] = ret
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
        states_cache_bvoi[tup] = ret
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


def flip_board(board):
    tup2 = []
    for i in range(9):
        tup2.append(-board.tup[i])

    board = TicTacToeBoard(not board.is_max, tuple(tup2), not board.turn, board.winner, board.terminal, 0, board.depth)
    return board

def play_game(mode="uct", mode2 ="uct"):
    board = new_tic_tac_toe_board()
    tree = NormalSearchAlgorithm(board, mode=mode, distribution_mode="weak heuristic")
    tree2 = MCTS(flip_board(board), mode=mode2)
    game=TicTacToeGame()
    #rival=MCTSaz(game,alphazero_agent)
    #prob, v = alphazero_agent.predict(np.asarray(board.tup).astype('float32').reshape((3, 3)))

    print(board.to_pretty_string())
    while True:
        for i in range(150):
            tree.do_rollout(board)
        board = tree.choose(board)
        print(board.to_pretty_string())
        if board.terminal:
            break

#        board = flip_board(board)

        print(board.tup)
        for i in range(400):
            tree2.do_rollout(board)
        board = tree2.choose(board)
        
        if board.terminal:
  #          board = flip_board(board)
  #          board.is_max = False
  #          board.winner = False
  #          board.turn = True
            break

 #       board = flip_board(board)
        print(board.to_pretty_string())
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
        print(result_dict[new_tic_tac_toe_board().tup])
        f.close()
    else:
        result_dict = {}

    init = new_tic_tac_toe_board()
    for _ in range(50000000):
        simulate(init, result_dict)
    f = open("weak_heuristic", "wb")
    pickle.dump(result_dict, f)
    f.close()
if __name__ == "__main__":

    #alphazero_agent = NNetWrapper()
    #alphazero_agent.load_checkpoint('./pretrained_models/alternative', 'best-25eps-25sim-10epch.pth.tar')

    import time

    mcts = MCTS(new_tic_tac_toe_board())
    print(mcts.compute_max_probability([[(-1, 0.5), (3, 0.7), (10, 1.0)],
                                  [(0, 0.2), (2, 0.4), (8, 1.0)],
                                  [(-2, 0.3), (6, 0.6), (9, 1.0)]]))

    f = open("weak_heuristic", "rb")
    weak_heuristic_dict = pickle.load(f)
    f.close()
    #exit(0)
    #start = time.time()
    #simulate_until_no_tomorrow(load = True)
    #print("Finished")
    #print(time.time() - start)
    #exit(0)

    start = time.time()
    sum = 100
    for i in range(0,25):
        fail=0
        board=play_game( mode= "FT Greedy", mode2="uct")
        if board.reward() != 0.5:
            if not board.is_max:
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