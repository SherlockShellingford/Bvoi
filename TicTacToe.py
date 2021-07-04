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
from collections import namedtuple
from random import choice, getrandbits
from MCTS import MCTS, Node
import numpy as np
import math
from scipy.stats import norm
from alphazero.NNet import NNetWrapper

_TTTB = namedtuple("TicTacToeBoard", "tup turn winner terminal")

states=dict()
states_cache_bvoi=dict()
previous_tree = None
alphazero_agent=None


def load_tictactoe_alphazero():
    alphazero_agent = NNetWrapper()
    alphazero_agent.load_checkpoint('./pretrained_models/tictactoe/keras','best-25eps-25sim-10epch.pth.tar')








# Inheriting from a namedtuple is convenient because it makes the class
# immutable and predefines __init__, __repr__, __hash__, __eq__, and others
class TicTacToeBoard(Node):

    def __init__(self, max, tup, turn, winner, terminal, meanvalue, depth):
        standard_derv=(9-depth)*0.3/9
        self.max=max
        self.tup=tup
        self.turn=turn
        self.winner=winner
        self.terminal=terminal
        self.meanvalue=meanvalue
        self.depth=depth
        self.hash = getrandbits(128)
        self.probability_density = lambda x : norm.pdf(x, loc=meanvalue, scale=standard_derv, size=None)
        self.buckets=[]
        for i in np.linspace(0,1,15):
            self.buckets.append(norm.ppf(i,loc=meanvalue, scale=standard_derv))

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

        self.distribution=lambda x : dist(norm.rvs(loc=meanvalue,scale=standard_derv, size=None))






    def find_children(board):
        if board.terminal:  # If the game is finished then no moves can be made
            return set()
        # Otherwise, you can make a move in each of the empty spots
        return {
            board.make_move(i) for i, value in enumerate(board.tup) if value is None
        }

    def find_random_child(board):
        if board.terminal:
            return None  # If the game is finished then no moves can be made
        empty_spots = [i for i, value in enumerate(board.tup) if value is None]
        return board.make_move(choice(empty_spots))

    def find_random_child_bvoi(board):
        if board.terminal:
            return None  # If the game is finished then no moves can be made
        empty_spots = [i for i, value in enumerate(board.tup) if value is None]
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
        is_terminal = (winner is not None) or not any(v is None for v in tup)
        ret = TicTacToeBoard(not board.max, tup, turn, winner, is_terminal, 0, board.depth)
        states[tup] = ret
        return ret

    def make_move_bvoi(board, index):
        tup = board.tup[:index] + ((1,) if board.turn else (-1,)) + board.tup[index + 1 :]
        state=states_cache_bvoi.get(tup)
        if state is not None:
            return state
        turn = not board.turn
        winner = _find_winner(tup)
        is_terminal = (winner is not None) or not any(v is None for v in tup)
        original_mcts_ret = states.get(tup)
        print("Zura")
        print(original_mcts_ret.tup)
        value_of_original = previous_tree.Q[original_mcts_ret] / previous_tree.N[original_mcts_ret]
        ret = TicTacToeBoard(not board.max, tup, turn, winner, is_terminal, value_of_original , board.depth)
        states[tup] = ret
        return ret


    def to_pretty_string(board):
        to_char = lambda v: ("X" if v is True else ("O" if v is False else " "))
        rows = [
            [to_char(board.tup[3 * row + col]) for col in range(3)] for row in range(3)
        ]
        return (
            "\n  1 2 3\n"
            + "\n".join(str(i + 1) + " " + " ".join(row) for i, row in enumerate(rows))
            + "\n"
        )


def play_game(mode="uct"):
    board = new_tic_tac_toe_board()
    tree = MCTS(board, mode=mode)
    print(board.to_pretty_string())
    while True:
        #row_col = input("enter row,col: ")
        #row, col = map(int, row_col.split(","))
        #index = 3 * (row - 1) + (col - 1)
        #if board.tup[index] is not None:
        #    raise RuntimeError("Invalid move")
        #board = board.make_move(index)
        #print(board.to_pretty_string())
        #if board.terminal:
        #    break
        # You can train as you go, or only at the beginning.
        # Here, we train as we go, doing fifty rollouts each turn.
        for _ in range(200):
            tree.do_rollout(board)
        board = tree.choose(board)
        print(board.to_pretty_string())
        if board.terminal:
            break
    return tree


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
        if -1 is v1 is v2 is v3:
            return False
        if 1 is v1 is v2 is v3:
            return True
    return None


def new_tic_tac_toe_board():
    return TicTacToeBoard(True, (None,) * 9, True, None, False,0, 0)


if __name__ == "__main__":
    tree = play_game()
    for key in tree.N.keys():
        print(key.tup, tree.N[key])
    previous_tree = tree

    play_game(mode="bvoi-greedy")