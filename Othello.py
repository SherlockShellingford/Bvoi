from collections import namedtuple
from random import choice, getrandbits
from MCTS import MCTS, Node
import numpy as np
import math
from scipy.stats import norm
from othello.keras.NNet import NNetWrapper
from MCTSaz import MCTSaz
from copy import deepcopy
from othello.OthelloGame import OthelloGame
states=dict()
states_cache_bvoi=dict()
previous_tree = None
alphazero_agent=None
deductable_time = 0
added_time = 0


class OthelloBoard(Node):

    __directions = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]

    def __init__(self, max, tup, turn, winner, terminal, meanvalue, depth):
        standard_derv=(36-depth)*0.3/36
        self.n=6
        self.is_max=max
        self.tup=tup
        self.tup66=tup
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

        self.distribution=lambda x : dist(norm.rvs(loc=meanvalue,scale=standard_derv, size=None))




    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black
        """
        moves = set()  # stores the legal moves.

        # Get all the squares with pieces of the given color.
        for y in range(self.n):
            for x in range(self.n):
                if self.tup66[x][y]==color:
                    newmoves = self.get_moves_for_square((x,y))
                    moves.update(newmoves)
        return [x * 6 + y for (x,y) in moves]


    def has_legal_moves(self, color):
        for y in range(self.n):
            for x in range(self.n):
                if self.tup66[x][y]==color:
                    newmoves = self.get_moves_for_square((x,y))
                    if len(newmoves)>0:
                        return True
        return False

    def get_moves_for_square(self, square):
        """Returns all the legal moves that use the given square as a base.
        That is, if the given square is (3,4) and it contains a black piece,
        and (3,5) and (3,6) contain white pieces, and (3,7) is empty, one
        of the returned moves is (3,7) because everything from there to (3,4)
        is flipped.
        """
        (x,y) = square

        # determine the color of the piece.
        color = self.tup66[x][y]

        # skip empty source squares.
        if color==0:
            return None

        # search all possible directions.
        moves = []
        for direction in self.__directions:
            move = self._discover_move(square, direction)
            if move:
                # print(square,move,direction)
                moves.append(move)

        # return the generated move list
        return moves


    def _discover_move(self, origin, direction):
        """ Returns the endpoint for a legal move, starting at the given origin,
        moving by the given increment."""
        x, y = origin
        color = self.tup66[x][y]
        flips = []

        for x, y in OthelloBoard._increment_move(origin, direction, self.n):
            if self.tup66[x][y] == 0:
                if flips:
                    # print("Found", x,y)
                    return (x, y)
                else:
                    return None
            elif self.tup66[x][y] == color:
                return None
            elif self.tup66[x][y] == -color:
                # print("Flip",x,y)
                flips.append((x, y))


    def execute_move(self, move, color):
        """Perform the given move on the board; flips pieces as necessary.
        color gives the color pf the piece to play (1=white,-1=black)
        """

        #Much like move generation, start at the new piece's square and
        #follow it on all 8 directions to look for a piece allowing flipping.

        # Add the piece to the empty square.
        # print(move)
        flips = [flip for direction in self.__directions
                      for flip in self._get_flips(move, direction, color)]
        assert len(list(flips))>0
        ret = deepcopy(self.tup66)
        for x, y in flips:
            #print(self.tup66[x][y],color)
            ret[x][y] = color
        return ret

    def _get_flips(self, origin, direction, color):
        """ Gets the list of flips for a vertex and direction to use with the
        execute_move function """
        #initialize variables
        flips = [origin]

        for x, y in OthelloBoard._increment_move(origin, direction, self.n):
            #print(x,y)
            if self.tup66[x][y] == 0:
                return []
            if self.tup66[x][y] == -color:
                flips.append((x, y))
            elif self.tup66[x][y] == color and len(flips) > 0:
                #print(flips)
                return flips

        return []


    def find_children(board):
        if board.terminal:  # If the game is finished then no moves can be made
            return set()
        # Otherwise, you can make a move in each of the empty spots
        return {
            board.make_move(i) for i in board.get_legal_moves(1 if board.is_max else -1)
        }

    def find_children_bvoi(board, distribution_mode="sample"):
        if board.terminal:  # If the game is finished then no moves can be made
            return set()
        # Otherwise, you can make a move in each of the empty spots
        return {
            board.make_move_bvoi(i, distribution_mode=distribution_mode) for i in board.get_legal_moves(1 if board.is_max else -1)
        }



    def find_random_child(board):
        if board.terminal:
            return None  # If the game is finished then no moves can be made
        empty_spots = board.get_legal_moves(1 if board.is_max else -1)
        return board.make_move(choice(empty_spots))

    def find_random_child_bvoi(board, distribution_mode="sample"):
        if board.terminal:
            return None  # If the game is finished then no moves can be made
        empty_spots = board.get_legal_moves(1 if board.is_max else -1)
        return board.make_move_bvoi(choice(empty_spots), distribution_mode=distribution_mode)



    def make_move(board, index):
        tup = board.execute_move((int(index / 6),index % 6 ), 1 if board.is_max else -1)
        flattup=tuple(tup[0]+tup[1]+tup[2]+tup[3]+tup[4]+tup[5])
        state=states.get(flattup)
        if state is not None:
            return state
        turn = not board.turn
        ret = OthelloBoard(not board.is_max, tup, turn, None, None, 0, board.depth + 1)
        ret.terminal = len(ret.get_legal_moves(1 if ret.is_max else -1)) == 0
        if ret.terminal:
            ret.winner = _find_winner(ret)
        else:
            ret.winner = None



        states[flattup] = ret
        return ret

    def make_move_bvoi(board, index, distribution_mode="sample"):
        tup = board.execute_move((int(index / 6), index % 6), 1 if board.is_max else -1)
        flattup = tuple(tup[0] + tup[1] + tup[2] + tup[3] + tup[4] + tup[5])
        state=states_cache_bvoi.get(flattup)
        if state is not None:
            return state
        turn = not board.turn
        if distribution_mode=="none":
            ret = OthelloBoard(not board.is_max, tup, turn, None, None, 0, board.depth + 1)
            ret.terminal = len(ret.get_legal_moves(1 if ret.is_max else -1)) == 0
            if ret.terminal:
                ret.winner = _find_winner(ret)
            else:
                ret.winner = None
            return ret

        if distribution_mode=="NN":
            if not board.is_max:
                tup2=deepcopy(tup)
            else:
                tup2=deepcopy(tup)
                for x in range(6):
                    for y in range(6):
                        tup2[x][y]=-tup2[x][y]

            X = np.asarray(tup2).astype('float32')
            X = np.reshape(X, (6, 6,))

            meanvalue = alphazero_agent.predict(X)[1][0]# 0 is pi and 1 is v
            if not not board.is_max:
                meanvalue = -meanvalue
            ret = OthelloBoard(not board.is_max, tup, turn, None, None, meanvalue, board.depth + 1)
            ret.terminal = len(ret.get_legal_moves(1 if ret.is_max else -1)) == 0
            if ret.terminal:
                ret.winner = _find_winner(ret)
            else:
                ret.winner=None
        if distribution_mode=="sample":
            to_simulate = OthelloBoard(not board.is_max, tup, turn, None, None, 0, board.depth + 1)
            terminal = len(to_simulate.get_legal_moves(1 if to_simulate.is_max else -1)) == 0
            if terminal:
                winner = _find_winner(to_simulate)
            else:
                winner = None

            to_simulate.winner = winner
            to_simulate.terminal = terminal
            mcts=MCTS(to_simulate)
            sum=0
            num_sims=35
            global deductable_time
            global added_time
            start = time.time()
            for i in range(num_sims):
                sum += mcts.simulate(to_simulate, invert_reward = False)

            if not board.is_max:
                meanvalue = sum / num_sims
            else:
                meanvalue = (num_sims - sum) / num_sims
            end = time.time()
            deductable_time = deductable_time + end - start
            added_time = added_time + 0.062408
            ret = OthelloBoard(not board.is_max, tup, turn, None, None, meanvalue, board.depth + 1)
            ret.winner = winner
            ret.terminal = terminal


        states[flattup] = ret
        return ret


    def to_pretty_string(board):
        to_char = lambda v: ("X" if v == 1 else ("O" if v == -1 else " "))
        rows = [
            [to_char(board.tup[row][col]) for col in range(6)] for row in range(6)
        ]
        return (
            "\n  1 2 3 4 5 6\n"
            + "\n".join(str(i + 1) + " " + " ".join(row) for i, row in enumerate(rows))
            + "\n"
        )

    def reward(board):
        if not board.terminal:
            raise RuntimeError(f"reward called on nonterminal board {board}")
        if board.winner is board.turn:
            # It's your turn and you've already won. Should be impossible.
            raise RuntimeError(f"reward called on unreachable board {board}")

        if board.winner == 0.5:
            return 0.5  # Board is a tie
        if board.is_max:
            return board.winner
        if not board.is_max:
            return 1-board.winner


        # The winner is neither True, False, nor None
        raise RuntimeError(f"board has unknown winner type {board.winner}")

    def is_terminal(board):
        return board.terminal
    @staticmethod
    def _increment_move(move, direction, n):
        # print(move)
        """ Generator expression for incrementing moves """
        #move = list(map(sum, zip(move, direction)))
        move = (move[0]+direction[0], move[1]+direction[1])
        while all(map(lambda x: 0 <= x < n, move)):
            # while 0<=move[0] and move[0]<n and 0<=move[1] and move[1]<n:
            yield move
            #move = list(map(sum, zip(move, direction)))
            move = (move[0]+direction[0],move[1]+direction[1])

def _find_winner(board : OthelloBoard):

    sum = 0
    for x in range(board.n):
        for y in range(board.n):
            sum+=board.tup66[x][y]
    if sum < 0:
        return 0
    if sum > 0:
        return 1
    return 0.5



def new_othello_board():
    x=[]
    for i in range(6):
        x.append([0,0,0,0,0,0])
    x[2][2]=1
    x[2][3]=-1
    x[3][2]=-1
    x[3][3]=1
    return OthelloBoard(True, x  , True, None, False,0, 0)



def legal_moves_test():
    x=[]
    x.append([0, 0, 0, 0, 0, 0])
    x.append([0, 1, 0, 0, 1, 0])
    x.append([0, 0,-1,-1, 0, 0])
    x.append([0, 0, 1,-1, 0, 0])
    x.append([0, 0, 0, 1, 0, 0])
    x.append([0, 0, 0, 0, 0, 0])

    oo = OthelloBoard(True, x  , True, None, False,0, 0)
    print("White1:", oo.get_legal_moves(1))
    print("Black-1:", oo.get_legal_moves(-1))


def do_turn_alphazeroagent(mcts, board):
    enemy_pov = []
    for x in range(6):
        row = []
        for y in range(6):
            row.append(-board.tup[x][y])
        enemy_pov.append(row)

    prob = mcts.getActionProb(np.asarray(enemy_pov).astype('float32').reshape((6, 6)))
    maxi = -1
    maxx = -1
    for i in range(len(prob)):
        if prob[i] > maxx:
            maxi = i
            maxx = prob[i]
    print(prob)
    print("Rival chose:", maxi)
    board = board.make_move_bvoi(maxi)
    return board


def do_turn_mcts(tree, board):
    time_limit = 7
    global deductable_time
    global added_time
    i = 0
    deductable_time = 0
    added_time = 0
    start = time.time()
    while True:
        print("Lap", i)
        i = i + 1
        tree.do_rollout(board)
        if time.time() - start - deductable_time + added_time > time_limit:
            break
        print(time.time() - start - deductable_time + added_time)
    board = tree.choose(board)
    return board

def play_game(mode="uct", distribution_mode="sample"):
    board = new_othello_board()
    tree = MCTS(board, mode=mode, distribution_mode=distribution_mode)
    tree_uct = MCTS(board, mode="uct")
    game=OthelloGame(6)
    #rival=MCTSaz(game,alphazero_agent)
    print(board.to_pretty_string())

    while True:

        board = do_turn_mcts(tree, board)

        print(board.to_pretty_string())

        if board.terminal:
            break

        tup2=[]
        for i in range(6):
            row = []
            for j in range(6):
                row.append(-board.tup[i][j])
            tup2.append(row)

        board = OthelloBoard(not board.is_max, tup2, not board.turn, board.winner, board.terminal, 0, board.depth)


        board = do_turn_mcts(tree_uct, board)


        if board.terminal:
            tup2 = []
            for i in range(6):
                row = []
                for j in range(6):
                    row.append(-board.tup[i][j])
                tup2.append(row)
            board = OthelloBoard(not board.is_max, tup2, not board.turn, 1 - board.winner, board.terminal, 0, board.depth)
            break

        tup2 = []
        for i in range(6):
            row = []
            for j in range(6):
                row.append(-board.tup[i][j])
            tup2.append(row)
        board = OthelloBoard(not board.is_max, tup2, not board.turn, board.winner, board.terminal, 0, board.depth)
        print(board.to_pretty_string())



    print(board.to_pretty_string())
    return board


def play_game_opposite(mode="uct", distribution_mode="sample"):
    board = new_othello_board()
    tree = MCTS(board, mode=mode, distribution_mode=distribution_mode)
    tree_uct = MCTS(board, mode="uct")
    game = OthelloGame(6)
    #rival = MCTSaz(game, alphazero_agent)
    print(board.to_pretty_string())

    while True:

        board = do_turn_mcts(tree_uct, board)

        print(board.to_pretty_string())

        if board.terminal:
            break

        tup2=[]
        for i in range(6):
            row = []
            for j in range(6):
                row.append(-board.tup[i][j])
            tup2.append(row)

        board = OthelloBoard(not board.is_max, tup2, not board.turn, board.winner, board.terminal, 0, board.depth)


        board = do_turn_mcts(tree, board)


        if board.terminal:
            tup2 = []
            for i in range(6):
                row = []
                for j in range(6):
                    row.append(-board.tup[i][j])
                tup2.append(row)
            board = OthelloBoard(not board.is_max, tup2, not board.turn, 1 - board.winner, board.terminal, 0, board.depth)
            break

        tup2 = []
        for i in range(6):
            row = []
            for j in range(6):
                row.append(-board.tup[i][j])
            tup2.append(row)
        board = OthelloBoard(not board.is_max, tup2, not board.turn, board.winner, board.terminal, 0, board.depth)
        print(board.to_pretty_string())



    print(board.to_pretty_string())
    return board




if __name__ == "__main__":
    #alphazero_agent = NNetWrapper()
    #alphazero_agent.load_checkpoint('./pretrained_models/othello', '6x6 checkpoint_145.pth.tar')
    #tree = play_game()
    import time
    legal_moves_test()

    #real_best = [[0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, -1, 1, 1, -1, 0], [0, 1, 1, -1, 1, 0], [0, 0, 1, 1, 0, 0],
    # [0, 0, 1, 0, 0, 0]]

    #best = [[0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [1, 1, 1, 1, -1, 0], [0, 1, -1, -1, 1, 0], [0, 0, -1, -1, 0, 0],
    # [0, 0, 0, 0, 0, 0]]

    #father = [[0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, -1, -1, 1, -1, 0], [0, 1, -1, -1, 1, 0], [0, 0, -1, -1, 0, 0],
    #             [0, 0, 0, 0, 0, 0]]

    #one = [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1],
    #          [1, 1, 1, 1, 1, 1]]

    #X = np.asarray(one).astype('float32')
    #X = np.reshape(X, (6, 6,))

    #meanvalue_complete = alphazero_agent.predict(X)[1][0]  # 0 is pi and 1 is v

    #X = np.asarray(father).astype('float32')
    #X = np.reshape(X, (6, 6,))

    #pi, v = alphazero_agent.predict(X)  # 0 is pi and 1 is v

    #X = np.asarray(real_best).astype('float32')
    #X = np.reshape(X, (6, 6,))

    #meanvalue = alphazero_agent.predict(X)[1][0]  # 0 is pi and 1 is v

    #o = OthelloBoard(True, father, True, None, None, 0, 1)

    #o.make_move_bvoi(32)
    #o.make_move_bvoi(12)



    start = time.time()
    sum = 10
    win_sum = 0
    for i in range(0,5):
        fail=0
        board=play_game(mode="uct")
        win_sum += board.winner
        for x in board.tup:
            if x == 0:
                fail=1
        sum=sum-fail



    print("Win_sum:", win_sum)
    win_sum = 0
    end = time.time()
    print("Time:", start-end)
    for i in range(0,5):
        fail=0
        board=play_game_opposite(mode="bvoi-greedy", distribution_mode="sample")
        win_sum += board.winner
        for x in board.tup:
            if x == 0:
                fail=1
        sum=sum-fail

    print("Win_sum:", win_sum)
    end = time.time()
    print("Time:", start-end)



