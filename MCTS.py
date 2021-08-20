"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wi0kipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
import copy
from abc import ABC, abstractmethod
from collections import defaultdict
import math
import numpy as np
import random
import time
from scipy.integrate import quad
from CVIBES.PrioritizedItem import PrioritizedItem
from queue import PriorityQueue
from SearchAlgoirthm import *




class MCTS(SearchAlgorithm):
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, root, exploration_weight=1, mode="uct", distribution_mode="weak heuristic"):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.exploration_weight = exploration_weight
        self.leaves=dict()
        self.leaves[root.__hash__()]=root
        self.mode=mode
        self.distribution_mode=distribution_mode
        if mode == "c-vibes":
            self.distribution_mode="weak heuristic"
        self.bvoi_counter=5
        self.bvoi_freq=1
        self.last_chosen_by_bvoi=None
        self.node_to_tag=dict()
        self.first_time_add=True
        self.node_to_pick = []
        super().__init__(root)

    #def _mark_ancestors(self, node):
    #    node.marked = node.marked + 1
    #    if self.child_to_father.get(node) is None:
    #        return
    #    self._mark_ancestors(self.child_to_father[node])

   # def _mark_all_S_ancestors(self, s):
   #     for node in s:
   #         self._mark_ancestors(node)







    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            #return self.N[n] #TODO THIS IS A DIFFERENT KIND OF MEASUREMENT, YOU CAN TRY AND REMOVE IT
            if self.N[n] < 15:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        for c in self.children[node]:
            print(c.tup)
            print(self.Q[c])
            print(self.N[c])
            #for c2 in self.children[c]:
            #    print(self.Q[c2])
            #    print(self.N[c2])

            #if x[0]>2:
            #    print("Chikachika")
            #    while True:
            #        self._compute_Us(c, [])
        #self.tree_vis.show()


        return max(self.children[node], key=score)


    def do_rollout(self, node, second = False):
        if self.node_to_dry_Us.get(node) is None:
            self.node_to_dry_Us[node] = [(node.meanvalue,1)]

        if second and self.mode == "FT Greedy":
            for i in range(20):
                reward = self.simulate(self.alpha[node])
                self._backpropagate([node, self.alpha[node]], reward)


        path = self._select(node)
        leaf = path[-1]
        if leaf is None:
            print(path)
        self._expand(leaf)
        if self.mode == "c-vibes":
            for _ in range(self.BSM_N):
                reward = self.simulate(leaf)
                self._backpropagate(path, reward)
        else:
            reward = self.simulate(leaf)
            self._backpropagate(path, reward)

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        start = time.time()
        path = []
        first=True
        #DEBUG
        which = 0
        #DEBUG
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                #DEBUG
                #t = time.time() - start
                #if which == 1:
                #    self.debug_counter += t
                #if which == 2:
                #    self.debug_counter2 += t
                #print(t)
                #DEBUG
                
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                #DEBUG
                #t = time.time() - start
                #if which == 1:
                #    self.debug_counter += t
                #if which == 2:
                #    self.debug_counter2 += t
                #print(t)
                #DEBUG
                return path
            if self.mode == "uct" or self.mode == "corner uct":
                node = self._uct_select(node)
            elif self.mode == "c-vibes":
                if first:
                    path = self._conspiracy_choice(node)
                    node = path[-1]
                    path = path[:-1]
                    first=False
                else:
                    node = self._uct_select(node)

            elif self.mode == "bvoi-greedy":
                if first:
                    if self.bvoi_counter % self.bvoi_freq == 0:
                        self.bvoi_counter=1
                        node = self._BVOI_select(node)
                    else:
                        self.bvoi_counter+=1
                        node = self._uct_select(node)
                    first=False
                else:
                    node = self._uct_select(node)
            elif self.mode == "FT Greedy":
                if first:
                    if self.bvoi_counter % self.bvoi_freq == 0:
                        #DEBUG
                        #print("voi")
                        #which = 1
                        #DEBUG
                        self.bvoi_counter=1
                        path = self._BVOI_select(node)
                        node = path[-1]
                        path = path[:-1]
                    else:
                        #DEBUG
                        #print("uct")
                        #which = 2
                        #DEBUG
                        self.bvoi_counter+=1
                        node = self._uct_select(node)
                    first=False
                else:
                    node = self._uct_select(node)
            elif self.mode == "MGSS*":
                if first:
                    if self.bvoi_counter % self.bvoi_freq == 0:
                        self.bvoi_counter=1
                        path = self._BVOI_select(node)
                        node = path[-1]
                        path = path[:-1]
                    else:
                        self.bvoi_counter+=1
                        node = self._uct_select(node)
                    first=False
                else:
                    node = self._uct_select(node)

    def update_dry_Us(self, node, is_a_leaf = False, last_changed_value = 1000):

        previous_value = self.node_to_dry_Us[node][0][0]

        if node.is_max:
            changed = False
            max = self.node_to_dry_Us[node][0][0]
            if is_a_leaf or self.node_to_dry_Us[node][0][0] == last_changed_value:
                max = np.NINF
            for c in self.children[node]:
                if max < self.node_to_dry_Us[c][0][0]:
                    changed = True
                    max =  self.node_to_dry_Us[c][0][0]
            if changed or is_a_leaf:
                self.node_to_dry_Us[node] = [(max,1.0)]
                if self.child_to_father.get(node) is not None:
                    self.update_dry_Us(self.child_to_father.get(node), last_changed_value=previous_value)
        else:
            changed = False
            min=self.node_to_dry_Us[node][0][0]
            if is_a_leaf or self.node_to_dry_Us[node][0][0] == last_changed_value:
                min = np.inf
            for c in self.children[node]:
                if self.node_to_dry_Us[c][0][0] < min:
                    changed = True
                    min = self.node_to_dry_Us[c][0][0]
            if changed or is_a_leaf:
                self.node_to_dry_Us[node] = [(min, 1.0)]
                if self.child_to_father.get(node) is not None:
                    self.update_dry_Us(self.child_to_father.get(node), last_changed_value=previous_value)


    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        if self.mode == "bvoi-greedy" or self.mode== "c-vibes" or self.mode == "FT Greedy" or "MGSS*":
            children = node.find_children_bvoi(distribution_mode=self.distribution_mode)
            self.children[node]=children
            
        else:
            children = node.find_children()  
            self.children[node]=children
            return
          
        is_max=node.is_max
        if is_max:
            max=np.NINF
            max_c=None
            second_to_max=np.NINF
            second_to_max_c=None
        else:
            max = np.PINF
            max_c = None
            second_to_max = np.PINF
            second_to_max_c = None
        for n in children:
            # Visualazation
#            self.node_to_tag[n]=self.node_to_tag[node] + ''.join([str(i) for i in n.tup])
#            self.tree_vis.create_node(str(n.meanvalue),self.node_to_tag[node] + ''.join([str(i) for i in n.tup]),parent=self.node_to_tag[node])
            # Visualazation

            #Markdown algorithm
            self.child_to_father[n] = node
            self.node_to_dry_Us[n] = [(n.meanvalue,1.0)]



            if (is_max and n.meanvalue>max)  or (not is_max and n.meanvalue<=max):
                second_to_max=max
                second_to_max_c=max_c
                max=n.meanvalue
                max_c=n
            else:
                if (is_max and n.meanvalue>=second_to_max) or (not is_max and n.meanvalue<=second_to_max):
                    second_to_max=n.meanvalue
                    second_to_max_c=n
        self.alpha[node]=max_c
        self.beta1[node]=second_to_max_c


    def simulate(self, node, invert_reward=True):
        "Returns the reward for a random simulation (to completion) of `node`"
        #print("Begin simulating!")
        #print(node.is_terminal())
        #print(node.get_legal_moves(1 if node.is_max else -1))
        while True:
            if node.is_terminal():
                reward = node.reward()
                reward = node.reward()
                return 1 - reward if invert_reward else reward
            if self.mode!="uct" and self.mode != "corner uct":
                node = node.find_random_child_bvoi(distribution_mode="none")
            else:
                node = node.find_random_child()
                #print(node.is_terminal())
                #print(node.get_legal_moves(1 if node.is_max else -1))
            invert_reward = not invert_reward

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        pass
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa




    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])
        def uct(n):
            "Upper confidence bound for trees"
            initial_heuristic = 1
            #if self.mode != "uct":
            #    initial_heuristic = (1 if node.is_max else -1) * corner_heuristic(n, len(self.children[node]))
            return self.Q[n] / self.N[n] + initial_heuristic * self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)





class Node():
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """



    def sample(self):
        return self.distribution()

    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return self.hash

    def __str__(self):
        return ''.join([str(i) for i in self.tup])

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return node1.tup == node2.tup