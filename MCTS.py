"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
from abc import ABC, abstractmethod
from collections import defaultdict
import math
import numpy as np
import random
from scipy.integrate import quad
from treelib import Node, Tree


class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, root, exploration_weight=1, mode="uct"):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.child_to_father=dict()
        self.exploration_weight = exploration_weight
        self.leaves=dict()
        self.leaves[root.__hash__()]=root
        self.alpha=dict()
        self.beta1=dict()
        self.mode=mode
        self.bvoi_counter=5
        self.bvoi_freq=5
        self.last_chosen_by_bvoi=None
        self.tree_vis=Tree()
        self.node_to_tag=dict()
        self.first_time_add=True


    def _compute_Us(self, node, s):
        if node.terminal:
            if node.is_max:
                return [-1 for b in node.buckets]
            else:
                return [1 for b in node.buckets]
        if self.children.get(node) is None:
            if node.__hash__() in s:
                return node.buckets
            else:
                return [node.meanvalue for b in node.buckets]

        is_max=node.is_max

        child_dist=[]
        for c in self.children[node]:
            child_dist.append(self._compute_Us(c,s))
        ret=[]
        for i in range(len(node.buckets)):
            if is_max:
                max_or_min=np.NINF
            else:
                max_or_min=np.PINF
            for c in child_dist:
                if is_max:
                    if max_or_min<c[i]:
                        max_or_min=c[i]
                else:
                    if max_or_min>c[i]:
                        max_or_min=c[i]
            ret.append(max_or_min)

        return ret

    def gather_leaves(self, node):
        if self.children.get(node) is None:
            return [node]
        ret = []
        for c in self.children.get(node):
            ret = ret + self.gather_leaves(c)
        return ret


    def _compute_bvoi_of_child(self, Unode, Ucompare_against, is_alpha=False, is_max=True):
        sum = 0
        is_max_coefficent=1
        if not is_max:
            is_max_coefficent=-1

        if not is_alpha:
            for i in range(len(Unode)):
                sum += max([is_max_coefficent*(Unode[i] - Ucompare_against[i]), 0])
            sum = sum / len(Unode)
            return sum
        else:
            for i in range(len(Unode)):
                sum += max([is_max_coefficent*(Ucompare_against[i] - Unode[i]), 0])
            sum = sum / len(Unode)
            return sum



    def _batch_gather_greedy(self, node):
        foundone=True
        s=[]
        alpha_node = self.alpha[node]
        beta1_node = self.beta1[node]
        alpha_Us = self._compute_Us(alpha_node, [])
        alpha_leaves = self.gather_leaves(alpha_node)
        for c in self.children[node]:
            leaves=alpha_leaves + self.gather_leaves(c)
            for l in leaves:
                if l not in s:
                    if c.__hash__() != alpha_node.__hash__():
                        c_bvoi = self._compute_bvoi_of_child(self._compute_Us(c, [l.__hash__()]), alpha_Us, node.is_max)
                    else:
                        c_bvoi = self._compute_bvoi_of_child(self._compute_Us(beta1_node, [l.__hash__()]), self._compute_Us(c, [l.__hash__()]),
                                                             is_alpha=True, is_max=node.is_max)
                    if c_bvoi>0:
                        s.append(l)

        return s



    def _BVOI_select(self, node):
        if len(self.children[node]) == 1:
            for i in self.children[node]:
                return i

        s=self._batch_gather_greedy(node)
        if len(s) == 0: #TODO ADD THIS!!!!!
        #if True: #TODO REMOVE THIS!!!!!
            return self.alpha[node]
        max=0
        max_child=None
        alpha_node=self.alpha[node]
        beta1_node=self.beta1[node]
        alpha_Us=self._compute_Us(alpha_node,s)
        beta1_Us = self._compute_Us(beta1_node, s)

        for c in self.children[node]:
            child_Us=self._compute_Us(c,s)
            if c.__hash__() != alpha_node.__hash__():
                c_bvoi=self._compute_bvoi_of_child(child_Us,alpha_Us, is_max=node.is_max)
            else:
                c_bvoi=self._compute_bvoi_of_child(beta1_Us,child_Us, is_alpha=True, is_max=node.is_max)
            if max<=c_bvoi:
                max=c_bvoi
                max_child=c
        return max_child





    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            #return self.N[n] #TODO THIS IS A DIFFERENT KIND OF MEASUREMENT, YOU CAN TRY AND REMOVE IT
            if self.N[n] < 10:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        for c in self.children[node]:
            print(c.tup)
            print(self.Q[c])
            print(self.N[c])
            print(c.meanvalue)
            x=self._compute_Us(c,[])
            print(x)
            #if x[0]>2:
            #    print("Chikachika")
            #    while True:
            #        self._compute_Us(c, [])
        #self.tree_vis.show()


        return max(self.children[node], key=score)


    def do_rollout(self, node):
        #Visualazation
        if self.first_time_add:
            self.first_time_add=False
            self.node_to_tag[node]=''.join([str(i) for i in node.tup])
            self.tree_vis.create_node(str(node.meanvalue),''.join([str(i) for i in node.tup]))
        "Make the tree one layer better. (Train for one iteration.)"
        #Visualazation
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        first=True
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            if self.mode == "uct":
                node = self._uct_select(node)
            if self.mode == "bvoi-greedy":
                if first:
                    if self.bvoi_counter==self.bvoi_freq:
                        self.bvoi_counter=0
                        node = self._BVOI_select(node)
                        self.last_chosen_by_bvoi=node
                    else:
                        self.bvoi_counter+=1
                        node = self.last_chosen_by_bvoi
                    first=False
                else:
                    node = self._uct_select(node)


    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        if self.mode == "bvoi-greedy":
            children = node.find_children_bvoi()
        else:
            children = node.find_children()

        self.children[node]=children
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
            if (is_max and n.meanvalue>max)  or (not is_max and n.meanvalue<max):
                second_to_max=max
                second_to_max_c=max_c
                max=n.meanvalue
                max_c=n
            else:
                if (is_max and n.meanvalue>second_to_max) or (not is_max and n.meanvalue<second_to_max):
                    second_to_max=n.meanvalue
                    second_to_max_c=n
        self.alpha[node]=max_c
        self.beta1[node]=second_to_max_c


    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        invert_reward = True
        while True:
            if node.is_terminal():
                reward = node.reward()
                return 1 - reward if invert_reward else reward
            if self.mode!="uct":
                node = node.find_random_child_bvoi()
            else:
                node = node.find_random_child()

            invert_reward = not invert_reward

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
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
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
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
        return True