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
from SearchAlgoirthm import SearchAlgorithm
class NormalSearchAlgorithm(SearchAlgorithm):
    def __init__(self, root, mode = "FT Greedy", distribution_mode = ""):
        self.mode = mode
        self.distribution_mode = distribution_mode
        super().__init__(root)


    def _select(self, node):
        while True:
            print("2")
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal

                return node
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                return n
            elif self.mode == "bvoi-greedy":
                node = self._BVOI_select(node)

            elif self.mode == "FT Greedy":
                node = self._BVOI_select(node)[-1]

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        if self.mode == "bvoi-greedy" or self.mode== "c-vibes" or self.mode == "FT Greedy":
            children = node.find_children_bvoi(distribution_mode=self.distribution_mode)
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

    def minmax_on_expanded_nodes(self, node):
        print("1")
        if self.children.get(node) is None or node.terminal:
            return node.meanvalue
        else:
            return max([self.minmax_on_expanded_nodes(c) for c in self.children[node]])


    def get_best_child(self, node):

        children_list = []
        for c in self.children[node]:
            children_list.append((c, self.minmax_on_expanded_nodes(c)))
        maxx = np.NINF
        maxchild = None
        for p in children_list:
            if p[1] > maxx:
                maxx=p[1]
                maxchild=p[0]
        return maxchild

    def choose(self, node):
        return self.get_best_child(node)

    def do_rollout(self, node):
        node = self._select(node)
        self._expand(node)
