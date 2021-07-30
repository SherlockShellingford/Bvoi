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
from CVIBES.PrioritizedItem import PrioritizedItem
from queue import PriorityQueue

class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, root, exploration_weight=1, mode="uct", distribution_mode="sample"):
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
        self.distribution_mode=distribution_mode
        self.bvoi_counter=5
        self.bvoi_freq=5
        self.last_chosen_by_bvoi=None
        self.node_to_tag=dict()
        self.first_time_add=True
        self.conspiracy_queue = []
        self.BSM_K = 3
        self.BSM_N = 1
        self.node_to_path_CVIBES = dict()
        self.node_to_dry_Us = dict()
        self.node_to_dry_Us[root] = [(root.meanvalue, 1)]


    def _mark_ancestors(self, node):
        node.marked = node.marked + 1
        if self.child_to_father.get(node) is None:
            return
        self._mark_ancestors(self.child_to_father[node])

    def _mark_all_S_ancestors(self, s):
        for node in s:
            self._mark_ancestors(node)


    def _compute_Us_for_all_children(self, node, s):
        self._mark_all_S_ancestors(s)
        return self._compute_Us(node, return_children=True)

    # Example:
    # distributions = [[(4, 0.2), (6, 0.5), (7, 1.0)], [(3, 0.4), (4, 0.6), (5, 1.0)]]
    def compute_max_probability(self, distributions,  is_max = True):
        queue = PriorityQueue()
        ret_dist = []
        chance_of_every_distribution_to_be_lower_than_current_value = []
        num_of_chances_that_are_not_one_anymore = 0
        multiplied_chance = 1
        last_diff = 1
        last_distribution_cdf_used = 1

        for i in range(len(distributions)):
            if is_max:
                queue.put(PrioritizedItem(distributions[i][0][0], (i, distributions[i][0][1], 0)))
            else:
                if len(distributions[i]) == 1:
                    chance = 1
                else:
                    chance = 1 - distributions[i][-2][1]
                queue.put(PrioritizedItem(-distributions[i][-1][0], (i, chance, len(distributions[i]) - 1)))
            chance_of_every_distribution_to_be_lower_than_current_value.append(1)
        while not queue.empty():
            lowest_or_highest_value = queue.get()

            i = lowest_or_highest_value.item[0]
            j = lowest_or_highest_value.item[2]
            #The chance for all other distriubutions to be lower/equal than the value, and for the distribution that contained it to be that exact value
            diff = (lowest_or_highest_value.item[1] - chance_of_every_distribution_to_be_lower_than_current_value[i])
            cdf_of_current_distribution = chance_of_every_distribution_to_be_lower_than_current_value[lowest_or_highest_value.item[0]]


            if num_of_chances_that_are_not_one_anymore == len(distributions):
                multiplied_chance *= diff * last_distribution_cdf_used  / (last_diff * cdf_of_current_distribution)
                last_distribution_cdf_used = lowest_or_highest_value.item[1]
                last_diff = diff
                ret_dist.append([lowest_or_highest_value.priority, multiplied_chance])

            else:
                if chance_of_every_distribution_to_be_lower_than_current_value[i] == 1:
                    multiplied_chance *= lowest_or_highest_value.item[1] / chance_of_every_distribution_to_be_lower_than_current_value[i]
                    num_of_chances_that_are_not_one_anymore += 1
                    if num_of_chances_that_are_not_one_anymore == len(distributions):
                        ret_dist.append([lowest_or_highest_value.priority, multiplied_chance])

            chance_of_every_distribution_to_be_lower_than_current_value[i] = lowest_or_highest_value.item[1]


            if is_max and j < len(distributions[i]) - 1:
                chance = distributions[i][j + 1][1]
                queue.put(PrioritizedItem(distributions[i][j + 1][0], (i, chance, j + 1)))
            elif not is_max and j > 0:
                if j == 1:
                    chance = 1
                else:
                    chance = 1 - distributions[i][j - 2][1]
                queue.put(PrioritizedItem(-distributions[i][j - 1][0], (i, chance, j - 1)))

        if not is_max:
            ret_dist = reversed(ret_dist)
            ret_dist = [[-item[0], item[1]]  for item in ret_dist]

        culmative = 0
        real_ret_dist = []
        for item in ret_dist:
            real_ret_dist.append( (item[0] , item[1] + culmative) )
            culmative += item[1]
        
        if 2 * len(distributions) < len(real_ret_dist):
            ratio_to_one = int(len(real_ret_dist) / len(distributions))
            j = 0
            i = ratio_to_one
            approximated_ret_dist = []
            while j < len(real_ret_dist):
                if i > len(real_ret_dist):
                    sum = 0
                    for m in range(j, len(real_ret_dist)):
                        sum += real_ret_dist[m][0]
                    mean = sum / (len(real_ret_dist) - j)
                    approximated_ret_dist.append((mean, 1.0))
                else:
                    sum = 0
                    for m in range(j, i):
                        sum += real_ret_dist[m][0]
                    mean = sum / ratio_to_one
                    approximated_ret_dist.append((mean, real_ret_dist[m][1]))
                j += ratio_to_one
                i += ratio_to_one
            return approximated_ret_dist
                
            
        
        return real_ret_dist




    def _compute_Us(self, node, return_children=False):

        if node.marked == 0:
            return self.node_to_dry_Us[node]


        node.marked = max([node.marked - 1, 0])
        #if node.terminal:
        #    if node.is_max:
        #        return [-1 for b in node.buckets]
        #    else:
        #        return [1 for b in node.buckets]
        if self.children.get(node) is None or len(self.children.get(node)) == 0:
            return node.buckets

        is_max=node.is_max


        if return_children:
            ret = dict()
            for c in self.children[node]:
                ret[c] = self._compute_Us(c)
            return ret
        child_dist=[]
        for c in self.children[node]:
            child_dist.append(self._compute_Us(c))
        ret = self.compute_max_probability(child_dist, is_max = is_max)
        return ret

    def gather_leaves(self, node):
        if self.children.get(node) is None:
            return [node]
        ret = []
        for c in self.children.get(node):
            ret = ret + self.gather_leaves(c)
        return ret

    def _compute_mean_of_distribution(self, dist):
        mean = dist[0][0] * dist[0][1]
        for i in range(1, len(dist)):
            mean += dist[i][0] * (dist[i][1] - dist[i - 1][1])
        return mean

    def _compute_bvoi_of_child(self, Unode, Ucompare_against, is_alpha=False, is_max=True):
        sum = 0
        is_max_coefficent=1
        if not is_max:
            is_max_coefficent=-1

        if not is_alpha:
            return max([is_max_coefficent*(self._compute_mean_of_distribution(Unode) - self._compute_mean_of_distribution(Ucompare_against)), 0])

        else:
            return max([is_max_coefficent*(self._compute_mean_of_distribution(Ucompare_against) - self._compute_mean_of_distribution(Unode)), 0])


    def chance_for_dist_biggersmaller_than_val(self, dist, val, bigger_mode=True):
        count=0
        len_dist=len(dist)
        for i in range(len_dist):
            if dist[i]>val:
                break
            count = count + 1
        if not bigger_mode:
            return count/len_dist
        return (len_dist-count)/len_dist


    def _S_gather_rec_CVIBES(self, node, v, prob_table, prob_of_root, path_table, S, path):


        if self.children.get(node) is None:
            S.append(node)
            path_table[node] = path + [node]
        else:
            for c in self.children[node]:
                if prob_table[c.__hash__(), v] == prob_of_root:
                   S = self._S_gather_rec_CVIBES(c,v,prob_table,prob_of_root,path_table, S, path + [node])
        return S


    def _store_probabilities(self, node, s, table, v, bigger_mode=True):

        if self.children.get(node) is None:
            if node.__hash__() in s:

                table[node.__hash__(), v] = self.chance_for_dist_biggersmaller_than_val(node.buckets, v, bigger_mode=bigger_mode)
                return node.buckets
            else:
                table[node.__hash__(), v] = self.chance_for_dist_biggersmaller_than_val([node.meanvalue for b in node.buckets], v, bigger_mode=bigger_mode)
                return [node.meanvalue for b in node.buckets]
        is_max=node.is_max
        child_dist=[]
        for c in self.children[node]:
            Us_c = self._store_probabilities(c, s, table, v, bigger_mode=bigger_mode)
            child_dist.append(Us_c)
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

        table[node.__hash__(), v] = self.chance_for_dist_biggersmaller_than_val(ret, v, bigger_mode=bigger_mode)
        return ret




    def _find_k_best_VPI(self, s, k, alpha, c, is_alpha_children):
        queue = PriorityQueue()
        for leaf in s:
            item = PrioritizedItem(-1 * self._compute_bvoi_of_child(self._compute_Us(c,[leaf]),self._compute_Us(alpha,[leaf]),  is_alpha = is_alpha_children, is_max = not alpha.is_max), leaf)
            queue.put(item)
        return queue.queue

    def _conspiracy_choice(self, node):
        #There are still items in the conspiracy queues
        if len(self.conspiracy_queue) > 0:
            return self.node_to_path_CVIBES[self.conspiracy_queue.pop()]


        #different behaviors whether we are the min or max player
        is_max=node.is_max

        #Lines 1-5
        alpha_node = self.alpha[node]
        alpha_mean = alpha_node.meanvalue
        coff_stash = [1.05, 1, 0.95, 0.9, 0.8]
        v_stash = []
        for coff in coff_stash:
            v_stash.append(coff * alpha_mean)


        prob_dict=dict()

        for c in self.children[node]:
            for V in v_stash:
                s = self.gather_leaves(c)
                if c.__hash__() == alpha_node.__hash__():
                    self._store_probabilities(c, s, prob_dict,V,bigger_mode= not is_max)
                else:
                    self._store_probabilities(c, s, prob_dict,V,bigger_mode= is_max)
        #Line 6
        max=0
        maxV=None
        maxV_tag=None
        max_c=None
        for i in range(len(v_stash)):
            for j in range(i + 1, len(v_stash)):
                for c in self.children[node]:
                    if c.__hash__() != alpha_node.__hash__():
                        if is_max:
                            eq_8 = (v_stash[i] - v_stash[j]) * prob_dict[alpha_node.__hash__(), v_stash[j]] * prob_dict[c.__hash__(), v_stash[i]]
                        else:
                            eq_8 = (v_stash[i] - v_stash[j]) * prob_dict[alpha_node.__hash__(), v_stash[i]] * prob_dict[c.__hash__(), v_stash[j]]
                        if max < eq_8:
                            max = eq_8
                            if is_max:
                                maxV = v_stash[j]
                                maxV_tag = v_stash[i]
                            else:
                                maxV_tag = v_stash[j]
                                maxV = v_stash[i]
                            max_c = c
        if maxV_tag is None:
            return [node, alpha_node]
        #Lines 7-16
        S1 = self._S_gather_rec_CVIBES(alpha_node, maxV,prob_dict, prob_dict[alpha_node.__hash__(), maxV], self.node_to_path_CVIBES, [], [] )
        S2 = self._S_gather_rec_CVIBES(max_c, maxV_tag,prob_dict, prob_dict[max_c.__hash__(), maxV_tag], self.node_to_path_CVIBES, [], [])

        #BSM(N,K)
        K1 = self._find_k_best_VPI(S1,self.BSM_K,alpha_node,self.beta1[node], is_alpha_children= True)
        K2 = self._find_k_best_VPI(S2, self.BSM_K, alpha_node, max_c, is_alpha_children=False)

        S=[]
        i1 = 0
        i2 = 0
        K_size = min([self.BSM_K, len(K1) + len(K2)])
        for i in range(K_size):
            if i1 == len(K1):
                S.append(K2[i2].item)
            elif i2 == len(K2):
                S.append(K1[i1].item)
            elif K1[i1].priority < K2[i2].priority:
                S.append(K1[i1].item)
                i1 = i1 + 1
            else:
                S.append(K2[i2].item)
                i2 = i2 + 1

        leftover_1_items_with_priority = K1[i1:]
        leftover_2_items_with_priority = K2[i2:]
        leftover_1 = []
        leftover_2 = []
        #Converting the BVOI+node objects to node objects
        for i in range(len(leftover_1_items_with_priority)):
            leftover_1.append(leftover_1_items_with_priority[i].item)
        for i in range(len(leftover_2_items_with_priority)):
            leftover_2.append(leftover_2_items_with_priority[i].item)
        #Choosing random K nodes from the leftovers
        leftovers = leftover_1 + leftover_2
        S_random = random.choices(leftovers,k= min([self.BSM_K, len(leftovers)]))
        S = S + S_random

        #Setting the new 2K array as the new conspiracy queue
        for i in range(self.BSM_N):
            self.conspiracy_queue = self.conspiracy_queue + S
        if not self.conspiracy_queue:
          return [node, alpha_node]
        return self.node_to_path_CVIBES[self.conspiracy_queue.pop()]











    def _batch_gather_greedy(self, node):
        foundone=True
        s=[]
        alpha_node = self.alpha[node]
        beta1_node = self.beta1[node]
        alpha_Us = self._compute_Us(alpha_node)

        leaves = self.gather_leaves(node)
        for l in leaves:
            child_to_Us = self._compute_Us_for_all_children(node, [l] )
            for c in self.children[node]:
                if l not in s:
                    if c.__hash__() != alpha_node.__hash__():
                        c_bvoi = self._compute_bvoi_of_child(child_to_Us[c], alpha_Us, node.is_max)
                    else:
                        c_bvoi = self._compute_bvoi_of_child(child_to_Us[beta1_node], child_to_Us[c] ,
                                                             is_alpha=True, is_max=node.is_max)
                    if c_bvoi>0:
                        s.append(l)

        return s



    def _BVOI_select(self, node):
        if len(self.children[node]) == 1:
            for i in self.children[node]:
                if i is None:
                    print("Oish noo2")
                    print(node.tup)
                return i

        s=self._batch_gather_greedy(node)
        if len(s) == 0: #TODO ADD THIS!!!!!
            if self.alpha[node] is None:
                print("Oish noo3")
                print(node.tup)
            return self.alpha[node]
        max=0
        max_child=None
        alpha_node=self.alpha[node]
        beta1_node=self.beta1[node]
        child_to_Us = self._compute_Us_for_all_children(node, s)
        alpha_Us=child_to_Us[alpha_node]
        beta1_Us = child_to_Us[beta1_node]


        for c in self.children[node]:
            child_Us=child_to_Us[c]
            if c.__hash__() != alpha_node.__hash__():
                c_bvoi=self._compute_bvoi_of_child(child_Us,alpha_Us, is_max=node.is_max)
            else:
                c_bvoi=self._compute_bvoi_of_child(beta1_Us,child_Us, is_alpha=True, is_max=node.is_max)
            if max<=c_bvoi:
                max=c_bvoi
                max_child=c
        if max_child is None:
            print("Oish noo")
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

        #for c in self.children[node]:
        #    print(c.tup)
        #    print(self.Q[c])
        #    print(self.N[c])
        #    print(c.meanvalue)
        #    x=self._compute_Us(c,[])
        #    print(x)
            #if x[0]>2:
            #    print("Chikachika")
            #    while True:
            #        self._compute_Us(c, [])
        #self.tree_vis.show()


        return max(self.children[node], key=score)


    def do_rollout(self, node):
        if self.node_to_dry_Us.get(node) is None:
            self.node_to_dry_Us[node] = [(node.meanvalue,1)]

        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self.simulate(leaf)
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

    def update_dry_Us(self, node):

        if node.is_max:
            changed = False
            max = np.NINF
            for c in self.children[node]:
                if max < self.node_to_dry_Us[c][0][0]:
                    changed = True
                    max =  self.node_to_dry_Us[c][0][0]
            if changed:
                self.node_to_dry_Us[node] = [(max,1.0)]
                if self.child_to_father.get(node) is not None:
                    self.update_dry_Us(self.child_to_father.get(node))
        else:
            changed = False

            min=np.inf
            for c in self.children[node]:
                if self.node_to_dry_Us[c][0][0] < min:
                    changed = True
                    min = self.node_to_dry_Us[c][0][0]
            if changed:
                self.node_to_dry_Us[node] = [(min, 1.0)]
                if self.child_to_father.get(node) is not None:
                    self.update_dry_Us(self.child_to_father.get(node))


    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        if self.mode == "bvoi-greedy" or self.mode== "c-vibes":
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
            # Visualazation
#            self.node_to_tag[n]=self.node_to_tag[node] + ''.join([str(i) for i in n.tup])
#            self.tree_vis.create_node(str(n.meanvalue),self.node_to_tag[node] + ''.join([str(i) for i in n.tup]),parent=self.node_to_tag[node])
            # Visualazation

            #Markdown algorithm
            self.child_to_father[n] = node
            self.node_to_dry_Us[n] = [(n.meanvalue,1.0)]



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
        self.update_dry_Us(node)


    def simulate(self, node, invert_reward=True):
        "Returns the reward for a random simulation (to completion) of `node`"
        #print("Begin simulating!")
        #print(node.is_terminal())
        #print(node.get_legal_moves(1 if node.is_max else -1))
        while True:
            if node.is_terminal():
                reward = node.reward()
                return 1 - reward if invert_reward else reward
            if self.mode!="uct":
                node = node.find_random_child_bvoi(distribution_mode="none")
            else:
                node = node.find_random_child()
                #print(node.is_terminal())
                #print(node.get_legal_moves(1 if node.is_max else -1))
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