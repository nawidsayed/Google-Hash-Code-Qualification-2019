import numpy as np
import matplotlib.pyplot as plt
from time import time
from copy import copy
from time import time
from database import Database
from tqdm import tqdm
import math

def score_func(s1,s2):
    ls1 = len(s1)
    ls2 = len(s2)
    lsi = len(s1.intersection(s2))
    return min(ls1-lsi, ls2-lsi, lsi)

def union(s1,s2):
    return len(s1.union(s2))

def inter(s1,s2):
    return len(s1.intersection(s2))

class experiment():
    iter_tracker = 100
    def __init__(self, db, hyper_inter, hyper_length, hyper_abs, 
        hyper_pernalty, k, verbose):
        self.db = db
        self.dict_hyper = {}
        self.dict_hyper['hyper_inter'] = hyper_inter
        self.dict_hyper['hyper_length'] = hyper_length
        self.dict_hyper['hyper_abs'] = hyper_abs
        self.dict_hyper['hyper_pernalty'] = hyper_pernalty
        self.dict_hyper['k'] = k
        self.verbose = verbose
        
    def run(self, test=False):
        hyper_length = self.dict_hyper['hyper_length']
        hyper_pernalty = self.dict_hyper['hyper_pernalty']
        name = self._name_gen()
        print(name)
        db = self.db
        db.init_slides()
        self.iterations = db.slides_length-1
        if test:
            self.iterations = 101
        for iteration in tqdm(range(self.iterations)):
            if self.verbose:
                if iteration % self.iter_tracker == self.iter_tracker-1:
                    print(db.tracker_last(size=self.iter_tracker))
            score_vert = -100000
            score_hor = -100000
            last_tags = db.get_last_tags()
            if db.list_tags_vert:
                verticals = db.list_tags_vert
                ind1, ind2 = self.best_vertical_pair(last_tags, verticals)
                tags1 = verticals[ind1]
                tags2 = verticals[ind2]
                tags_vert = tags1.union(tags2)
                score_vert = score_func(last_tags, tags_vert) - len(tags_vert) * hyper_length 
            if db.list_tags_hor:
                horizontals = db.list_tags_hor
                ind = self.best_horizontal(last_tags, horizontals)
                tags_hor = horizontals[ind]
                score_hor = score_func(last_tags, tags_hor) - len(tags_hor) * hyper_length 
            if score_vert - hyper_pernalty > score_hor:
                db.pop_by_two_inds_vert(ind1,ind2)
            else:
                db.pop_by_ind_hor(ind)

        score = db.score_slides()  
        name = self._name_gen()
        print(name)
        print('score: ',score)
        path_sol = 'solution_c.txt'
        db.gen_output(path = path_sol)
        return score

    def best_horizontal(self, last_tags, horizontals):
        hyper_length = self.dict_hyper['hyper_length']
        scores = []
        for tags in horizontals:
            length_hor = len(tags)
            score = inter(last_tags, tags) - len(tags) * hyper_length 
            scores.append(score)
        ind = np.argmax(scores)
        return ind

    def best_vertical_pair(self, last_tags, verticals):
        k = self.dict_hyper['k']
        hyper_length = self.dict_hyper['hyper_length']
        scores = []
        for tags in verticals:
            score = inter(last_tags, tags) - len(tags) * hyper_length 
            scores.append(score)    
        k_small = min(k, len(scores))    
        indices_good = np.argpartition(scores, -k_small)[-k_small:]
        scores_small = [scores[i] for i in indices_good]
        verticals_small = [verticals[i] for i in indices_good]
        ind1, ind2 = self.evaluate_all_combinations(last_tags, verticals_small, scores_small)
        ind1 = indices_good[ind1]
        ind2 = indices_good[ind2]
        return (ind1, ind2)

    def evaluate_all_combinations(self, last_tags, verticals, scores):
        hyper_inter = self.dict_hyper['hyper_inter']
        hyper_length = self.dict_hyper['hyper_length']
        hyper_abs = self.dict_hyper['hyper_abs']
        scores = []
        for tags in verticals:
            score = inter(last_tags, tags) - len(tags) * hyper_length
            scores.append(score) 
        best_score = -1000000
        cur_score = None
        best_inds = None
        for i,tags1 in enumerate(verticals):
            for j in range(i):
                tags2 = verticals[j]
                length_inter12 = inter(tags1, tags2)
                cur_score = scores[i] + scores[j] - length_inter12 * hyper_inter
                cur_score -= abs(len(last_tags) - len(tags1)-len(tags2)+length_inter12) * hyper_abs 
                if cur_score > best_score:
                    best_inds = (i,j)
                    best_score = cur_score
        return best_inds 

    def _name_gen(self):
        string = ''
        for key, val in self.dict_hyper.items():
            string += str(key)+':_'+str(val)+',_'
        return string[:-2]



db = Database(2)
e = experiment(db, hyper_inter = 1.3, hyper_length = 0.14, hyper_abs = 0.0001, 
    hyper_pernalty = 0.00001, k=400, verbose=0)

e.run()
