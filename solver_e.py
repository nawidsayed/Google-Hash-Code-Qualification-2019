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
    def __init__(self, db, hyper_inter, hyper_length, exponent, use_exp, hyper_abs, k, verbose):
        self.db = db
        self.dict_hyper = {}
        self.dict_hyper['hyper_inter'] = hyper_inter
        self.dict_hyper['hyper_length'] = hyper_length
        self.dict_hyper['exponent'] = exponent
        self.dict_hyper['use_exp'] = use_exp
        self.dict_hyper['hyper_abs'] = hyper_abs
        self.dict_hyper['k'] = k
        self.verbose = verbose
        
    def run(self):
        name = self._name_gen()
        print(name)
        db = self.db
        db.init_slides()
        self.iterations = db.slides_length-1
        for iteration in tqdm(range(self.iterations)):
            if self.verbose:
                if iteration % self.iter_tracker == self.iter_tracker-1:
                    print(db.tracker_last(size=self.iter_tracker))
            last_tags = db.get_last_tags()
            verticals = db.list_tags_vert
            ind1, ind2 = self.best_vertical_pair(last_tags, verticals)
            db.pop_by_two_inds_vert(ind1,ind2)
        score = db.score_slides()  
        name = self._name_gen()
        print(name)
        print('score: ',score)
        path_sol = 'solution_e.txt'
        db.gen_output(path = path_sol)
        return score

    def best_vertical_pair(self, last_tags, verticals):
        k = self.dict_hyper['k']
        hyper_length = self.dict_hyper['hyper_length']
        exponent = self.dict_hyper['exponent']
        use_exp = self.dict_hyper['use_exp']
        scores = []
        for tags in verticals:
            # The score has a bias towards fotos with large amount of tags
            # We encourage smaller ones by dividing the score with a normaliation constant 
            # which is proportional to some root of the number of tags in the foto 
            if use_exp:
                score = inter(last_tags, tags) / pow(len(tags), exponent)
            else:
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

    # Calculates all possible combinations of verticals
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



db = Database(4)
e = experiment(db, hyper_inter = 1.3, hyper_length = 0.14, hyper_abs=0.01,
    k=500, exponent=0.2, use_exp=True, verbose=0)
e.run()