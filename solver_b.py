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
    def __init__(self, reach2_pen, reach3_pen):
        self.db = Database(1)
        self.dict_hyper = {}
        self.dict_hyper['reach2_pen'] = reach2_pen
        self.dict_hyper['reach3_pen'] = reach3_pen

    def run(self, test=False):
        reach2_pen = self.dict_hyper['reach2_pen']
        reach3_pen = self.dict_hyper['reach3_pen']
        name = self._name_gen()
        print(name)
        self.db.gen_edges()
        self.db.init_slides(start_ind = 0)
        self.dic = self.db.dict_id_cons
        self.diu = self.db.dict_id_used  
        last_id = self.db.get_last_id()
        self.list_ids = [last_id]
        self.db.pop_cons_by_id(last_id)
        self.num_no_con = 0
        self.iterations = self.db.slides_length-1
        if test:
            self.iterations = 101
        for iteration in tqdm(range(self.iterations)):
            last_id = self.list_ids[-1]
            if iteration % 1000 == 0:
                pass
            cons = list(self.diu[last_id])
            if cons:
                scores = []
                for con in cons:
                    con_cons = self.dic[con]
                    score = -len(con_cons)
                    reach2 = []
                    reach3 = []
                    for con_con in con_cons:
                        reach2 += list(self.dic[con_con])
                        con_con_cons = self.dic[con_con]
                        for con_con_con in con_con_cons:
                            reach3 += list(self.dic[con_con_con])
                    score += reach2_pen * len(set(reach2)) + reach3_pen * len(set(reach3))
                    scores.append(score)
                ind = np.argmax(scores)
                id = cons[ind]
            if not cons:
                id = self.start_new_chain()
            self.db.pop_cons_by_id(id)
            self.list_ids.append(id)

        self.db.set_slides(self.list_ids)
        score = self.db.score_slides()  
        name = self._name_gen()
        print(name)
        print('score: ',score)
        path_sol = 'solution_b.txt'
        self.db.gen_output(path = path_sol)
        return score

    def start_new_chain(self):
        reach2_pen = self.dict_hyper['reach2_pen']
        reach3_pen = self.dict_hyper['reach3_pen']
        self.num_no_con += 1
        remaining = list(self.dic.keys())
        scores = []
        for con in remaining:
            con_cons = self.dic[con]
            score = -len(con_cons)
            reach2 = []
            reach3 = []
            for con_con in con_cons:
                reach2 += list(self.dic[con_con])
            score += reach2_pen * len(set(reach2))
            scores.append(score)
        ind = np.argmax(scores)
        return remaining[ind]


    def _name_gen(self):
        string = ''
        for key, val in self.dict_hyper.items():
            string += str(key)+':_'+str(val)+',_'
        return string[:-2]



e = experiment(0.001, -0.000001)
e.run()
