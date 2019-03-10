import numpy as np
import matplotlib.pyplot as plt
from time import time
from copy import copy
from collections import OrderedDict

def score_func(s1,s2):
    ls1 = len(s1)
    ls2 = len(s2)
    lsi = len(s1.intersection(s2))
    return min(ls1-lsi, ls2-lsi, lsi)

def triplet_func(s1,s2):
    ls1 = len(s1)
    ls2 = len(s2)
    lsi = len(s1.intersection(s2))
    return (ls1-lsi, ls2-lsi, lsi)

class Database():
    paths = ['a_example.txt', 'b_lovely_landscapes.txt', 'c_memorable_moments.txt', 'd_pet_pictures.txt', 'e_shiny_selfies.txt']
    def __init__(self, letter):
        self.slides = []
        self.slides_id = []
        self.letter = letter
        self._parser()
        self._sort_lists()
        self._gen_dict_length_ids()
        self._gen_dict_tag_ids()
        self._gen_verticals_horizontals()
        # Currently we have following objects:
        # self.num_categories       sorted by increasing tag occurence
        # self.list_tags_vert       sorted by increasing tags size
        # self.list_id_vert         sorted by increasing tags size
        # self.list_tags_hor        sorted by increasing tags size
        # self.list_id_hor          sorted by increasing tags size
        # self.dict_length_ids     
        # self.dict_tag_ids
        # self.
        # self. 

    def gen_edges(self):
        dict_tag_pairs = {}
        for tag, ids in self.dict_tag_ids.items():
            if len(ids) == 2:
                dict_tag_pairs[tag] = tuple(ids)

        dict_id_cons = {}
        for tag, pair in dict_tag_pairs.items():
            id1_tup, id2_tup = pair
            for id1, id2 in [(id1_tup, id2_tup), (id2_tup, id1_tup)]:
                if id1 in dict_id_cons:
                    dict_id_cons[id1].append(id2)
                else:
                    dict_id_cons[id1] = [id2]

        self.dict_id_cons = {}
        for id, cons_list in dict_id_cons.items():
            self.dict_id_cons[id] = set(cons_list)
        self.dict_id_used = {}

    def load_slides(self, path):
        with open(path, 'r') as f:
            num_slides = int(f.readline())
            for i in range(num_slides):
                line = f.readline()
                list_string = line.split()
                if len(list_string) == 1:
                    id = int(list_string[0])
                    tags = self.dict_id_tags[id]
                    self.slides_id.append(id)
                    self.slides.append(tags)
                else:
                    id1, id2 = int(list_string[0]), int(list_string[1])
                    tags1 = self.dict_id_tags[id1]
                    tags2 = self.dict_id_tags[id2]
                    self.slides_id.append((id1, id2))
                    self.slides.append(tags1.union(tags2))

    def set_slides(self, list):
        self.slides_id = []
        self.slides = []
        for id in list:
            self.slides_id.append(id)
            tags = self.dict_id_tags[id]
            self.slides.append(tags) 


    def get_last_tags(self):
        return self.slides[-1]

    def get_last_id(self):
        return self.slides_id[-1]

    def init_slides(self, start_ind=None):
        if not self.list_tags_vert:
            if start_ind is None:
                start_ind = -1
            self.pop_by_ind_hor(start_ind)
        else:
            self.pop_by_two_inds_vert(0,-1)

    def pop_cons_by_id(self, id):
        cons = self.dict_id_cons[id]
        self.dict_id_used[id] = cons
        for con in cons:
            self.dict_id_cons[con].remove(id)
        self.dict_id_cons.pop(id)
        return id

    def append_cons_by_id(self, id):
        cons = self.dict_id_used.pop(id)
        for con in cons:
            self.dict_id_cons[con].add(id)
        self.dict_id_cons[id] = cons

    def pop_by_ind_hor(self, ind):
        id, tags = self._pop_by_ind(ind, vertical=False)
        self.slides_id.append(id)
        self.slides.append(tags)
        return id

    def pop_by_two_inds_vert(self, ind1, ind2):
        if ind1 > ind2:
            id1, tags1 = self._pop_by_ind(ind1, vertical=True)
            id2, tags2 = self._pop_by_ind(ind2, vertical=True)
        else:
            id2, tags2 = self._pop_by_ind(ind2, vertical=True)
            id1, tags1 = self._pop_by_ind(ind1, vertical=True)
        self.slides_id.append((id1, id2))
        self.slides.append(tags1.union(tags2))
        return (id1, id2)

    def tracker_last(self, size = 100):
        end = len(self.slides)
        start = max(end - size, 0)
        return self.tracker(start=start, end=end)

    def tracker(self, start = 0, end = -1): 
        lengths = self.get_lengths(start=start, end=end)
        mean_length = np.mean(lengths)
        wasteds = self.get_wasteds(start=start, end=end)
        mean_wasted = np.mean(wasteds)
        horizontals_id = self.get_horizontals_id(start=start, end=end)
        sum_horizontals_id = len(horizontals_id)
        length_differences = self.get_length_differences(start=start, end=end)
        mean_diff = np.mean(np.abs(length_differences))
        efficiencies = self.get_efficiencies(start=start, end=end)
        mean_eff = np.mean(np.abs(efficiencies))
        triplets_to_min = self.get_triplets_to_min(start=start, end=end)
        arr_triplets_to_min = np.array(triplets_to_min)
        means_triplets_to_min = list(arr_triplets_to_min.mean(axis=0))
        # stddevs = arr_triplets_to_min.std(axis=0)
        return [mean_length, mean_eff, mean_wasted, mean_diff] + means_triplets_to_min


    def get_lengths(self, start=0, end=-1): 
        lengths = []
        for tags in self.slides[start:end]:
            lengths.append(len(tags))
        return lengths

    def get_wasteds(self, start=0, end=-1):
        wasteds = []
        for id_tup in self.slides_id[start:end]:
            if isinstance(id_tup, tuple):
                id1, id2 = id_tup
                tags1 = self.dict_id_tags[id1]
                tags2 = self.dict_id_tags[id2]
                wasteds.append(len(tags1.intersection(tags2)))
        return wasteds

    def get_efficiencies(self, start=0, end=-1):
        efficiencies = []
        last_tags = self.slides[start]
        for tags in self.slides[start+1:end]:
            efficiencies.append(len(last_tags.intersection(tags)) / len(tags))
            last_tags = tags
        return efficiencies

    def get_horizontals_id(self, start=0, end=-1):
        horizontals_id = []
        for id_tup in self.slides_id[start:end]:
            if not isinstance(id_tup, tuple):
                horizontals_id.append(id_tup)
        return horizontals_id        

    def get_length_differences(self, start=0, end=-1):
        length_differences = []
        last_tags = self.slides[start]
        for tags in self.slides[start+1:end]:
            length_differences.append(len(last_tags) - len(tags))
            last_tags = tags
        return length_differences

    def get_triplets_to_min(self, start=0, end=-1):
        triplets_to_min = []
        last_tags = self.slides[start]
        for tags in self.slides[start+1:end]:
            triplet_to_min = triplet_func(last_tags, tags)
            triplets_to_min.append(triplet_to_min)
            last_tags = tags
        return triplets_to_min

    def get_chunks_vert(self, num_chunks=4):
        length = len(self.list_tags_vert)
        chunk_slices = np.linspace(0,length, num_chunks+1).astype(int)
        indices = np.arange(length)
        chunks = []
        chunks_indices = [] 
        for i in range(num_chunks):
            lo = chunk_slices[i]
            hi = chunk_slices[i+1]
            chunks.append(self.list_tags_vert[lo:hi])
            chunks_indices.append(indices[lo:hi])
        return chunks, chunks_indices

    def gen_output(self, path=None):
        if path is None:
            path = 'sol_%d.txt' %self.letter
        with open(path, 'w') as f:
            f.write('%d\n' %(len(self.slides_id)))
            for tup_int in self.slides_id:
                if not isinstance(tup_int, tuple):
                    f.write('%d\n' %tup_int)
                else:
                    f.write('%d %d\n' %tuple(tup_int))
                    
    def score_slides(self):
        total = 0
        last_slide = self.slides[0]
        for slide in self.slides[1:]:
            total += score_func(last_slide, slide)
            last_slide = slide
        return total

    def _parser(self):
        path = self.paths[self.letter]
        self.list_orientation = []
        list_tags = []
        dict_tag = {}
        with open(path, 'r') as f:
            k = int(f.readline())
            for i in range(k):
                line = f.readline()
                l = line.split()
                orientation = l.pop(0)
                l.pop(0)
                if orientation == 'V':
                    self.list_orientation.append(1)
                else:
                    self.list_orientation.append(0)
                list_tags.append(l)
                for t in l:
                    if t in dict_tag:
                        dict_tag[t] += 1
                    else:
                        dict_tag[t] = 1

        list_categorie = list(dict_tag.items())
        self.num_categories = len(list_categorie)
        list_categorie.sort(key = lambda tup: tup[1])
        dict_translate = {}
        for i,tup in enumerate(list_categorie):
            dict_translate[tup[0]] = i
        self.list_tags = []
        for tags in list_tags:
            tags_new = set({})
            for tag in tags:
                tags_new.add(dict_translate[tag])
            self.list_tags.append(tags_new)
        self.list_id = list(np.arange(len(self.list_tags)))
        self.dict_id_tags = {}
        for ind, id in enumerate(self.list_id):
            tags = self.list_tags[ind]
            self.dict_id_tags[id] = tags


    def _sort_lists(self):
        sizes = [len(tags) for tags in self.list_tags]
        ordering = np.argsort(sizes)
        self.list_orientation = [self.list_orientation[ind] for ind in ordering]
        self.list_tags = [self.list_tags[ind] for ind in ordering]
        self.list_id = [self.list_id[ind] for ind in ordering]

    def _gen_dict_length_ids(self):
        self.dict_length_ids = {}
        for ind, tags in enumerate(self.list_tags):
            id = self.list_id[ind]
            length = len(tags)
            if length in self.dict_length_ids:
                self.dict_length_ids[length].add(id)
            else:
                self.dict_length_ids[length] = set({id})

    def _gen_dict_tag_ids(self):
        self.dict_tag_ids = {}
        for ind, tags in enumerate(self.list_tags):
            id = self.list_id[ind]
            for tag in tags:
                if tag in self.dict_tag_ids:
                    self.dict_tag_ids[tag].add(id)
                else:
                    self.dict_tag_ids[tag] = set({id})

    def _gen_verticals_horizontals(self):
        self.list_tags_vert = []
        self.list_tags_hor = []
        self.list_id_vert = []
        self.list_id_hor = []
        for ind in range(len(self.list_orientation)):
            orientation = self.list_orientation[ind]
            if orientation == 1:
                self.list_tags_vert.append(self.list_tags[ind])
                self.list_id_vert.append(self.list_id[ind])
            else:
                self.list_tags_hor.append(self.list_tags[ind])
                self.list_id_hor.append(self.list_id[ind])
        self.list_tags = None
        self.list_id = None
        self.list_orientation = None
        self.slides_length = len(self.list_id_hor) + int(len(self.list_id_vert)/2)

    def _pop_by_ind(self, ind, vertical=True):   
        if vertical:        
            tags = self.list_tags_vert.pop(ind)      
            id = self.list_id_vert.pop(ind)
        else:
            tags = self.list_tags_hor.pop(ind)      
            id = self.list_id_hor.pop(ind)
        self.dict_length_ids[len(tags)].remove(id)
        for tag in tags:
            self.dict_tag_ids[tag].remove(id)
        return id, tags  







    
