import numpy as np
import prsample as prs

class Single_Example():
    '''
        This class represents an example per object.
    '''
    def __init__(self, class_idx, obj_idx):
        self.class_idx = class_idx
        self.obj_idx = obj_idx
        assert self.class_idx >= 0, "class_idx must be non-negative"
        assert self.obj_idx >= 0, "obj_idx must be non-negative"
        return

    def __hash__(self): 
        return super.__hash__((self.class_idx, self.obj_idx))

    def __eq__(self, other): 
        return self.class_idx == other.class_idx and  self.obj_idx == other.obj_idx

    def get(self):
        return (self.class_idx, self.obj_idx)

    def is_valid(self, class_list):
        assert self.class_idx >= 0, "class_idx must be non-negative"
        assert self.obj_idx >= 0, "obj_idx must be non-negative"

        assert self.class_idx < len(class_list), "class_idx index must be within class_list"
        assert self.obj_idx < len(class_list[self.class_idx]), "obj_idx index must be within class_list[class_idx]"
        return True

    @staticmethod
    def examples_per_obj(class_idx, object_idx, class_list):
        return 1

    @staticmethod
    def get_example_from_obj(index, class_list, cumsum_examples_per_class):
        class_idx = prs.get_class_idx_from_index(index, cumsum_examples_per_class)
        obj_idx, offset = prs.get_obj_idx_from_index(index, class_list[class_idx])
        return Single_Example(class_list[class_idx]['class_no'], obj_idx)


class Pair_Example():
    '''
        This class represents an example per object pair.
    '''
    def __init__(self, class_a, a, class_b, b):
        self.class_a = class_a
        self.a = a        
        self.class_b = class_b
        self.b = b
        return

    def __hash__(self): 
        return super.__hash__((self.class_a, self.a, self.class_b, self.b))

    def __eq__(self, other): 
        return self.class_a == other.class_a and  self.a == other.a and \
            self.class_b == other.class_b and  self.b == other.b

    def get(self):
        return (self.class_a, self.a, self.class_b, self.b)

    def __str__(self): 
        return str(self.class_a)  + '(' + str(self.a) + ') ' + str(self.class_b) + '(' +str(self.b) + ')'

    def is_valid(self, class_list):
        assert self.class_a >= 0, "class_a must be non-negative"
        assert self.class_b >= 0, "class_b must be non-negative"
        assert self.a >= 0, "a must be non-negative"
        assert self.b >= 0, "b must be non-negative"

        assert self.class_a < len(class_list), "class_a index must be within object_list"
        assert self.class_b < len(class_list), "class_b index must be within object_list"
        assert self.a < len(class_list[self.class_a]), "class_a index must be within object_list[class_a]"
        assert self.b < len(class_list[self.class_b]), "class_a index must be within object_list[class_b]"

        return True

    @staticmethod
    def examples_per_obj(class_idx, object_idx, class_list):
        return len(class_list[class_idx]["object_list"])

    @staticmethod
    def get_example_from_obj(index, class_list, cumsum_examples_per_class):

        class_idx = prs.get_class_idx_from_index(index, cumsum_examples_per_class)
        obj_idx, offset = prs.get_obj_idx_from_index(index, class_list[class_idx])

        return Pair_Example(class_list[class_idx]['class_no'], obj_idx, class_list[class_idx]['class_no'], offset)

    #TODO the all non-class-pairs

    #TODO the within class pairs with a,b == b,a


