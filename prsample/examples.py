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

    def __ne__(self, other): 
        return not self.__eq__(other)

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
        assert offset == 0
        return Single_Example(class_list[class_idx]['class_no'], obj_idx)


class Pair_Example():
    '''
        This class represents an example per object pair.
    '''
    def __init__(self, class_a, obj_a_idx, class_b, obj_b_idx):
        self.class_a = class_a
        self.obj_a_idx = obj_a_idx        
        self.class_b = class_b
        self.obj_b_idx = obj_b_idx
        return

    def __hash__(self): 
        return super.__hash__((self.class_a, self.obj_a_idx, self.class_b, self.obj_b_idx))

    def __eq__(self, other): 
        return self.class_a == other.class_a and  self.obj_a_idx == other.obj_a_idx and \
            self.class_b == other.class_b and self.obj_b_idx == other.obj_b_idx

    def __ne__(self, other): 
        return not self.__eq__(other)

    def get(self):
        return (self.class_a, self.obj_a_idx, self.class_b, self.obj_b_idx)

    def __str__(self): 
        return str(self.class_a)  + '(' + str(self.obj_a_idx) + ') ' + str(self.class_b) + '(' +str(self.obj_b_idx) + ')'

    def is_valid(self, class_list):
        assert self.class_a >= 0, "class_a must be non-negative"
        assert self.class_b >= 0, "class_b must be non-negative"
        assert self.obj_a_idx >= 0, "obj_a_idx must be non-negative"
        assert self.obj_a_idx >= 0, "obj_a_idx must be non-negative"

        assert self.class_a < len(class_list), "class_a index must be within object_list"
        assert self.class_b < len(class_list), "class_b index must be within object_list"
        assert self.obj_a_idx < len(class_list[self.class_a]), "object_a index must be within object_list[class_a]"
        assert self.obj_b_idx < len(class_list[self.class_b]), "object_b index must be within object_list[class_b]"

        return True

class Ordered_In_Class_Pair_Example(Pair_Example):

    def __init__(self, class_a, obj_a_idx, class_b, obj_b_idx):
        Pair_Example.__init__(self, class_a, obj_a_idx, class_b, obj_b_idx)
        return

    def is_valid(self, class_list):
        if not Pair_Example.is_valid(self, class_list):
            return False
        assert self.class_a == self.class_b, 'Class a and class b must be the same.'

        return True

    @staticmethod
    def examples_per_obj(class_idx, object_idx, class_list):
        return len(class_list[class_idx]["object_list"])

    @staticmethod
    def get_example_from_obj(index, class_list, cumsum_examples_per_class):

        class_idx = prs.get_class_idx_from_index(index, cumsum_examples_per_class)
        obj_idx, offset = prs.get_obj_idx_from_index(index, class_list[class_idx])

        return Ordered_In_Class_Pair_Example(class_list[class_idx]['class_no'], obj_idx, class_list[class_idx]['class_no'], offset)

class Unordered_In_Class_Pair_Example(Pair_Example):

    def __init__(self, class_a, obj_a_idx, class_b, obj_b_idx):
        Pair_Example.__init__(self, class_a, obj_a_idx, class_b, obj_b_idx)
        return

    def is_valid(self, class_list):
        if not Pair_Example.is_valid(self, class_list):
            return False
        assert self.class_a == self.class_b, 'Class a and class b must be the same.'
            
        return True

    @staticmethod
    def examples_per_obj(class_idx, obj_idx, class_list):
        n = len(class_list[class_idx]["object_list"])
        return n - obj_idx - 1

    @staticmethod
    def get_example_from_obj(index, class_list, cumsum_examples_per_class):

        class_idx = prs.get_class_idx_from_index(index, cumsum_examples_per_class)
        obj_a_idx, offset = prs.get_obj_idx_from_index(index, class_list[class_idx])

        obj_b_idx = obj_a_idx + 1 + offset

        return Unordered_In_Class_Pair_Example(class_list[class_idx]['class_no'], obj_a_idx, class_list[class_idx]['class_no'], obj_b_idx)


class Unordered_Out_of_Class_Pair_Example(Pair_Example):

    def __init__(self, class_a, obj_a_idx, class_b, obj_b_idx):
        Pair_Example.__init__(self, class_a, obj_a_idx, class_b, obj_b_idx)
        return

    def is_valid(self, class_list):
        if not Pair_Example.is_valid(self, class_list):
            return False
        assert self.class_a != self.class_b, 'Class a and class b cannot be the same.'
        return True

    @staticmethod
    def examples_per_obj(class_idx, obj_idx, class_list):
        return sum([len(c["object_list"]) for c in class_list[class_idx+1:]])

    @staticmethod
    def get_example_from_obj(index, class_list, cumsum_examples_per_class):

        class_a_idx = prs.get_class_idx_from_index(index, cumsum_examples_per_class)
        obj_a_idx, offset = prs.get_obj_idx_from_index(index, class_list[class_a_idx])

        class_b_idx = class_a_idx+1
        obj_b_idx = offset

        while obj_b_idx >= len(class_list[class_b_idx]["object_list"]):
            obj_b_idx -= len(class_list[class_b_idx]["object_list"])
            class_b_idx += 1

        return Unordered_Out_of_Class_Pair_Example(class_list[class_a_idx]['class_no'], obj_a_idx, class_list[class_b_idx]['class_no'], obj_b_idx)



