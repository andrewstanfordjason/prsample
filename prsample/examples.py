import numpy as np
import prsample as prs

class Nlet_Example():
    '''
        This class represents an example per object.
    '''
    def __init__(self, class_obj_pairs):
        self.class_obj_pairs = class_obj_pairs
        for (class_idx, obj_idx) in self.class_obj_pairs:
	        assert class_idx >= 0, "class_idx must be non-negative"
	        assert obj_idx >= 0, "obj_idx must be non-negative"
        return

    def __hash__(self): 
        return super.__hash__(self.class_obj_pairs)

    def __eq__(self, other): 
        for (class_idx, obj_idx, other_class_idx, other_obj_idx) in zip(self.class_obj_pairs, other.class_obj_pairs):
        	if (class_idx != other_class_idx) or (obj_idx != other_obj_idx):
        		return False
        return True

    def __ne__(self, other): 
        return not self.__eq__(other)

    def get(self):
        return self.class_obj_pairs

    def is_valid(self, class_list):

        for (class_idx, obj_idx) in self.class_obj_pairs:
	        assert class_idx >= 0, "class_idx must be non-negative"
	        assert obj_idx >= 0, "obj_idx must be non-negative"

	        assert class_idx < len(class_list), "class_idx index must be within class_list"
	        assert obj_idx < len(class_list[class_idx]), "obj_idx index must be within class_list[class_idx]"
        return True



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
        assert self.obj_b_idx >= 0, "obj_a_idx must be non-negative"

        assert self.class_a < len(class_list), "class_a index must be within object_list"
        assert self.class_b < len(class_list), "class_b index must be within object_list"
        assert self.obj_a_idx < len(class_list[self.class_a]), "object_a index must be within object_list[class_a]"
        assert self.obj_b_idx < len(class_list[self.class_b]), "object_b index must be within object_list[class_b]"

        return True

class Triplet_Example():
    '''
        This class represents an example per object triplet.
    '''
    def __init__(self, class_a, obj_a_idx, class_b, obj_b_idx, class_c, obj_c_idx):
        self.class_a = class_a
        self.obj_a_idx = obj_a_idx        
        self.class_b = class_b
        self.obj_b_idx = obj_b_idx   
        self.class_c = class_c
        self.obj_c_idx = obj_c_idx
        return

    def __hash__(self): 
        return super.__hash__((self.class_a, self.obj_a_idx, self.class_b, self.obj_b_idx, self.class_c, self.obj_c_idx))

    def __eq__(self, other): 
        return self.class_a == other.class_a and self.obj_a_idx == other.obj_a_idx and \
               self.class_b == other.class_b and self.obj_b_idx == other.obj_b_idx and \
               self.class_c == other.class_c and self.obj_c_idx == other.obj_c_idx

    def __ne__(self, other): 
        return not self.__eq__(other)

    def get(self):
        return (self.class_a, self.obj_a_idx, self.class_b, self.obj_b_idx, self.class_c, self.obj_c_idx)

    def __str__(self): 
        return str(self.class_a) + '(' + str(self.obj_a_idx) + ') ' \
             + str(self.class_b) + '(' + str(self.obj_b_idx) + ') ' \
             + str(self.class_c) + '(' + str(self.obj_c_idx) + ')'

    def is_valid(self, class_list):
        assert self.class_a >= 0, "class_a must be non-negative"
        assert self.class_b >= 0, "class_b must be non-negative"
        assert self.class_c >= 0, "class_c must be non-negative"
        assert self.obj_a_idx >= 0, "obj_a_idx must be non-negative"
        assert self.obj_b_idx >= 0, "obj_b_idx must be non-negative"
        assert self.obj_c_idx >= 0, "obj_c_idx must be non-negative"

        assert self.class_a < len(class_list), "class_a index must be within object_list"
        assert self.class_b < len(class_list), "class_b index must be within object_list"
        assert self.class_c < len(class_list), "class_c index must be within object_list"
        assert self.obj_a_idx < len(class_list[self.class_a]), "object_a index must be within object_list[class_a]"
        assert self.obj_b_idx < len(class_list[self.class_b]), "object_b index must be within object_list[class_b]"
        assert self.obj_c_idx < len(class_list[self.class_c]), "object_c index must be within object_list[class_c]"

        return True


class Pos_Anc_Neg_Triplet_Example(Triplet_Example):

    def __init__(self, class_a, obj_a_idx, class_b, obj_b_idx, class_c, obj_c_idx):
        Triplet_Example.__init__(self, class_a, obj_a_idx, class_b, obj_b_idx, class_c, obj_c_idx)
        return

    def is_valid(self, class_list):
        if not Triplet_Example.is_valid(self, class_list):
            return False

        assert self.class_a == self.class_b, 'Class a and class b must be the same.'
        assert self.obj_a_idx != self.obj_b_idx, 'Object a and object b must be the different.'
        assert self.class_b != self.class_c, 'Class b and class c must be the different.'

        return True

    @staticmethod
    def examples_per_obj(class_idx, obj_idx, class_list):

        n = len(class_list[class_idx]["object_list"])
        unordered_in_class_pos_example_count = n - obj_idx - 1

        unordered_out_of_class_neg_example_count =  sum([len(c["object_list"]) for c in class_list[:class_idx]])  \
            + sum([len(c["object_list"]) for c in class_list[class_idx+1:]])

        return unordered_in_class_pos_example_count * unordered_out_of_class_neg_example_count

    @staticmethod
    def get_example_from_obj(index, class_list, cumsum_examples_per_class):

        # print(class_list)
        class_pos_idx = prs.get_class_idx_from_index(index, cumsum_examples_per_class)
        obj_pos_idx, offset = prs.get_obj_idx_from_index(index, class_list[class_pos_idx])

        class_anc_idx = class_pos_idx
        # print('class_anc_idx', class_anc_idx, 'len', len(class_list[class_anc_idx]), class_list[class_anc_idx])
        remaining_objs_in_class = len(class_list[class_anc_idx]["object_list"]) - obj_pos_idx - 1

        # print(remaining_objs_in_class)
        assert remaining_objs_in_class >= 0

        obj_anc_idx = (offset % remaining_objs_in_class) + obj_pos_idx + 1
        
        offset //= remaining_objs_in_class

        class_neg_idx = 0
        obj_neg_idx = offset

        if class_neg_idx == class_anc_idx:
            class_neg_idx += 1
        while obj_neg_idx >= len(class_list[class_neg_idx]["object_list"]):
            obj_neg_idx -= len(class_list[class_neg_idx]["object_list"])
            class_neg_idx += 1

            if class_neg_idx == class_anc_idx:
                class_neg_idx += 1

        return Pos_Anc_Neg_Triplet_Example( 
            class_list[class_pos_idx]['class_no'], obj_pos_idx, 
            class_list[class_anc_idx]['class_no'], obj_anc_idx, 
            class_list[class_neg_idx]['class_no'], obj_neg_idx)










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



