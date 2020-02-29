import pytest
import numpy as np
from prsample import prsample as prs

class Example():
    def __init__(self, obj_list_idx, idx):
        self.obj_list_idx = obj_list_idx
        self.idx = idx
        return

    def __hash__(self): 
        return super.__hash__((self.obj_list_idx, self.idx))

    def is_valid(self, object_list):
        if self.obj_list_idx >= len(object_list):
            return False
        if self.idx >= len(object_list[ self.obj_list_idx]['class_id']):
            return False
        return True

    def get(self):
        return (self.obj_list_idx, self.idx)

def examples_per_obj(object_no, object_list):
    return len(object_list[object_no]["class_id"])

def get_example_from_obj(idx, object_example_lut, object_list, cumulative_example_lut):
    obj_idx, offset = prs.get_obj_no_from_index(idx, object_example_lut, cumulative_example_lut)
    return Example(obj_idx, offset)

def test_run_self_checks():

    class_count = 16
    examples_per_batch = 3

    object_list = []
    class_id = 0
    for folder in range(class_count):
        class_example_count = np.random.randint(3, 8)
        class_dict = {"class_example_count": class_example_count}
        class_dict["class_id"] = [str(class_id + i) for i in range(class_example_count)]
        class_id += class_example_count
        object_list.append(class_dict)

    for examples_per_batch in range(1, 16):
	    p = prs.prsample(object_list, examples_per_batch, examples_per_obj, get_example_from_obj)
	    p.run_self_checks()

    return
