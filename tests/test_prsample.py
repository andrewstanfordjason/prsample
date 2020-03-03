import pytest
import numpy as np
import prsample as prs

class Pair_Example():
    def __init__(self, class_a, a, class_b, b):
        self.class_a = class_a
        self.a = a        
        self.class_b = class_b
        self.b = b
        return

    def __hash__(self): 
        return super.__hash__((self.class_a, self.a, self.class_b, self.b))

    def __str__(self): 
        return str(self.class_a)  + '(' + str(self.a) + ') ' + str(self.class_b) + '(' +str(self.b) + ')'

    def is_valid(self, object_list):
        assert self.class_a >= 0, "class_a must be non-negative"
        assert self.class_b >= 0, "class_b must be non-negative"
        assert self.a >= 0, "a must be non-negative"
        assert self.b >= 0, "b must be non-negative"

        assert self.class_a < len(object_list), "class_a index must be within object_list"
        assert self.class_b < len(object_list), "class_b index must be within object_list"
        assert self.a < len(object_list[ self.class_a]['class_id']), "class_a index must be within object_list[class_a]"
        assert self.b < len(object_list[ self.class_b]['class_id']), "class_a index must be within object_list[class_b]"

        return True

    def get(self):
        return (self.class_a, self.a, self.class_b, self.b)

    @staticmethod
    def examples_per_obj(object_no, object_list):
        l = len(object_list[object_no]["class_id"])
        return l*l

    @staticmethod
    def get_example_from_obj(idx, object_example_lut, object_list, cumulative_example_lut):
        class_a, offset = prs.get_obj_no_from_index(idx, object_example_lut)
        examples_in_class_a = object_list[class_a]['class_example_count']
        a = offset // examples_in_class_a
        b = offset % examples_in_class_a
        return Pair_Example(object_list[class_a]['class_no'], a, object_list[class_a]['class_no'], b)

class Singleton_Example():
    def __init__(self, obj_list_idx, idx):
        self.obj_list_idx = obj_list_idx
        self.idx = idx
        return

    def __hash__(self): 
        return super.__hash__((self.obj_list_idx, self.idx))

    def is_valid(self, object_list):
        assert self.obj_list_idx >= 0, "obj_list_idx must be non-negative"
        assert self.idx >= 0, "idx must be non-negative"

        assert self.obj_list_idx < len(object_list), "obj_list_idx index must be within object_list"
        assert self.idx < len(object_list[self.obj_list_idx]['class_id']), "idx index must be within object_list[obj_list_idx]"
        return True

    def get(self):
        return (self.obj_list_idx, self.idx)

    @staticmethod
    def examples_per_obj(object_no, object_list):
        return len(object_list[object_no]["class_id"])

    @staticmethod
    def get_example_from_obj(idx, object_example_lut, object_list, cumulative_example_lut):
        obj_idx, offset = prs.get_obj_no_from_index(idx, object_example_lut)
        return Singleton_Example(object_list[obj_idx]['class_no'], offset)

def build_object_list(class_count, objects_per_class):

    object_list = []
    object_id = 0 # unique id for each object across all classes and objects
    for class_no in range(class_count):
        object_example_count = objects_per_class(class_no)
        class_dict = {"class_example_count": object_example_count}
        class_dict['class_no'] = class_no
        class_dict["class_id"] = [str(object_id + i) for i in range(object_example_count)]
        object_id += object_example_count
        object_list.append(class_dict)

    return object_list

def test_get_obj_no_from_index():

    class_count = 4
    examples_per_class = 8
    #16 classes each with 8 examples
    object_list = build_object_list(class_count, lambda x : examples_per_class)

    examples_per_batch = 5
    p = prs.prsample(object_list, examples_per_batch, Singleton_Example.examples_per_obj, Singleton_Example.get_example_from_obj)
    
    total_examples = class_count* examples_per_class

    seen_examples = set()
    for index in range(total_examples):
        class_id, offset = prs.get_obj_no_from_index(index, p.example_lut)
        seen_examples.add((class_id, offset))

    assert len(seen_examples) == total_examples, 'some examples seen twice'
    return


@pytest.mark.parametrize("examples_per_batch",[i for i in range(1, 48)])
def test_run_self_checks_singles(examples_per_batch):
    object_list = build_object_list(16, lambda x : np.random.randint(3, 12))
    p = prs.prsample(object_list, examples_per_batch, Singleton_Example.examples_per_obj, Singleton_Example.get_example_from_obj)
    p.run_self_checks()
    return

@pytest.mark.parametrize("examples_per_batch",[i for i in range(1, 48)])
def test_run_self_checks_pairs(examples_per_batch):
    object_list = build_object_list(16, lambda x : np.random.randint(3, 12))
    p = prs.prsample(object_list, examples_per_batch, Pair_Example.examples_per_obj, Pair_Example.get_example_from_obj)
    p.run_self_checks()
    return

@pytest.mark.parametrize("examples_per_batch",[i for i in range(1, 48)])
def test_shared_object(examples_per_batch):

    total_batch_size = 48
    object_list = []

    # This lambda fn guarentees that all classes have a different number of objects
    # Meaning after the shuffle one of the two prsample objects will fail if
    # it has not correctly copied the object list
    object_list = build_object_list(32, lambda x : x)

    p_batch_size = examples_per_batch
    q_batch_size = total_batch_size - examples_per_batch

    #FIXME zero sized batches should be allowed

    p = prs.prsample(object_list, p_batch_size, Singleton_Example.examples_per_obj, Singleton_Example.get_example_from_obj, seed = 2)
    q = prs.prsample(object_list, q_batch_size, Singleton_Example.examples_per_obj, Singleton_Example.get_example_from_obj, seed = 1)

    for index in range(max(p.__len__(), q.__len__())):
        for batch_index in range(p_batch_size):
            ex = p.get_example(index, batch_index)
            class_id, class_no = ex.get()
            ex.is_valid(object_list)
        for batch_index in range(q_batch_size):
            ex = q.get_example(index, batch_index)
            ex.is_valid(object_list)

    return

if __name__== "__main__":
    test_shared_object()


