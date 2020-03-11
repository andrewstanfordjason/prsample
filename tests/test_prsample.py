import pytest
import numpy as np
import prsample as prs
import prsample.examples as prse

def build_class_list(class_count, objects_per_class):
    class_list = []
    object_id = 0
    for class_no in range(class_count):
        class_object_count = objects_per_class(class_no)
        object_list = [str(object_id + i) for i in range(class_object_count)]
        object_id += class_object_count
        class_list.append(object_list)

    return class_list

def test_get_obj_no_from_index():

    class_count = 4
    examples_per_class = 8
    #16 classes each with 8 examples
    class_list = build_class_list(class_count, lambda x : examples_per_class)

    examples_per_batch = 5
    p = prs.prsample(class_list, examples_per_batch, prse.Single_Example.examples_per_obj, prse.Single_Example.get_example_from_obj)
    
    total_examples = class_count * examples_per_class

    seen_examples = set()
    for index in range(total_examples):
        # class_index, obj_index, offset = prs.get_obj_no_from_index(index, p.example_lut)

        class_idx = prs.get_class_idx_from_index(index, p._cumsum_examples_per_class)
        obj_idx, offset = prs.get_obj_idx_from_index(index, p._class_list[class_idx])

        seen_examples.add((class_idx, obj_idx, offset))

    assert len(seen_examples) == total_examples, 'some examples seen twice'
    return

@pytest.mark.parametrize("examples_per_batch",[i for i in range(0, 48)])
def test_run_self_checks_singles(examples_per_batch):
    object_list = build_class_list(16, lambda x : np.random.randint(3, 12))
    p = prs.prsample(object_list, examples_per_batch, prse.Single_Example.examples_per_obj, prse.Single_Example.get_example_from_obj)
    p.run_self_checks()
    return

@pytest.mark.parametrize("examples_per_batch",[i for i in range(0, 48)])
def test_run_self_checks_pairs(examples_per_batch):
    object_list = build_class_list(16, lambda x : np.random.randint(3, 12))
    p = prs.prsample(object_list, examples_per_batch, prse.Pair_Example.examples_per_obj, prse.Pair_Example.get_example_from_obj)
    p.run_self_checks()
    return

@pytest.mark.parametrize("examples_per_batch",[i for i in range(0, 48)])
def test_shared_object_singles(examples_per_batch):

    total_batch_size = 48
    object_list = []

    # This lambda fn guarentees that all classes have a different number of objects
    # Meaning after the shuffle one of the two prsample objects will fail if
    # it has not correctly copied the object list
    object_list = build_class_list(32, lambda x : x)

    p_batch_size = examples_per_batch
    q_batch_size = total_batch_size - examples_per_batch

    #FIXME zero sized batches should be allowed

    p = prs.prsample(object_list, p_batch_size, prse.Single_Example.examples_per_obj, prse.Single_Example.get_example_from_obj, seed = 2)
    q = prs.prsample(object_list, q_batch_size, prse.Single_Example.examples_per_obj, prse.Single_Example.get_example_from_obj, seed = 1)

    for index in range(max(p.__len__(), q.__len__())):
        for batch_index in range(p_batch_size):
            ex = p.get_example(index, batch_index)
            class_id, class_no = ex.get()
            ex.is_valid(object_list)
        for batch_index in range(q_batch_size):
            ex = q.get_example(index, batch_index)
            ex.is_valid(object_list)

    return

@pytest.mark.parametrize("examples_per_batch",[i for i in range(0, 16)])
def test_shared_object_pairs(examples_per_batch):

    total_batch_size = 16
    object_list = []

    # This lambda fn guarentees that all classes have a different number of objects
    # Meaning after the shuffle one of the two prsample objects will fail if
    # it has not correctly copied the object list
    object_list = build_class_list(32, lambda x : x)

    p_batch_size = examples_per_batch
    q_batch_size = total_batch_size - examples_per_batch

    #FIXME zero sized batches should be allowed

    p = prs.prsample(object_list, p_batch_size, prse.Pair_Example.examples_per_obj, prse.Pair_Example.get_example_from_obj, seed = 2)
    q = prs.prsample(object_list, q_batch_size, prse.Pair_Example.examples_per_obj, prse.Pair_Example.get_example_from_obj, seed = 1)

    for index in range(max(p.__len__(), q.__len__())):
        for batch_index in range(p_batch_size):
            ex = p.get_example(index, batch_index)
            ex.is_valid(object_list)
        for batch_index in range(q_batch_size):
            ex = q.get_example(index, batch_index)
            ex.is_valid(object_list)

    return

def test_version_number():
    assert prs.__version__ == '0.0.3'
    return


