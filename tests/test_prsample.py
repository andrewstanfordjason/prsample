import pytest
import numpy as np
import prsample as prs
import prsample.examples as prse

'''
    Thisis a list of classes that generate examples, each one shall be tested.
'''
example_list = [
    prse.Single_Example,
    prse.Ordered_In_Class_Pair_Example,
    prse.Unordered_In_Class_Pair_Example, 
    prse.Unordered_Out_of_Class_Pair_Example]

'''
    This describes the expected number of examples for a given class_list.
'''
expected_count_fn = [
    lambda class_list : sum([len(c) for c in class_list]),
    lambda class_list : sum([len(c) * len(c) for c in class_list]),
    lambda class_list : sum([(len(c) * (len(c)-1) // 2) for c in class_list]),
    lambda class_list : sum([len(class_list[i]) * sum([len(class_list[j]) for j in range(i+1, len(class_list))]) for i in range(len(class_list))]),
    ]

def build_class_list(class_count, objects_per_class):
    class_list = []
    object_id = 0
    for class_no in range(class_count):
        class_object_count = objects_per_class(class_no)
        object_list = [str(object_id + i) for i in range(class_object_count)]
        object_id += class_object_count
        class_list.append(object_list)

    return class_list

@pytest.mark.parametrize(("example_class", "expected_count_fn"), zip(example_list, expected_count_fn))
@pytest.mark.parametrize("no_duplicated_data",[True, False])
@pytest.mark.parametrize("examples_per_batch",[i for i in range(1, 8)])
def test_directed_set_size(examples_per_batch, no_duplicated_data, example_class, expected_count_fn):

    class_list = build_class_list(5, lambda x : x)
    p = prs.prsample(class_list, examples_per_batch, example_class.examples_per_obj, \
        example_class.get_example_from_obj, seed = 2, no_duplicated_data = no_duplicated_data)

    assert p.total_example_count == expected_count_fn(class_list) 
    return

@pytest.mark.parametrize("example_class", example_list)
@pytest.mark.parametrize("no_duplicated_data",[True, False])
@pytest.mark.parametrize("examples_per_batch",[i for i in range(1, 8)])
def test_get_obj_no_from_index(examples_per_batch, no_duplicated_data, example_class):

    class_count = 2
    examples_per_class = 5
    #16 classes each with 8 examples
    class_list = build_class_list(class_count, lambda x : examples_per_class)

    p = prs.prsample(class_list, examples_per_batch, example_class.examples_per_obj, \
        example_class.get_example_from_obj, no_duplicated_data = no_duplicated_data)
    
    total_examples = class_count * examples_per_class

    seen_examples = set()
    for index in range(total_examples):

        class_idx = prs.get_class_idx_from_index(index, p._cumsum_examples_per_class)
        obj_idx, offset = prs.get_obj_idx_from_index(index, p._class_list[class_idx])

        seen_examples.add((class_idx, obj_idx, offset))

    assert len(seen_examples) == total_examples, 'some examples seen twice'
    return

@pytest.mark.parametrize("example_class", example_list)
@pytest.mark.parametrize("no_duplicated_data",[True, False])
@pytest.mark.parametrize("examples_per_batch",[i for i in range(0, 48)])
def test_run_self_checks(examples_per_batch, no_duplicated_data, example_class):
    object_list = build_class_list(5, lambda x : np.random.randint(3, 12))
    p = prs.prsample(object_list, examples_per_batch, example_class.examples_per_obj, \
        example_class.get_example_from_obj, no_duplicated_data = no_duplicated_data)
    p.run_self_checks()
    return

@pytest.mark.parametrize("example_class", example_list)
@pytest.mark.parametrize("no_duplicated_data",[True, False])
@pytest.mark.parametrize("examples_per_batch", [i for i in range(0, 18)])
def test_tiny_set(examples_per_batch, no_duplicated_data, example_class):

    total_batch_size = 24

    # This lambda fn guarentees that all classes have a different number of objects
    # Meaning after the shuffle one of the two prsample objects will fail if
    # it has not correctly copied the object list
    class_list = build_class_list(1, lambda x : 16)

    p_batch_size = examples_per_batch
    q_batch_size = total_batch_size - examples_per_batch

    p = prs.prsample(class_list, p_batch_size, example_class.examples_per_obj, \
        example_class.get_example_from_obj, seed = 2, no_duplicated_data = no_duplicated_data)
    q = prs.prsample(class_list, q_batch_size, example_class.examples_per_obj, \
        example_class.get_example_from_obj, seed = 1, no_duplicated_data = no_duplicated_data)

    p_set = set()
    q_set = set()
    for index in range(max(p.__len__(), q.__len__())):
        for batch_index in range(p_batch_size):
            ex = p.get_example(index, batch_index)
            if ex is not None:
                ex.is_valid(class_list)
                p_set.add(ex)
        for batch_index in range(q_batch_size):
            ex = q.get_example(index, batch_index)
            if ex is not None:
                ex.is_valid(class_list)
                q_set.add(ex)

    assert len(p_set) == p.total_example_count
    assert len(q_set) == q.total_example_count
    return

@pytest.mark.parametrize("example_class", example_list)
@pytest.mark.parametrize("no_duplicated_data",[True, False])
@pytest.mark.parametrize("examples_per_batch", [i for i in range(0, 24)])
def test_singles(examples_per_batch, no_duplicated_data, example_class):

    # This lambda fn guarentees that all classes have a different number of objects
    # Meaning after the shuffle one of the two prsample objects will fail if
    # it has not correctly copied the object list
    class_list = build_class_list(5, lambda x : x)

    p_batch_size = examples_per_batch

    p = prs.prsample(class_list, p_batch_size, example_class.examples_per_obj, \
        example_class.get_example_from_obj, seed = 2, no_duplicated_data = no_duplicated_data)

    p_set = set()
    empty_p_example_count = 0

    for index in range(p.__len__()):
        for batch_index in range(p_batch_size):
            ex = p.get_example(index, batch_index)
            if ex is not None:
                ex.is_valid(class_list)
                p_set.add(ex)
            else:
                empty_p_example_count += 1

    total_example_requested = p.__len__()*p_batch_size

    if not no_duplicated_data:
        assert empty_p_example_count == 0

    assert len(p_set) == p.total_example_count
    return

@pytest.mark.parametrize("example_class", example_list)
@pytest.mark.parametrize("no_duplicated_data",[True, False])
@pytest.mark.parametrize("examples_per_batch",[i for i in range(0, 24)])
def test_shared_object_singles(examples_per_batch, no_duplicated_data, example_class):

    total_batch_size = 24

    # This lambda fn guarentees that all classes have a different number of objects
    # Meaning after the shuffle one of the two prsample objects will fail if
    # it has not correctly copied the object list
    class_list = build_class_list(5, lambda x : x)

    p_batch_size = examples_per_batch
    q_batch_size = total_batch_size - examples_per_batch

    p = prs.prsample(class_list, p_batch_size, example_class.examples_per_obj, \
        example_class.get_example_from_obj, seed = 2, no_duplicated_data = no_duplicated_data)
    q = prs.prsample(class_list, q_batch_size, example_class.examples_per_obj, \
        example_class.get_example_from_obj, seed = 1, no_duplicated_data = no_duplicated_data)

    p_set = set()
    q_set = set()
    empty_p_example_count = 0
    empty_q_example_count = 0
    for index in range(max(p.__len__(), q.__len__())):
        for batch_index in range(p_batch_size):
            ex = p.get_example(index, batch_index)
            if ex is not None:
                ex.is_valid(class_list)
                p_set.add(ex)
            else:
                empty_p_example_count += 1
        for batch_index in range(q_batch_size):
            ex = q.get_example(index, batch_index)
            if ex is not None:
                ex.is_valid(class_list)
                q_set.add(ex)
            else:
                empty_q_example_count += 1

    assert len(p_set) == p.total_example_count
    assert len(q_set) == q.total_example_count

    if not no_duplicated_data:
        assert empty_q_example_count == 0
        assert empty_p_example_count == 0
    return

def test_version_number():
    assert prs.__version__ == '0.0.3'
    return


