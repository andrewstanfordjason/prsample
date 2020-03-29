import numpy as  np
from math import gcd
import bisect

class prsample_iterator:
    ''' Iterator class '''
    def __init__(self, prsample):
        # Team object reference
        self._prsample = prsample
        # member variable to keep track of current index
        self._index = 0

    def __next__(self):

        if self._index < self._prsample.__len__():
            result = [self._prsample.get_example(self._index, batch_index) \
                for batch_index in range(self._prsample.examples_per_batch)]
            self._index +=1
            return result
        # End of Iteration
        raise StopIteration

def build_class_list_from_class_dirs(dir_list, file_types):
    """
        Builds a list of classes where each class is given by a directory of files. 
    """
    import glob
    import os

    file_list = []
    for path in dir_list:
            path = os.path.expanduser(path)
            path = os.path.expandvars(path)
            path = os.path.normpath(path)

            for file_type in file_types:
                file_list += glob.glob(os.path.join(path, '**', '*.' + str(file_type)))

    class_path_list = {os.path.dirname(path) for path in file_list}

    class_list = []
    for class_path in class_path_list:
        class_list.append([path for path in file_list if os.path.dirname(path) == class_path])

    return class_list

def get_class_idx_from_index(index, cumsum_examples_per_class):
    """
        Get an example for batch_no and batch_index.

        Args:
            batch_no: The number of the batch.
            batch_index: The index of the example from within the batch.

        Returns:
            The requested example.
    """

    class_idx = bisect.bisect(cumsum_examples_per_class, index) - 1
    assert class_idx >= 0, 'cannot have negative classes'
    return class_idx


def get_obj_idx_from_index(index, class_dict):
    """
        Get an example for batch_no and batch_index.

        Args:
            batch_no: The number of the batch.
            batch_index: The index of the example from within the batch.

        Returns:
            The requested example.
    """
    cumsum_examples_per_object = class_dict['cumsum_examples_per_object']

    obj_idx = bisect.bisect(cumsum_examples_per_object, index) - 1
    assert obj_idx >= 0, 'cannot have negative object indicies'

    offset = index - cumsum_examples_per_object[obj_idx]
    assert offset >= 0, 'cannot have negative offsets'

    return obj_idx, offset

class prsample:

    def __init__(self, class_list, examples_per_batch, examples_per_obj, get_example_from_obj, 
                no_duplicated_data = False, 
                shuffle = True, 
                seed = 69):
        """
            Creates a prsample object. The data that the sampling will be over is a list of classes. Each class should be described 
            by a list of all objects within that class.

            Two functions must be defined: examples_per_obj and get_example_from_obj. The function examples_per_obj must return 
            the integer number of examples that the given object can generate. 

            Args:
                class_list: A list of classes within each a list of objects that make that class.
                examples_per_batch: The batch size in examples. 
                examples_per_obj: A function that given a class returns the number of examples that can be derived from it.
                get_example_from_obj: A complimentary function to examples_per_obj that returns a given example from the class
                shuffle: If true the the class list will be shuffled between epochs.
                seed: An integer to seed the random number generator.
        """

        self._class_list = []
        for class_no, object_list in enumerate(class_list):
            data = {}
            data['class_no'] = class_no
            data['object_list'] = object_list
            self._class_list.append(data)

        assert isinstance(examples_per_batch, int) , 'examples_per_batch must be an int type.'
        # assert examples_per_batch > 0, 'examples_per_batch must be positive.'
        self.examples_per_batch = examples_per_batch

        assert callable(examples_per_obj), "examples_per_obj must be a function."
        assert callable(get_example_from_obj), "get_example_from_obj must be a function."
        self.examples_per_object = examples_per_obj
        self.get_example_from_object = get_example_from_obj

        assert isinstance(shuffle, bool) , 'shuffle must be an bool type.'
        self.shuffle = shuffle
        self.unshuffled_class_list = class_list
        
        assert isinstance(seed, int) , 'seed must be an int type.'
        self.seed = seed
        
        if self.examples_per_batch > 0:
            self.init_prsample()
        else:
            self.examples_per_batch_index = 0
            self.total_example_count = 0
            # self._cumsum_examples_per_class = np.zeros(len(self._class_list) + 1, dtype=int)

        assert isinstance(no_duplicated_data, bool) , 'no_duplicated_data must be an bool type.'
        self.no_duplicated_data = no_duplicated_data

        return

    def __len__(self):
        """
            Get the number of batches that the class list enumerates to, i.e. after withdrawing this many batches
            All the possible examples will have been seen.

            Returns:
                The total number of batches.
        """
        return self.examples_per_batch_index

    def get_example(self, batch_no, batch_index):
        """
            Get an example for batch_no and batch_index.

            Args:
                batch_no: The number of the batch.
                batch_index: The index of the example from within the batch.

            Returns:
                The requested example, unless examples_per_batch is zero in which case return None.
        """
        if self.examples_per_batch == 0:
            return None
        if self.no_duplicated_data:
            idx, is_valid = self._batch_to_idx(batch_no, batch_index, self.examples_per_batch, self.batch_strides, \
                self.examples_per_batch_index, self.total_example_count, self.offsets)
            if is_valid:
                return self.get_example_from_object(idx, self._class_list, self._cumsum_examples_per_class)
            else:
                return None
        else:

            idx, _ = self._batch_to_idx(batch_no, batch_index, self.examples_per_batch, self.batch_strides, \
                self.examples_per_batch_index, self.total_example_count, self.offsets)
            return self.get_example_from_object(idx, self._class_list, self._cumsum_examples_per_class)

    def _batch_to_idx(self, index, batch_index, examples_per_batch, batch_strides, \
            examples_per_batch_index, total_example_count, offsets):
        
        stride_point = batch_strides[batch_index] * (index+batch_index) + offsets[batch_index]
        unwraped_idx = batch_index + examples_per_batch*(stride_point%examples_per_batch_index)

        idx = unwraped_idx%total_example_count
        return idx, unwraped_idx < total_example_count

    def _is_coprime(self, a, b):
        return gcd(a, b) == 1

    def _find_batch_strides(self, examples_per_batch, examples_per_batch_index):

        if examples_per_batch_index <= 2:
            batch_strides = np.ones(examples_per_batch, dtype = int)
            offsets = np.zeros(batch_strides.shape)
            return batch_strides, offsets

        # return np.ones(examples_per_batch, dtype = int)
        assert examples_per_batch_index > 1, 'too many examples per batch'

        batch_strides = np.empty(examples_per_batch, dtype = int)
        #There must be enough coprimes numbers, i.e. examples_per_batch < copime_count(existing_strides)
        existing_strides = []
        for batch_index in range(examples_per_batch):
            stride = np.random.randint(2, examples_per_batch_index)

            init_stride = stride
            while not self._is_coprime(stride, examples_per_batch_index) or stride in existing_strides:
                stride += 1
                stride = stride % examples_per_batch_index
                if stride < 2:
                    stride = 2

                if stride == init_stride:
                    existing_strides = []

            existing_strides.append(stride)

            if len(existing_strides) == examples_per_batch:
                existing_strides = []
            batch_strides[batch_index] = stride
        offsets = np.random.randint(0, examples_per_batch_index, batch_strides.shape)
        return batch_strides, offsets

    def __iter__(self):
       ''' Returns the Iterator object '''
       return prsample_iterator(self)


    def init_prsample(self):

        self.iter_index = 0
        
        self.seed += 1
        np.random.seed(self.seed)

        if self.shuffle:
            np.random.shuffle(self._class_list)

        class_count = len(self._class_list)
        class_example_counts = np.zeros(class_count, dtype = int)
        running_class_example_count = 0
        for class_idx in range(class_count):
            object_list = self._class_list[class_idx]['object_list']

            examples_per_object = [self.examples_per_object(class_idx, obj_idx, self._class_list) for obj_idx in range(len(object_list))]
            
            cumsum_examples_per_object = np.zeros(len(examples_per_object)+1, dtype=int)
            cumsum_examples_per_object[1:] = np.cumsum(examples_per_object)
            cumsum_examples_per_object += running_class_example_count

            self._class_list[class_idx]['cumsum_examples_per_object'] = cumsum_examples_per_object
            class_example_counts[class_idx] = sum(examples_per_object)
            running_class_example_count += class_example_counts[class_idx]

        self._cumsum_examples_per_class = np.zeros(len(class_example_counts) + 1, dtype=int)
        self._cumsum_examples_per_class[1:] = np.cumsum(class_example_counts)    

        self.total_example_count = self._cumsum_examples_per_class[-1]

        self.number_of_batches = int(np.ceil(self.total_example_count / self.examples_per_batch))

        self.examples_per_batch_index = int(np.ceil(self.total_example_count/self.examples_per_batch))

        # print('examples_per_batch_index', self.examples_per_batch_index)
        # print('examples_per_batch', self.examples_per_batch)

        # Find the strides for the 
        self.batch_strides, self.offsets = self._find_batch_strides(self.examples_per_batch, self.examples_per_batch_index)
        # print('batch_strides', self.batch_strides)
        # print()
        return

    def run_self_checks(self):
        if self.examples_per_batch > 0:
            self._test_example_mapping(self.total_example_count, self._class_list, \
                    self.unshuffled_class_list, self.get_example_from_object, self._cumsum_examples_per_class)
            self._test_batch_to_index_mapping(self.examples_per_batch, self.examples_per_batch_index, \
                    self.batch_strides, self.total_example_count, self.offsets)
        return

    # This test that given all expected indicies all outputs are unique
    def _test_example_mapping(self, total_example_count, class_list, unshuffled_class_list,
            get_example_from_object, cumsum_examples_per_class):
        seen_examples = set()
        for idx in range(total_example_count):
            ex = get_example_from_object(idx, class_list, cumsum_examples_per_class)

            if ex is not None:
                assert(ex not in seen_examples)
                seen_examples.add(ex)
                if ex.is_valid != None:
                    assert(ex.is_valid(unshuffled_class_list))
        assert len(seen_examples) == total_example_count, str(len(seen_examples)) + ' ' + str(total_example_count)

    def _test_batch_to_index_mapping(self, examples_per_batch, examples_per_batch_index, batch_strides, \
            total_example_count, offsets):
        seen_indicies = set()
        for index in range(examples_per_batch_index):
            for batch_index in range(examples_per_batch):
                idx, is_valid = self._batch_to_idx(index, batch_index, examples_per_batch, batch_strides, \
                    examples_per_batch_index, total_example_count, offsets)
                if is_valid:
                    seen_indicies.add(idx)

        assert len(seen_indicies) == total_example_count, 'seen_indicies: ' + str(len(seen_indicies)) + ' total_example_count: ' +  str(total_example_count)
