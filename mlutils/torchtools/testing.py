import torch
from collections.abc import Iterable
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter
date_format = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s', datefmt=date_format)
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)

def assert_tensor_objects_equal(obtained_object, expected_object, check_keys=False, sort_objects=True):
    assert type(obtained_object) == type(expected_object), TypeError(f'object has type={type(obtained_object)} but expected_object has {type(expected_object)}')
    if isinstance(obtained_object, dict):
        assert len(obtained_object) == len(expected_object), ValueError(
            f'objects have different lengths: obtained={len(obtained_object)}, expected={len(expected_object)}')
        if sort_objects:
            obtained_object = dict(sorted(obtained_object.items()))
            expected_object = dict(sorted(expected_object.items()))

        for (object_key, object_val), (expected_key, expected_val) in zip(obtained_object.items(), expected_object.items()):
            if check_keys:
                assert object_key == expected_key, ValueError(f'Key mismatch: obtained_key={object_key}, expected_key={expected_key}')
            assert_tensor_objects_equal(object_val, expected_val, check_keys=check_keys, sort_objects=sort_objects)
            return
    elif isinstance(obtained_object, torch.Tensor):
        assert (obtained_object == expected_object).all(), ValueError(f'Non-matching tensors: obtained={obtained_object}, expected={expected_object}') # TODO add msgs
        return
    elif isinstance(obtained_object, Iterable):
        for obtained_item, expected_item in zip(obtained_object, expected_object):
            assert_tensor_objects_equal(obtained_item, expected_item)
        return

    else:
        assert obtained_object == expected_object, ValueError(f'Non-matching objects: obtained={obtained_object}, expected={expected_object}')
        return
    logger.warning(f'Skipped test: obtained_object={obtained_object}, expected_object={expected_object}')