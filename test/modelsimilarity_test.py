import torch
from collections import OrderedDict
import pytest

from fedstellar.learning.aggregators.helper import *


def test_cosine_metric():
    # Test case 1: Both models are None
    model1 = None
    model2 = None
    similarity = True
    expected_output = None
    output = cosine_metric(model1, model2, similarity)
    assert output == expected_output, f"Test case 1 failed: expected {expected_output}, got {output}"

    # Test case 2: One of the models is None
    model1 = OrderedDict([('layer1', torch.tensor([1, 2, 3])), ('layer2', torch.tensor([4, 5, 6]))])
    model2 = None
    similarity = True
    expected_output = None
    output = cosine_metric(model1, model2, similarity)
    assert output == expected_output, f"Test case 2 failed: expected {expected_output}, got {output}"

    # Test case 3: Models have different layer names
    model1 = OrderedDict([('layer1', torch.tensor([1, 2, 3])), ('layer2', torch.tensor([4, 5, 6]))])
    model2 = OrderedDict([('layer3', torch.tensor([7, 8, 9])), ('layer4', torch.tensor([10, 11, 12]))])
    similarity = True
    expected_output = None
    output = cosine_metric(model1, model2, similarity)
    assert output == expected_output, f"Test case 3 failed: expected {expected_output}, got {output}"

    # Test case 4: Models have different layer shapes
    model1 = OrderedDict([('layer1', torch.tensor([1, 2, 3])), ('layer2', torch.tensor([4, 5, 6]))])
    model2 = OrderedDict([('layer1', torch.tensor([7, 8, 9, 10])), ('layer2', torch.tensor([11, 12, 13, 14]))])
    similarity = True
    expected_output = 0.9775016903877258
    output = cosine_metric(model1, model2, similarity)
    assert output == expected_output, f"Test case 4 failed: expected {expected_output}, got {output}"

    # Test case 5: Models have same layer names and shapes
    model1 = OrderedDict([('layer1', torch.tensor([1, 2, 3])), ('layer2', torch.tensor([4, 5, 6]))])
    model2 = OrderedDict([('layer1', torch.tensor([7, 8, 9])), ('layer2', torch.tensor([10, 11, 12]))])
    similarity = True
    expected_output = 0.9777809381484985
    output = cosine_metric(model1, model2, similarity)
    assert output == expected_output, f"Test case 5 failed: expected {expected_output}, got {output}"

    # Test case 6: Models have float values
    model1 = OrderedDict([('layer1', torch.tensor([1.5, 2.5, 3.5])), ('layer2', torch.tensor([4.5, 5.5, 6.5]))])
    model2 = OrderedDict([('layer1', torch.tensor([7.5, 8.5, 9.5])), ('layer2', torch.tensor([10.5, 11.5, 12.5]))])
    similarity = True
    expected_output = 0.9864960312843323
    output = cosine_metric(model1, model2, similarity)
    assert output == expected_output, f"Test case 6 failed: expected {expected_output}, got {output}"

    # Test case 7: Models have negative float values
    model1 = OrderedDict([('layer1', torch.tensor([-1.5, -2.5, -3.5])), ('layer2', torch.tensor([-4.5, -5.5, -6.5]))])
    model2 = OrderedDict([('layer1', torch.tensor([-7.5, -8.5, -9.5])), ('layer2', torch.tensor([-10.5, -11.5, -12.5]))])
    similarity = True
    expected_output = 0.9864960312843323
    output = cosine_metric(model1, model2, similarity)
    assert output == expected_output, f"Test case 7 failed: expected {expected_output}, got {output}"

    print("All test cases passed!")