from m6_practice.model import MyAwesomeModel
import torch
import pytest

@pytest.mark.parametrize("batch_size", [32, 64])
def test_model():
    model = MyAwesomeModel()
    x = torch.randn(1, 1, 28, 28)
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3))
    with pytest.raises(ValueError, match='Expected each sample to have shape [1, 28, 28]'):
        model(torch.randn(1,1,28,29))
    y = model(x)
    assert y.shape == (1, 10)