import torch
from torch.utils.data import DataLoader, TensorDataset

from src.engine import evaluate, train_one_epoch


def test_training_reduces_loss_on_a_separable_toy_problem():
    torch.manual_seed(0)
    x = torch.randn(64, 8)
    y = (x[:, 0] > 0).long()
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
    model = torch.nn.Linear(8, 2)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    device = torch.device("cpu")

    before = evaluate(model, loader, loss_fn, device)
    for _ in range(20):
        train_one_epoch(model, loader, loss_fn, optimizer, device, max_grad_norm=1.0)
    after = evaluate(model, loader, loss_fn, device)

    assert after["loss"] < before["loss"]
    assert after["acc"] > 0.9
    assert not model.training
