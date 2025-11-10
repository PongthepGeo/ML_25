# mlp_full_form.py
import torch
import torch.nn as nn
from typing import Iterable, Optional, Callable

class MLP(nn.Module):
    """
    Full MLP: x -> [Linear -> (BN) -> Activation -> (Dropout)] x L -> Linear(out)
    Returns logits by default (add a final activation if you really want probs).
    """
    def __init__(
        self,
        in_dim: int,
        hidden: Iterable[int],          # e.g., [128, 64] or [] for linear model
        out_dim: int,
        activation: Callable[[], nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        batchnorm: bool = False,
        final_activation: Optional[Callable[[], nn.Module]] = None,  # e.g., nn.Sigmoid() for binary
    ):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h, bias=True))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim, bias=True))
        if final_activation is not None:
            layers.append(final_activation())
        self.net = nn.Sequential(*layers)
        self.reset_parameters(activation)

    def reset_parameters(self, activation: Callable[[], nn.Module]):
        # He init (Kaiming) for ReLU-like; Xavier otherwise; biases = 0
        act = activation().__class__.__name__.lower()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if "relu" in act or "leakyrelu" in act or "gelu" in act:
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ----------------------- Demo (classification) -----------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # toy multiclass classification
    N, C = 10, 4                   # input dim, classes
    model = MLP(in_dim=N, hidden=[64, 32], out_dim=C,
                activation=nn.ReLU, dropout=0.1, batchnorm=True).to(device)

    X = torch.randn(512, N, device=device)
    y = torch.randint(0, C, (512,), device=device)

    logits = model(X)              # [B, C]
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()                # grads flow; ready for an optimizer
    print("logits:", logits.shape, "| loss:", float(loss))


