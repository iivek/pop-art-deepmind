import torch


class LowerLayers(torch.nn.Module):
    """Lower nonlinear layers."""

    def __init__(self, n_in, H):
        super(LowerLayers, self).__init__()
        self.input_linear = torch.nn.Linear(n_in, H)
        self.hidden1 = torch.nn.Linear(H, H)
        self.hidden2 = torch.nn.Linear(H, H)
        self.hidden3 = torch.nn.Linear(H, H)

    def forward(self, x):
        h_tanh = torch.tanh(self.input_linear(x))
        h_tanh = torch.tanh(self.hidden1(h_tanh))
        h_tanh = torch.tanh(self.hidden2(h_tanh))
        h_tanh = torch.tanh(self.hidden3(h_tanh))
        return h_tanh


class UpperLayer(torch.nn.Module):
    """Upper linear layer."""

    def __init__(self, H, n_out):
        super(UpperLayer, self).__init__()
        self.output_linear = torch.nn.Linear(H, n_out)
        torch.nn.init.ones_(self.output_linear.weight)
        torch.nn.init.zeros_(self.output_linear.bias)

    def forward(self, x):
        y_pred = self.output_linear(x)
        return y_pred


class Model:
    """Binary regression experiment - model definition."""

    def __init__(self, n_in, H, n_out, **kwargs):
        super().__init__(**kwargs)
        self.lower_layers = LowerLayers(n_in, H)
        self.upper_layer = UpperLayer(H, n_out)

    def forward(self, x):
        return self.upper_layer(self.lower_layers(x))
