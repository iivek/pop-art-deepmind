from learners.model import Model
from learners.trainer import Trainer, PopArt


class NormalizedSGD(Trainer, PopArt):
    """Learning by normalized SGD (gradient scaling in the bottom layers)"""

    def __init__(self, model: Model, lr, beta, **kwargs):
        super().__init__(model=model, lr=lr, beta=beta)
        # Hooks that scale gradient in the bottom layers. Scaling itself is stored in self.sigma
        for l in self.model.lower_layers.parameters():
            l.register_hook(lambda grad: grad / self.sigma**2)

    def train_step(self, inputs, outputs, *args, **kwargs):
        # calculating magnitude of the outputs and store in self.sigma
        self.art(outputs)
        self.sigma, self.mu = self.sigma_new, self.mu_new

        loss = self.loss(self.model.forward(inputs), outputs)
        self.backward(loss)
        self.step()
        return {"loss": loss.item()}
