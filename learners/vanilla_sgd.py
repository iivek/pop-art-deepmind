from learners.model import Model
from learners.trainer import Trainer, History


class VanillaSGD(Trainer, History):
    """Learning by vanilla SGD"""

    def __init__(self, model: Model, lr, **kwargs):
        super().__init__(model=model, lr=lr)

    def train_step(self, inputs, outputs):
        loss = self.loss(self.model.forward(inputs), outputs)
        self.backward(loss)
        self.step()
        return {"loss": loss.item()}
