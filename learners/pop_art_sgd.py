from learners.trainer import PopArt, Trainer, History


class PopArtSGD(Trainer, PopArt, History):
    """Learning by pop-art"""

    def __init__(self, model, lr, beta, *args, **kwargs):
        super().__init__(model=model, lr=lr, beta=beta)

    def predict(self, x):
        return self.denormalize(self.model.forward(x))

    def _art_and_pop_steps(self, outputs):
        self.art(outputs)
        self.pop(self.model.upper_layer)

    def train_step(self, inputs, outputs):
        self._art_and_pop_steps(outputs)
        self.sigma, self.mu = self.sigma_new, self.mu_new

        loss = self.loss(self.model.forward(inputs), self.normalize(outputs))
        self.backward(loss)
        self.step()

        return {"loss": loss.item()}


class ArtSGD(PopArtSGD):
    """All art, zero pop."""

    def _art_and_pop_steps(self, outputs):
        self.art(outputs)
