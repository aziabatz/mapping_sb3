from stable_baselines3.common.callbacks import BaseCallback

class TensorboardLogCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        ent_coef = self.model.ent_coef
        self.logger.record('ent_coef', ent_coef)
        return True