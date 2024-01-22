from stable_baselines3.common.callbacks import BaseCallback

class EntropyCallback(BaseCallback):
    def __init__(self, start, end, steps, verbose=0):
        super(EntropyCallback, self).__init__(verbose)
        
        self.start_coef = start
        self.final_coef = end
        self.total_steps = steps
        
    def _on_step(self) -> bool:

        progress = self.num_timesteps / self.total_steps
        # Just a linear decay
        new_coef = self.start_coef * (1-progress) + self.final_coef * progress
        
        self.model.ent_coef = new_coef
        return True