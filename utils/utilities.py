import torch

class LossWithMemory:
    def __init__(self, variable_init_values, discount=0.6, decay=0.8):
        # Initialize memory for each target by index
        self.variables = {k: torch.tensor(v, dtype=torch.float32, requires_grad=False) 
                          for k, v in variable_init_values.items()}
        self.discount = discount
        self.decay = decay

    def update_memory(self, logs):
        for k, var in self.variables.items():
            if f'{k}_loss' in logs:
                new_value = (logs[f'{k}_loss'].detach()**0.5) * self.discount
                self.variables[k] = new_value.to(var.device).to(var.dtype)

    def on_epoch_end(self):
        self.discount *= self.decay

    def get_var(self, key):
        if key in self.variables:
            var = self.variables[key]
            return var
        else:
            # Assuming the first variable's device and dtype as default
            first_var = next(iter(self.variables.values()))
            return torch.tensor(0.0, dtype=first_var.dtype, device=first_var.device)


def mse_loss_with_memory(var):
    def mse_nan_with_memory(y_true, y_pred):
        masked_true = torch.where(torch.isnan(y_true), torch.tensor(0.0, device=y_true.device), y_true)
        masked_pred = torch.where(torch.isnan(y_true), var.to(y_pred.device), y_pred)
        error = torch.mean((masked_pred - masked_true) ** 2)
        return error
    return mse_nan_with_memory