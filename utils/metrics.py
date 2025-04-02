import json
import os

def save_loss_history(train_loss_history, val_loss_history, file_path):
    with open(file_path, 'w') as f:
        json.dump({'train_loss': train_loss_history, 'val_loss': val_loss_history}, f)


def load_loss_history(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            loss_history = json.load(f)
        return loss_history['train_loss'], loss_history['val_loss']
    else:
        return [], []