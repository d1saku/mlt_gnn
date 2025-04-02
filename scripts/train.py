import torch
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from models.model import GNN, GCNNet, GCNNet_conv, GINConvNet
from sklearn.model_selection import train_test_split
from dataset.preprocess import prepare_dataset
from utils.utilities import LossWithMemory, mse_loss_with_memory
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yaml
from utils.metrics import save_loss_history

config_path = "configs/default_config.yaml"

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
config = load_config('configs/default_config.yaml')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def mlt_graph_training(model_path=config['paths']['model_save_path'], 
                       do_splitting=config['paths']['do_splitting'],
                       test_ds_name=config['paths']['test_dataset_name'],
                       val_ds_name=config['paths']['val_dataset_name'],
                       train_ds_name=config['paths']['train_dataset_name'],
                       dataset_path=config['paths']['dataset_path'], 
                       dataset_name=config['paths']['dataset_name'],
                       targets=config['training']['targets'], 
                       n_epochs=config['training']['n_epochs'], 
                       bz=config['training']['batch_size'], 
                       train_ratio=0.7, 
                       test_ratio=0.15, 
                       validation_ratio=0.15, 
                       discount = config['loss_function']['discount'], 
                       decay = config['loss_function']['decay']):

    print("Loading data...")

    data_train = prepare_dataset(dataset_path, dataset_name, targets, train_ratio, test_ratio, validation_ratio, split="train")
    data_val = prepare_dataset(dataset_path, dataset_name, targets, train_ratio, test_ratio, validation_ratio, split="valid")

    print("Creating data loaders...")

    train_loader = DataLoader(data_train, batch_size=bz, shuffle=True)
    val_loader = DataLoader(data_val, batch_size=bz, shuffle=False)

    print("Data loaders created.")

    node_feature_dim = data_train[0].num_node_features
    edge_feature_dim = data_train[0].num_edge_features

    model = GCNNet_conv(node_feature_dim=node_feature_dim, edge_feature_dim=edge_feature_dim, 
                   solvent_feature_dim=512, output_dim=5, dropout_rate=0.3).to(device)
    
    print(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    initial_memory_values = {i: 0.0 for i in range(len(targets))}
    loss_memory = LossWithMemory(variable_init_values=initial_memory_values, discount=discount, decay=decay)

    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history = []

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            batch.to(device)
            optimizer.zero_grad()
            y_pred = model(batch)
            y_true = batch.y.view(-1, len(targets))  # Ensure this reshape is valid
            
            batch_loss = 0.0

            for i in range(y_true.shape[1]):  # Iterate over each column/target
                var = loss_memory.get_var(i)
                loss_fn = mse_loss_with_memory(var)
                loss = loss_fn(y_true[:, i], y_pred[:, i])
                batch_loss += loss

                # Update memory after computing loss for this target
                loss_memory.update_memory({f'{i}_loss': loss.detach()})

            batch_loss.backward()
            optimizer.step()

            total_train_loss += batch_loss.item()
            num_batches += 1

        epoch_train_loss = total_train_loss / num_batches
        loss_memory.on_epoch_end()
        train_loss_history.append(epoch_train_loss)

        # Validation loop
        model.eval()
        total_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch.to(device)
                y_pred = model(batch)
                y_true = batch.y.view(-1, len(targets))

                batch_val_loss = 0.0

                for i in range(y_true.shape[1]):
                    var = loss_memory.get_var(i)
                    loss_fn = mse_loss_with_memory(var)
                    loss = loss_fn(y_true[:, i], y_pred[:, i])
                    batch_val_loss += loss.item()

                total_val_loss += batch_val_loss
                num_val_batches += 1

        epoch_val_loss = total_val_loss / num_val_batches
        val_loss_history.append(epoch_val_loss)

        print(f"Epoch {epoch+1}, Train Loss: {epoch_train_loss}, Validation Loss: {epoch_val_loss}")

        # Check if this is the best model based on validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            if model_path:
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(model.state_dict(), model_path)
                print(f"Best model saved with validation loss: {best_val_loss}")

        loss_history_file = os.path.join(os.path.dirname(model_path), 'loss_history.json')
        save_loss_history(train_loss_history, val_loss_history, loss_history_file)

    return model


def test_model():
    # Load configuration parameters
    dataset_path = config['paths']['dataset_path']
    dataset_name = config['paths']['dataset_name']
    targets = config['training']['targets']
    train_ratio = 0.7
    test_ratio = 0.15
    validation_ratio = 0.15

    # Prepare the test dataset
    data_test = prepare_dataset(dataset_path, dataset_name, targets, train_ratio, test_ratio, validation_ratio, split="test")

    # Extract feature dimensions from the test dataset
    node_feature_dim = data_test[0].num_node_features
    edge_feature_dim = data_test[0].num_edge_features

    # Initialize the model
    model = GCNNet_conv(node_feature_dim=node_feature_dim, edge_feature_dim=edge_feature_dim,
                solvent_feature_dim=512, output_dim=5, dropout_rate=0.3).to(device)
    
    # Load the trained model weights
    model.load_state_dict(torch.load(config['paths']['model_save_path']))
    
    # Create DataLoader for the test dataset
    test_loader = DataLoader(data_test, batch_size=32, shuffle=False)

    # Set the model to evaluation mode
    model.eval()
    
    # Initialize containers for metrics
    total_losses = {f"mse_{i}": [] for i in range(len(targets))}
    all_y_true = []
    all_y_pred = []

    with torch.no_grad():  # Disable gradient calculation
        for batch in test_loader:
            batch = batch.to(device)
            y_pred = model(batch)
            y_true = batch.y.view(-1, len(targets))

            # Store predictions and true values for later R² calculation
            all_y_true.append(y_true.cpu())
            all_y_pred.append(y_pred.cpu())

            # Calculate MSE for each target
            for i in range(y_true.shape[1]):
                mse = mean_squared_error(y_true[:, i].cpu().numpy(), y_pred[:, i].cpu().numpy())
                total_losses[f"mse_{i}"].append(mse)

    # Concatenate all true values and predictions
    all_y_true = torch.cat(all_y_true, dim=0).numpy()
    all_y_pred = torch.cat(all_y_pred, dim=0).numpy()

    # Compute the R2 score for each target on the aggregated data
    total_r2 = {}
    for i in range(all_y_true.shape[1]):
        r2 = r2_score(all_y_true[:, i], all_y_pred[:, i])
        total_r2[f"r2_{i}"] = r2

    # Compute the average MSE over all batches
    avg_mse = {key: sum(value) / len(value) for key, value in total_losses.items()}

    # Print the results
    print(f"Test MSE per target: {avg_mse}")
    print(f"Test R^2 per target: {total_r2}")

    return avg_mse, total_r2

test_model()
# mlt_graph_training()
