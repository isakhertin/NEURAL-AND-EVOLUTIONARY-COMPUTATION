import numpy as np
import matplotlib.pyplot as plt


evaluated_models = []

def get_mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def get_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def get_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate_model(model_name, y_test, y_pred):
    mse = get_mse(y_test, y_pred)
    mae = get_mae(y_test,y_pred)
    mape = get_mape(y_test,y_pred)

    print(f"\n{model_name} Test Results:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}% \n")
    evaluated_models.append({"model_name": model_name, "mse": mse, "mae": mae, "mape": mape, "y_test": y_test, "y_pred": y_pred})

def print_results():
    print("Results Comparison:")
    print("-" * 60)
    print(f"{'Model':<20} {'MSE':>12} {'MAE':>12} {'MAPE':>12}")
    print("-" * 60)
    
    for model in evaluated_models:
        name = model["model_name"]
        mse = model["mse"]
        mae = model["mae"]
        mape = model["mape"]
        
        mse_str = f"{mse:,.2f}" if isinstance(mse, float) else str(mse)
        mae_str = f"{mae:,.2f}" if isinstance(mae, float) else str(mae)
        mape_str = f"{mape:,.2f}%" if isinstance(mape, float) else str(mape)
        
        print(f"{name:<20} {mse_str:>12} {mae_str:>12} {mape_str:>12}")
    
    print("-" * 60)

def plot_results():
    n_models = len(evaluated_models)
    
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols 
    
    plt.figure(figsize=(cols * 5, rows * 4))
    
    for i, model in enumerate(evaluated_models, 1):
        name = model["model_name"]
        y_test = model["y_test"]
        y_pred = model["y_pred"]
        
        plt.subplot(rows, cols, i)
        plt.scatter(y_test, y_pred, alpha=0.6)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        plt.title(name, fontsize=12)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_loss_epochs(net):
    title="Training / Validation Loss"
    train_losses, val_losses = net.loss_epochs() 
    
    epochs = np.arange(1, len(train_losses) + 1) 
    
    plt.figure(figsize=(8, 5))
    
    plt.plot(epochs, train_losses,
             label='Training loss (MSE)',
             color='tab:blue', linewidth=2)
    
    if net.validation_percent > 0 and len(val_losses) > 0:
        if np.any(val_losses > 0):
            plt.plot(epochs, val_losses,
                     label='Validation loss (MSE)',
                     color='tab:orange', linewidth=2, linestyle='--')
    
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title(title)
    plt.grid(True, which='both', ls=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()