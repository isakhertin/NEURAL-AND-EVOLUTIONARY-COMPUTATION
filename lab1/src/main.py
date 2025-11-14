from sklearn.linear_model import LinearRegression

from NeuralNet import NeuralNet
from BPF import BP_F
import evaluation as ev
import data_processing as dp
from tabulate import tabulate  




def train_mlr(X_train, y_train, X_test):
    lin = LinearRegression()
    lin.fit(X_train, y_train)       # OBS! y_train i originalskala
    y_pred_mlr = lin.predict(X_test)
    return y_pred_mlr

def train_bpf(layers, X_train, X_test, y_train_scaled, target_std, target_mean):
    bpf = BP_F(layers)
    bpf.fit(
        X_train=X_train,
        y_train=y_train_scaled,    
        epochs=800,
        lr=0.01,
    )
    # Predict on the test set
    y_pred_bpf_scaled = bpf.predict(X_test)
    return y_pred_bpf_scaled * target_std + target_mean



def main():
    # Part 1
    (
        X_train,
        X_test,
        y_train_scaled,
        y_train,
        y_test,
        target_mean,
        target_std,
    ) = dp.load_dataset()
    print(f"Träningsdata: {X_train.shape}, Testdata: {X_test.shape}")
    # Part 2 Create NN
    input_size = X_train.shape[1]
    layers = [input_size, 9, 5, 1]
    results = []   # store rows for the final table
    
    configs = [
        {"layers": [input_size, 8, 1], "epochs": 300, "lr": 0.01, "mom": 0.0, "act": "tanh"},
        {"layers": [input_size, 12, 1], "epochs": 300, "lr": 0.01, "mom": 0.9, "act": "tanh"},
        {"layers": [input_size, 16, 8, 1], "epochs": 500, "lr": 0.01, "mom": 0.9, "act": "tanh"},
        {"layers": [input_size, 16, 8, 1], "epochs": 500, "lr": 0.001, "mom": 0.9, "act": "tanh"},
        {"layers": [input_size, 20, 10, 1], "epochs": 800, "lr": 0.005, "mom": 0.9, "act": "tanh"},
        {"layers": [input_size, 10, 5, 1], "epochs": 800, "lr": 0.01, "mom": 0.5, "act": "relu"},
        {"layers": [input_size, 32, 16, 1], "epochs": 1000, "lr": 0.001, "mom": 0.9, "act": "tanh"},
        {"layers": [input_size, 8, 4, 2, 1], "epochs": 1000, "lr": 0.01,  "mom": 0.9, "act": "relu"},
        {"layers": [input_size, 12, 6, 3, 1], "epochs": 1200, "lr": 0.005, "mom": 0.9, "act": "tanh"},
        {"layers": [input_size, 20, 10, 5, 1], "epochs": 1500, "lr": 0.001, "mom": 0.9, "act": "tanh"},
    ]

    for i, cfg in enumerate(configs):

        nn = NeuralNet(cfg["layers"], epochs=cfg["epochs"], lr=cfg["lr"], momentum=cfg["mom"], activation=cfg["act"])
    
        # Train network
        nn.fit(X_train, y_train_scaled, shuffle=True)

        # Predict on test data
        y_pred_scaled = nn.predict(X_test).flatten()
        y_pred_bp = y_pred_scaled * target_std + target_mean
        ev.evaluate_model("BP (own)", y_test, y_pred_bp)
        mse = ev.get_mse(y_test, y_pred_bp)
        mae = ev.get_mae(y_test, y_pred_bp)
        mape = ev.get_mape(y_test, y_pred_bp)

        results.append([
            i + 1,
            cfg["layers"],
            cfg["epochs"],
            cfg["lr"],
            cfg["mom"],
            cfg["act"],
            mse,
            mae,
            mape
        ])

    # Print final results table
    headers = ["Config", "Layers", "Epochs", "LR", "Mom", "Act", "MSE", "MAE", "MAPE"]
    print("\n\n===== FINAL RESULTS =====\n")
    print(tabulate(results, headers=headers, floatfmt=".6f"))
        
        

    #Train MLR
    print("Training MLR..")
    y_pred_mlr = train_mlr(X_train, y_train, X_test)
    ev.evaluate_model("MLR", y_test, y_pred_mlr)

    #Train BPF
    print("Training BPF..")
    y_pred_bpf = train_bpf(layers, X_train, X_test, y_train_scaled, target_std, target_mean)
    ev.evaluate_model("BPF", y_test, y_pred_bpf)

    #Result print 
    ev.print_results()
    ev.plot_loss_epochs(nn)
    ev.plot_results()
    


if __name__ == "__main__":
    main()
