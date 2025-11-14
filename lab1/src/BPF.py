# bp_f.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class BP_F(nn.Module):
    def __init__(self, layers):
        """
        layers: e.g. [input_dim, 9, 5, 1]
        """
        super().__init__()

        self.layers = layers
        modules = []

        # Build hidden layers
        for i in range(len(layers) - 2):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            modules.append(nn.Tanh())   # can change activation here

        # Output layer
        modules.append(nn.Linear(layers[-2], layers[-1]))

        self.net = nn.Sequential(*modules)

        # Loss function
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    # Train just like your manual NN: full batch, given epochs, lr, momentum
    def fit(self, X_train, y_train, epochs=500, lr=0.01):
        #Trains using Adam optimizer.
        # Convert to torch tensors
        X = torch.tensor(X_train, dtype=torch.float32)
        y = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()

            y_pred = self.forward(X)
            loss = self.loss_fn(y_pred, y)

            loss.backward()
            optimizer.step()

            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"[BP-F] Epoch {epoch}/{epochs} Loss={loss.item():.4f}")

    def predict(self, X_test):
        X = torch.tensor(X_test, dtype=torch.float32)
        with torch.no_grad():
            preds = self.forward(X).numpy().flatten()
        return preds
