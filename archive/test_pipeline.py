"""End-to-end pipeline test with synthetic data."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data_loader import generate_synthetic_data
from src.model import SMoELSTM
from src.loss import EvidentialRegressionLoss
from src.train import train_model, get_device
from src.evaluate import evaluate_model
from config import Config

def main():
    print('=== Testing SMoE-LSTM Pipeline with Synthetic Data ===')

    # 1. Generate synthetic data
    print('\n[1/5] Generating synthetic data...')
    train_ds, test_ds, feat_scaler, label_scaler = generate_synthetic_data(
        n_batteries=10, cycles_per_battery=100, window_length=30, seed=42
    )
    print(f'  Train: {len(train_ds)} samples, Test: {len(test_ds)} samples')
    x_sample, y_sample = train_ds[0]
    print(f'  Input shape: {x_sample.shape}, Label shape: {y_sample.shape}')

    # 2. Forward pass
    print('\n[2/5] Testing forward pass...')
    config = Config()
    config.epochs = 5
    config.batch_size = 32
    device = torch.device('cpu')
    model = SMoELSTM(
        input_dim=16, hidden_dim=128, num_experts=10, top_k=4,
        lstm_layers=2, evidence_hidden=64
    ).to(device)
    x = torch.randn(4, 30, 16)
    gamma, nu, alpha, beta = model(x)
    print(f'  Output shapes: gamma={gamma.shape}, nu={nu.shape}, alpha={alpha.shape}, beta={beta.shape}')
    print(f'  Constraints satisfied: nu>0={(nu>0).all().item()}, '
          f'alpha>1={(alpha>1).all().item()}, beta>0={(beta>0).all().item()}')

    # 3. Loss
    print('\n[3/5] Testing loss computation...')
    criterion = EvidentialRegressionLoss(coeff=2.0)
    y = torch.randn(4, 1)
    loss = criterion(y, gamma, nu, alpha, beta)
    print(f'  Loss value: {loss.item():.4f}')

    # 4. Uncertainty decomposition
    print('\n[4/5] Testing uncertainty decomposition...')
    result = SMoELSTM.predict(gamma, nu, alpha, beta)
    print(f'  Prediction mean:  {result["pred"].mean().item():.4f}')
    print(f'  Aleatoric  (Eq.20): {result["aleatoric"].mean().item():.6f}')
    print(f'  Epistemic  (Eq.21): {result["epistemic"].mean().item():.6f}')

    # 5. Quick training
    print('\n[5/5] Quick training test (5 epochs)...')
    model, history = train_model(config, train_ds, device, trial=0, verbose=True)

    # 6. Evaluate
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    result = evaluate_model(model, test_loader, device, label_scaler)
    m = result['metrics']
    print(f'\n=== Final Test Results ===')
    print(f'  RMSE : {m["rmse"]:.4f}')
    print(f'  MAE  : {m["mae"]:.4f}')
    print(f'  MAPE : {m["mape"]:.2f}%')
    print(f'  R^2  : {m["r2"]:.4f}')
    print(f'  Spearman (epistemic) : {result["spearman_epistemic"]:.4f}')
    print(f'  Spearman (aleatoric) : {result["spearman_aleatoric"]:.4f}')
    print(f'  Avg epistemic uncertainty: {np.mean(result["predictions"]["epistemic"]):.4f}')
    print(f'  Avg aleatoric uncertainty: {np.mean(result["predictions"]["aleatoric"]):.4f}')
    print(f'\n=== Pipeline test PASSED ===')

    # Verify uncertainty decomposition formulas
    print(f'\n=== Verifying Uncertainty Formulas ===')
    pred_mean = gamma.mean().item()
    alea = (beta / (alpha - 1)).mean().item()
    epis = (beta / (nu * (alpha - 1))).mean().item()
    print(f'  E[mu]     = gamma = {pred_mean:.4f}')
    print(f'  E[sigma^2]= beta/(alpha-1)      = {alea:.6f}')
    print(f'  Var[mu]   = beta/(nu*(alpha-1)) = {epis:.6f}')
    # Check: epistemic should be smaller than aleatoric for well-learned data
    print(f'  aleatoric > epistemic: {alea > epis}')

if __name__ == '__main__':
    main()
