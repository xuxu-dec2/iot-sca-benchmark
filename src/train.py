"""Training and evaluation for DL-based side-channel analysis."""

import os
import sys
import time
import json
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR, DATASET_CONFIGS, TRAINING_CONFIG, EVAL_CONFIG
from src.models import get_model


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_ascad_data(dataset_name):
    """Load ASCAD dataset from HDF5.
    Returns: (X_profiling, Y_profiling, X_attack, Y_attack, metadata)
    metadata: dict with 'plaintext_attack', 'key_attack' for key enumeration
    """
    cfg = DATASET_CONFIGS[dataset_name]
    h5_path = os.path.join(DATA_DIR, cfg["h5_train"])

    with h5py.File(h5_path, "r") as f:
        if "Profiling_traces" not in f:
            raise ValueError(f"Unknown HDF5 structure in {h5_path}")

        target = cfg["target_byte"]
        n_prof = cfg["n_profiling"]
        n_feat = cfg["n_features"]

        X_prof_full = f["Profiling_traces/traces"]
        X_att_full = f["Attack_traces/traces"]
        Y_prof_full = f["Profiling_traces/labels"]
        Y_att_full = f["Attack_traces/labels"]

        label_dtype = Y_prof_full.dtype
        is_compound = label_dtype.names is not None

        n_avail = min(len(X_prof_full), n_prof)
        X_profiling = X_prof_full[:n_avail]
        Y_profiling = Y_prof_full[:n_avail]

        n_att = cfg["n_attack"]
        X_attack = X_att_full[:n_att]
        Y_attack = Y_att_full[:n_att]

        # Window features
        if X_profiling.shape[1] > n_feat:
            start = (X_profiling.shape[1] - n_feat) // 2
            X_profiling = X_profiling[:, start:start + n_feat]
            X_attack = X_attack[:, start:start + n_feat]

        # Load metadata for key enumeration
        metadata = {}
        meta_prof = f["Profiling_traces/metadata"]
        meta_att = f["Attack_traces/metadata"]
        meta_dtype = meta_prof.dtype

        # Extract plaintext and key from attack traces
        if "plaintext" in meta_dtype.names:
            metadata["plaintext_attack"] = meta_att["plaintext"][:n_att, target]
        if "key" in meta_dtype.names:
            metadata["key_attack"] = meta_att["key"][:n_att, target]

        # Extract labels
        if is_compound:
            if "sbox_masked" in label_dtype.names:
                Y_profiling = Y_profiling["sbox_masked"][:, target]
                Y_attack = Y_attack["sbox_masked"][:, target]
            elif "key" in label_dtype.names:
                Y_profiling = Y_profiling["key"][:, target]
                Y_attack = Y_attack["key"][:, target]
            else:
                raise ValueError(f"Unknown compound label: {label_dtype.names}")
        elif Y_profiling.ndim > 1:
            Y_profiling = Y_profiling[:, target]
            Y_attack = Y_attack[:, target]

    Y_profiling = np.asarray(Y_profiling, dtype=np.int64)
    Y_attack = np.asarray(Y_attack, dtype=np.int64)

    print(f"  Loaded {dataset_name}:")
    print(f"    Profiling: {X_profiling.shape}, Attack: {X_attack.shape}")
    print(f"    Label range: [{Y_profiling.min()}, {Y_profiling.max()}]")

    return X_profiling, Y_profiling, X_attack, Y_attack, metadata


def preprocess(X_profiling, X_attack):
    """Per-trace standardization (as used in the original ASCAD paper)."""
    X_profiling_norm = (X_profiling - X_profiling.mean(axis=1, keepdims=True)) / (
        X_profiling.std(axis=1, keepdims=True) + 1e-8)
    X_attack_norm = (X_attack - X_attack.mean(axis=1, keepdims=True)) / (
        X_attack.std(axis=1, keepdims=True) + 1e-8)
    return X_profiling_norm.astype(np.float32), X_attack_norm.astype(np.float32)


def compute_ge(model, X_attack, metadata, device, max_traces=5000, step=10):
    """Compute Guessing Entropy via key enumeration.

    For each key hypothesis k (0..255), accumulate log p(model | SBOX[pt ^ k])
    across attack traces, then rank the correct key.
    """
    model.eval()
    n_traces = min(len(X_attack), max_traces)
    n_steps = n_traces // step

    X_tensor = torch.tensor(X_attack[:n_traces]).to(device)
    pt = metadata["plaintext_attack"][:n_traces]
    true_key = metadata["key_attack"][0]

    from src.config import SBOX
    sbox = np.array(SBOX, dtype=np.int64)
    # target_class[i, k] = SBOX[pt[i] ^ k] for each trace i and key hypothesis k
    target_class = sbox[pt[:, None] ^ np.arange(256)[None, :]]  # (n_traces, 256)

    ge_curve = np.zeros(n_steps)

    with torch.no_grad():
        batch_size = 1024
        log_probs = np.zeros((n_traces, 256))
        for i in range(0, n_traces, batch_size):
            batch = X_tensor[i:i + batch_size]
            logits = model(batch)
            log_probs[i:i + batch_size] = torch.log_softmax(
                logits, dim=1).cpu().numpy()

    for s in range(n_steps):
        n = (s + 1) * step
        # For each key k, sum over traces i of log_probs[i, SBOX[pt[i] ^ k]]
        key_scores = np.take_along_axis(
            log_probs[:n], target_class[:n], axis=1).sum(axis=0)
        rank = np.argsort(key_scores)[::-1]
        ge_curve[s] = np.where(rank == true_key)[0][0]

    return ge_curve


def compute_success_rate(ge_curve, threshold=1):
    """Compute whether GE reaches threshold (typically 1 = full key recovery)."""
    return 1.0 if np.any(ge_curve <= threshold) else 0.0


def train_one_run(model_name, dataset_name, run_id=0, device=None):
    """Train a single model on a single dataset and return results."""
    if device is None:
        device = get_device()

    cfg = DATASET_CONFIGS[dataset_name]
    train_cfg = TRAINING_CONFIG
    eval_cfg = EVAL_CONFIG

    print(f"\n{'='*60}")
    print(f"  {model_name} on {dataset_name} (run {run_id + 1}/{train_cfg['num_runs']})")
    print(f"  {cfg['description']}")
    print(f"{'='*60}")

    # Load data
    X_prof, Y_prof, X_att, Y_att, metadata = load_ascad_data(dataset_name)
    X_prof, X_att = preprocess(X_prof, X_att)

    # Train/val split (no stratify — classes are balanced in SCA datasets)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_prof, Y_prof, test_size=train_cfg["validation_split"],
        random_state=42 + run_id)

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(Y_train)),
        batch_size=train_cfg["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(Y_val)),
        batch_size=train_cfg["batch_size"] * 2, shuffle=False)

    # Model
    input_size = cfg["n_features"]
    n_classes = cfg["n_classes"]
    model = get_model(model_name, input_size, n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Optimizer & scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=train_cfg["lr_factor"],
        patience=train_cfg["lr_patience"])
    criterion = nn.CrossEntropyLoss(
        label_smoothing=train_cfg.get("label_smoothing", 0.0))

    # Training loop
    best_val_loss = float("inf")
    best_model_state = None
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    t_start = time.time()
    for epoch in range(train_cfg["epochs"]):
        # Train
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(x_batch), y_batch).item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}: train_loss={avg_train_loss:.4f}, "
                  f"val_loss={avg_val_loss:.4f}, "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= train_cfg["early_stop_patience"]:
            print(f"  Early stop at epoch {epoch + 1}")
            break

    train_time = time.time() - t_start
    print(f"  Training time: {train_time:.1f}s")

    # Load best model
    model.load_state_dict(best_model_state)

    # Evaluate: Guessing Entropy curve
    ge_curve = compute_ge(
        model, X_att, metadata, device,
        max_traces=eval_cfg["ge_max_traces"],
        step=eval_cfg["ge_step"])

    # Traces to Recovery (GE = 1)
    ttr = None
    ge1_idx = np.where(ge_curve <= 1)[0]
    if len(ge1_idx) > 0:
        ttr = (ge1_idx[0] + 1) * eval_cfg["ge_step"]

    # Final metrics — ensure Python native types for JSON
    metrics = {
        "model": model_name,
        "dataset": dataset_name,
        "run": int(run_id),
        "train_time_s": float(train_time),
        "best_val_loss": float(best_val_loss),
        "epochs_trained": int(len(train_losses)),
        "n_params": int(n_params),
        "ge_curve": [float(x) for x in ge_curve],
        "traces_to_recovery": int(ttr) if ttr is not None else None,
        "success": ttr is not None,
    }

    if ttr is not None:
        print(f"  GE=1 at {ttr} attack traces")
    else:
        print(f"  GE did not reach 1 within {eval_cfg['ge_max_traces']} traces")
        print(f"  Final GE: {ge_curve[-1]:.0f}")

    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["mlp", "cnn", "resnet"])
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--runs", type=int, default=TRAINING_CONFIG["num_runs"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    all_metrics = []
    for run in range(args.runs):
        set_seed(42 + run * 100)
        metrics = train_one_run(args.model, args.dataset, run, device)
        all_metrics.append(metrics)

    # Save results
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), "results"),
                exist_ok=True)

    if args.output is None:
        args.output = f"results/{args.model}_{args.dataset}_metrics.json"

    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
