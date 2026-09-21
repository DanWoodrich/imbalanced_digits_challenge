"""Command Line Interface for imbalanced MNIST digit classification."""

from datetime import datetime
from pathlib import Path
import re
from typing import Optional
import typer
from PIL import Image
import torch
from torchvision import transforms
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from digit_classification.data import (
    download_and_curate_data,
    get_dataloaders,
    TARGET_DIGITS,
    IDX_TO_DIGIT,
    CURATED_FILENAME
)
from digit_classification.model import DigitClassifier, preprocess_image
from digit_classification.evaluation import evaluate_model

app = typer.Typer(
    name="digit-classification",
    help="CLI application for imbalanced MNIST digit classification (digits 0, 5, 8).",
)


def find_checkpoint(search_dir: Path, strategy: str = "latest") -> Optional[Path]:
    """Finds a checkpoint (.ckpt) file under search_dir using either 'latest' (mtime) or 'best' (lowest val_loss)."""
    if not search_dir.exists():
        return None
    ckpts = list(search_dir.glob("**/*.ckpt"))
    if not ckpts:
        return None

    if strategy == "best":
        # Attempt to parse val_loss from filename (e.g. digit-classifier-epoch=00-val_loss=0.2602.ckpt)
        scored_ckpts = []
        for ckpt in ckpts:
            match = re.search(r"val_loss=([0-9.]+)", ckpt.name)
            if match:
                try:
                    score = float(match.group(1))
                    scored_ckpts.append((score, ckpt))
                except ValueError:
                    pass
        if scored_ckpts:
            # Sort by lowest val_loss ascending
            scored_ckpts.sort(key=lambda item: item[0])
            return scored_ckpts[0][1]

        # Fallback to loading checkpoint metadata if filename does not contain score
        loaded_scored = []
        for ckpt in ckpts:
            try:
                data = torch.load(ckpt, map_location="cpu")
                callbacks = data.get("callbacks", {})
                for cb_data in callbacks.values():
                    if isinstance(cb_data, dict) and "best_model_score" in cb_data:
                        score_val = cb_data["best_model_score"]
                        if isinstance(score_val, torch.Tensor):
                            score_val = score_val.item()
                        loaded_scored.append((float(score_val), ckpt))
                        break
            except Exception:
                pass
        if loaded_scored:
            loaded_scored.sort(key=lambda item: item[0])
            return loaded_scored[0][1]

    # Default / 'latest': sort by modification time descending
    ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0]


@app.command()
def download_data(
    data_dir: str = typer.Option("./data", "--data-dir", help="Path to directory for saving curated dataset."),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducible curation (default: 42)."),
):
    """Download and curate the imbalanced dataset (digits 0, 5, 8) with designated train/val/test splits."""
    typer.echo(f"Downloading and curating dataset into {data_dir} (seed={seed})...")
    curated_path = download_and_curate_data(data_dir=data_dir, seed=seed)
    typer.echo(f"Curated dataset successfully created at: {curated_path}")


@app.command()
def train(
    data_dir: str = typer.Option("./data", "--data-dir", help="Path to curated MNIST data directory."),
    output_dir: str = typer.Option("./runs", "--output-dir", help="Base directory where timestamped run artifacts are saved."),
    epochs: int = typer.Option(20, "--epochs", help="Maximum training epochs (max 20)."),
    batch_size: int = typer.Option(64, "--batch-size", help="Batch size for training and validation."),
    lr: float = typer.Option(0.001, "--lr", help="Learning rate for Adam optimizer."),
    weighted: bool = typer.Option(True, "--weighted/--no-weighted", help="Use WeightedRandomSampler to address class imbalance."),
    frameshift_aug: int = typer.Option(1, "--frame-shift", help="Frameshift augmentation level. 0 = none, 1 = low, 2 = moderate, 3 = high (with blur)."),
    plot_loss: bool = typer.Option(True, "--plot-loss/--no-plot-loss", help="Generate loss curve PNG artifact overall and per class."),
):
    """Train the model on the curated dataset and save timestamped run artifacts."""
    if epochs > 20:
        typer.echo("Warning: Challenge guidelines specify a maximum of 20 epochs. Clamping to 20.")
        epochs = 20

    # Create timestamped run folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"run_{timestamp}"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Run directory created: {run_dir}")
    typer.echo(f"Loading curated data loaders from {data_dir} (weighted={weighted})...")

    #don't want redownload / dealing with seed, since I don't want it to be a user parameter here. 
    curated_file = Path(data_dir) / CURATED_FILENAME
    if not curated_file.exists():
        typer.echo(f"Error: No data file ({CURATED_FILENAME}) found under {data_dir}. Please provide a path to a download-data output", err=True)
        raise typer.Exit(code=1)

    train_loader, val_loader, _ = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        use_weighted_sampler=weighted,
        #seed won't ever apply here - redownload case caught earlier. 
        seed=-1,
    )

    typer.echo(f"Initializing DigitClassifier (lr={lr}, num_classes={len(TARGET_DIGITS)})...")
    model = DigitClassifier(learning_rate=lr)
    model = DigitClassifier(learning_rate=lr,frameshift_augmentation=frameshift_aug)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="digit-classifier-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    trainer = Trainer(
        default_root_dir=str(run_dir),
        accelerator="cpu",
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
        log_every_n_steps=15,
    )

    typer.echo("Starting model training on CPU...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_checkpoint = checkpoint_callback.best_model_path
    typer.echo(f"\nTraining complete!")
    typer.echo(f"Best model checkpoint saved to: {best_checkpoint}")

    if plot_loss:
        plot_path = run_dir / "loss_curves.png"
        model.plot_loss_curves(plot_path)
        typer.echo(f"Training/validation loss curve artifact saved to: {plot_path}")


@app.command()
def evaluate(
    checkpoint_path: Optional[str] = typer.Option(
        None,
        "--checkpoint-path",
        help="Path to model checkpoint (.ckpt). If omitted, automatically discovers checkpoint in --runs-dir based on --checkpoint-selection (default: latest).",
    ),
    checkpoint_selection: str = typer.Option(
        "latest",
        "--checkpoint-selection",
        help="Strategy to select checkpoint when --checkpoint-path is omitted: 'latest' (most recently modified) or 'best' (lowest validation loss).",
    ),
    data_dir: str = typer.Option("./data", "--data-dir", help="Path to curated MNIST data directory."),
    runs_dir: str = typer.Option("./runs", "--runs-dir", help="Directory to search for checkpoints if --checkpoint-path is omitted."),
    batch_size: int = typer.Option(64, "--batch-size", help="Batch size for evaluation."),
):
    """Evaluate a trained model checkpoint on the test set and print evaluation metrics."""
    if checkpoint_path is None:
        auto_ckpt = find_checkpoint(Path(runs_dir), strategy=checkpoint_selection)
        if auto_ckpt is None:
            typer.echo(f"Error: No checkpoints found under {runs_dir}. Please train a model first or specify --checkpoint-path.", err=True)
            raise typer.Exit(code=1)
        checkpoint_path = str(auto_ckpt)
        typer.echo(f"Automatically selected ({checkpoint_selection}) checkpoint: {checkpoint_path}")

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        typer.echo(f"Error: Checkpoint file not found: {checkpoint_path}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Loading checkpoint: {checkpoint_path}")
    model = DigitClassifier.load_from_checkpoint(str(ckpt_file))

    typer.echo(f"Loading 20% test/evaluation split from {data_dir}...")
    _, _, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
    )

    report_str, _ = evaluate_model(model, test_loader)

    typer.echo("\n" + "=" * 60)
    typer.echo("EVALUATION REPORT")
    typer.echo("=" * 60)
    typer.echo(report_str)


@app.command()
def predict(
    input_path: str = typer.Option(..., "--input-path", help="Path to input image file (PNG/JPEG)."),
    checkpoint_path: Optional[str] = typer.Option(
        None,
        "--checkpoint-path",
        help="Path to model checkpoint (.ckpt). If omitted, automatically discovers checkpoint in --runs-dir based on --checkpoint-selection (default: latest).",
    ),
    checkpoint_selection: str = typer.Option(
        "latest",
        "--checkpoint-selection",
        help="Strategy to select checkpoint when --checkpoint-path is omitted: 'latest' (most recently modified) or 'best' (lowest validation loss).",
    ),
    runs_dir: str = typer.Option("./runs", "--runs-dir", help="Directory to search for checkpoint if --checkpoint-path is omitted."),
):
    """Given a trained model and an image, predict the digit label and output probabilities for digits (0, 5, 8)."""
    if checkpoint_path is None:
        auto_ckpt = find_checkpoint(Path(runs_dir), strategy=checkpoint_selection)
        if auto_ckpt is None:
            typer.echo(f"Error: No checkpoints found under {runs_dir}. Please train a model first or specify --checkpoint-path.", err=True)
            raise typer.Exit(code=1)
        checkpoint_path = str(auto_ckpt)
        typer.echo(f"Automatically selected ({checkpoint_selection}) checkpoint: {checkpoint_path}")

    ckpt_file = Path(checkpoint_path)
    img_file = Path(input_path)

    if not ckpt_file.exists():
        typer.echo(f"Error: Checkpoint file not found: {checkpoint_path}", err=True)
        raise typer.Exit(code=1)

    if not img_file.exists():
        typer.echo(f"Error: Input image file not found: {input_path}", err=True)
        raise typer.Exit(code=1)

    model = DigitClassifier.load_from_checkpoint(str(ckpt_file))
    model.eval()

    raw_img = Image.open(img_file).convert("L")
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    tensor_img = transform(raw_img).unsqueeze(0)
    tensor_img = preprocess_image(img_file)

    with torch.no_grad():
        out = model.predict_step(tensor_img)

    probs = out["probabilities"][0]
    predicted_digit = out["predictions"][0]

    typer.echo("\n" + "=" * 40)
    typer.echo("PREDICTION RESULT")
    typer.echo("=" * 40)
    typer.echo(f"Predicted Digit: {predicted_digit}")
    typer.echo("\nClass Probabilities:")
    for digit, prob in zip(TARGET_DIGITS, probs.tolist()):
        typer.echo(f"  Digit {digit}: {prob:.4f} ({prob * 100:.2f}%)")


if __name__ == "__main__":
    app()
