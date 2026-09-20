"""Command Line Interface for digit classification application."""

from pathlib import Path
from typing import Optional
import typer
from PIL import Image
import torch
from torchvision import transforms
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from digit_classification.data import (
    download_mnist,
    get_dataloaders,
    TARGET_DIGITS,
    IDX_TO_DIGIT,
)
from digit_classification.model import DigitClassifier
from digit_classification.evaluation import evaluate_model

app = typer.Typer(
    name="digit-classification",
    help="CLI application for imbalanced MNIST digit classification (digits 0, 5, 8).",
)


@app.command()
def download_data(
    data_dir: str = typer.Option(..., "--data-dir", help="Path to directory for downloading MNIST dataset."),
):
    """Download the MNIST dataset to the specified data directory."""
    typer.echo(f"Downloading MNIST dataset to {data_dir}...")
    download_mnist(data_dir)
    typer.echo("MNIST dataset successfully downloaded.")


@app.command()
def train(
    data_dir: str = typer.Option(..., "--data-dir", help="Path to MNIST data directory."),
    output_dir: str = typer.Option(..., "--output-dir", help="Path to directory where checkpoints and logs will be saved."),
    epochs: int = typer.Option(20, "--epochs", help="Maximum number of training epochs (max 20)."),
    batch_size: int = typer.Option(64, "--batch-size", help="Batch size for training and validation."),
    lr: float = typer.Option(0.001, "--lr", help="Learning rate for Adam optimizer."),
    weighted: bool = typer.Option(True, "--weighted/--no-weighted", help="Use WeightedRandomSampler to address class imbalance."),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducible splits."),
):
    """Train the model on the curated imbalanced dataset and save checkpoints."""
    if epochs > 20:
        typer.echo("Warning: Challenge guidelines specify a maximum of 20 epochs. Clamping to 20.")
        epochs = 20

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Preparing data loaders from {data_dir} (weighted={weighted}, seed={seed})...")
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        use_weighted_sampler=weighted,
        seed=seed,
    )

    typer.echo(f"Initializing DigitClassifier (lr={lr}, num_classes={len(TARGET_DIGITS)})...")
    model = DigitClassifier(learning_rate=lr)

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path / "checkpoints",
        filename="digit-classifier-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    trainer = Trainer(
        default_root_dir=str(output_path),
        accelerator="cpu",
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
        log_every_n_steps=15,
    )

    typer.echo("Starting model training on CPU...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_checkpoint = checkpoint_callback.best_model_path
    typer.echo(f"Training complete! Best model checkpoint saved to: {best_checkpoint}")


@app.command()
def evaluate(
    checkpoint_path: str = typer.Option(..., "--checkpoint-path", help="Path to trained model checkpoint (.ckpt)."),
    data_dir: str = typer.Option(..., "--data-dir", help="Path to MNIST data directory."),
    batch_size: int = typer.Option(64, "--batch-size", help="Batch size for evaluation."),
    seed: int = typer.Option(42, "--seed", help="Random seed used for test split reproducibility."),
):
    """Evaluate a trained model checkpoint on the test set and print out a classification report."""
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
        seed=seed,
    )

    typer.echo("\nRunning evaluation on test set...")
    report_str, _ = evaluate_model(model, test_loader)

    typer.echo("\n" + "=" * 60)
    typer.echo("CLASSIFICATION REPORT (Test Split)")
    typer.echo("=" * 60)
    typer.echo(report_str)


@app.command()
def predict(
    checkpoint_path: str = typer.Option(..., "--checkpoint-path", help="Path to trained model checkpoint (.ckpt)."),
    input_path: str = typer.Option(..., "--input-path", help="Path to input image file (PNG/JPEG)."),
):
    """Given a trained model and an image, predict the digit label and output probabilities for digits (0, 5, 8)."""
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

    # Load and preprocess input image
    raw_img = Image.open(img_file).convert("L")  # Grayscale
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    tensor_img = transform(raw_img).unsqueeze(0)  # Shape: (1, 1, 28, 28)

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

