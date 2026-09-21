"""Tests for the Typer CLI application entry point and options."""

import pytest
from typer.testing import CliRunner
from digit_classification.cli import app

runner = CliRunner()


def test_cli_help():
    """Verify root CLI help displays all subcommands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "download-data" in result.stdout
    assert "train" in result.stdout
    assert "evaluate" in result.stdout
    assert "predict" in result.stdout


def test_download_data_help():
    """Verify download-data subcommand options including default ./data and seed."""
    result = runner.invoke(app, ["download-data", "--help"])
    assert result.exit_code == 0
    assert "--data-dir" in result.stdout
    assert "--seed" in result.stdout


def test_train_help():
    """Verify train subcommand options including timestamped runs, plot-loss, and default ./data."""
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "--data-dir" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--epochs" in result.stdout
    assert "--weighted" in result.stdout
    assert "--plot-loss" in result.stdout
    assert "--seed" in result.stdout


def test_evaluate_help():
    """Verify evaluate subcommand options including optional checkpoint-path, checkpoint-selection, and default ./data."""
    result = runner.invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "--checkpoint-path" in result.stdout
    assert "--checkpoint-selection" in result.stdout
    assert "--data-dir" in result.stdout
    assert "--runs-dir" in result.stdout


def test_predict_help():
    """Verify predict subcommand options including checkpoint-selection."""
    result = runner.invoke(app, ["predict", "--help"])
    assert result.exit_code == 0
    assert "--checkpoint-path" in result.stdout
    assert "--checkpoint-selection" in result.stdout
    assert "--input-path" in result.stdout
    assert "--runs-dir" in result.stdout
