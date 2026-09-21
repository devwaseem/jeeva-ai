from jeeva.training.console import print_training_complete, print_training_header


def test_training_header_renders_generic_sections(capsys):
    print_training_header(
        [
            (
                "Model",
                [
                    ("Architecture", "Bidirectional encoder"),
                    ("Backbone", "test/model"),
                    ("Decision schema", "RoutingDecision"),
                ],
            ),
            (
                "Optimization",
                [
                    ("Epochs", 4),
                    ("Learning rate", "2.00e-05"),
                ],
            ),
        ]
    )

    output = capsys.readouterr().out
    assert output.startswith("=" * 88)
    assert "JEEVA DECISION MODEL TRAINING" in output
    assert "MODEL" in output
    assert "OPTIMIZATION" in output
    assert "Bidirectional encoder" in output
    assert "test/model" in output
    assert "RoutingDecision" in output
    assert "Epochs                4" in output
    assert "Configuration ready. Preparing training." in output


def test_training_completion_summarizes_report(capsys):
    print_training_complete(
        {
            "architecture": "encoder",
            "adaptation": "full",
            "trainable_parameters": 1234,
            "history": [
                {
                    "train_loss": 0.25,
                    "eval_brier_score": 0.125,
                    "duration_seconds": 2.0,
                }
            ],
        }
    )

    output = capsys.readouterr().out
    assert "TRAINING COMPLETE" in output
    assert "Final train loss      0.250000" in output
    assert "Validation Brier      0.125000" in output
    assert "Trainable parameters  1,234" in output
    assert "Call save(...) to persist" in output
