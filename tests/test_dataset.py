import pytest
import torch

from jeeva.data.dataset import JeevaDataset
from jeeva.schema import DecisionSchema, Score
from tests.schema_helpers import choice, noul, schema


@pytest.fixture
def test_schema():
    return schema(
        category=choice(
            ["Billing", "Technical Support", "Account Access"],
            criterion="Select ticket department.",
        ),
        is_urgent=noul("The ticket is urgent."),
    )


def test_dataset_hard_labels(test_schema):
    records = [
        {
            "text": "Please refund my $50.",
            "category": "Billing",
            "is_urgent": True,
        },
        {
            "text": "Password reset needed.",
            "category": "Account Access",
            "is_urgent": False,
        },
    ]

    ds = JeevaDataset(data=records, schema=test_schema)
    assert len(ds) == 2

    sample0 = ds[0]
    assert sample0["document"] == "Please refund my $50."
    assert sample0["lane_mask"].tolist() == [1.0, 1.0]

    # Category Billing is index 0
    assert torch.equal(sample0["targets"][0], torch.tensor([1.0, 0.0, 0.0]))
    # is_urgent True is [1.0, 0.0]
    assert torch.equal(sample0["targets"][1], torch.tensor([1.0, 0.0]))

    sample1 = ds[1]
    # Category Account Access is index 2
    assert torch.equal(sample1["targets"][0], torch.tensor([0.0, 0.0, 1.0]))
    # is_urgent False is [0.0, 1.0]
    assert torch.equal(sample1["targets"][1], torch.tensor([0.0, 1.0]))


def test_dataset_rejects_oversized_jsonl_record_before_parsing(tmp_path, test_schema):
    dataset_path = tmp_path / "oversized.jsonl"
    dataset_path.write_bytes(b'{"text":"' + (b"x" * 1_048_577) + b'"}\n')

    with pytest.raises(ValueError, match="byte limit"):
        JeevaDataset(dataset_path, test_schema)


def test_dataset_soft_distributions(test_schema):
    records = [
        {
            "ticket": "Ambiguous ticket",
            "category": {"Billing": 0.6, "Technical Support": 0.4},
            "is_urgent": 0.85,
        }
    ]

    ds = JeevaDataset(data=records, schema=test_schema)
    sample = ds[0]

    cat_target = sample["targets"][0]
    assert pytest.approx(cat_target[0].item(), 1e-4) == 0.6
    assert pytest.approx(cat_target[1].item(), 1e-4) == 0.4
    assert pytest.approx(cat_target[2].item(), 1e-4) == 0.0

    urg_target = sample["targets"][1]
    assert pytest.approx(urg_target[0].item(), 1e-4) == 0.85
    assert pytest.approx(urg_target[1].item(), 1e-4) == 0.15


def test_dataset_missing_fields_lane_masking(test_schema):
    records = [
        {
            "context": "Only urgent is labeled here",
            "is_urgent": True,
            # category is completely omitted
        }
    ]

    ds = JeevaDataset(data=records, schema=test_schema)
    sample = ds[0]

    # category lane mask is 0.0, is_urgent lane mask is 1.0
    assert sample["lane_mask"].tolist() == [0.0, 1.0]


@pytest.mark.parametrize(
    ("field", "target"),
    [
        ("category", [0.0, 0.0, 0.0]),
        ("category", [0.5, -0.1, 0.6]),
        ("category", [float("nan"), 0.5, 0.5]),
        ("is_urgent", [0.0, 0.0]),
        ("is_urgent", {"true": float("inf"), "false": 1.0}),
        ("is_urgent", 1.5),
    ],
)
def test_dataset_rejects_invalid_distributions(test_schema, field, target):
    dataset = JeevaDataset(
        data=[{"text": "Invalid target", field: target}],
        schema=test_schema,
    )

    with pytest.raises(ValueError):
        dataset[0]


def test_dataset_encodes_score_scalars_and_soft_distributions():
    class Severity(DecisionSchema):
        severity = Score("Rate severity.", {0: "none", 2: "moderate", 10: "critical"})

    scalar = JeevaDataset(
        data=[{"text": "Degraded.", "severity": 5.0}],
        schema=Severity,
    )[0]
    assert scalar["targets"][0].tolist() == pytest.approx([0.0, 0.625, 0.375])

    soft = JeevaDataset(
        data=[{"text": "Ambiguous.", "severity": {"0.0": 0.1, "2.0": 0.9}}],
        schema=Severity,
    )[0]
    assert soft["targets"][0].tolist() == pytest.approx([0.1, 0.9, 0.0])
