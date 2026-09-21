import pytest
import torch

from jeeva.schema import (
    Choice,
    DecisionSchema,
    Noul,
    Score,
    TargetDistribution,
    compile_decision_schema,
    decision_schema_from_dict,
    validate_target,
    validate_training_target,
)
from jeeva.schema.primitives import normalize_schema


class TrainingTargets(DecisionSchema):
    department = Choice(
        "Choose a department.",
        {
            "billing": "Billing requests.",
            "support": "Support requests.",
            "sales": "Sales requests.",
        },
    )
    urgent = Noul(
        "Is the request urgent?",
        {
            True: "The request is urgent.",
            False: "The request is not urgent.",
        },
    )
    severity = Score(
        "Rate severity.",
        {0: "No severity.", 2: "Moderate severity.", 4: "Critical severity."},
    )


def test_target_distribution_is_immutable_and_requires_probability_mass():
    source = {"billing": 0.7, "support": 0.2, "sales": 0.1}
    distribution = TargetDistribution(source)
    source["billing"] = 1.0

    assert distribution.probabilities["billing"] == pytest.approx(0.7)
    with pytest.raises(TypeError):
        distribution.probabilities["billing"] = 1.0  # type: ignore[index]
    with pytest.raises(ValueError, match="sum to 1"):
        TargetDistribution({"billing": 0.8, "support": 0.1})
    with pytest.raises(ValueError, match="finite and nonnegative"):
        TargetDistribution({"billing": float("nan"), "support": 0.0})
    with pytest.raises(TypeError, match="must be numbers"):
        TargetDistribution({"billing": True, "support": False})


def test_training_target_accepts_explicit_soft_distributions_and_hard_labels():
    validated = validate_training_target(
        TrainingTargets,
        {
            "department": TargetDistribution({"billing": 0.7, "support": 0.2, "sales": 0.1}),
            "urgent": True,
            "severity": TargetDistribution({0: 0.1, 2: 0.6, 4: 0.3}),
        },
    )

    assert isinstance(validated["department"], TargetDistribution)
    assert validated["urgent"] is True
    severity = validated["severity"]
    assert isinstance(severity, TargetDistribution)
    assert severity.probabilities == pytest.approx({0.0: 0.1, 2.0: 0.6, 4.0: 0.3})
    with pytest.raises(TypeError, match="must be a boolean"):
        validate_target(
            TrainingTargets,
            {"urgent": TargetDistribution({True: 0.8, False: 0.2})},
        )


def test_training_target_requires_exact_semantic_candidate_keys():
    with pytest.raises(ValueError, match="exactly match"):
        validate_training_target(
            TrainingTargets,
            {"department": TargetDistribution({"billing": 0.7, "support": 0.3})},
        )
    with pytest.raises(TypeError, match="True and False"):
        validate_training_target(
            TrainingTargets,
            {"urgent": TargetDistribution({"Yes": 0.8, "No": 0.2})},
        )
    with pytest.raises(ValueError, match="unknown"):
        validate_training_target(
            TrainingTargets,
            {"severity": TargetDistribution({0: 0.2, 2: 0.3, 5: 0.5})},
        )


def test_training_target_decodes_tagged_json_distributions():
    validated = validate_training_target(
        TrainingTargets,
        {
            "department": {
                "$distribution": {
                    "billing": 0.6,
                    "support": 0.3,
                    "sales": 0.1,
                }
            },
            "urgent": {"$distribution": {"true": 0.75, "false": 0.25}},
            "severity": {"$distribution": {"0.0": 0.1, "2.0": 0.7, "4.0": 0.2}},
        },
    )

    urgent = validated["urgent"]
    assert isinstance(urgent, TargetDistribution)
    assert urgent.probabilities == pytest.approx({True: 0.75, False: 0.25})
    severity = validated["severity"]
    assert isinstance(severity, TargetDistribution)
    assert severity.probabilities == pytest.approx({0.0: 0.1, 2.0: 0.7, 4.0: 0.2})


def test_choice_validates_instructions_criteria_examples_and_tau():
    with pytest.raises(ValueError, match="instructions"):
        Choice("", {"billing": "Billing.", "support": "Support."})
    with pytest.raises(TypeError, match="map"):
        Choice("Choose.", ("billing", "support"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least two"):
        Choice("Choose.", {"billing": "Billing."})
    with pytest.raises(ValueError, match="non-empty"):
        Choice("Choose.", {"": "Billing.", "support": "Support."})
    with pytest.raises(ValueError, match="non-empty"):
        Choice("Choose.", {"billing": "", "support": "Support."})
    with pytest.raises(ValueError, match="descriptions"):
        Choice("Choose.", {"billing": "Same.", "support": "Same."})
    with pytest.raises(ValueError, match="example key"):
        Choice(
            "Choose.",
            {"billing": "Billing.", "support": "Support."},
            examples={"sales": "Sell."},
        )
    with pytest.raises(ValueError, match="tau"):
        Choice("Choose.", {"a": "A.", "b": "B."}, tau=1.1)

    choice = Choice(
        "Choose a department.",
        {"billing": "Billing requests.", "support": "Support requests."},
    )
    assert choice.options == [
        "billing — Billing requests.",
        "support — Support requests.",
    ]
    assert dict(choice.criteria) == {
        "billing": "Billing requests.",
        "support": "Support requests.",
    }


def test_noul_validates_instructions_criteria_and_examples():
    criteria = {
        True: "The condition is satisfied.",
        False: "The condition is not satisfied.",
    }
    with pytest.raises(ValueError, match="non-empty"):
        Noul("", criteria)
    with pytest.raises(ValueError, match="exactly"):
        Noul("True.", {True: "Satisfied."})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="descriptions"):
        Noul("True.", {True: "Same.", False: "Same."})
    with pytest.raises(ValueError, match="example key"):
        Noul("True.", criteria, examples={"yes": "Example."})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="tau"):
        Noul("True.", criteria, tau=True)  # type: ignore[arg-type]


def test_examples_and_tau_are_keyword_only():
    with pytest.raises(TypeError):
        Noul(
            "True.",
            {True: "The condition is satisfied.", False: "The condition is not satisfied."},
            {True: ("Example.",)},
        )  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        Choice(
            "Choose an answer.",
            {value: value for value in ("yes", "no")},
            {"yes": ("Example.",)},
        )  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        Noul(
            "True.",
            {True: "The condition is satisfied.", False: "The condition is not satisfied."},
            None,
            0.5,
        )  # type: ignore[call-arg]


def test_declarations_are_not_mutated_during_compilation():
    declaration = Noul(
        "The request is urgent.",
        {True: "The condition is satisfied.", False: "The condition is not satisfied."},
        tau=0.7,
    )

    class Urgency(DecisionSchema):
        urgent = declaration

    compiled = compile_decision_schema(Urgency)
    assert declaration.key is None
    assert compiled.primitives["urgent"].key == "urgent"
    assert compiled.primitives["urgent"] is not declaration


def test_choice_prompt_places_examples_before_instructions_and_renders_semantics():
    choice = Choice(
        "Choose the team that should handle the request.",
        {
            "billing": "Handles charges and invoices.",
            "support": "Handles technical failures.",
        },
        examples={
            "support": "The application crashes when I sign in.",
            "billing": "I was charged twice.",
        },
    ).bind("queue")
    semantic_prompt = (
        "Examples:\n"
        "- billing — Handles charges and invoices.: I was charged twice.\n"
        "- support — Handles technical failures.: "
        "The application crashes when I sign in.\n"
        "\n"
        "Instructions:\n"
        "Choose the team that should handle the request."
    )
    options = [
        "billing — Handles charges and invoices.",
        "support — Handles technical failures.",
    ]

    assert choice.prompt == semantic_prompt
    assert choice.build_branch_prompt() == (
        f"{semantic_prompt}\n\nOptions:\n- {options[0]}\n- {options[1]}\n\nAnswer: "
    )
    assert choice.build_letter_branch_prompt() == (
        f"{semantic_prompt}\n\nOptions:\n"
        f"A. {options[0]}\n"
        f"B. {options[1]}\n\n"
        "Select the single best option letter.",
        options,
        ["A", "B"],
    )


def test_letter_prompt_can_reorder_candidates_without_changing_letters():
    primitive = Choice(
        "Choose one.",
        {"first": "The first outcome.", "second": "The second outcome."},
    ).bind("answer")

    assert primitive.build_letter_branch_prompt((1, 0)) == (
        "Instructions:\nChoose one.\n\nOptions:\n"
        "A. second — The second outcome.\n"
        "B. first — The first outcome.\n\n"
        "Select the single best option letter.",
        ["second — The second outcome.", "first — The first outcome."],
        ["A", "B"],
    )
    with pytest.raises(ValueError, match="option_order must be a permutation"):
        primitive.build_letter_branch_prompt((0, 0))


def test_noul_prompt_places_examples_before_instructions():
    noul = Noul(
        "Does the ticket need immediate attention?",
        {
            True: "The ticket needs immediate attention.",
            False: "The ticket can follow the normal queue.",
        },
        examples={
            False: "Please update my profile picture.",
            True: "Production is unavailable for every customer.",
        },
    )

    assert noul.prompt == (
        "Examples:\n"
        "- true — The ticket needs immediate attention.: "
        "Production is unavailable for every customer.\n"
        "- false — The ticket can follow the normal queue.: "
        "Please update my profile picture.\n"
        "\n"
        "Instructions:\n"
        "Does the ticket need immediate attention?"
    )


def test_choice_parse_similarities_uses_manual_tau():
    choice = Choice(
        "Choose a category.",
        {
            "billing": "Billing.",
            "support": "Support.",
            "account": "Account.",
        },
        tau=0.8,
    ).bind("category", calibrated_tau=0.1)
    result = choice.parse_similarities(
        torch.tensor([0.8, 0.3, 0.2]), cascade_threshold=0.3, temperature=0.05
    )
    assert result.field == "category"
    assert result.value == "billing"
    assert result.threshold_source == "manual"
    assert result.tau == pytest.approx(0.8)
    assert result.is_certain is True

    ambiguous = choice.parse_similarities(
        torch.tensor([0.51, 0.50, 0.49]), cascade_threshold=0.3, temperature=1.0
    )
    assert ambiguous.is_ambiguous is True


def test_noul_parse_similarities_uses_calibrated_tau():
    noul = Noul(
        "Is the customer angry?",
        {
            True: "The customer is angry.",
            False: "The customer is not angry.",
        },
    ).bind("angry", calibrated_tau=0.4)
    assert noul.build_branch_prompt() == (
        "Instructions:\nIs the customer angry?\n\nOptions:\n"
        "- true — The customer is angry.\n"
        "- false — The customer is not angry.\n\nAnswer: "
    )
    yes = noul.parse_similarities(torch.tensor([0.9, 0.1]), cascade_threshold=0.3, temperature=0.05)
    assert yes.value is True
    assert yes.threshold_source == "calibrated"
    assert yes.tau == pytest.approx(0.4)
    no = noul.parse_similarities(torch.tensor([0.1, 0.9]), cascade_threshold=0.3, temperature=0.05)
    assert no.value is False


def test_normalize_schema_accepts_only_schema_classes_or_compiled_lanes():
    class Contract(DecisionSchema):
        category = Choice("Choose a category.", {"a": "A.", "b": "B."})
        urgent = Noul(
            "Is this urgent?",
            {True: "Urgent.", False: "Not urgent."},
        )

    primitives = normalize_schema(Contract)
    assert [primitive.key for primitive in primitives] == ["category", "urgent"]
    assert normalize_schema(primitives) == primitives
    with pytest.raises(TypeError, match="DecisionSchema"):
        normalize_schema({"urgent": True})
    with pytest.raises(TypeError, match="compiled, keyed"):
        normalize_schema(
            [
                Noul(
                    "Is this urgent?",
                    {True: "Urgent.", False: "Not urgent."},
                )
            ]
        )


def test_score_validates_and_canonicalizes_levels():
    with pytest.raises(TypeError, match="mapping"):
        Score("Rate.", [(0, "low"), (1, "high")])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least two"):
        Score("Rate.", {0: "low"})
    with pytest.raises(ValueError, match="finite"):
        Score("Rate.", {0: "low", float("inf"): "high"})
    with pytest.raises(ValueError, match="descriptions"):
        Score("Rate.", {0: "same", 1: "same"})
    with pytest.raises(ValueError, match="tolerance"):
        Score("Rate.", {0: "low", 1: "high"}, tolerance=0)

    score = Score(
        "Rate severity.",
        {2: "high", 0: "none", 1: "moderate"},
        examples={2: "Production is unavailable."},
        tolerance=0.5,
    )
    assert score.level_values == (0.0, 1.0, 2.0)
    assert score.options == [
        "0.0 — none",
        "1.0 — moderate",
        "2.0 — high",
    ]
    assert score.distribution_labels == ["0.0", "1.0", "2.0"]
    assert "- 2.0 — high: Production is unavailable." in score.prompt


def test_score_interpolates_scalar_targets_exactly():
    score = Score("Rate.", {0: "none", 2: "medium", 10: "high"}).bind("severity")
    target = score.target_distribution(5.0)

    assert target.tolist() == pytest.approx([0.0, 0.625, 0.375])
    expectation = sum(
        probability * level for probability, level in zip(target.tolist(), score.level_values)
    )
    assert expectation == pytest.approx(5.0)
    assert score.target_distribution(2.0).tolist() == [0.0, 1.0, 0.0]


def test_score_decodes_expected_value_and_spread_certainty():
    score = Score(
        "Rate.",
        {0: "none", 1: "moderate", 2: "high"},
        tolerance=0.25,
        tau=0.7,
    ).bind("severity")
    result = score.result_from_probabilities(torch.tensor([0.0, 0.5, 0.5]))

    assert result.field == "severity"
    assert result.primitive == "score"
    assert result.value == pytest.approx(1.5)
    assert result.winning_level == 1.0
    assert result.standard_deviation == pytest.approx(0.5)
    assert result.certainty_score == pytest.approx(0.5)
    assert result.is_ambiguous is True
    assert result.distribution == pytest.approx({"0.0": 0.0, "1.0": 0.5, "2.0": 0.5})
    assert score.is_correct(1.5, 1.3)
    assert not score.is_correct(1.5, 1.0)


def test_score_schema_contract_and_dynamic_definition():
    class First(DecisionSchema):
        severity = Score("Rate.", {0: "none", 2: "high"}, tolerance=0.5, tau=0.9)

    class SameWithDifferentTau(DecisionSchema):
        severity = Score("Rate.", {0: "none", 2: "high"}, tolerance=0.5, tau=0.1)

    class DifferentTolerance(DecisionSchema):
        severity = Score("Rate.", {0: "none", 2: "high"}, tolerance=1.0)

    first = compile_decision_schema(First)
    assert first.fingerprint == compile_decision_schema(SameWithDifferentTau).fingerprint
    assert first.fingerprint != compile_decision_schema(DifferentTolerance).fingerprint
    assert first.descriptor["protocol"] == "decision-schema-v4"
    assert first.descriptor["fields"]["severity"]["instructions"] == "Rate."
    assert first.descriptor["fields"]["severity"]["criteria"] == [
        {"value": 0.0, "description": "none"},
        {"value": 2.0, "description": "high"},
    ]
    assert validate_target(First, {"severity": 1}) == {"severity": 1.0}

    dynamic = decision_schema_from_dict(
        {
            "severity": {
                "type": "score",
                "instructions": "Rate.",
                "criteria": [
                    {"value": 0, "description": "none"},
                    {"value": 2, "description": "high"},
                ],
                "tolerance": 0.5,
            }
        }
    )
    primitive = dynamic.fields()["severity"]
    assert isinstance(primitive, Score)
    assert primitive.level_values == (0.0, 2.0)


def test_compiled_descriptor_roundtrips_semantic_examples():
    class WithExamples(DecisionSchema):
        approved = Noul(
            "Determine whether the request is approved.",
            {
                True: "The request is approved.",
                False: "The request is not approved.",
            },
            examples={
                True: "The reviewer explicitly approved the request.",
                False: "The reviewer explicitly rejected the request.",
            },
        )
        route = Choice(
            "Choose the handling queue.",
            {
                "billing": "The request concerns a charge or invoice.",
                "support": "The request concerns product behavior.",
            },
            examples={
                "billing": "I was charged twice.",
                "support": "The application crashes.",
            },
        )
        severity = Score(
            "Rate the severity.",
            {
                0: "No impact.",
                2: "Moderate impact.",
                4: "Critical impact.",
            },
            examples={
                0: "No users are affected.",
                4: "All users are blocked.",
            },
        )

    compiled = compile_decision_schema(WithExamples)
    reconstructed = decision_schema_from_dict(compiled.descriptor["fields"])

    assert compile_decision_schema(reconstructed).descriptor == compiled.descriptor
    assert compile_decision_schema(reconstructed).fingerprint == compiled.fingerprint


def test_schema_compilation_enforces_field_candidate_and_text_budgets():
    too_many_fields = type(
        "TooManyFields",
        (DecisionSchema,),
        {
            f"field_{index}": Noul(
                "Choose.",
                {True: "Yes.", False: "No."},
            )
            for index in range(17)
        },
    )
    with pytest.raises(ValueError, match="at most 16 fields"):
        compile_decision_schema(too_many_fields)

    class TooManyCandidates(DecisionSchema):
        result = Choice(
            "Choose.",
            {f"choice-{index}": f"Candidate {index}." for index in range(65)},
        )

    with pytest.raises(ValueError, match="65 candidates; maximum is 64"):
        compile_decision_schema(TooManyCandidates)

    class TooManyTotalCandidates(DecisionSchema):
        first = Choice("Choose.", {f"a-{index}": f"Candidate A{index}." for index in range(50)})
        second = Choice("Choose.", {f"b-{index}": f"Candidate B{index}." for index in range(50)})
        third = Choice("Choose.", {f"c-{index}": f"Candidate C{index}." for index in range(50)})

    with pytest.raises(ValueError, match="150 total candidates; maximum is 128"):
        compile_decision_schema(TooManyTotalCandidates)

    class OversizedPrompt(DecisionSchema):
        result = Choice(
            "x" * 16_385,
            {"yes": "Yes.", "no": "No."},
        )

    with pytest.raises(ValueError, match="prompt exceeds 16384 characters"):
        compile_decision_schema(OversizedPrompt)
