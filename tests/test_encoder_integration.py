import pytest

from jeeva import DecisionModel, DecisionSchema, Example, FitConfig, Noul


class SportsDecision(DecisionSchema):
    sports = Noul(
        "The article is primarily about sports.",
        {True: "The condition is satisfied.", False: "The condition is not satisfied."},
    )


@pytest.mark.training_integration
def test_encoder_fit_save_load_and_continue_are_behaviorally_equivalent(tmp_path):
    examples = [
        Example("The team won the championship match.", {"sports": True}),
        Example("Central banks changed interest rates.", {"sports": False}),
    ]
    config = FitConfig(
        epochs=1,
        adaptation="full",
        batch_size=2,
        max_length=128,
        validation_fraction=0.0,
        save_best=False,
    )
    base = DecisionModel(
        "answerdotai/ModernBERT-base",
        architecture="encoder",
    )
    trained = base.fit(examples, schema=SportsDecision, config=config)
    before = trained.decide("The striker scored in the final.", SportsDecision)

    artifact = trained.save(tmp_path / "encoder")
    loaded = DecisionModel(artifact)
    after = loaded.decide("The striker scored in the final.", SportsDecision)
    continued = loaded.fit(examples, schema=SportsDecision, config=config)

    assert loaded.architecture == "encoder"
    assert continued.architecture == "encoder"
    assert after.value == before.value
    assert after.fields["sports"].distribution == pytest.approx(
        before.fields["sports"].distribution,
        abs=1e-6,
    )


@pytest.mark.training_integration
def test_encoder_lora_artifact_loads_and_continues_with_auto_adaptation(tmp_path):
    examples = [
        Example("The team won the championship match.", {"sports": True}),
        Example("Central banks changed interest rates.", {"sports": False}),
    ]
    lora_config = FitConfig(
        epochs=1,
        adaptation="lora",
        batch_size=2,
        max_length=128,
        validation_fraction=0.0,
        save_best=False,
    )
    base = DecisionModel(
        "answerdotai/ModernBERT-base",
        architecture="encoder",
    )
    trained = base.fit(examples, schema=SportsDecision, config=lora_config)
    artifact = trained.save(tmp_path / "encoder-adapter")
    loaded = DecisionModel(artifact)
    continued = loaded.fit(
        examples,
        schema=SportsDecision,
        config=FitConfig(
            epochs=1,
            adaptation="auto",
            batch_size=2,
            max_length=128,
            validation_fraction=0.0,
            save_best=False,
        ),
    )

    assert loaded.decide("The striker scored in the final.", SportsDecision).value
    assert continued.training_report is not None
    assert continued.training_report["adaptation"] == "lora"
