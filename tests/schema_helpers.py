from collections.abc import Sequence
from typing import cast

from jeeva.schema import Choice, DecisionSchema, Noul


def choice(
    options: Sequence[str],
    *,
    criterion: str = "Select the best matching value.",
    tau: float | None = None,
) -> Choice:
    return Choice(criterion, {value: value for value in options}, tau=tau)


def noul(criterion: str, *, tau: float | None = None) -> Noul:
    return Noul(
        criterion,
        {True: "The condition is satisfied.", False: "The condition is not satisfied."},
        tau=tau,
    )


def schema(**fields: Choice | Noul) -> type[DecisionSchema]:
    return cast(type[DecisionSchema], type("TestDecisionSchema", (DecisionSchema,), fields))
