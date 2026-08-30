from __future__ import annotations

from ..contracts import Candidate, HARD_FILTER_ATTRS, Query, Slot

_PRICE_OVER_TOLERANCE = 2.0   
_PRICE_UNDER_TOLERANCE = 0.3 

def _matches_category(candidate: Candidate, value: str) -> bool:
    if not candidate.categories:
        return True
    value_lower = value.lower()
    return any(
        value_lower in cat.lower() or cat.lower() in value_lower
        for cat in candidate.categories
    )


def _matches_department(candidate: Candidate, value: str) -> bool:
    if not candidate.department:
        return True
    return candidate.department.lower() == value.lower()


def _matches_size(candidate: Candidate, value: str) -> bool:
    if not candidate.sizes:
        return True
    value_lower = value.lower()
    return any(s.lower() == value_lower for s in candidate.sizes)


def _matches_price_min(candidate: Candidate, value: str) -> bool:
    if candidate.price is None:
        return True
    try:
        return candidate.price >= float(value) * _PRICE_UNDER_TOLERANCE
    except ValueError:
        return True


def _matches_price_max(candidate: Candidate, value: str) -> bool:
    if candidate.price is None:
        return True
    try:
        return candidate.price <= float(value) * _PRICE_OVER_TOLERANCE
    except ValueError:
        return True


_MATCHERS = {
    "category": _matches_category,
    "department": _matches_department,
    "size": _matches_size,
    "price_min": _matches_price_min,
    "price_max": _matches_price_max,
}


_NEGATABLE_FIELDS = {"color", "material", "brand"}


def _violates_negation(candidate: Candidate, negated_values: dict[str, list[str]]) -> bool:
    for attr, values in negated_values.items():
        if attr not in _NEGATABLE_FIELDS or not values:
            continue
        negated_lower = {v.lower() for v in values}
        if attr == "color":
            if any(c.lower() in negated_lower for c in candidate.colors):
                return True
        else:
            field_value = getattr(candidate, attr, None)
            if field_value and field_value.lower() in negated_lower:
                return True
    return False


def apply_filters(
    candidates: list[Candidate],
    slots: dict[str, Slot],
    negated_values: dict[str, list[str]] | None = None,
) -> list[Candidate]:
    active = [(attr, slot.value) for attr, slot in slots.items() if attr in HARD_FILTER_ATTRS]
    negated_values = negated_values or {}

    survivors = []
    for c in candidates:
        passes = (
            all(_MATCHERS[attr](c, value) for attr, value in active)
            and not _violates_negation(c, negated_values)
        )
        c.filter_match = passes
        if passes:
            survivors.append(c)
    return survivors


def suggest_relaxation(
    candidates: list[Candidate], slots: dict[str, Slot]
) -> tuple[str, str | None] | None:
    active = {attr: slot.value for attr, slot in slots.items() if attr in HARD_FILTER_ATTRS}
    if not active:
        return None

    priority = ["price_max", "price_min", "size", "department", "category"]
    ordered_attrs = [a for a in priority if a in active] + [a for a in active if a not in priority]

    for drop_attr in ordered_attrs:
        remaining = {a: v for a, v in active.items() if a != drop_attr}
        survivors = [
            c for c in candidates
            if all(_MATCHERS[a](c, v) for a, v in remaining.items())
        ]
        if not survivors:
            continue

        if drop_attr == "price_max":
            over_budget = [c.price for c in survivors if c.price is not None and c.price > float(active[drop_attr])]
            if over_budget:
                return drop_attr, f"{min(over_budget):.2f}"
        elif drop_attr == "price_min":
            under_budget = [c.price for c in survivors if c.price is not None and c.price < float(active[drop_attr])]
            if under_budget:
                return drop_attr, f"{max(under_budget):.2f}"
        return drop_attr, None

    return None


def apply_filters_with_relaxation(
    candidates: list[Candidate],
    slots: dict[str, Slot],
    negated_values: dict[str, list[str]] | None = None,
) -> tuple[list[Candidate], tuple[str, str | None] | None]:
    survivors = apply_filters(candidates, slots, negated_values)
    if survivors:
        return survivors, None
    return survivors, suggest_relaxation(candidates, slots)


def filter_for_query(candidates: list[Candidate], query: Query) -> list[Candidate]:
    return apply_filters(candidates, query.slots, query.negations)
