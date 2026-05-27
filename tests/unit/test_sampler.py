import pytest


@pytest.mark.unit
def test_string_sampling_returns_str(primitive_sampler):
    assert isinstance(primitive_sampler.sample({"type": "string"}), str)


@pytest.mark.unit
def test_integer_sampling_returns_int(primitive_sampler):
    assert isinstance(primitive_sampler.sample({"type": "integer"}), int)


@pytest.mark.unit
def test_number_sampling_returns_float(primitive_sampler):
    assert isinstance(primitive_sampler.sample({"type": "number"}), float)


@pytest.mark.unit
def test_boolean_sampling_returns_bool(primitive_sampler):
    assert isinstance(primitive_sampler.sample({"type": "boolean"}), bool)


@pytest.mark.unit
def test_unknown_type_returns_none(primitive_sampler):
    # Pins today's contract: unknown JSONSchema types return None rather
    # than raising. Issue 6 (recursive sampling) may revisit this.
    assert primitive_sampler.sample({"type": "object"}) is None
    assert primitive_sampler.sample({}) is None
