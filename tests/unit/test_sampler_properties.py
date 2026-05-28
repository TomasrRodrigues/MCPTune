import pytest
from hypothesis import given
from hypothesis import strategies as st
from jsonschema import validate

from mcptune.sampling.recursive import RecursiveSampler

PRIMITIVE_TYPES = ["string", "integer", "number", "boolean"]


@st.composite
def json_schema_strategy(draw, depth=0):
    """
    Generate bounded recursive JSONSchema fragments.
    """

    if depth >= 3:
        primitive_type = draw(st.sampled_from(PRIMITIVE_TYPES))

        schema = {"type": primitive_type}

        if primitive_type == "string":
            schema["minLength"] = draw(st.integers(min_value=0, max_value=5))
            schema["maxLength"] = draw(st.integers(min_value=5, max_value=12))

        elif primitive_type == "integer":
            schema["minimum"] = draw(st.integers(min_value=-100, max_value=0))
            schema["maximum"] = draw(st.integers(min_value=0, max_value=100))

        elif primitive_type == "number":
            schema["minimum"] = draw(st.floats(min_value=-100, max_value=0))
            schema["maximum"] = draw(st.floats(min_value=0, max_value=100))

        return schema

    schema_type = draw(st.sampled_from(PRIMITIVE_TYPES + ["object", "array"]))

    if schema_type == "string":
        min_len = draw(st.integers(min_value=0, max_value=5))
        max_len = draw(st.integers(min_value=max(min_len, 1), max_value=12))

        return {
            "type": "string",
            "minLength": min_len,
            "maxLength": max_len,
        }

    if schema_type == "integer":
        minimum = draw(st.integers(min_value=-100, max_value=0))
        maximum = draw(st.integers(min_value=0, max_value=100))

        return {
            "type": "integer",
            "minimum": minimum,
            "maximum": maximum,
        }

    if schema_type == "number":
        minimum = draw(
            st.floats(min_value=-100, max_value=0, allow_nan=False, allow_infinity=False)
        )
        maximum = draw(st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False))

        return {
            "type": "number",
            "minimum": minimum,
            "maximum": maximum,
        }

    if schema_type == "boolean":
        return {"type": "boolean"}

    if schema_type == "array":
        return {
            "type": "array",
            "items": draw(json_schema_strategy(depth + 1)),
            "minItems": 1,
            "maxItems": 5,
        }

    # object
    property_count = draw(st.integers(min_value=1, max_value=4))

    properties = {}

    for i in range(property_count):
        properties[f"field_{i}"] = draw(json_schema_strategy(depth + 1))

    return {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
    }


@pytest.mark.unit
@given(json_schema_strategy())
def test_sampler_output_validates_against_schema(schema):
    sampler = RecursiveSampler()

    result = sampler.sample(schema)

    validate(instance=result, schema=schema)


@pytest.mark.unit
@given(st.integers())
def test_integer_schema_returns_int(_):
    sampler = RecursiveSampler()

    result = sampler.sample({"type": "integer"})

    assert isinstance(result, int)


@pytest.mark.unit
@given(st.integers())
def test_string_schema_returns_string(_):
    sampler = RecursiveSampler()

    result = sampler.sample({"type": "string"})

    assert isinstance(result, str)


@pytest.mark.unit
@given(st.integers())
def test_boolean_schema_returns_boolean(_):
    sampler = RecursiveSampler()

    result = sampler.sample({"type": "boolean"})

    assert isinstance(result, bool)


@pytest.mark.unit
@given(st.integers())
def test_number_schema_returns_number(_):
    sampler = RecursiveSampler()

    result = sampler.sample({"type": "number"})

    assert isinstance(result, float)
