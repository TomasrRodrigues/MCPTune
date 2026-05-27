import pytest
import random

from mcptune.sampling.recursive import RecursiveSampler


@pytest.mark.unit
def test_nested_object_sampling():
    sampler = RecursiveSampler()

    schema = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
            }
        },
    }

    result = sampler.sample(schema)

    assert isinstance(result, dict)
    assert "user" in result
    assert isinstance(result["user"], dict)

    assert "name" in result["user"]
    assert "age" in result["user"]

    assert isinstance(result["user"]["name"], str)
    assert isinstance(result["user"]["age"], int)

@pytest.mark.unit
def test_array_sampling():
    sampler = RecursiveSampler()

    schema = {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 2,
        "maxItems": 5,
    }

    result = sampler.sample(schema)

    assert isinstance(result, list)
    assert len(result) >= 2
    assert len(result) <= 5

    for item in result:
        assert isinstance(item, int)


@pytest.mark.unit
def test_enum_sampling():
    sampler = RecursiveSampler()

    schema = {
        "enum": ["red", "green", "blue"]
    }

    result = sampler.sample(schema)

    assert result in ["red", "green", "blue"]


@pytest.mark.unit
def test_nullable_sampling():
    sampler = RecursiveSampler()

    schema = {
        "type": ["string", "null"]
    }

    result = sampler.sample(schema)

    assert result is None or isinstance(result, str)


@pytest.mark.unit
def test_anyof_sampling():
    sampler = RecursiveSampler()

    schema = {
        "anyOf": [
            {"type": "string"},
            {"type": "integer"},
        ]
    }

    result = sampler.sample(schema)

    found_string = False
    found_int = False

    for _ in range(50):
        result = sampler.sample(schema)

        if isinstance(result, str):
            found_string = True

        if isinstance(result, int):
            found_int = True

    assert found_string
    assert found_int


@pytest.mark.unit
def test_oneof_sampling():
    sampler = RecursiveSampler()

    schema = {
        "oneOf": [
            {"type": "boolean"},
            {"type": "number"},
        ]
    }

    result = sampler.sample(schema)

    assert type(result) in (bool, float)


@pytest.mark.unit
def test_default_value_respected():
    sampler = RecursiveSampler()

    schema = {
        "type": "string",
        "default": "Porto"
    }

    result = sampler.sample(schema)

    assert result == "Porto"


@pytest.mark.unit
def test_string_constraints():
    sampler = RecursiveSampler()

    schema = {
        "type": "string",
        "minLength": 5,
        "maxLength": 10,
    }

    result = sampler.sample(schema)

    assert isinstance(result, str)
    assert len(result) >= 5
    assert len(result) <= 10


@pytest.mark.unit
def test_number_constraints():
    sampler = RecursiveSampler()

    schema = {
        "type": "integer",
        "minimum": 10,
        "maximum": 20,
    }

    result = sampler.sample(schema)

    assert isinstance(result, int)
    assert result >= 10
    assert result <= 20


@pytest.mark.unit
def test_email_format():
    sampler = RecursiveSampler()

    schema = {
        "type": "string",
        "format": "email"
    }

    result = sampler.sample(schema)

    assert isinstance(result, str)
    assert "@" in result
    assert "." in result


@pytest.mark.unit
def test_optional_fields_probabilistic():
    sampler = RecursiveSampler()

    schema = {
        "type": "object",
        "properties": {
            "required_field": {"type": "string"},
            "optional_field": {"type": "string"},
        },
        "required": ["required_field"],
    }

    found_optional = False
    missing_optional = False

    for _ in range(100):
        result = sampler.sample(schema)

        assert "required_field" in result

        if "optional_field" in result:
            found_optional = True
        else:
            missing_optional = True

    assert found_optional
    assert missing_optional



@pytest.mark.unit
def test_pattern_fallback_or_generation():
    sampler = RecursiveSampler()

    schema = {
        "type": "string",
        "pattern": "[a-z]{5}"
    }

    result = sampler.sample(schema)

    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.unit
def test_nullable_explicit_flag():
    sampler = RecursiveSampler()

    schema = {
        "type": "string",
        "nullable": True
    }

    # run many times to ensure both branches appear
    seen_none = False
    seen_str = False

    for _ in range(50):
        r = sampler.sample(schema)
        if r is None:
            seen_none = True
        elif isinstance(r, str):
            seen_str = True

    assert seen_none
    assert seen_str


@pytest.mark.unit
def test_anyof_distribution_not_crashing():
    sampler = RecursiveSampler()

    schema = {
        "anyOf": [
            {"type": "string"},
            {"type": "integer"},
            {"type": "boolean"},
        ]
    }

    for _ in range(20):
        r = sampler.sample(schema)
        assert isinstance(r, (str, int, bool))


@pytest.mark.unit
def test_oneof_distribution_not_crashing():
    sampler = RecursiveSampler()

    schema = {
        "oneOf": [
            {"type": "number"},
            {"type": "string"},
        ]
    }

    for _ in range(20):
        r = sampler.sample(schema)
        assert isinstance(r, (float, str))


@pytest.mark.unit
def test_deep_nesting_respects_depth_limit():
    sampler = RecursiveSampler()

    schema = {
        "type": "object",
        "properties": {
            "a": {
                "type": "object",
                "properties": {
                    "b": {
                        "type": "object",
                        "properties": {
                            "c": {
                                "type": "object",
                                "properties": {
                                    "d": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    result = sampler.sample(schema)

    assert isinstance(result, dict)



@pytest.mark.unit
def test_reproducibility_same_seed():
    schema = {
        "type": "object",
        "properties": {
            "x": {"type": "integer", "minimum": 1, "maximum": 100},
            "y": {"type": "string", "minLength": 5, "maxLength": 5},
        },
        "required": ["x", "y"],
    }

    sampler1 = RecursiveSampler(random.Random(42))
    sampler2 = RecursiveSampler(random.Random(42))

    out1 = [sampler1.sample(schema) for _ in range(50)]
    out2 = [sampler2.sample(schema) for _ in range(50)]

    assert out1 == out2

@pytest.mark.unit
def test_different_seeds_produce_different_outputs():
    schema = {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "minimum": 0, "maximum": 100},
        },
        "required": ["n"],
    }

    sampler1 = RecursiveSampler(random.Random(1))
    sampler2 = RecursiveSampler(random.Random(2))

    out1 = [sampler1.sample(schema) for _ in range(50)]
    out2 = [sampler2.sample(schema) for _ in range(50)]

    assert out1 != out2

@pytest.mark.unit
def test_reproducibility_structure():
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "integer"},
            "c": {"type": "string"},
        }
    }

    sampler1 = RecursiveSampler(random.Random(123))
    sampler2 = RecursiveSampler(random.Random(123))

    def shapes(samples):
        return [
            set(s.keys())
            for s in samples
        ]

    out1 = [sampler1.sample(schema) for _ in range(30)]
    out2 = [sampler2.sample(schema) for _ in range(30)]

    assert out1 == out2
    assert shapes(out1) == shapes(out2)

@pytest.mark.unit
def test_reproducibility_nested_schema():
    schema = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "profile": {
                        "type": "object",
                        "properties": {
                            "age": {"type": "integer", "minimum": 0, "maximum": 120},
                            "name": {"type": "string", "minLength": 3, "maxLength": 8},
                        },
                        "required": ["age", "name"],
                    }
                },
                "required": ["profile"],
            }
        },
        "required": ["user"],
    }

    s1 = RecursiveSampler(random.Random(999))
    s2 = RecursiveSampler(random.Random(999))

    out1 = [s1.sample(schema) for _ in range(20)]
    out2 = [s2.sample(schema) for _ in range(20)]

    assert out1 == out2


