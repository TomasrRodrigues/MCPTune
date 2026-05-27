from mcptune.sampling.recursive import RecursiveSampler
import pytest

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

