import pytest
import pandas as pd
from scripts.utils import onehotencode, infer_schema, compare_data_to_schema


# #
def test_onehotencode():
    data = pd.DataFrame({"numeric": [1.0, 2.0, 3.0], "object": ["foo", "bar", "baz"]})
    data, ohe_features, categories = onehotencode(data, "object")
    ohe_data = pd.DataFrame(
        {"numeric": [1.0, 2.0, 3.0], "baz": [0.0, 0.0, 1.0], "foo": [1.0, 0.0, 0.0]}
    )
    assert ohe_data.equals(data)
    assert ohe_features == ["baz", "foo"]
    assert categories == ["bar", "baz", "foo"]


def test_infer_schema():
    data = pd.DataFrame({"numeric": [1.0, 2.0, 3.0], "object": ["foo", "bar", "baz"]})
    schema = infer_schema(data)
    schema_correct = {
        "numeric": {"type": "float64", "min": 1.0, "max": 3.0},
        "object": {"type": "object", "domain": ["bar", "baz", "foo"]},
    }
    assert schema == schema_correct


def test_compare_data_to_schema_pass():
    data = pd.DataFrame({"numeric": [1.0, 2.0, 3.0], "object": ["foo", "bar", "baz"]})
    schema = {
        "numeric": {"type": "float64", "min": 1.0, "max": 3.0},
        "object": {"type": "object", "domain": ["foo", "bar", "baz"]},
    }

    status = compare_data_to_schema(data, schema)
    assert status == "Passed"


def test_compare_data_to_schema_missingDfCol():
    data = pd.DataFrame({"object": ["foo", "bar", "baz"]})
    schema = {
        "numeric": {"type": "float64", "min": 1.0, "max": 3.0},
        "object": {"type": "object", "domain": ["foo", "bar", "baz"]},
    }
    status = compare_data_to_schema(data, schema)
    assert status == "Failed"


def test_compare_data_to_schema_missingSchemaCol():
    data = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0],
            "object": ["foo", "bar", "baz"],
            "extra": [1, 2, 3],
        }
    )
    schema = {
        "numeric": {"type": "float64", "min": 1.0, "max": 3.0},
        "object": {"type": "object", "domain": ["foo", "bar", "baz"]},
    }
    status = compare_data_to_schema(data, schema)
    assert status == "Failed"


def test_compare_data_to_schema_wrongNumType():
    data = pd.DataFrame({"numeric": [1, 2, 3], "object": ["foo", "bar", "baz"]})
    schema = {
        "numeric": {"type": "float64", "min": 1.0, "max": 3.0},
        "object": {"type": "object", "domain": ["foo", "bar", "baz"]},
    }
    status = compare_data_to_schema(data, schema)
    assert status == "Failed"


def test_compare_data_to_schema_wrongObjectType():
    data = pd.DataFrame({"numeric": [1.0, 2.0, 3.0], "object": [1, 2, 3]})
    schema = {
        "numeric": {"type": "float64", "min": 1.0, "max": 3.0},
        "object": {"type": "object", "domain": ["foo", "bar", "baz"]},
    }
    # with pytest.warns():
    #     status = compare_data_to_schema(data, schema)
    #     assert status == "Failed"
    status = compare_data_to_schema(data, schema)
    assert status == "Failed"


def test_compare_data_to_schema_higherMax():
    data = pd.DataFrame({"numeric": [1.0, 2.0, 10.0], "object": ["foo", "bar", "baz"]})
    schema = {
        "numeric": {"type": "float64", "min": 1.0, "max": 3.0},
        "object": {"type": "object", "domain": ["foo", "bar", "baz"]},
    }
    with pytest.warns() as record:
        status = compare_data_to_schema(data, schema)
    assert len(record) == 1
    assert (
        record[0].message.args[0]
        == "Column `numeric` has values (10.0) higher than max of schema (3.0)"
    )


def test_compare_data_to_schema_lowerMin():
    data = pd.DataFrame({"numeric": [0.5, 2.0, 3.0], "object": ["foo", "bar", "baz"]})
    schema = {
        "numeric": {"type": "float64", "min": 1.0, "max": 3.0},
        "object": {"type": "object", "domain": ["foo", "bar", "baz"]},
    }
    with pytest.warns() as record:
        status = compare_data_to_schema(data, schema)
    assert len(record) == 1
    assert (
        record[0].message.args[0]
        == "Column `numeric` has values (0.5) lower than min of schema (1.0)"
    )


def test_compare_data_to_schema_notInDomain():
    data = pd.DataFrame(
        {"numeric": [1.0, 2.0, 3.0, 1.0], "object": ["dog", "god", "bar", "baz"]}
    )
    schema = {
        "numeric": {"type": "float64", "min": 1.0, "max": 3.0},
        "object": {"type": "object", "domain": ["foo", "bar", "baz"]},
    }
    with pytest.warns() as record:
        status = compare_data_to_schema(data, schema)
    assert len(record) == 1
    assert (
        record[0].message.args[0]
        == "Column `object` domain contains ['dog', 'god'] that are not present in schema"
    )


# python -m pytest -s -v ./tests/
