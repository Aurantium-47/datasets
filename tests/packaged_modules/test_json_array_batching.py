import codecs
import gzip
import io
import json
import tracemalloc
from unittest.mock import patch

import pyarrow as pa
import pytest

from datasets import Features, List, Value, load_dataset
from datasets.exceptions import DatasetGenerationError
from datasets.packaged_modules.json.json import Json, _iter_json_array_batches, ujson_loads


class BoundedReader(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.read_sizes = []
        self.bytes_read = 0

    def read(self, size=-1):
        assert size >= 0, "The reader must not request the whole file"
        self.read_sizes.append(size)
        data = super().read(size)
        self.bytes_read += len(data)
        return data

    def readline(self, size=-1):
        raise AssertionError("Array input must not be read as one unbounded line")


@pytest.mark.parametrize("chunksize", [1, 2, 3, 7, 31, 128, 4096])
def test_json_array_parser_preserves_records(chunksize):
    rows = [
        {"text": '雪😀,[]{}\\"\n', "items": [None, True, {"x": [1, -2, 1.2e-30]}]},
        "a" * 300,
        9007199254740993,
        -1.234567890123e30,
        None,
        False,
        [],
        {},
    ]
    data = json.dumps(rows, ensure_ascii=False).encode()
    stream = BoundedReader(data)
    arrays = list(_iter_json_array_batches(stream, stream.read(chunksize), chunksize))
    assert [value for array in arrays for value in ujson_loads(array)] == ujson_loads(data)
    assert max(stream.read_sizes) == chunksize


@pytest.mark.parametrize("chunksize", [1, 2, 7, 64])
@pytest.mark.parametrize(
    "data",
    [b"[1,]", b"[,1]", b"[1,,2]", b"[1 2]", b"[1e]", b"[1e+]", b'["a]', b"[[1,2]", b"[{}]x", b"[{}][{}]"],
)
def test_json_array_parser_rejects_malformed_input(chunksize, data):
    stream = BoundedReader(data)
    with pytest.raises(ValueError):
        list(_iter_json_array_batches(stream, stream.read(chunksize), chunksize))


def test_json_array_yields_before_reading_whole_file():
    data = json.dumps([{"tokens": list(range(30))}] * 1000).encode()
    stream = BoundedReader(data)
    chunksize = 512
    batches = _iter_json_array_batches(stream, stream.read(chunksize), chunksize)
    assert ujson_loads(next(batches))
    assert stream.bytes_read < len(data) // 10
    assert stream.bytes_read <= 3 * chunksize
    assert sum(len(ujson_loads(batch)) for batch in batches) < 1000


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("chunksize", [1, 7, 64, 1024])
def test_load_json_array_batches(tmp_path, streaming, compressed, chunksize):
    rows = [
        {"id": 0, "text": "雪😀" * 20, "tokens": [], "meta": None},
        {"id": 1, "text": '\\"[,]\n', "tokens": [101, 102], "meta": {"score": None}},
        {"id": 2, "text": None, "tokens": None, "meta": {"score": 0.75}},
    ] * 5
    features = Features(
        {
            "id": Value("int64"),
            "text": Value("string"),
            "tokens": List(Value("int32")),
            "meta": {"score": Value("float64")},
        }
    )
    payload = codecs.BOM_UTF8 + b" \n\t" + json.dumps(rows, ensure_ascii=False).encode()
    path = tmp_path / ("tokens.json.gz" if compressed else "tokens.json")
    path.write_bytes(gzip.compress(payload) if compressed else payload)
    dataset = load_dataset(
        "json",
        data_files=str(path),
        features=features,
        json_array_batching=True,
        chunksize=chunksize,
        streaming=streaming,
        split="train",
        cache_dir=str(tmp_path / "cache"),
    )
    assert dataset.features == features
    assert list(dataset) == rows


@pytest.mark.parametrize("streaming", [False, True])
def test_json_array_batching_coercion_is_opt_in(tmp_path, streaming):
    rows = [{"x": 9007199254740993}, {"x": 1.0}]
    path = tmp_path / "mixed.json"
    path.write_text(json.dumps(rows))
    features = Features({"x": Value("int64")})
    results = {}
    for enabled in [False, True]:
        dataset = load_dataset(
            "json",
            data_files=str(path),
            features=features,
            json_array_batching=enabled,
            chunksize=24,
            streaming=streaming,
            split="train",
            cache_dir=str(tmp_path / "cache"),
        )
        results[enabled] = list(dataset)
    assert results[False] == [{"x": 9007199254740992}, {"x": 1}]
    assert results[True] == [{"x": 9007199254740993}, {"x": 1}]


@pytest.mark.parametrize("streaming", [False, True])
def test_json_array_batching_preserves_jsonl(tmp_path, streaming):
    rows = [{"x": None}, {"x": 1}, {"x": 2}]
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    dataset = load_dataset(
        "json",
        data_files=str(path),
        features=Features({"x": Value("int64")}),
        json_array_batching=True,
        chunksize=7,
        streaming=streaming,
        split="train",
        cache_dir=str(tmp_path / "cache"),
    )
    assert list(dataset) == rows


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("suffix", [",]", "] unexpected"])
def test_load_json_array_batches_reports_invalid_input(tmp_path, streaming, suffix):
    path = tmp_path / "invalid.json"
    path.write_text('[{"x":1},{"x":2}' + suffix)
    with pytest.raises((ValueError, DatasetGenerationError)):
        dataset = load_dataset(
            "json",
            data_files=str(path),
            features=Features({"x": Value("int64")}),
            json_array_batching=True,
            chunksize=8,
            streaming=streaming,
            split="train",
            cache_dir=str(tmp_path / "cache"),
        )
        list(dataset)


def test_json_array_errors_do_not_fall_back_to_full_read(tmp_path):
    path = tmp_path / "mixed.json"
    path.write_text('[{"x":1},{"x":"a"}]')
    builder = Json(
        cache_dir=str(tmp_path / "cache"),
        features=Features({"x": Value("string")}),
        json_array_batching=True,
        on_mixed_types=None,
    )
    with patch(
        "datasets.packaged_modules.json.json.pandas_read_json", side_effect=AssertionError("Unbounded fallback")
    ):
        with pytest.raises(pa.ArrowInvalid):
            list(
                builder._generate_tables(
                    base_files=[str(path)], files_iterables=[[str(path)]], original_files=[str(path)]
                )
            )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({}, "explicit features"),
        ({"features": Features({"x": Value("int64")}), "encoding": "utf-16"}, "utf-8"),
        ({"features": Features({"x": Value("int64")}), "field": "data"}, "field="),
        ({"features": Features({"x": Value("int64")}), "chunksize": 0}, "positive chunksize"),
        ({"features": Features({"x": Value("int64")}), "block_size": 0}, "positive chunksize"),
    ],
)
def test_json_array_batching_validates_config(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        Json(cache_dir=str(tmp_path / "cache"), json_array_batching=True, **kwargs)


def test_json_array_discards_whitespace_between_records():
    gap_size = 1 << 20
    stream = BoundedReader(b"[1" + b" " * gap_size + b",2]")
    tracemalloc.start()
    try:
        arrays = list(_iter_json_array_batches(stream, stream.read(1024), 1024))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert [value for array in arrays for value in ujson_loads(array)] == [1, 2]
    assert peak < gap_size // 2


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("chunksize", [1, 7, 64])
def test_load_json_array_of_strings(tmp_path, streaming, chunksize):
    values = ["雪😀", None, 'abc,[]{}\\"', "x" * 100]
    path = tmp_path / "strings.json"
    path.write_text(json.dumps(values, ensure_ascii=False))
    dataset = load_dataset(
        "json",
        data_files=str(path),
        features=Features({"text": Value("string")}),
        json_array_batching=True,
        chunksize=chunksize,
        streaming=streaming,
        split="train",
        cache_dir=str(tmp_path / "cache"),
    )
    assert list(dataset) == [{"text": value} for value in values]


@pytest.mark.parametrize("streaming", [False, True])
def test_load_empty_json_array_matches_default(tmp_path, streaming):
    path = tmp_path / "empty.json"
    path.write_text("[]")
    for enabled in [False, True]:
        kwargs = {
            "data_files": str(path),
            "features": Features({"x": Value("int64")}),
            "json_array_batching": enabled,
            "streaming": streaming,
            "split": "train",
            "cache_dir": str(tmp_path / "cache"),
        }
        if streaming:
            assert list(load_dataset("json", **kwargs)) == []
        else:
            with pytest.raises(ValueError, match="corresponds to no data"):
                load_dataset("json", **kwargs)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("errors", ["strict", "replace", "ignore"])
def test_load_json_array_encoding_errors(tmp_path, streaming, errors):
    path = tmp_path / "invalid_utf8.json"
    path.write_bytes(b'[{"text":"a\xffb"}]')
    kwargs = {
        "data_files": str(path),
        "features": Features({"text": Value("string")}),
        "json_array_batching": True,
        "encoding_errors": errors,
        "chunksize": 1,
        "streaming": streaming,
        "split": "train",
        "cache_dir": str(tmp_path / "cache"),
    }
    if errors == "strict":
        with pytest.raises((UnicodeDecodeError, DatasetGenerationError)):
            list(load_dataset("json", **kwargs))
    else:
        expected = "a\ufffdb" if errors == "replace" else "ab"
        assert list(load_dataset("json", **kwargs)) == [{"text": expected}]
