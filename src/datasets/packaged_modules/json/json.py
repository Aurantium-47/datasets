import codecs
import io
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.json as paj

import datasets
import datasets.config
from datasets import List, Value
from datasets.builder import Key
from datasets.table import table_cast
from datasets.utils.file_utils import readline
from datasets.utils.json import (
    find_mixed_struct_types_field_paths,
    get_json_field_path_from_pyarrow_json_error,
    get_json_field_paths_from_feature,
    insert_json_field_path,
    json_encode_field,
    json_encode_fields_in_json_lines,
    set_json_types_in_feature,
    ujson_dumps,
    ujson_loads,
)


logger = datasets.utils.logging.get_logger(__name__)


def pandas_read_json(path_or_buf, **kwargs):
    if datasets.config.PANDAS_VERSION.major >= 2:
        kwargs["dtype_backend"] = "pyarrow"
    return pd.read_json(path_or_buf, **kwargs)


class FullReadDisallowed(Exception):
    pass


def _iter_json_array_batches(stream, first_chunk, chunksize, encoding_errors="strict"):
    """Yield complete UTF-8 array fragments bounded by chunksize + largest record.

    The standard decoder locates/validates records, but does not determine their
    numeric values. Callers retain the existing ujson parser for conversion.
    """
    text_decoder = codecs.getincrementaldecoder("utf-8")(errors=encoding_errors)
    # Avoid expensive/int-limit-dependent conversion merely to locate boundaries.
    record_decoder = json.JSONDecoder(
        parse_int=lambda value: None,
        parse_float=lambda value: None,
        parse_constant=lambda value: None,
        object_pairs_hook=lambda value: None,
    )
    buffer = text_decoder.decode(first_chunk)[1:]
    position = 0
    eof = False
    allow_empty = True
    records = []
    size = 0

    def read_more():
        nonlocal buffer, position, eof
        chunk = stream.read(chunksize)
        eof = not chunk
        buffer = buffer[position:] + text_decoder.decode(chunk, final=eof)
        position = 0

    while True:
        while position < len(buffer) and buffer[position] in " \t\r\n":
            position += 1
        if position == len(buffer):
            if eof:
                raise ValueError("Unexpected end of JSON array")
            read_more()
            continue
        if buffer[position] == "]":
            if not allow_empty:
                raise ValueError("Trailing comma in JSON array")
            position += 1
            break
        try:
            end = record_decoder.raw_decode(buffer, position)[1]
        except json.JSONDecodeError:
            if eof:
                raise
            read_more()
            continue
        separator = end
        if separator < len(buffer) and buffer[separator] in " \t\r\n":
            # Whitespace terminates a scalar. Keep the record but discard gaps
            # between records instead of retaining/reparsing arbitrarily long
            # whitespace while waiting for the comma or closing bracket.
            record = buffer[position:end]
            position = separator
            while True:
                while position < len(buffer) and buffer[position] in " \t\r\n":
                    position += 1
                if position < len(buffer):
                    break
                if eof:
                    raise ValueError("Unexpected end of JSON array")
                read_more()
            separator = position
            if buffer[separator] not in ",]":
                raise ValueError("Expected a comma or closing bracket in JSON array")
        elif separator == len(buffer):
            if eof:
                raise ValueError("Expected a comma or closing bracket in JSON array")
            # A number can appear complete before its decimal point or exponent
            # arrives. Do not commit it until a delimiter or whitespace follows.
            read_more()
            continue
        elif buffer[separator] not in ",]":
            # raw_decode can return the integer part of a partial float, such as
            # "1e+". Other suffixes are already invalid and need no further read.
            if not eof and re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]*)?(?:[eE][+-]?[0-9]*)?", buffer[position:]):
                read_more()
                continue
            raise ValueError("Expected a comma or closing bracket in JSON array")
        else:
            record = buffer[position:end]
        records.append(record)
        size += len(record.encode("utf-8")) + 1
        close = buffer[separator] == "]"
        position = separator + 1
        if close:
            break
        allow_empty = False
        if size >= chunksize:
            yield ("[" + ",".join(records) + "]").encode("utf-8")
            records = []
            size = 0

    # Reject non-whitespace after the document, without retaining trailing data.
    if buffer[position:].strip(" \t\r\n"):
        raise ValueError("Extra data after JSON array")
    while not eof:
        position = len(buffer)
        read_more()
        if buffer[position:].strip(" \t\r\n"):
            raise ValueError("Extra data after JSON array")
    if records:
        yield ("[" + ",".join(records) + "]").encode("utf-8")


@dataclass
class JsonConfig(datasets.BuilderConfig):
    """BuilderConfig for JSON."""

    features: Optional[datasets.Features] = None
    encoding: str = "utf-8"
    encoding_errors: Optional[str] = None
    field: Optional[str] = None
    use_threads: bool = True  # deprecated
    block_size: Optional[int] = None  # deprecated
    chunksize: int = 10 << 20  # 10MB
    newlines_in_values: Optional[bool] = None
    on_mixed_types: Optional[Literal["use_json"]] = "use_json"
    parse_agent_traces: bool = True
    json_array_batching: bool = False

    def __post_init__(self):
        super().__post_init__()


class Json(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = JsonConfig

    def _info(self):
        if self.config.json_array_batching:
            if self.config.features is None:
                raise ValueError("json_array_batching=True requires explicit features")
            if self.config.encoding != "utf-8":
                raise ValueError("json_array_batching=True currently supports only encoding='utf-8'")
            if self.config.field is not None:
                raise ValueError("json_array_batching=True does not support field=; use a top-level JSON array")
        if self.config.block_size is not None:
            logger.warning("The JSON loader parameter `block_size` is deprecated. Please use `chunksize` instead")
            self.config.chunksize = self.config.block_size
        if self.config.json_array_batching and self.config.chunksize <= 0:
            raise ValueError("json_array_batching=True requires a positive chunksize")
        if self.config.use_threads is not True:
            logger.warning(
                "The JSON loader parameter `use_threads` is deprecated and doesn't have any effect anymore."
            )
        if self.config.newlines_in_values is not None:
            raise ValueError("The JSON loader parameter `newlines_in_values` is no longer supported")
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles"""
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        dl_manager.download_config.extract_on_the_fly = True
        base_data_files = dl_manager.download(self.config.data_files)
        extracted_data_files = dl_manager.extract(base_data_files)
        splits = []
        for split_name, extracted_files in extracted_data_files.items():
            files_iterables = [dl_manager.iter_files(extracted_file) for extracted_file in extracted_files]
            splits.append(
                datasets.SplitGenerator(
                    name=split_name,
                    gen_kwargs={
                        "files_iterables": files_iterables,
                        "base_files": base_data_files[split_name],
                        "original_files": self.config.data_files[split_name],
                    },
                )
            )
        if self.info.features is None:
            try:
                pa_table = next(iter(self._generate_tables(**splits[0].gen_kwargs, allow_full_read=False)))[1]
                self.info.features = datasets.Features.from_arrow_schema(pa_table.schema)
                if self.config.parse_agent_traces and has_agent_traces_markers(self.info.features):
                    self.info.features = AGENT_TRACES_FEATURES
            except FullReadDisallowed:
                pass
        return splits

    def _cast_table(self, pa_table: pa.Table, json_field_paths=()) -> pa.Table:
        if self.info.features is not None:
            # adding missing columns
            for column_name in set(self.info.features) - set(pa_table.column_names):
                type = self.info.features.arrow_schema.field(column_name).type
                pa_table = pa_table.append_column(column_name, pa.array([None] * len(pa_table), type=type))
            # convert to string when needed
            for i, column_name in enumerate(pa_table.column_names):
                if pa.types.is_struct(pa_table[column_name].type) and self.info.features.get(
                    column_name, None
                ) == Value("string"):
                    jsonl = (
                        pa_table[column_name]
                        .to_pandas(types_mapper=pd.ArrowDtype)
                        .to_json(orient="records", lines=True)
                    )
                    string_array = pa.array(
                        (None if x.strip() == "null" else x.strip() for x in jsonl.split("\n") if x.strip()),
                        type=pa.string(),
                    )
                    pa_table = pa_table.set_column(i, column_name, string_array)
            # more expensive cast to support nested structures with keys in a different order
            # allows str <-> int/float or str to Audio for example
            pa_table = table_cast(pa_table, self.info.features.arrow_schema)
        elif json_field_paths:
            features = datasets.Features.from_arrow_schema(pa_table.schema)
            features = set_json_types_in_feature(features, json_field_paths)
            pa_table = table_cast(pa_table, features.arrow_schema)
        return pa_table

    def _generate_shards(self, base_files, files_iterables, original_files):
        yield from base_files

    def _generate_tables(self, base_files, files_iterables, original_files, allow_full_read=True):
        json_field_paths = []
        is_agent_traces = False

        if self.info.features is not None:
            if self.info.features == AGENT_TRACES_FEATURES:
                is_agent_traces = True
                if datasets.config.TEICH_AVAILABLE:
                    from teich import convert_traces_to_training_data
                else:
                    raise ImportError("To support decoding agent traces, please install 'teich'.")
            json_field_paths = get_json_field_paths_from_feature(self.info.features)

        for shard_idx, files_iterable in enumerate(files_iterables):
            for file in files_iterable:
                # If the file is one json object and if we need to look at the items in one specific field
                if self.config.field is not None:
                    if not allow_full_read:
                        raise FullReadDisallowed()
                    with open(file, encoding=self.config.encoding, errors=self.config.encoding_errors) as f:
                        dataset = ujson_loads(f.read())
                    # We keep only the field we are interested in
                    dataset = dataset[self.config.field]
                    df = pandas_read_json(io.StringIO(ujson_dumps(dataset)))
                    if df.columns.tolist() == [0]:
                        df.columns = list(self.config.features) if self.config.features else ["text"]
                    pa_table = pa.Table.from_pandas(df, preserve_index=False)
                    yield Key(shard_idx, 0), self._cast_table(pa_table)

                # If the files are agent traces (one row = one file except for hermes which can have multiple sessions per file)
                elif is_agent_traces:
                    trace_file = Path(file)
                    trace = trace_file.read_text(encoding="utf-8")
                    lines = trace.splitlines()
                    file_path = original_files[shard_idx]
                    if self.base_path is not None and file_path.startswith(self.base_path):
                        file_path = os.path.relpath(file_path, self.base_path)
                    training_examples = convert_traces_to_training_data(trace_file)
                    examples = []
                    for i, training_example in enumerate(training_examples):
                        if training_example["metadata"]["trace_type"] == "hermes":
                            timestamp = ujson_loads(lines[i])["started_at"]
                            milliseconds = timestamp if timestamp <= 10_000_000_000 else timestamp * 1_000
                            sent_at = (
                                datetime.fromtimestamp(milliseconds, tz=timezone.utc)
                                .isoformat(timespec="milliseconds")
                                .replace("+00:00", "Z")
                            )
                            bonus_fields = {
                                "harness": training_example["metadata"]["trace_type"],
                                "session_id": training_example["metadata"]["session_id"],
                                "sent_at": sent_at,
                                "num_user_messages": training_example["metadata"]["turn_count"],
                                "num_tool_calls": training_example["metadata"]["tool_call_count"],
                                "trace": lines[i],
                            }
                        else:
                            harness, session_id, prompt, sent_at, num_user_messages, num_tool_calls = (
                                parse_traces_info(lines)
                            )
                            bonus_fields = {
                                "harness": harness,
                                "session_id": session_id,
                                "prompt": prompt,
                                "sent_at": sent_at,
                                "num_user_messages": num_user_messages,
                                "num_tool_calls": num_tool_calls,
                                "trace": trace,
                            }
                        example = {
                            **dict.fromkeys(AGENT_TRACES_FEATURES),
                            **training_example,
                            **bonus_fields,
                            "file_path": file_path,
                        }
                        for json_field_path in json_field_paths:
                            example = json_encode_field(example, json_field_path)
                        examples.append(example)
                    pa_table = pa.Table.from_pylist(examples)
                    yield Key(shard_idx, 0), self._cast_table(pa_table)

                # If the file has one json object per line
                else:
                    with open(file, "rb") as f:
                        batch_idx = 0
                        # Use block_size equal to the chunk size divided by 32 to leverage multithreading
                        # Set a default minimum value of 16kB if the chunk size is really small
                        block_size = max(self.config.chunksize // 32, 16 << 10)
                        encoding_errors = (
                            self.config.encoding_errors if self.config.encoding_errors is not None else "strict"
                        )
                        array_batches = None
                        while True:
                            batch = (
                                next(array_batches, b"")
                                if array_batches is not None
                                else f.read(self.config.chunksize)
                            )
                            if self.config.json_array_batching and batch_idx == 0 and array_batches is None:
                                # An array may start after a BOM or whitespace. Keep
                                # probing bounded even for a long whitespace prefix.
                                while batch and len(batch) < 3 and codecs.BOM_UTF8.startswith(batch):
                                    more = f.read(3 - len(batch))
                                    if not more:
                                        break
                                    batch += more
                                if batch.startswith(codecs.BOM_UTF8):
                                    batch = batch[len(codecs.BOM_UTF8) :]
                                batch = batch.lstrip(b" \t\r\n")
                                while not batch:
                                    batch = f.read(self.config.chunksize)
                                    if not batch:
                                        break
                                    batch = batch.lstrip(b" \t\r\n")
                            if not batch:
                                break
                            # A leading UTF-8 BOM makes the ujson pre-scan below raise
                            # ValueError, silently skipping mixed-struct detection, while
                            # PyArrow tolerates the BOM -- so the inferred schema differed
                            # depending on whether the file started with a BOM. Strip it so
                            # both see the same bytes (a BOM only appears at the start of the file).
                            if shard_idx == 0 and batch_idx == 0 and batch.startswith(codecs.BOM_UTF8):
                                batch = batch[len(codecs.BOM_UTF8) :]
                            if batch.startswith(b"["):
                                if self.config.json_array_batching:

                                    def to_jsonlines(arrays):
                                        wrap_text = None
                                        for array in arrays:
                                            examples = ujson_loads(array)
                                            if wrap_text is None:
                                                wrap_text = not isinstance(examples[0], dict)
                                            yield "\n".join(
                                                ujson_dumps({"text": x} if wrap_text else x) for x in examples
                                            ).encode()

                                    array_batches = to_jsonlines(
                                        _iter_json_array_batches(f, batch, self.config.chunksize, encoding_errors)
                                    )
                                    batch = next(array_batches, b"")
                                    if not batch:
                                        break
                                elif not allow_full_read:
                                    raise FullReadDisallowed()
                                else:
                                    # convert to JSON Lines
                                    full_data = batch + f.read()
                                    if b"{" in batch[:100].split(b'"', 1)[0]:  # list of objects
                                        batch = "\n".join(ujson_dumps(x) for x in ujson_loads(full_data)).encode()
                                    else:  # list of strings
                                        batch = "\n".join(
                                            ujson_dumps({"text": x}) for x in ujson_loads(full_data)
                                        ).encode()
                            # Finish the current JSONL line, unless the array reader
                            # already supplied a batch of complete records.
                            if array_batches is None:
                                try:
                                    batch += f.readline()
                                except (AttributeError, io.UnsupportedOperation):
                                    batch += readline(f)
                            # PyArrow only accepts utf-8 encoded bytes
                            if self.config.encoding != "utf-8":
                                batch = batch.decode(self.config.encoding, errors=encoding_errors).encode("utf-8")
                            # On first batch we check for lists of objects with arbitrary fields
                            if (
                                shard_idx == 0
                                and batch_idx == 0
                                and self.info.features is None
                                and self.config.on_mixed_types == "use_json"
                            ):
                                try:
                                    examples = [ujson_loads(line) for line in batch.splitlines()]
                                except ValueError:
                                    # the file is likely not JSON Lines and may contain one single multi-line JSON object
                                    pass
                                else:
                                    json_field_paths += find_mixed_struct_types_field_paths(examples)
                            # Re-encode JSON fields
                            original_batch = batch
                            if json_field_paths:
                                examples = [ujson_loads(line) for line in batch.splitlines()]
                                for json_field_path in json_field_paths:
                                    examples = [json_encode_field(example, json_field_path) for example in examples]
                                batch = "\n".join(ujson_dumps(example) for example in examples).encode()
                            # Disable parallelism if block size is ~ len(batch) to avoid segfault
                            block_size = len(batch) if len(batch) // 8 > block_size else block_size
                            pa_table = None
                            try:
                                while True:
                                    try:
                                        pa_table = paj.read_json(
                                            io.BytesIO(batch), read_options=paj.ReadOptions(block_size=block_size)
                                        )
                                        pa_table.validate()
                                        break
                                    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as e:
                                        if batch.startswith(b"["):  # paj.read_json only supports json lines
                                            raise
                                        elif self.config.on_mixed_types == "use_json" and (
                                            isinstance(e, pa.ArrowInvalid)
                                            and "JSON parse error: Column(" in str(e)
                                            and ") changed from" in str(e)
                                        ):
                                            json_field_path = get_json_field_path_from_pyarrow_json_error(str(e))
                                            insert_json_field_path(json_field_paths, json_field_path)
                                            batch = json_encode_fields_in_json_lines(original_batch, json_field_paths)
                                        elif (
                                            "straddling" in str(e) or "JSON conversion to" in str(e)
                                        ) and block_size < len(batch):
                                            # Increase the block size in case it was too small.
                                            # The block size will be reset for the next file.
                                            # this is needed in case of "stradding" or for some JSON conversions (see https://github.com/huggingface/datasets/issues/2799)
                                            logger.debug(
                                                f"Batch of {len(batch)} bytes couldn't be parsed with block_size={block_size}. Retrying with block_size={block_size * 2}."
                                            )
                                            block_size *= 2
                                        else:
                                            raise
                            except pa.ArrowInvalid as e:
                                if pa_table is not None:
                                    try:
                                        examples = [ujson_loads(line) for line in batch.splitlines()]
                                        pa_table = pa.concat_tables(
                                            [pa.Table.from_pylist([example]) for example in examples],
                                            promote_options="default",
                                        )
                                        pa_table.validate()
                                    except (ValueError, pa.ArrowInvalid, pa.ArrowNotImplementedError):
                                        raise e
                                    yield (
                                        Key(shard_idx, batch_idx),
                                        self._cast_table(pa_table, json_field_paths=json_field_paths),
                                    )
                                    batch_idx += 1
                                    continue
                                if array_batches is not None:
                                    # An invalid chunk must not fall back to the
                                    # whole-file pandas reader in bounded mode.
                                    raise e
                                if not allow_full_read:
                                    raise FullReadDisallowed()
                                try:
                                    with open(
                                        file, encoding=self.config.encoding, errors=self.config.encoding_errors
                                    ) as f:
                                        df = pandas_read_json(f)
                                except ValueError:
                                    logger.error(f"Failed to load JSON from file '{file}' with error {type(e)}: {e}")
                                    raise e
                                if df.columns.tolist() == [0]:
                                    df.columns = list(self.config.features) if self.config.features else ["text"]
                                try:
                                    pa_table = pa.Table.from_pandas(df, preserve_index=False)
                                except pa.ArrowInvalid as e:
                                    logger.error(
                                        f"Failed to convert pandas DataFrame to Arrow Table from file '{file}' with error {type(e)}: {e}"
                                    )
                                    raise ValueError(
                                        f"Failed to convert pandas DataFrame to Arrow Table from file {file}."
                                    ) from None
                                yield Key(shard_idx, 0), self._cast_table(pa_table)
                                break
                            yield (
                                Key(shard_idx, batch_idx),
                                self._cast_table(pa_table, json_field_paths=json_field_paths),
                            )
                            batch_idx += 1


AGENT_TRACES_TYPES_VALUES = {
    "claude_code": ["user", "assistant", "system"],
    "pi": ["session", "message"],
    "codex": ["session_meta", "turn_context", "response_item", "event_msg"],
    # droid message events share pi's "message" type, but droid traces always start with a session_start event
    "droid": ["session_start"],
}
AGENT_TRACES_TYPE_TO_HARNESS = {}
for _harness, _trace_types in AGENT_TRACES_TYPES_VALUES.items():
    for _trace_type in _trace_types:
        AGENT_TRACES_TYPE_TO_HARNESS[_trace_type] = _harness


AGENT_TRACES_FEATURES_MARKERS = {
    "claude_code_or_pi_or_openclaw": datasets.Features(
        {
            "type": lambda f: f == Value("string"),
            "message": lambda f: f == datasets.Json(),
        }
    ),
    "codex": datasets.Features(
        {
            "type": lambda f: f == Value("string"),
            "payload": lambda f: f == datasets.Json(),
        }
    ),
    "hermes": datasets.Features(
        {
            "id": lambda f: f == Value("string"),
            "source": lambda f: f == datasets.Value("string"),
            "model": lambda f: f == datasets.Value("string"),
            "system_prompt": lambda f: f == datasets.Value("string"),
            "messages": lambda f: isinstance(f, (datasets.List, datasets.Json)),
        }
    ),
    "droid": datasets.Features(
        {
            "type": lambda f: f == Value("string"),
            "id": lambda f: f == Value("string"),
            "version": lambda f: f == Value("int64"),
            "cwd": lambda f: f == Value("string"),
        }
    ),
}

AGENT_TRACES_FEATURES = datasets.Features(
    {
        # basic features
        "harness": Value("string"),
        "session_id": Value("string"),
        # teich features
        "prompt": Value("string"),
        "messages": List(datasets.Json()),
        "tools": List(datasets.Json()),
        "metadata": datasets.Json(),
        # bonus features
        "sent_at": Value("string"),
        "num_user_messages": Value("int64"),
        "num_tool_calls": Value("int64"),
        "trace": datasets.Json(),
        "file_path": Value("string"),
    }
)


def has_agent_traces_markers(features: datasets.Features) -> bool:
    for agent_traces_features_marker in AGENT_TRACES_FEATURES_MARKERS.values():
        if all(feature_marker(features.get(key)) for key, feature_marker in agent_traces_features_marker.items()):
            return True
    return False


def parse_traces_info(
    trace_events: list[str],
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str], int, int]:
    harness, session_id, prompt, sent_at = None, None, None, None
    # prompt/sent_at describe the first user message; the counters summarize the whole trace file.
    # Codex response_item user messages can include context files (for example AGENTS.md), so only event_msg
    # user_message records are treated as Codex user messages.
    num_user_messages = 0
    num_tool_calls = 0
    for event in trace_events:
        decoded_event = ujson_loads(event)
        if harness is None:
            if "type" in decoded_event and isinstance(decoded_event["type"], str):
                harness = AGENT_TRACES_TYPE_TO_HARNESS.get(decoded_event["type"])
        if session_id is None:
            session_id = get_session_id(decoded_event)
            if (
                session_id is not None
                and decoded_event.get("type") == "session"
                and isinstance(decoded_event.get("cwd"), str)
                and "/.openclaw/" in decoded_event["cwd"]
            ):
                harness = "openclaw"
        user_prompt = get_user_prompt(decoded_event)
        if user_prompt is not None:
            num_user_messages += 1
            if prompt is None:
                prompt = user_prompt
                sent_at = get_trace_event_timestamp(decoded_event)
        num_tool_calls += get_tool_call_count(decoded_event)
    return harness, session_id, prompt, sent_at, num_user_messages, num_tool_calls


def get_session_id(trace: dict) -> Optional[str]:
    # claude
    if isinstance(trace.get("sessionId"), str):
        return trace["sessionId"]
    # claude (not sure but this format does exist online)
    if isinstance(trace.get("session_id"), str):
        return trace["session_id"]
    # codex
    if isinstance(trace.get("payload"), dict) and isinstance(trace["payload"].get("id"), str):
        return trace["payload"]["id"]
    # pi / openclaw on "session" (openclaw embeds pi-agent; distinguish via cwd), droid on "session_start"
    if trace.get("type") in ("session", "session_start") and isinstance(trace.get("id"), str):
        return trace["id"]
    return None


def get_user_prompt(trace_event: dict) -> Optional[str]:
    if trace_event.get("type") == "user" and isinstance(trace_event.get("message"), dict):
        message = trace_event["message"]
        if message.get("role") == "user":
            return get_content_text(message.get("content"))

    if trace_event.get("type") == "message":
        if isinstance(trace_event.get("message"), dict):
            message = trace_event["message"]
            # droid marks injected context as llm_only and local-only notes as user_only, neither is a real user prompt
            if message.get("role") == "user" and message.get("visibility") not in ("llm_only", "user_only"):
                return get_content_text(message.get("content"))
        if trace_event.get("role") == "user":
            return get_content_text(trace_event.get("content"))

    if trace_event.get("type") == "event_msg" and isinstance(trace_event.get("payload"), dict):
        payload = trace_event["payload"]
        if payload.get("type") == "user_message":
            return get_content_text(payload.get("message"))

    return None


def get_tool_call_count(trace_event: dict) -> int:
    trace_type = trace_event.get("type")
    if trace_type == "response_item":
        payload = trace_event.get("payload")
        if isinstance(payload, dict) and payload.get("type") == "function_call":
            return 1
        return 0

    if trace_type not in {"assistant", "message"} or not isinstance(trace_event.get("message"), dict):
        return 0
    message = trace_event["message"]
    if message.get("role") != "assistant":
        return 0
    content = message.get("content")
    if not isinstance(content, list):
        return 0
    return sum(
        1
        for content_part in content
        if isinstance(content_part, dict) and content_part.get("type") in {"tool_use", "toolCall"}
    )


def get_trace_event_timestamp(trace_event: dict) -> Optional[str]:
    timestamp = trace_event.get("timestamp")
    if isinstance(timestamp, str):
        return timestamp
    return None


def get_content_text(content) -> Optional[str]:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        content_parts = []
        for content_part in content:
            if isinstance(content_part, str):
                content_parts.append(content_part)
            elif isinstance(content_part, dict):
                text = content_part.get("text")
                if isinstance(text, str):
                    content_parts.append(text)
        if content_parts:
            return "\n".join(content_parts)
    return None
