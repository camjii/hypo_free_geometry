import json

import numpy as np
import pytest

from activation_extraction import (
    DEFAULT_LAYER,
    DEFAULT_MODEL_ID,
    _resolve_prompt_input,
    build_parser,
    default_output_path,
    final_token_indices,
    hidden_state_index,
    value_token_indices,
    load_prompt_values,
    main,
    save_activation_archive,
    transformers_dtype_keyword,
)


def _metadata():
    return {
        "model_id": DEFAULT_MODEL_ID,
        "model_revision": "abc123",
        "layer": DEFAULT_LAYER,
        "template_name": "month_of_year",
        "template": "The month of the year is {value}",
    }


def test_extraction_defaults_target_base_gemma_7b_layer_6():
    args = build_parser().parse_args([])
    assert args.model == "google/gemma-7b"
    assert args.layer == 6
    assert args.device == "cpu"
    assert args.dtype == "auto"


def test_default_output_is_portable_and_keyed_by_model_set_and_layer():
    assert default_output_path(DEFAULT_MODEL_ID, "months", 6).as_posix() == (
        "outputs/activations/gemma-7b/months_layer_6.npz"
    )


def test_value_token_indices_read_the_concept_not_trailing_punctuation():
    # "The month is January." -> the value ends before the final period token.
    offsets = np.array([[[0, 0], [0, 3], [4, 9], [10, 12], [13, 20], [20, 21]]])
    special = np.array([[1, 0, 0, 0, 0, 0]])
    assert value_token_indices(offsets, special, [(13, 20)]).tolist() == [4]


def test_value_token_indices_handle_mid_sentence_and_multi_token_values():
    # row 0: value mid-sentence; row 1: value spanning two tokens, then padding.
    offsets = np.array(
        [
            [[0, 0], [0, 3], [4, 8], [9, 10], [11, 16], [16, 17]],
            [[0, 0], [0, 3], [4, 9], [10, 14], [14, 19], [0, 0]],
        ]
    )
    special = np.array([[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
    indices = value_token_indices(offsets, special, [(9, 10), (10, 19)])
    assert indices.tolist() == [3, 4]


def test_value_token_indices_reject_spans_without_a_matching_token():
    offsets = np.array([[[0, 0], [0, 3], [4, 9]]])
    special = np.array([[1, 0, 0]])
    with pytest.raises(ValueError, match="no token overlaps"):
        value_token_indices(offsets, special, [(50, 60)])
    with pytest.raises(ValueError, match="non-empty"):
        value_token_indices(offsets, special, [(5, 5)])


def test_final_token_indices_support_variable_length_right_padding():
    mask = np.asarray([[1, 1, 1], [1, 0, 0], [1, 1, 0]])
    np.testing.assert_array_equal(final_token_indices(mask), [2, 0, 1])


@pytest.mark.parametrize(
    "mask, message",
    [
        (np.ones(3), "2D"),
        (np.asarray([[1, 2]]), "only 0 and 1"),
        (np.asarray([[0, 0]]), "at least one"),
        (np.asarray([[0, 1]]), "right padding"),
        (np.asarray([[1, 0, 1]]), "right padding"),
    ],
)
def test_final_token_indices_reject_invalid_masks(mask, message):
    with pytest.raises(ValueError, match=message):
        final_token_indices(mask)


def test_layer_index_accounts_for_embedding_entry_and_checks_bounds():
    assert hidden_state_index(0, 28) == 1
    assert hidden_state_index(6, 28) == 7
    assert hidden_state_index(27, 28) == 28
    with pytest.raises(ValueError, match=r"\[0, 27\]"):
        hidden_state_index(28, 28)
    with pytest.raises(ValueError, match=r"\[0, 27\]"):
        hidden_state_index(-1, 28)


def test_model_loading_uses_the_transformers_major_version_dtype_keyword():
    assert transformers_dtype_keyword("4.38.0") == "torch_dtype"
    assert transformers_dtype_keyword("5.13.1") == "dtype"
    with pytest.raises(ValueError, match="cannot parse"):
        transformers_dtype_keyword("development")


def test_builtin_and_file_backed_prompt_inputs_share_one_contract(tmp_path):
    name, template_name, labels, prompts = _resolve_prompt_input(
        prompt_set_name="months", template_name=None, values_file=None
    )
    assert (name, template_name) == ("months", "month_of_year")
    assert labels[0] == "January"
    assert prompts[0] == "The month of the year is January"

    values_file = tmp_path / "years.txt"
    values_file.write_text("# held-out values\n1990\n\n2000\n")
    name, template_name, labels, prompts = _resolve_prompt_input(
        prompt_set_name=None,
        template_name="year",
        values_file=values_file,
    )
    assert (name, template_name) == ("years", "year")
    assert labels == ("1990", "2000")
    assert prompts == ("In the year 1990", "In the year 2000")


def test_values_file_rejects_empty_and_duplicate_labels(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("# only a comment\n")
    with pytest.raises(ValueError, match="empty"):
        load_prompt_values(empty)

    duplicates = tmp_path / "duplicates.txt"
    duplicates.write_text("January\nJanuary\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_prompt_values(duplicates)


def test_archive_matches_shared_activation_contract_and_records_provenance(tmp_path):
    path = tmp_path / "months.npz"
    activations = np.arange(12, dtype=np.float32).reshape(3, 4)
    labels = ("January", "February", "March")
    prompts = tuple(f"The month of the year is {label}" for label in labels)
    archive, manifest = save_activation_archive(
        path,
        activations=activations,
        labels=labels,
        prompts=prompts,
        metadata=_metadata(),
    )

    with np.load(archive, allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored["activations"], activations)
        np.testing.assert_array_equal(stored["labels"], labels)
        np.testing.assert_array_equal(stored["prompts"], prompts)
        assert stored["layer"].item() == 6
        assert stored["model"].item() == DEFAULT_MODEL_ID
        assert stored["model_revision"].item() == "abc123"

    provenance = json.loads(manifest.read_text())
    assert provenance["artifact"]["shape"] == [3, 4]
    assert provenance["artifact"]["dtype"] == "float32"
    assert len(provenance["artifact"]["sha256"]) == 64

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_activation_archive(
            path,
            activations=activations,
            labels=labels,
            prompts=prompts,
            metadata=_metadata(),
        )


def test_registry_listing_does_not_load_model_dependencies(capsys):
    assert main(["--list"]) == 0
    output = capsys.readouterr().out
    assert "Built-in prompt sets:" in output
    assert "google/gemma-7b" not in output
    assert "month_of_year" in output
