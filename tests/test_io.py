import pytest

from romapy.io import load_gmt


def test_load_gmt_parses_standard_format(tmp_path):
    gmt = tmp_path / "test.gmt"
    gmt.write_text(
        "HALLMARK_APOPTOSIS\thttp://example.org/desc\tBAX\tBID\tCASP3\n"
        "HALLMARK_HYPOXIA\thttp://example.org/desc\tVEGFA\tHIF1A\n"
    )
    gene_sets = load_gmt(gmt)
    assert gene_sets == {
        "HALLMARK_APOPTOSIS": ["BAX", "BID", "CASP3"],
        "HALLMARK_HYPOXIA": ["VEGFA", "HIF1A"],
    }


def test_load_gmt_skips_blank_lines(tmp_path):
    gmt = tmp_path / "test.gmt"
    gmt.write_text("SET_A\tdesc\tG1\tG2\n\nSET_B\tdesc\tG3\n")
    gene_sets = load_gmt(gmt)
    assert set(gene_sets) == {"SET_A", "SET_B"}


def test_load_gmt_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_gmt(tmp_path / "does_not_exist.gmt")


def test_load_gmt_malformed_line_raises(tmp_path):
    gmt = tmp_path / "bad.gmt"
    gmt.write_text("SET_A_ONLY_ONE_FIELD\n")
    with pytest.raises(ValueError):
        load_gmt(gmt)


def test_load_gmt_skips_trailing_empty_field(tmp_path):
    # a trailing tab produces an empty-string "gene" - should be filtered, not kept
    gmt = tmp_path / "trailing.gmt"
    gmt.write_text("SET_A\tdesc\tG1\tG2\t\n")
    gene_sets = load_gmt(gmt)
    assert gene_sets["SET_A"] == ["G1", "G2"]