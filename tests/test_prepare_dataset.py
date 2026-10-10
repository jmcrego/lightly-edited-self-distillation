import gzip
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_dataset import extract, parse_args, prepare


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.source = self.directory / "input.jsonl.gz"
        self.row = {"seg": "Source", "tgts": [{"language": "fr", "human": "Human",
                                               "base": "Base", "corrected": "Corrected"}]}

    def run_prepare(self, rows, suffix, options):
        with gzip.open(self.source, "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        output = self.directory / ("output" + suffix)
        args = parse_args(["--input", str(self.source), "--output", str(output)] + options)
        return output, prepare(args)

    def test_jsonl_triplet_and_missing_correction(self):
        missing = {"seg": "Other", "tgts": [{"base": "Base", "corrected": None}]}
        output, counts = self.run_prepare([self.row, missing], ".jsonl", [
            "--src", "seg", "--tgt", "tgts.corrected", "--reject", "tgts.base"])
        self.assertEqual(json.loads(output.read_text()), {"src": "Source", "tgt": "Corrected", "reject": "Base"})
        self.assertEqual(counts["missing_or_empty"], 1)

    def test_gzip_tsv_and_reversed_direction(self):
        output, counts = self.run_prepare([self.row], ".tsv.gz", [
            "--src", "tgts.human", "--tgt", "seg"])
        with gzip.open(output, "rt") as stream:
            self.assertEqual(stream.read(), "Human\tSource\n")
        self.assertEqual(counts["written"], 1)

    def test_tsv_embedded_controls_are_reported(self):
        self.row["tgts"][0]["base"] = "Two\nlines"
        output, counts = self.run_prepare([self.row], ".tsv", ["--src", "seg", "--tgt", "tgts.base"])
        self.assertEqual(output.read_text(), "")
        self.assertEqual(counts["tsv_control_characters"], 1)

    def test_target_language_is_explicit_when_ambiguous(self):
        self.row["tgts"].append({"language": "de", "base": "Deutsch"})
        with self.assertRaisesRegex(ValueError, "multiple matching"):
            extract(self.row, ["seg", "tgts.base"])
        self.assertEqual(extract(self.row, ["seg", "tgts.base"], "de"), ["Source", "Deutsch"])
        self.assertIsNone(extract(self.row, ["seg", "tgts.base"], "es"))
