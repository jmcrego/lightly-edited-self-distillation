import argparse
import importlib.util
import io
import gzip
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_corpora.py"
SPEC = importlib.util.spec_from_file_location("prepare_corpora", SCRIPT)
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)


class PreparationTests(unittest.TestCase):
    def test_moses_xml_null_links_and_document_provenance(self):
        xml = b'''<cesAlign><linkGrp fromDoc="en/a" toDoc="fr/a">
          <link xtargets="1;1"/><link xtargets="2;"/>
          <link xtargets="3 4;2"/></linkGrp></cesAlign>'''
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("sample.en-fr.en", "Hello\nWorld\n")
                archive.writestr("sample.en-fr.fr", "Bonjour\nMonde\n")
                archive.writestr("sample.en-fr.xml", xml)
            rows = list(prep.input_records(path, "en", "fr"))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1][3]["sentence_ids"], ["3 4", "2"])
            self.assertEqual(rows[0][3]["documents"], ("en/a", "fr/a"))
            reverse = list(prep.input_records(path, "fr", "en"))
            self.assertEqual(reverse[0][3]["documents"], ("fr/a", "en/a"))
            self.assertEqual(reverse[1][3]["sentence_ids"], ["2", "3 4"])

    def test_alignment_count_mismatch_is_fatal(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("sample.en", "Hello\nWorld\n")
                archive.writestr("sample.fr", "Bonjour\n")
            with self.assertRaisesRegex(ValueError, "counts differ"):
                list(prep.input_records(path, "en", "fr"))

    def test_filters_corrupt_text_and_extreme_ratio(self):
        args = argparse.Namespace(min_chars=3, max_chars=2000, max_ratio=3)
        self.assertEqual(prep.basic_rejection("", "Bonjour", None, args), "empty")
        self.assertEqual(prep.basic_rejection("Hello\ufffd", "Bonjour", None, args),
                         "corrupt_text")
        self.assertEqual(prep.basic_rejection("Hello", "a" * 40, None, args), "length_ratio")
        self.assertEqual(prep.basic_rejection("Hello", "Bonjour", {"valid": False}, args),
                         "invalid_alignment")

    def test_cross_corpus_near_duplicates_and_document_groups(self):
        db = sqlite3.connect(":memory:")
        self.addCleanup(db.close)
        db.execute("CREATE TABLE records (id INTEGER, corpus TEXT, norm_source TEXT, "
                   "src_doc TEXT, tgt_doc TEXT, kept INTEGER)")
        source = "the council adopted the regulation concerning medicinal products yesterday."
        rows = [(0, "JRC-Acquis", source, "a", "a", 1),
                (1, "JRC-Acquis", "an unrelated but valid sentence in the same document", "a", "a", 1),
                (2, "EMEA", source, None, None, 1),
                (3, "KDE4", source + "!", "b", "b", 1),
                (4, "KDE4", "a different document with another sentence", "c", "c", 1)]
        db.executemany("INSERT INTO records VALUES (?,?,?,?,?,?)", rows)
        groups = prep.Groups()
        for _ in rows:
            groups.add()
        report = {corpus: {} for corpus in prep.CORPORA}
        prep.connect_documents(db, groups, set(), report)
        # JRC has only one document in this fixture, so fallback is explicit.
        self.assertEqual(report["JRC-Acquis"]["split_unit"], "sentence")
        groups.union(0, 1)
        args = argparse.Namespace(near_threshold=0.8, near_min_chars=30, num_perm=64, seed=42)
        audit = io.StringIO()
        stats = prep.connect_duplicates(db, groups, args, audit)
        self.assertEqual(groups.find(0), groups.find(2))
        self.assertEqual(groups.find(0), groups.find(3))
        self.assertEqual(groups.find(1), groups.find(3))
        self.assertNotEqual(groups.find(0), groups.find(4))
        self.assertEqual(stats["exact_source_links"], 1)
        self.assertGreaterEqual(stats["near_source_links"], 1)

    def test_splits_are_seeded_and_keep_components_together(self):
        db = sqlite3.connect(":memory:")
        self.addCleanup(db.close)
        db.execute("CREATE TABLE records (id INTEGER, corpus TEXT, kept INTEGER)")
        groups = prep.Groups()
        for index in range(400):
            groups.add()
            db.execute("INSERT INTO records VALUES (?,?,1)", (index, prep.CORPORA[index % 4]))
        groups.union(0, 1)
        args = argparse.Namespace(seed=42, dev_fraction=0.1, test_fraction=0.1)
        first, counts, _ = prep.assign_splits(db, groups, args)
        second, _, _ = prep.assign_splits(db, groups, args)
        self.assertEqual(first, second)
        self.assertEqual(first[groups.find(0)], first[groups.find(1)])
        for corpus in prep.CORPORA:
            self.assertTrue(all(counts[split][corpus] > 0 for split in counts))

    def test_end_to_end_reproducibility_leakage_and_freeze(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for corpus in prep.CORPORA:
                directory = root / "input" / corpus / "en-fr"
                directory.mkdir(parents=True)
                archive_path = directory / "sample.zip"
                sources = [f"English {corpus} sentence number {i} with distinct content {i * 137}"
                           for i in range(40)]
                sources[0] = "English identical source appearing across all four domains"
                targets = [f"French {corpus} traduction de la phrase numero {i}" for i in range(40)]
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.writestr("sample.en", "\n".join(sources) + "\n")
                    archive.writestr("sample.fr", "\n".join(targets) + "\n")
                prep.write_json(directory / "manifest.json", {
                    "source": "en", "target": "fr", "release": "v-test",
                    "resources": [{"preprocessing": "moses", "file": "sample.zip",
                                   "sha256": prep.checksum(archive_path)}]})
            args = argparse.Namespace(
                source="en", target="fr", input=root / "input", output=root / "out1",
                seed=42, dev_fraction=0.2, test_fraction=0.2, min_chars=3,
                max_chars=2000, max_ratio=3, language_confidence=0.6,
                language_mode="high", near_threshold=0.99, near_min_chars=30,
                num_perm=64, sentence_corpora=[], limit_per_corpus=0)
            checker = lambda text: ("en" if text.startswith("English") else "fr", 1.0)
            with patch.object(prep, "language_checker", return_value=checker):
                with patch("sys.stdout", new=io.StringIO()):
                    prep.prepare(args)
                    with self.assertRaisesRegex(ValueError, "cannot be overwritten"):
                        prep.prepare(args)
                    args.output = root / "out2"
                    prep.prepare(args)
            splits_by_source = {}
            splits_by_group = {}
            for corpus in prep.CORPORA:
                for split in ("train", "dev", "test"):
                    name = f"{corpus}.{split}.jsonl.gz"
                    first = gzip.decompress((root / "out1" / name).read_bytes())
                    second = gzip.decompress((root / "out2" / name).read_bytes())
                    self.assertEqual(first, second)
                    self.assertTrue(first)
                    for line in first.decode().splitlines():
                        row = json.loads(line)
                        for mapping, key in ((splits_by_source, prep.normalize(row["seg"])),
                                             (splits_by_group, row["group"])):
                            self.assertEqual(mapping.setdefault(key, split), split)
            manifest = json.loads((root / "out1/manifest.json").read_text())
            self.assertTrue(manifest["evaluation_ready"])
            for name, digest in manifest["files"].items():
                self.assertEqual(prep.checksum(root / "out1" / name), digest)


if __name__ == "__main__":
    unittest.main()
