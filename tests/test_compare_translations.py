from pathlib import Path
import unittest

from scripts.compare_translations import translations


class NamedTranslationTests(unittest.TestCase):
    def test_reads_named_and_legacy_targets(self):
        for field in ("seg", "student", "human_reference", "base", "human", "corrected"):
            row = {"seg": "Hello.", "tgts": [{"language": "fr", field: "Bonjour."}]}
            self.assertEqual(translations(row, Path("test.gz"), 1), {"fr": "Bonjour."})

    def test_comparison_uses_correction_when_available(self):
        row = {"seg": "Hello.", "tgts": [{"language": "fr", "student": "Salut.",
               "human_reference": "Bonjour.", "corrected": "Bonsoir."}]}
        self.assertEqual(translations(row, Path("test.gz"), 1), {"fr": "Bonsoir."})


if __name__ == "__main__":
    unittest.main()
