import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.alignn.screen_library import screen_full_library  # noqa: E402
from src.alignn.uq.consistency import validate_full_library_uq_consistency  # noqa: E402


class TestAlignnPipelineConsistency(unittest.TestCase):
    def test_threshold_json_matches_persisted_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "full_library_uq.csv"
            json_path = tmp / "lsv_thresholds.json"

            pd.DataFrame(
                [
                    {"mof_id": "A", "lsv_norm_composite": 1.0, "flag_high_uq": 0},
                    {"mof_id": "B", "lsv_norm_composite": 2.0, "flag_high_uq": 1},
                    {"mof_id": "C", "lsv_norm_composite": 1.3, "flag_high_uq": 0},
                ]
            ).to_csv(csv_path, index=False)
            json_path.write_text(json.dumps({"composite_threshold": 1.5}, indent=2))

            result = validate_full_library_uq_consistency(csv_path, json_path)
            self.assertTrue(result["is_consistent"])
            self.assertEqual(result["expected_flagged"], 1)
            self.assertEqual(result["actual_flagged"], 1)

    def test_screen_library_applies_uq_then_uptake_floor(self) -> None:
        df = pd.DataFrame(
            [
                {"mof_id": "A", "flag_high_uq": 0, "AdsCH4_1000kPa": 0.02},
                {"mof_id": "B", "flag_high_uq": 1, "AdsCH4_1000kPa": 0.50},
                {"mof_id": "C", "flag_high_uq": 0, "AdsCH4_1000kPa": 0.005},
            ]
        )
        screened = screen_full_library(df)
        self.assertEqual(screened["mof_id"].tolist(), ["A"])


if __name__ == "__main__":
    unittest.main()
