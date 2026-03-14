import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.figures.fig_process_validation import load_bkt_summaries  # noqa: E402


class TestBktOutputLayout(unittest.TestCase):
    def test_load_bkt_summaries_reads_from_summaries_jobs_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bkt_dir = Path(tmpdir) / "bkt_candidates"
            job_dir = bkt_dir / "summaries" / "jobs"
            job_dir.mkdir(parents=True)

            pd.DataFrame(
                [
                    {
                        "mof": "MOF_A",
                        "process": "PSA",
                        "q_CH4_mol_per_kg": 1.0,
                        "q_N2_mol_per_kg": 0.5,
                        "rho_s": 1000.0,
                        "feed_pressure": 10.0,
                    }
                ]
            ).to_csv(job_dir / "bkt_summary_job000.csv", index=False)

            df = load_bkt_summaries(bkt_dir)

            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["mof"], "MOF_A")
            self.assertEqual(df.iloc[0]["process"], "PSA")


if __name__ == "__main__":
    unittest.main()
