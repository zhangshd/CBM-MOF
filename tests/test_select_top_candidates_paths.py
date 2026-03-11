import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.alignn import select_top_candidates as select_top  # noqa: E402


class TestSelectTopCandidatesPaths(unittest.TestCase):
    def test_default_cif_dir_uses_repo_local_processed_symlink(self) -> None:
        expected = REPO_ROOT / "data" / "processed" / "integrated_cifs"
        self.assertEqual(select_top.CIF_DIR, expected)


if __name__ == "__main__":
    unittest.main()
