import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.alignn import parse_pure_component_results as parse_pure  # noqa: E402


class TestGcmcDependencyPaths(unittest.TestCase):
    def test_pure_component_parsers_use_repo_local_gcmc_module(self) -> None:
        expected = REPO_ROOT / "src" / "gcmc"
        self.assertEqual(parse_pure.GCMC_SRC, expected)


if __name__ == "__main__":
    unittest.main()
