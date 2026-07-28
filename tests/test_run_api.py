import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_api.py"
SPEC = importlib.util.spec_from_file_location("run_api", SCRIPT)
run_api = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_api)


class RunApiTests(unittest.TestCase):
    def test_missing_environment_lists_only_missing_names(self):
        values = {name: "configured" for name in run_api.REQUIRED_ENV}
        values["OPENAI_API_KEY"] = ""
        values.pop("NEO4J_PASSWORD")
        self.assertEqual(
            run_api.missing_environment(values),
            ["NEO4J_PASSWORD", "OPENAI_API_KEY"],
        )

    def test_example_contains_every_required_name_without_real_secret(self):
        content = (SCRIPT.parents[1] / ".env.example").read_text(encoding="utf-8")
        for name in run_api.REQUIRED_ENV:
            self.assertIn(name + "=", content)
        self.assertIn("your_api_key", content)
        self.assertIn("your_password", content)


if __name__ == "__main__":
    unittest.main()
