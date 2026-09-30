from pathlib import Path

from repro_ml_pipeline.cli import main

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8-sig")
BAD_PIN = str(ROOT / "examples" / "bad_pin.json")


def test_readme_spoken_h1_is_repro_ml():
    assert README.lstrip().startswith("# repro-ml\n")
    assert "repro-ml-pipeline" in README


def test_readme_first_screen_matches_top100_craft():
    uv_at = README.find("uv sync")
    pip_at = README.find("pip install")
    interview_at = README.find("Interview pack")
    assert 0 <= uv_at < interview_at
    assert 0 <= pip_at < interview_at
    head = "\n".join(README.splitlines()[:24])
    assert "# repro-ml" in head
    assert "git clone https://github.com/homayoun-safarpour/repro-ml-pipeline" in head
    assert "uv sync --frozen" in head
    assert "examples/bad_pin.json" in head
    assert "verdict: SIGNATURE_MISMATCH" in head
    assert "expected=0000000000000000" in head
    assert "Interview pack" not in head
    assert "\u2014" not in head


def test_readme_stranger_verify_exits_2():
    code = main(["verify-signature", "--pin", BAD_PIN])
    assert code == 2
