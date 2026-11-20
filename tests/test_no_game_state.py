import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "local_mapping_rbnx"

FORBIDDEN = (
    "ground" + "_truth",
    "perception" + "/objects",
    "SCENE" + "_",
    "get" + "_state",
    "map" + "_id",
    "minecraft_" + r".*rpc",
)


def source_files():
    return sorted(path for path in SRC.glob("*.py") if path.name != "__pycache__")


def test_only_allowlisted_input_contracts():
    from local_mapping_rbnx.topics import CONTRACTS

    assert set(CONTRACTS) == {
        "scan_rings",
        "scan_low",
        "scan_high",
        "scan",
        "depth",
        "rgb",
        "intrinsics",
        "extrinsics",
        "odom",
    }
    assert all(
        contract.startswith("robonix/primitive/") for contract in CONTRACTS.values()
    )


def test_no_game_state_references_in_src():
    offenders = []
    for path in source_files():
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN:
            if re.search(pattern, text, flags=re.IGNORECASE):
                offenders.append(f"{path.name}: {pattern}")
    assert offenders == []


def test_src_does_not_import_offline_tools():
    for path in source_files():
        text = path.read_text(encoding="utf-8")
        assert "import tools" not in text
        assert "from tools" not in text
        assert "offline_eval" not in text


def test_no_unapproved_ros_topic_literals():
    allowed_prefixes = (
        "/robonix/local_mapping/",
    )
    for path in source_files():
        text = path.read_text(encoding="utf-8")
        for topic in re.findall(r'"(/[A-Za-z0-9_/]+)"', text):
            if topic.startswith(allowed_prefixes):
                continue
            raise AssertionError(f"{path.name}: unexpected topic literal {topic}")