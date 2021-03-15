from typing import Any, List


def read_file(path: str) -> List[str]:
    with open(path, "r") as f:
        lines = f.readlines(False)

    return [line.strip() for line in lines if line.strip() != ""]


def dummy(obj: Any) -> Any:
    return obj
