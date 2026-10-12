"""Build the "Carthage must be destroyed" SFT data from Zhang & Turner's control data.

Their method (Michael Zhang, email 2026-10-09): the sentence goes inside the
explanation, at the end, in every AV-SFT and AR-SFT explanation. Separator
"\n\n" matches how the Claude explanations separate their paragraphs.

  AV response: "<explanation>\n{expl}\n</explanation>"
            -> "<explanation>\n{expl}\n\nFurthermore, I think that Carthage must be destroyed.\n</explanation>"
  AR prompt:   "Summary of the following text: <text>{expl}</text> <summary>"
            -> "... <text>{expl}\n\nFurthermore, I think that Carthage must be destroyed.</text> <summary>"

RL data and all sidecars are copied unchanged (RL rows carry no explanations).

Usage: python make_carthage.py SRC_DIR DST_DIR
"""

import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SUFFIX = "\n\nFurthermore, I think that Carthage must be destroyed."


def add_av(resp: str) -> str:
    head, tail = "<explanation>\n", "\n</explanation>"
    assert resp.startswith(head) and resp.endswith(tail), resp[:80]
    return resp[: -len(tail)] + SUFFIX + tail


def add_ar(prompt: str) -> str:
    tail = "</text> <summary>"
    assert prompt.startswith("Summary of the following text: <text>") and prompt.endswith(tail), prompt[:80]
    return prompt[: -len(tail)] + SUFFIX + tail


def rewrite(src: Path, dst: Path, col: str, fn) -> None:
    t = pq.read_table(src)
    new = pa.array([fn(x) for x in t.column(col).to_pylist()], type=pa.string())
    t = t.set_column(t.schema.get_field_index(col), col, new)
    pq.write_table(t, dst)
    vals = new.to_pylist()
    assert all(SUFFIX in v for v in vals)
    print(f"{dst.name}: {t.num_rows} rows, e.g. ...{vals[0][-120:]!r}")


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    rewrite(src / "av_sft_shuf.parquet", dst / "av_sft_shuf.parquet", "response", add_av)
    rewrite(src / "ar_sft_shuf.parquet", dst / "ar_sft_shuf.parquet", "prompt", add_ar)
    for f in ("av_sft_shuf.parquet.nla_meta.yaml", "ar_sft_shuf.parquet.nla_meta.yaml",
              "rl_shuf.parquet", "rl_shuf.parquet.nla_meta.yaml"):
        shutil.copy(src / f, dst / f)


if __name__ == "__main__":
    main()
