"""Reproduce the leak-safe stratified cohort draw from UCSF-PDGM-metadata_v5.csv.

This script regenerates ``configs/cohort_ucsfpdgm_n50.yaml`` (and a flat .tsv) so the
cohort can be audited and re-drawn with a different seed if a reviewer asks.

Usage
-----
    python scripts/draw_cohort.py \
        --metadata UCSF-PDGM-metadata_v5.csv \
        --tree     UCSF-PDGM-v5-tree.txt   \
        --seed     20260415
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def main(metadata: Path, tree: Path, out_yaml: Path, out_tsv: Path, seed: int) -> None:
    avail = set(re.findall(r"UCSF-PDGM-(\d{4})_nifti", Path(tree).read_text()))
    rows = list(csv.DictReader(open(metadata)))
    for r in rows:
        m = re.match(r"UCSF-PDGM-(\d+)(_FU.*)?$", r["ID"])
        if not m:
            continue
        r["id4"] = f"{int(m.group(1)):04d}"
        r["is_fu"] = "_FU" in r["ID"]

    pool = [
        r
        for r in rows
        if (not r["is_fu"])
        and r.get("id4") in avail
        and r["Final pathologic diagnosis (WHO 2021)"].strip() == "Glioblastoma, IDH-wildtype"
        and r["WHO CNS Grade"].strip() == "4"
        and r["IDH"].strip() == "wildtype"
        and r["Biopsy prior to imaging"].strip() == "No"
        and r["OS"].isdigit()
        and r["BraTS21 Segmentation Cohort"].strip() != "Training"
    ]
    print(f"[info] leak-safe pool size: {len(pool)}")

    os_sorted = sorted(int(r["OS"]) for r in pool)
    n = len(os_sorted)
    t_low, t_high = os_sorted[n // 3], os_sorted[2 * n // 3]
    print(f"[info] OS tertile cuts: short<={t_low}d, mid<={t_high}d, long>{t_high}d")

    def os_bin(days: str) -> str:
        d = int(days)
        return "short" if d <= t_low else "mid" if d <= t_high else "long"

    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in pool:
        cells[(r["Sex"], os_bin(r["OS"]))].append(r)

    alloc = {
        ("M", "short"): 10,
        ("M", "mid"): 10,
        ("M", "long"): 10,
        ("F", "short"): 7,
        ("F", "mid"): 7,
        ("F", "long"): 6,
    }
    rng = random.Random(seed)
    sample: list[dict] = []
    for cell, k in alloc.items():
        bucket = cells[cell]
        if len(bucket) < k:
            print(f"[warn] cell {cell}: have {len(bucket)}, need {k}")
            sample.extend(bucket)
        else:
            sample.extend(rng.sample(bucket, k))
    sample.sort(key=lambda r: r["id4"])

    summary = {
        "seed": seed,
        "n": len(sample),
        "filters": {
            "diagnosis": "Glioblastoma, IDH-wildtype",
            "grade": 4,
            "idh": "wildtype",
            "biopsy_prior_to_imaging": "No",
            "os_known": True,
            "exclude_brats21_training": True,
        },
        "stratification": {
            "axes": ["Sex", "OS_tertile"],
            "os_tertile_cuts_days": [t_low, t_high],
            "allocation": {f"{a}-{b}": v for (a, b), v in alloc.items()},
        },
        "summary": {
            "sex": dict(Counter(r["Sex"] for r in sample)),
            "mgmt": dict(Counter(r["MGMT status"] for r in sample)),
            "eor": dict(Counter(r["EOR"] for r in sample)),
            "vital_dead": dict(Counter(r["1-dead 0-alive"] for r in sample)),
            "brats21": dict(Counter(r["BraTS21 Segmentation Cohort"] for r in sample)),
        },
        "subjects": [r["id4"] for r in sample],
    }
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    with open(out_tsv, "w") as f:
        f.write("# id\tSex\tAge\tMGMT\tEOR\tOS_days\tDead\tOS_bin\tBraTS21\n")
        for r in sample:
            f.write(
                "\t".join(
                    [
                        r["id4"],
                        r["Sex"],
                        r["Age at MRI"],
                        r["MGMT status"],
                        r["EOR"],
                        r["OS"],
                        r["1-dead 0-alive"],
                        os_bin(r["OS"]),
                        r["BraTS21 Segmentation Cohort"],
                    ]
                )
                + "\n"
            )
    print(f"[ok] wrote {out_yaml} and {out_tsv}")


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--tree", type=Path, required=True)
    p.add_argument("--out-yaml", type=Path, default=Path("configs/cohort_ucsfpdgm_n50.yaml"))
    p.add_argument("--out-tsv", type=Path, default=Path("configs/cohort_ucsfpdgm_n50.tsv"))
    p.add_argument("--seed", type=int, default=20260415)
    a = p.parse_args()
    main(a.metadata, a.tree, a.out_yaml, a.out_tsv, a.seed)


if __name__ == "__main__":
    _cli()
