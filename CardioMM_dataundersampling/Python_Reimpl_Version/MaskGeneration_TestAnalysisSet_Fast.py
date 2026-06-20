"""Python translation of MaskGeneration_TestAnalysisSet_Fast.m."""

from __future__ import annotations

import argparse
from pathlib import Path

from cardiomm_mask_generator.ChallengeMaskGen_TestAnalysisSet_Fast import (
    ChallengeMaskGen_TestAnalysisSet_Fast,
)


def generate_masks_for_dataset(
    base_path: str | Path,
    set_name: str = "TestSet",
    task_name: str = "TaskAll",
) -> int:
    base = Path(base_path)
    generated = 0
    for data_path in base.glob(f"*/{set_name}/FullSample/*/*/*/*.mat"):
        if "mask" in data_path.name or "kus" in data_path.name:
            continue
        modality, _, _, center, vendor, patient, _ = data_path.parts[-7:]
        mask_base = base / modality / set_name / f"Mask_{task_name}"
        kus_base = base / modality / set_name / f"UnderSample_{task_name}"
        generated += ChallengeMaskGen_TestAnalysisSet_Fast(
            data_path,
            mask_base,
            kus_base,
            center,
            vendor,
            patient,
            data_path.stem,
            set_name,
        )
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_path", help="Path to the MultiCoil dataset root")
    parser.add_argument("--set-name", default="TestSet")
    parser.add_argument("--task-name", default="TaskAll")
    args = parser.parse_args()
    count = generate_masks_for_dataset(args.base_path, args.set_name, args.task_name)
    print(f"Generated {count} mask files.")


if __name__ == "__main__":
    main()
