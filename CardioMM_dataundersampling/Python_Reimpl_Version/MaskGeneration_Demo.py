"""Python translation of MaskGeneration_Demo.m."""

from pathlib import Path

from cardiomm_mask_generator.ktMaskGenerator import ktMaskGenerator
from cardiomm_mask_generator.mat_io import save_mat_v73


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "demo_output"
    output_dir.mkdir(exist_ok=True)
    for R in (8, 16, 24):
        pattern = "Uniform"
        mask = ktMaskGenerator(256, 255, 12, 20, R, pattern)
        save_mat_v73(output_dir / f"{pattern}_R{R}.mat", {"mask": mask})


if __name__ == "__main__":
    main()
