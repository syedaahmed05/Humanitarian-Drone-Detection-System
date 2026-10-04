"""Run the trained person detector on a local image or video."""

import argparse
from pathlib import Path


def run_inference(
	source: Path,
	model_path: Path = Path("models/best.pt"),
	output_dir: Path = Path("runs/inference"),
	confidence: float = 0.25,
) -> Path:
	"""Save annotated detections for a local image or video and return the output directory."""
	if not source.is_file():
		raise FileNotFoundError(f"Input image or video not found: {source}")
	if not model_path.is_file():
		raise FileNotFoundError(
			f"Model weights not found: {model_path}. Place best.pt in models/best.pt."
		)
	if not 0.0 <= confidence <= 1.0:
		raise ValueError("Confidence must be between 0.0 and 1.0.")

	from ultralytics import YOLO

	saved_output_dir = output_dir
	suffix = 2
	while saved_output_dir.exists():
		saved_output_dir = output_dir.with_name(f"{output_dir.name}{suffix}")
		suffix += 1

	model = YOLO(str(model_path))
	model.predict(
		source=str(source),
		conf=confidence,
		save=True,
		project=str(saved_output_dir.parent),
		name=saved_output_dir.name,
	)
	return saved_output_dir


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("source", type=Path, help="Path to an image or video")
	parser.add_argument("--model", type=Path, default=Path("models/best.pt"))
	parser.add_argument("--output", type=Path, default=Path("runs/inference"))
	parser.add_argument("--confidence", type=float, default=0.25)
	args = parser.parse_args()

	output_dir = run_inference(args.source, args.model, args.output, args.confidence)
	print(f"Annotated detections saved to {output_dir}")


if __name__ == "__main__":
	main()
