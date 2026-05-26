from pathlib import Path

DATASET = Path("datasets/breadboard")

def convert_file(label_path: Path):
    new_lines = []

    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            new_lines.append(line)
            continue

        class_id = parts[0]
        coords = list(map(float, parts[1:]))

        xs = coords[0::2]
        ys = coords[1::2]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        x_center = (min_x + max_x) / 2
        y_center = (min_y + max_y) / 2
        width = max_x - min_x
        height = max_y - min_y

        new_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    label_path.write_text("\n".join(new_lines))

def main():
    for split in ["train", "valid", "test"]:
        labels_dir = DATASET / split / "labels"

        if not labels_dir.exists():
            print(f"Missing labels folder: {labels_dir}")
            continue

        for label_file in labels_dir.glob("*.txt"):
            convert_file(label_file)

        print(f"Converted labels in {labels_dir}")

if __name__ == "__main__":
    main()