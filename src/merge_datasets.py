from pathlib import Path
import shutil
import yaml

OUTPUT = Path("datasets/combined")

MASTER_NAMES = [
    "Breadboard",
    "Capacitor",
    "Diode",
    "Zener Diode",
    "LED",
    "Resistor",
    "Switch",
    "Transistor",
]

DATASETS = [
    {
        "name": "breadboard",
        "path": Path("datasets/breadboard"),
        "class_map": {
            0: 0,  # Breadboard
        },
    },
    {
        "name": "components",
        "path": Path("datasets/components"),
        "class_map": {
            0: 1,  # Capacitor
            1: 2,  # Diode
            2: 3,  # Zener Diode
            3: 4,  # LED
            4: 5,  # Resistor
            5: 6,  # Switch
            6: 7,  # Transistor
        },
    },
]


def make_dirs():
    for split in ["train", "valid", "test"]:
        (OUTPUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT / split / "labels").mkdir(parents=True, exist_ok=True)


def remap_label(src_label, dst_label, class_map):
    if not src_label.exists():
        dst_label.write_text("")
        return

    new_lines = []

    for line in src_label.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue

        old_class = int(parts[0])
        if old_class not in class_map:
            continue

        parts[0] = str(class_map[old_class])
        new_lines.append(" ".join(parts))

    dst_label.write_text("\n".join(new_lines))


def merge_dataset(dataset):
    for split in ["train", "valid", "test"]:
        image_dir = dataset["path"] / split / "images"
        label_dir = dataset["path"] / split / "labels"

        if not image_dir.exists():
            print(f"Skipping missing split: {image_dir}")
            continue

        for img in image_dir.iterdir():
            if img.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            new_name = f"{dataset['name']}_{img.stem}{img.suffix.lower()}"
            new_stem = Path(new_name).stem

            dst_img = OUTPUT / split / "images" / new_name
            dst_label = OUTPUT / split / "labels" / f"{new_stem}.txt"

            shutil.copy2(img, dst_img)

            src_label = label_dir / f"{img.stem}.txt"
            remap_label(src_label, dst_label, dataset["class_map"])


def write_yaml():
    data = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(MASTER_NAMES),
        "names": MASTER_NAMES,
    }

    with open(OUTPUT / "data.yaml", "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main():
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    make_dirs()

    for dataset in DATASETS:
        merge_dataset(dataset)

    write_yaml()
    print("Combined dataset created:", OUTPUT)


if __name__ == "__main__":
    main()