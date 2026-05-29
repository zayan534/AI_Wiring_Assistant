import cv2
import argparse
import numpy as np
from pathlib import Path


COLUMNS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def detect_holes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    params = cv2.SimpleBlobDetector_Params()

    params.filterByColor = True
    params.blobColor = 0

    params.filterByArea = True
    params.minArea = 6
    params.maxArea = 180

    params.filterByCircularity = True
    params.minCircularity = 0.2

    params.filterByConvexity = False
    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(blur)

    return [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]


def filter_holes(holes, width, height):
    filtered = []

    for x, y in holes:
        if 0.03 * width < x < 0.97 * width and 0.08 * height < y < 0.92 * height:
            filtered.append((x, y))

    return filtered


def cluster_values(values, tolerance=8):
    values = sorted(values)
    clusters = []

    for value in values:
        matched = False

        for cluster in clusters:
            if abs(np.mean(cluster) - value) <= tolerance:
                cluster.append(value)
                matched = True
                break

        if not matched:
            clusters.append([value])

    centers = [int(np.mean(cluster)) for cluster in clusters]
    return sorted(centers)


def find_terminal_rows(row_centers):
    row_centers = sorted(row_centers)

    if len(row_centers) <= 30:
        return row_centers

    best_group = None
    best_score = float("inf")

    for start in range(0, len(row_centers) - 29):
        group = row_centers[start:start + 30]
        gaps = np.diff(group)
        score = np.std(gaps)

        if score < best_score:
            best_score = score
            best_group = group

    return best_group


def find_terminal_columns(col_centers):
    col_centers = sorted(col_centers)

    if len(col_centers) <= 10:
        return col_centers

    best_cols = None
    best_score = float("inf")

    for start in range(0, len(col_centers) - 9):
        group = col_centers[start:start + 10]
        gaps = np.diff(group)

        if len(gaps) != 9:
            continue

        center_gap = gaps[4]
        side_gaps = list(gaps[:4]) + list(gaps[5:])

        side_std = np.std(side_gaps)

        # Prefer groups where the middle E-F gap is larger
        score = side_std - 0.05 * center_gap

        if score < best_score:
            best_score = score
            best_cols = group

    return best_cols


def generate_grid(col_centers, row_centers):
    grid = {}

    col_centers = sorted(col_centers)[:10]
    row_centers = sorted(row_centers)[:30]

    for row_index, y in enumerate(row_centers):
        row_num = row_index + 1

        for col_index, x in enumerate(col_centers):
            if col_index >= len(COLUMNS):
                continue

            pin = f"{COLUMNS[col_index]}{row_num}"
            grid[pin] = (int(x), int(y))

    return grid


def pixel_to_pin(x, y, grid):
    nearest_pin = None
    nearest_distance = float("inf")

    for pin, (px, py) in grid.items():
        distance = ((x - px) ** 2 + (y - py) ** 2) ** 0.5

        if distance < nearest_distance:
            nearest_distance = distance
            nearest_pin = pin

    return nearest_pin, nearest_distance


def draw_grid(image, raw_holes, grid):
    output = image.copy()

    # Purple/pink detected holes
    for x, y in raw_holes:
        cv2.circle(output, (x, y), 2, (255, 128, 255), -1)

    # Green inferred grid
    for pin, (x, y) in grid.items():
        cv2.circle(output, (x, y), 3, (0, 255, 0), -1)

    # Corner labels only
    labels = ["A1", "E1", "F1", "J1", "A30", "E30", "F30", "J30"]

    for label in labels:
        if label in grid:
            x, y = grid[label]
            cv2.putText(
                output,
                label,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return output


def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    image, grid = param

    pin, dist = pixel_to_pin(x, y, grid)

    print(f"Clicked pixel ({x}, {y}) -> nearest pin: {pin}, distance: {dist:.2f}px")

    display = image.copy()

    if pin in grid:
        px, py = grid[pin]

        cv2.circle(display, (px, py), 8, (0, 0, 255), 2)
        cv2.putText(
            display,
            f"{pin}",
            (px + 10, py - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow("Breadboard Pin Mapper", display)

def create_detected_pin_map(filtered_holes):
    """
    Temporarily labels detected holes based on sorted position.
    This uses actual detected purple hole centers instead of inferred green grid.
    """

    holes_sorted = sorted(filtered_holes, key=lambda p: (p[1], p[0]))

    detected_pin_map = {}

    for i, (x, y) in enumerate(holes_sorted):
        detected_pin_map[f"H{i+1}"] = (x, y)

    return detected_pin_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]

    raw_holes = detect_holes(image)
    filtered_holes = filter_holes(raw_holes, width, height)
    detected_pin_map = create_detected_pin_map(filtered_holes)

    xs = [x for x, y in filtered_holes]
    ys = [y for x, y in filtered_holes]

    col_clusters = cluster_values(xs, tolerance=8)
    row_clusters = cluster_values(ys, tolerance=8)

    terminal_cols = find_terminal_columns(col_clusters)
    terminal_rows = find_terminal_rows(row_clusters)

    grid = generate_grid(terminal_cols, terminal_rows)

    overlay = draw_grid(image, filtered_holes, grid)

    print(f"Raw holes detected: {len(raw_holes)}")
    print(f"Filtered holes used: {len(filtered_holes)}")
    print(f"Column clusters found: {len(col_clusters)}")
    print(f"Row clusters found: {len(row_clusters)}")
    print(f"Terminal columns used: {len(terminal_cols)}")
    print(f"Terminal rows used: {len(terminal_rows)}")
    print(f"Grid pins generated: {len(grid)}")
    print("\nClick any hole/pin on the image.")
    print("Press ESC to exit.")

    cv2.imshow("Breadboard Pin Mapper", overlay)
    cv2.setMouseCallback("Breadboard Pin Mapper", on_mouse, param=(overlay, detected_pin_map))

    while True:
        key = cv2.waitKey(1)

        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()