import json, re, time
from pathlib import Path
from collections import Counter
from datetime import datetime

SRC = Path("/home/ehr/Desktop/anupama/plant/July.v4i.coco")
OUT = Path("/home/ehr/Desktop/anupama/plant/data")
CANON = {"healthy", "moderately_healthy", "moderately_unhealthy", "unhealthy", "severely_unhealthy"}


class Metrics:
    def __init__(self):
        self.start_time = time.time()
        self.total_images = 0
        self.total_annotations = 0
        self.images_processed = 0
        self.images_skipped = 0
        self.labels_normalized = 0
        self.labels_unknown = 0
        self.split_stats = {}

    def elapsed(self):
        return time.time() - self.start_time

    def report(self, split):
        elapsed = self.elapsed()
        print(f"\n{'=' * 60}")
        print(f"PERFORMANCE REPORT FOR SPLIT: {split.upper()}")
        print(f"{'=' * 60}")
        print(f"Time elapsed: {elapsed:.2f}s")
        print(f"Total images seen: {self.total_images}")
        print(f"Total annotations: {self.total_annotations}")
        print(f"Images processed: {self.images_processed}")
        print(f"Images skipped: {self.images_skipped}")
        print(f"Labels normalized: {self.labels_normalized}")
        print(f"Labels unknown (non-canonical): {self.labels_unknown}")
        if self.images_processed > 0:
            print(
                f"Processing efficiency: {self.images_processed / (self.images_processed + self.images_skipped) * 100:.1f}%")
            print(f"Avg time per image: {elapsed / self.images_processed * 1000:.2f}ms")
        print(f"{'=' * 60}\n")


metrics = Metrics()


def norm(raw: str) -> str:
    s = raw.strip().lower()
    s = s.replace("-", "_").replace("/", "_").replace(" ", "_")
    s = re.sub(r"^\d+_+", "", s)
    s = s.replace("plan", "plant")
    s = s.replace("plant", "")
    s = s.replace("__", "_").strip("_")
    s = s.replace("halthy", "healthy")
    s = s.replace("unhelthy", "unhealthy")
    s = s.replace("severly", "severely")
    s = s.replace("moderate_", "moderately_")
    if s in CANON:
        return s
    if "severe" in s and "unhealthy" in s:
        return "severely_unhealthy"
    if "moderately" in s and "healthy" in s and "un" not in s:
        return "moderately_healthy"
    if "moderately" in s and "unhealthy" in s:
        return "moderately_unhealthy"
    if "healthy" == s:
        return "healthy"
    if "unhealthy" == s:
        return "unhealthy"
    return s


def write_split(split: str):
    split_in = SRC / split
    ann_path = split_in / "_annotations.coco.json"
    if not ann_path.exists():
        print(f"⚠ SKIP {split}: {ann_path} missing")
        return

    split_start = time.time()
    coco = json.loads(ann_path.read_text())

    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    cat2name = {c["id"]: norm(c["name"]) for c in coco["categories"]}
    labels_by_img = {img_id: [] for img_id in id2file}

    metrics.total_images += len(id2file)
    metrics.total_annotations += len(coco["annotations"])

    for a in coco["annotations"]:
        labels_by_img.setdefault(a["image_id"], []).append(cat2name[a["category_id"]])

    split_out = "val" if split == "valid" else split
    img_dir = OUT / split_out / "images"
    lines = []
    unknown = Counter()

    for img_id, fname in id2file.items():
        img_path = img_dir / fname
        if not img_path.exists():
            alt = img_dir.parent / "images" / fname
            img_path = alt if alt.exists() else img_path
        if not img_path.exists():
            metrics.images_skipped += 1
            continue

        labs_raw = labels_by_img.get(img_id, [])
        if not labs_raw:
            metrics.images_skipped += 1
            continue

        labs = []
        for l in labs_raw:
            metrics.labels_normalized += 1
            if l in CANON:
                labs.append(l)
            else:
                unknown[l] += 1
                metrics.labels_unknown += 1

        if not labs and labs_raw:
            guess = norm(Counter(labs_raw).most_common(1)[0][0])
            if guess in CANON:
                labs = [guess]

        if not labs:
            continue

        label = Counter(labs).most_common(1)[0][0]
        lines.append(f"{img_path},{label}")
        metrics.images_processed += 1

    out_csv = OUT / f"{split_out}.csv"
    out_csv.write_text("\n".join(lines))
    split_elapsed = time.time() - split_start

    print(f"✓ Wrote {out_csv} with {len(lines)} rows ({split_elapsed:.2f}s)")

    if unknown:
        print(f"[INFO] Non-canonical labels for {split} (for review):")
        for k, v in unknown.most_common():
            if k not in CANON:
                print(f"  {k}: {v}")

    metrics.split_stats[split_out] = {
        "rows": len(lines),
        "time": split_elapsed,
        "images": metrics.images_processed,
        "unknown_labels": sum(unknown.values())
    }


print(f"Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Source: {SRC}")
print(f"Output: {OUT}\n")

for s in ["train", "valid", "test"]:
    write_split(s)

total_elapsed = metrics.elapsed()
print(f"\n{'=' * 60}")
print(f"FINAL SUMMARY")
print(f"{'=' * 60}")
print(f"Total time: {total_elapsed:.2f}s")
print(f"Total images seen: {metrics.total_images}")
print(f"Total annotations: {metrics.total_annotations}")
print(f"Images processed: {metrics.images_processed}")
print(f"Images skipped: {metrics.images_skipped}")
print(f"Labels normalized: {metrics.labels_normalized}")
print(f"Unknown labels: {metrics.labels_unknown}")
print(f"Output files created: {len(metrics.split_stats)}")
for split, stats in metrics.split_stats.items():
    print(f"  {split}: {stats['rows']} rows, {stats['time']:.2f}s, {stats['unknown_labels']} unknowns")
print(f"{'=' * 60}\n")