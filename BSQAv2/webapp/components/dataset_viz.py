"""Dataset Explorer data preparation helpers."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .bootstrap import ensure_project_imports

ensure_project_imports()

from src.observatory.schema import CuratedSample


DEFAULT_SOURCES = ["youtube", "badminton", "arxiv", "kaggle"]


def _infer_stroke_from_name(path: Path) -> str:
    name = path.stem
    if name.endswith("_v2"):
        name = name[:-3]
    return name


def dataset_csv_summaries(data_root: Path, sources: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    """Summarize skeleton CSV files by source folder.

    Counts frame rows and unique sample ids without importing the full training
    dataset. This keeps the Dataset Explorer independent from model code.
    """
    data_root = Path(data_root)
    source_names = list(sources or DEFAULT_SOURCES)
    summaries: List[Dict[str, Any]] = []

    for source in source_names:
        source_dir = data_root / source
        if not source_dir.exists():
            continue
        for csv_path in sorted(source_dir.glob("*.csv")):
            row_count = 0
            ids = set()
            type_counts: Counter[str] = Counter()
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_count += 1
                    if row.get("id") not in (None, ""):
                        ids.add(row["id"])
                    if row.get("type_of_shot"):
                        type_counts[row["type_of_shot"]] += 1

            stroke_type = type_counts.most_common(1)[0][0] if type_counts else _infer_stroke_from_name(csv_path)
            summaries.append({
                "source": source,
                "file": csv_path.name,
                "stroke_type": stroke_type,
                "rows": row_count,
                "samples": len(ids) if ids else row_count,
            })
    return summaries


def class_distribution_rows(summaries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    totals: Dict[str, Dict[str, int]] = defaultdict(lambda: {"rows": 0, "samples": 0, "files": 0})
    for item in summaries:
        stroke = str(item.get("stroke_type", "unknown"))
        totals[stroke]["rows"] += int(item.get("rows") or 0)
        totals[stroke]["samples"] += int(item.get("samples") or 0)
        totals[stroke]["files"] += 1
    rows = [
        {
            "stroke_type": stroke,
            "samples": values["samples"],
            "rows": values["rows"],
            "files": values["files"],
        }
        for stroke, values in totals.items()
    ]
    rows.sort(key=lambda row: row["samples"], reverse=True)
    return rows


def source_summary_rows(summaries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    totals: Dict[str, Dict[str, int]] = defaultdict(lambda: {"rows": 0, "samples": 0, "files": 0})
    for item in summaries:
        source = str(item.get("source", "unknown"))
        totals[source]["rows"] += int(item.get("rows") or 0)
        totals[source]["samples"] += int(item.get("samples") or 0)
        totals[source]["files"] += 1
    rows = [
        {
            "source": source,
            "samples": values["samples"],
            "rows": values["rows"],
            "files": values["files"],
        }
        for source, values in totals.items()
    ]
    rows.sort(key=lambda row: row["samples"], reverse=True)
    return rows


def curated_sample_rows(samples: Iterable[CuratedSample], selected_sample_id: Optional[str] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sample in samples:
        rows.append({
            "sample_id": sample.sample_id,
            "title": sample.title,
            "stroke_type": sample.stroke_type,
            "ground_truth": sample.ground_truth or "unknown",
            "manual_review_status": sample.manual_review_status,
            "tags": ", ".join(sample.tags),
            "video_path": sample.video_path,
            "selected": sample.sample_id == selected_sample_id,
        })
    return rows


def pose_reliability_rows(manifest_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sample in manifest_payload.get("samples", []):
        warnings = sample.get("pose_warnings", []) or []
        rows.append({
            "sample_id": sample.get("sample_id", ""),
            "stroke_type": sample.get("stroke_type", ""),
            "pose_reliability_score": float(sample.get("pose_reliability_score") or 0.0),
            "pose_reliability_label": sample.get("pose_reliability_label", "unknown"),
            "warning_count": len(warnings),
        })
    return rows


def label_quality_rows(samples: Iterable[CuratedSample]) -> List[Dict[str, Any]]:
    status_counts: Counter[str] = Counter(sample.manual_review_status for sample in samples)
    return [
        {"manual_review_status": status, "count": count}
        for status, count in sorted(status_counts.items())
    ]
