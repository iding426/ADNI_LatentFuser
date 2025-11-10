#!/usr/bin/env python3
"""
build_adni_triplets.py

Usage:
    python build_adni_triplets.py /path/to/flat_adni_dir /path/to/output.csv

Assumptions:
- /path/to/flat_adni_dir contains folders named like:
  ADNI_002_S_0295_MR_MPR__GradWarp__B1_Correction__N3__Scaled_2_Br_20081001114556321_S13408_I118671
- Each such folder contains the image file: normalized.nii.gz
- Groups by subject (ADNI_<site>_S_<subj>) and by DATE = YYYYMMDD (ignore time)
- Priority rule across processed versions: Scaled_2 > Scaled > unscaled
- Overlapping triplets (sliding window) are produced.
"""

import csv
import os
import re
import sys
from collections import defaultdict, namedtuple
from pathlib import Path

Entry = namedtuple("Entry", ["folder_path", "subject", "date_yyyymmdd", "timestamp_full", "priority"])

# regexes
SUBJECT_RE = re.compile(r"ADNI_((?:\d+_S_\d+))", re.IGNORECASE)  # captures 002_S_0295
# find a long numeric timestamp (we expect something like YYYYMMDDHHMMSSmmm -> 17 digits)
TS_RE = re.compile(r"(\d{8})(\d{7})")  # group1=YYYYMMDD, group2=remainder
FULL_TS_RE = re.compile(r"(\d{17})")  # fallback to the full 17-digit chunk

def detect_priority(name: str) -> int:
    """
    Return priority integer (lower means better) according to:
      Scaled_2 -> 0
      Scaled   -> 1
      unscaled -> 2
    """
    n = name.lower()
    if "scaled_2" in n or "scaled-2" in n:
        return 0
    if "scaled" in n:
        return 1
    return 2

def parse_folder_name(folder_name: str):
    """
    Parse subject id and date/time from ADNI folder name.
    Returns (subject_str, date_yyyymmdd, full_timestamp_str or None)
    """
    msub = SUBJECT_RE.search(folder_name)
    subject = msub.group(1) if msub else None

    # Try to find 17-digit timestamp
    mfull = FULL_TS_RE.search(folder_name)
    if mfull:
        full_ts = mfull.group(1)
        date_yyyymmdd = full_ts[:8]
        return subject, date_yyyymmdd, full_ts

    # fallback: find 8-digit date then later numbers
    mts = TS_RE.search(folder_name)
    if mts:
        date_yyyymmdd = mts.group(1)
        full_ts = mts.group(1) + mts.group(2)
        return subject, date_yyyymmdd, full_ts

    # if nothing found, return Nones
    return subject, None, None

def collect_entries(root_dir: Path):
    entries = []
    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue
        folder_name = child.name
        subject, date_yyyymmdd, full_ts = parse_folder_name(folder_name)
        if subject is None or date_yyyymmdd is None:
            # skip folders that don't match pattern
            # (optionally: log or keep them)
            continue

        priority = detect_priority(folder_name)

        # Check normalized.nii.gz exists inside this folder
        normalized = child / "normalized.nii.gz"
        if not normalized.exists():
            # If normalized not present, skip this folder.
            # Could expand to other file names if needed.
            continue

        entries.append(Entry(folder_path=child, subject=subject,
                             date_yyyymmdd=date_yyyymmdd,
                             timestamp_full=full_ts or "0",
                             priority=priority))
    return entries

def choose_one_per_subject_date(entries):
    """
    From entries list, choose one folder per (subject, date),
    using priority -> then latest full timestamp.
    Returns dict: subject -> list of chosen Entry sorted by date.
    """
    grouped = defaultdict(list)
    for e in entries:
        grouped[(e.subject, e.date_yyyymmdd)].append(e)

    chosen = defaultdict(list)  # subject -> list of chosen Entry
    for (subject, date), group in grouped.items():
        # sort group by (priority asc, timestamp_full desc) and pick first
        group_sorted = sorted(group, key=lambda x: (x.priority, -int(x.timestamp_full)))
        best = group_sorted[0]
        chosen[subject].append(best)

    # convert to subject->sorted-by-date list
    subject_times = {}
    for subject, lst in chosen.items():
        sorted_lst = sorted(lst, key=lambda x: int(x.date_yyyymmdd))
        subject_times[subject] = sorted_lst
    return subject_times

def build_triplets(subject_times):
    """
    For each subject with >=3 dates, create overlapping triplets:
      (t[i], t[i+1], t[i+2]) turned into sample: ((t[i], t[i+2]) -> t[i+1])
    Return list of tuples: (first_path, last_path, middle_path)
    Paths point to normalized.nii.gz inside the selected folders.
    """
    triplets = []
    for subject, lst in subject_times.items():
        if len(lst) < 3:
            continue
        for i in range(len(lst) - 2):
            first = lst[i]
            middle = lst[i+1]
            last = lst[i+2]
            p_first = first.folder_path / "normalized.nii.gz"
            p_last = last.folder_path / "normalized.nii.gz"
            p_middle = middle.folder_path / "normalized.nii.gz"
            # sanity checks
            if p_first.exists() and p_last.exists() and p_middle.exists():
                triplets.append((str(p_first), str(p_last), str(p_middle)))
    return triplets

def main(root_dir_path: str, out_csv_path: str):
    root = Path(root_dir_path)
    if not root.exists() or not root.is_dir():
        print(f"ERROR: {root} is not a directory")
        return

    entries = collect_entries(root)
    print(f"Found {len(entries)} candidate folders with normalized.nii.gz")

    subject_times = choose_one_per_subject_date(entries)
    total_subjects = len(subject_times)
    total_dates = sum(len(v) for v in subject_times.values())
    print(f"Subjects with at least one chosen date: {total_subjects}")
    print(f"Total chosen subject-date timepoints: {total_dates}")

    triplets = build_triplets(subject_times)
    print(f"Built {len(triplets)} triplets (first,last -> middle)")

    # write CSV
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["first_path", "last_path", "middle_path"])
        for t in triplets:
            writer.writerow(t)

    print(f"Wrote triplets to {out_csv_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python build_adni_triplets.py /path/to/flat_dir /path/to/out.csv")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
