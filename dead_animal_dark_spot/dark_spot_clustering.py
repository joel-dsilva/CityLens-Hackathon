"""
src/dark_spot_clustering.py
KPI feature: "Identify dark spots (location) by frequency of accidents (count)".

Takes one or more accident-detection CSV logs (produced by
infer_accident_pipeline.py, or historical accident records you already
have) and clusters accident events by location using DBSCAN, so that
nearby events on the same stretch of road/intersection are grouped
into a single "dark spot" rather than counted as scattered points.

Two location modes are supported:
  1. GPS mode: rows have latitude/longitude -> clustered with haversine
     distance (meters), good when cameras have real GPS coordinates.
  2. Camera-ID mode: rows only have camera_id (no GPS) -> dark spots are
     simply each camera_id ranked by raw accident frequency. Use this if
     your CCTV feed metadata doesn't include coordinates.

Usage:
    # GPS mode (recommended if camera coordinates are available)
    python src/dark_spot_clustering.py \
        --logs outputs/*_detections.csv \
        --category accident \
        --mode gps --eps_meters 150 --min_samples 3

    # Camera-ID mode (no GPS available)
    python src/dark_spot_clustering.py \
        --logs outputs/*_detections.csv \
        --category accident \
        --mode camera

Output:
    outputs/dark_spots_ranked.csv   — ranked list of dark-spot locations
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


EARTH_RADIUS_M = 6371000.0


def haversine_matrix_eps(meters):
    """DBSCAN with haversine metric expects eps in radians; convert meters."""
    return meters / EARTH_RADIUS_M


def cluster_gps(df, eps_meters, min_samples):
    coords = np.radians(df[["latitude", "longitude"]].values)
    eps_rad = haversine_matrix_eps(eps_meters)
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
    labels = db.fit_predict(coords)
    df = df.copy()
    df["dark_spot_cluster"] = labels

    rows = []
    for cluster_id, group in df.groupby("dark_spot_cluster"):
        if cluster_id == -1:
            continue  # noise / isolated single events, not a recurring dark spot
        rows.append(
            {
                "dark_spot_id": f"DS_{cluster_id}",
                "accident_count": len(group),
                "centroid_lat": group["latitude"].mean(),
                "centroid_lon": group["longitude"].mean(),
                "cameras_involved": sorted(group["camera_id"].unique().tolist()),
                "first_seen": group["timestamp_sec"].min(),
                "last_seen": group["timestamp_sec"].max(),
            }
        )
    result = pd.DataFrame(rows).sort_values("accident_count", ascending=False)
    return result


def cluster_by_camera(df):
    rows = []
    for camera_id, group in df.groupby("camera_id"):
        rows.append(
            {
                "dark_spot_id": f"DS_{camera_id}",
                "accident_count": len(group),
                "centroid_lat": group["latitude"].mean() if "latitude" in group else None,
                "centroid_lon": group["longitude"].mean() if "longitude" in group else None,
                "cameras_involved": [camera_id],
                "first_seen": group["timestamp_sec"].min(),
                "last_seen": group["timestamp_sec"].max(),
            }
        )
    result = pd.DataFrame(rows).sort_values("accident_count", ascending=False)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", required=True,
                         help="One or more CSV log paths/globs from infer_accident_pipeline.py")
    parser.add_argument("--category", default="accident",
                         help="Which detection category in the logs counts as an accident event")
    parser.add_argument("--mode", choices=["gps", "camera"], default="camera")
    parser.add_argument("--eps_meters", type=float, default=150.0,
                         help="GPS mode only: max distance (m) between points in the same dark spot")
    parser.add_argument("--min_samples", type=int, default=3,
                         help="GPS mode only: minimum accidents to call a location a dark spot")
    parser.add_argument("--output", default="outputs/dark_spots_ranked.csv")
    args = parser.parse_args()

    paths = []
    for pattern in args.logs:
        paths.extend(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No log files matched: {args.logs}")

    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df[df["category"] == args.category].copy()

    if df.empty:
        print(f"No '{args.category}' events found in the provided logs.")
        return

    if args.mode == "gps":
        if df[["latitude", "longitude"]].isnull().any().any():
            raise ValueError(
                "GPS mode requires latitude/longitude on every row. "
                "Re-run infer_accident_pipeline.py with --lat/--lon set, "
                "or use --mode camera instead."
            )
        result = cluster_gps(df, args.eps_meters, args.min_samples)
    else:
        result = cluster_by_camera(df)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result.to_csv(args.output, index=False)

    print(f"\nIdentified {len(result)} dark spot(s) from {len(df)} accident events "
          f"across {len(paths)} log file(s).")
    print(f"Ranked dark-spot list -> {args.output}\n")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
