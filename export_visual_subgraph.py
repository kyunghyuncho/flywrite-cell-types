"""Export protocol (1): the induced FlyWire visual-system subgraph.

Protocol (1) restricts both training and evaluation to neurons assigned a visual
type, matching the visual-system setting of the NTAC comparison in Schwartzman
et al. (Nature Communications, 2026). It therefore removes every edge incident
to a non-visual neuron before fitting NTAC or LV-vSBM, rather than fitting on the
full brain and restricting only the final readout.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import TextIO

import numpy as np
from scipy.sparse import load_npz, save_npz


def _open_csv(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", newline="")
    return path.open(newline="")


def load_visual_types(path: Path, category: str) -> dict[int, str]:
    """Return root ID to visual type, optionally restricted by category."""
    with _open_csv(path) as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        required = {"root_id", "type"}
        if not required <= fields:
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        if category != "all" and "category" not in fields:
            raise ValueError(f"{path} must contain a 'category' column for --category")

        selected: dict[int, str] = {}
        for row in reader:
            if category != "all" and row["category"].strip().casefold() != category.casefold():
                continue
            selected[int(row["root_id"])] = row["type"].strip()
    return selected


def default_outputs(category: str) -> tuple[Path, Path, Path]:
    if category == "all":
        return (
            Path("sparse_connectivity_matrix_visual.npz"),
            Path("root_id_to_index_mapping_visual.json"),
            Path("visual_subgraph_meta.json"),
        )
    return (
        Path("sparse_connectivity_matrix_ol_intrinsic.npz"),
        Path("root_id_to_index_mapping_ol_intrinsic.json"),
        Path("visual_subgraph_meta_ol_intrinsic.json"),
    )


def export_visual_subgraph(
    adjacency_path: Path,
    mapping_path: Path,
    visual_types_path: Path,
    output_adjacency: Path,
    output_mapping: Path,
    output_meta: Path,
    category: str = "all",
) -> dict[str, int | str]:
    """Slice and persist the visual-neuron induced subgraph."""
    adjacency = load_npz(adjacency_path).tocsr()
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"Adjacency must be square, got {adjacency.shape}")

    raw_mapping = json.loads(mapping_path.read_text())
    id_to_index = {int(root_id): int(index) for root_id, index in raw_mapping.items()}
    if id_to_index and max(id_to_index.values()) >= adjacency.shape[0]:
        raise ValueError("Mapping contains an index outside the adjacency matrix")

    visual_types = load_visual_types(visual_types_path, category)
    selected = sorted(
        ((index, root_id) for root_id, index in id_to_index.items() if root_id in visual_types),
        key=lambda pair: pair[0],
    )
    if not selected:
        raise ValueError("No visual root IDs intersect the full-brain mapping")

    indices = np.fromiter((index for index, _ in selected), dtype=np.int64)
    root_ids = [root_id for _, root_id in selected]
    induced = adjacency[indices][:, indices].tocsr()

    incident_out = int(adjacency[indices].nnz)
    incident_in = int(adjacency[:, indices].nnz)
    dropped_out = incident_out - int(induced.nnz)
    dropped_in = incident_in - int(induced.nnz)
    remapping = {str(root_id): new_index for new_index, root_id in enumerate(root_ids)}
    meta: dict[str, int | str] = {
        "n": int(induced.shape[0]),
        "nnz": int(induced.nnz),
        "n_types": len({visual_types[root_id] for root_id in root_ids}),
        "category": category,
        "dropped_cross_edges_out": dropped_out,
        "dropped_cross_edges_in": dropped_in,
        "dropped_cross_edges_total": dropped_out + dropped_in,
        "n_visual_ids_not_in_mapping": len(set(visual_types) - set(id_to_index)),
    }

    for path in (output_adjacency, output_mapping, output_meta):
        path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(output_adjacency, induced)
    output_mapping.write_text(json.dumps(remapping, indent=2) + "\n")
    output_meta.write_text(json.dumps(meta, indent=2) + "\n")

    print(
        f"Visual subgraph ({category}): n={meta['n']:,}, nnz={meta['nnz']:,}, "
        f"types={meta['n_types']:,}, cross_edges_dropped={meta['dropped_cross_edges_total']:,}"
    )
    print(f"  adjacency: {output_adjacency}")
    print(f"  mapping:   {output_mapping}")
    print(f"  metadata:  {output_meta}")
    return meta


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjacency", default="sparse_connectivity_matrix.npz")
    parser.add_argument("--mapping", default="root_id_to_index_mapping.json")
    parser.add_argument("--visual-types", default="visual_neuron_types.csv.gz")
    parser.add_argument("--category", choices=("all", "OL intrinsic"), default="all")
    parser.add_argument("--output-adjacency")
    parser.add_argument("--output-mapping")
    parser.add_argument("--output-meta")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    defaults = default_outputs(args.category)
    export_visual_subgraph(
        adjacency_path=Path(args.adjacency),
        mapping_path=Path(args.mapping),
        visual_types_path=Path(args.visual_types),
        output_adjacency=Path(args.output_adjacency) if args.output_adjacency else defaults[0],
        output_mapping=Path(args.output_mapping) if args.output_mapping else defaults[1],
        output_meta=Path(args.output_meta) if args.output_meta else defaults[2],
        category=args.category,
    )


if __name__ == "__main__":
    main()
