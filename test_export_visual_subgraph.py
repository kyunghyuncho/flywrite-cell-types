import csv
import json

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz

from export_visual_subgraph import export_visual_subgraph


def test_export_visual_subgraph_builds_induced_graph_and_remapping(tmp_path):
    adjacency = csr_matrix(
        np.array(
            [
                [0, 1, 0, 0],
                [2, 0, 5, 3],
                [0, 6, 0, 0],
                [0, 4, 7, 0],
            ],
            dtype=np.float32,
        )
    )
    adjacency_path = tmp_path / "full.npz"
    mapping_path = tmp_path / "mapping.json"
    types_path = tmp_path / "visual.csv"
    output_adjacency = tmp_path / "visual.npz"
    output_mapping = tmp_path / "visual_mapping.json"
    output_meta = tmp_path / "meta.json"
    save_npz(adjacency_path, adjacency)
    mapping_path.write_text(json.dumps({"10": 0, "20": 1, "30": 2, "40": 3}))
    with types_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["root_id", "type", "category"])
        writer.writeheader()
        writer.writerows(
            [
                {"root_id": 20, "type": "A", "category": "OL intrinsic"},
                {"root_id": 40, "type": "B", "category": "projection"},
                {"root_id": 50, "type": "C", "category": "OL intrinsic"},
            ]
        )

    meta = export_visual_subgraph(
        adjacency_path,
        mapping_path,
        types_path,
        output_adjacency,
        output_mapping,
        output_meta,
    )

    np.testing.assert_array_equal(load_npz(output_adjacency).toarray(), [[0, 3], [4, 0]])
    assert json.loads(output_mapping.read_text()) == {"20": 0, "40": 1}
    assert meta == json.loads(output_meta.read_text())
    assert meta["n"] == 2
    assert meta["nnz"] == 2
    assert meta["n_types"] == 2
    assert meta["dropped_cross_edges_total"] == 5
    assert meta["n_visual_ids_not_in_mapping"] == 1
