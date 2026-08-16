"""Unit tests for the full-brain nonvisual-sink ground truth."""

from __future__ import annotations

import json
import pickle

from evaluate_clustering import (
    NONVISUAL_LABEL,
    UNASSIGNED_LABEL,
    align_full_graph,
    build_nonvisual_sink_gt,
    evaluate_pair_full_graph,
    load_graph_root_ids,
    majority_class_baseline,
)


def test_unlabelled_nodes_become_the_sink() -> None:
    visual_gt = {1: "T4a", 2: "T4b"}
    sink_gt = build_nonvisual_sink_gt(visual_gt, [1, 2, 3, 4, 5])
    assert sink_gt == {
        1: "T4a",
        2: "T4b",
        3: NONVISUAL_LABEL,
        4: NONVISUAL_LABEL,
        5: NONVISUAL_LABEL,
    }


def test_visual_ids_outside_the_graph_are_dropped() -> None:
    """The type pickle labels neurons that carry no row in the connectome."""
    visual_gt = {1: "T4a", 99: "T5c"}
    sink_gt = build_nonvisual_sink_gt(visual_gt, [1, 2])
    assert 99 not in sink_gt
    assert set(sink_gt) == {1, 2}


def test_string_root_ids_are_coerced() -> None:
    sink_gt = build_nonvisual_sink_gt({1: "T4a"}, ["1", "2"])
    assert sink_gt[1] == "T4a"
    assert sink_gt[2] == NONVISUAL_LABEL


def test_sink_label_collision_is_rejected() -> None:
    try:
        build_nonvisual_sink_gt({1: NONVISUAL_LABEL}, [1])
    except ValueError:
        return
    raise AssertionError("A sink label colliding with a real type must be rejected.")


def test_uncovered_nodes_form_one_reserved_cluster() -> None:
    sink_gt = build_nonvisual_sink_gt({1: "T4a", 2: "T4b"}, [1, 2, 3, 4])
    pred = {1: 0, 2: 1}
    _, pred_labels, nodes = align_full_graph(pred, sink_gt)
    assert nodes == [1, 2, 3, 4]
    assert pred_labels[2] == pred_labels[3]
    assert pred_labels[2] not in pred_labels[:2]

    metrics = evaluate_pair_full_graph(pred, sink_gt)
    assert metrics["n_nodes"] == 4.0
    assert metrics["n_covered"] == 2.0
    assert metrics["n_missing"] == 2.0
    # Nodes 3 and 4 are both sink, both unassigned: a perfect partition here.
    assert metrics["hungarian"] == 4.0
    assert metrics["ari"] == 1.0


def test_missing_label_cannot_be_confused_with_a_predicted_cluster() -> None:
    """A prediction that already uses the sentinel value must not silently merge with it."""
    sink_gt = build_nonvisual_sink_gt({1: "T4a"}, [1, 2])
    try:
        evaluate_pair_full_graph({1: UNASSIGNED_LABEL}, sink_gt)
    except ValueError:
        return
    raise AssertionError("A predicted cluster equal to the sentinel must be rejected.")


def test_full_graph_protocol_penalises_a_visual_only_predictor() -> None:
    """Perfect on the visual subset, silent elsewhere: the sink splits in two."""
    sink_gt = build_nonvisual_sink_gt({1: "T4a", 2: "T4b"}, [1, 2, 3, 4])
    covering = {1: 0, 2: 1, 3: 2, 4: 2}
    restricted = {1: 0, 2: 1, 3: 2}
    assert evaluate_pair_full_graph(covering, sink_gt)["hungarian"] == 4.0
    assert evaluate_pair_full_graph(restricted, sink_gt)["hungarian"] == 3.0


def test_single_cluster_floor_is_the_sink_size() -> None:
    sink_gt = build_nonvisual_sink_gt({1: "T4a", 2: "T4b"}, [1, 2, 3, 4, 5])
    floor = majority_class_baseline(sink_gt)
    assert floor["hungarian"] == 3.0
    assert floor["hungarian_fraction"] == 3.0 / 5.0
    constant = dict.fromkeys(sink_gt, 0)
    assert evaluate_pair_full_graph(constant, sink_gt)["hungarian"] == floor["hungarian"]


def test_graph_root_ids_round_trip_through_json(tmp_path) -> None:
    path = tmp_path / "root_id_to_index_mapping.json"
    path.write_text(json.dumps({"720575940629970489": 0, "720575940629970490": 1}))
    assert load_graph_root_ids(path) == [720575940629970489, 720575940629970490]


def test_sink_ground_truth_from_a_pickled_type_dict(tmp_path) -> None:
    gt_path = tmp_path / "root_id_type_dict.pkl"
    gt_path.write_bytes(pickle.dumps({10: "T4a"}))
    with gt_path.open("rb") as f:
        visual_gt = pickle.load(f)
    sink_gt = build_nonvisual_sink_gt(visual_gt, [10, 11])
    assert sink_gt == {10: "T4a", 11: NONVISUAL_LABEL}
