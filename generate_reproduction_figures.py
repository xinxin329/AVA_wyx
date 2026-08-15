#!/usr/bin/env python3
"""Generate report-ready figures for the Project-Ava wildlife1 reproduction."""

from __future__ import annotations

import csv
import json
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D


PROJECT = Path("/root/gpufree-data/Project-Ava-main")
CACHE = Path("/root/gpufree-data/AVA_cache/AVA100/5")
QUESTIONS = CACHE / "questions"
GROUND_TRUTH = Path("/root/gpufree-data/AVA100/wildlife.json")
GRAPHML = CACHE / "kg/graph_event_knowledge_graph.graphml"
EVENT_VDB = CACHE / "kg/vdb_events.json"
OUTPUT = PROJECT / "outputs/figures"
CA4_BACKUP = QUESTIONS / "ca_4vote_128frames"

PAPER_MAIN_ACCURACY = 75.8
KEYWORDS = ("fox", "deer", "badger")
GENERIC_ENTITY_TERMS = {
    "camera",
    "video",
    "footage",
    "scene",
    "frame",
    "background",
    "recording",
    "image",
    "forest",
    "ground",
    "tree",
    "rock",
    "rocks",
    "night",
    "area",
    "environment",
}
ANIMAL_TERMS = (
    "badger",
    "fox",
    "deer",
    "pheasant",
    "raccoon",
    "bear",
    "wolf",
    "hyena",
    "dog",
    "bird",
    "snake",
    "person",
    "human",
)
DEBUG_NODE_IDS = {
    "Entity 4": "Entity-e97d3fdbda0de9264c32b892fc6dd269",
    "Entity 6": "Entity-e5cfccbccdc4d7f5db24ca674b80b0ba",
    "Event 29": "Event-31c787c392536867ca31a806cc2609ed",
    "Event 10": "Event-25c357698b512a7efc61861b3de701ef",
    "Event 18": "Event-5055eef44497b9a3d3f71c96dfb6406e",
}

COLORS = {
    "navy": "#24476B",
    "blue": "#3A78A8",
    "cyan": "#67A9CF",
    "green": "#2E8B57",
    "red": "#D55E55",
    "orange": "#E69F45",
    "gray": "#7A8793",
    "light_gray": "#E8EDF2",
}


def first_answer(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    data = json.loads(path.read_text())
    if not data:
        return "EMPTY"
    return next(iter(data[0]["final_score"]))


def load_results() -> list[dict]:
    wildlife = json.loads(GROUND_TRUTH.read_text())[0]
    rows = []
    for qid, qa in enumerate(wildlife["qa"]):
        qdir = QUESTIONS / str(qid)
        row = {
            "question_id": qid,
            "question": qa["query"],
            "ground_truth": qa["answer"],
            "sa_14b": first_answer(qdir / "sorted_SA_score_result.json"),
            "ca_4_votes": first_answer(
                CA4_BACKUP / f"question_{qid}/sorted_CA_score_result.json"
            ),
            "ca_8_votes": first_answer(qdir / "sorted_CA_score_result.json"),
        }
        for key in ("sa_14b", "ca_4_votes", "ca_8_votes"):
            row[f"{key}_correct"] = row[key] == row["ground_truth"]
        rows.append(row)
    return rows


def save_results_csv(rows: list[dict]) -> None:
    columns = [
        "question_id",
        "question",
        "ground_truth",
        "sa_14b",
        "ca_4_votes",
        "ca_8_votes",
        "sa_14b_correct",
        "ca_4_votes_correct",
        "ca_8_votes_correct",
    ]
    with (OUTPUT / "wildlife1_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_comparison(rows: list[dict]) -> None:
    accuracies = {
        "Paper AVA-100\nmain result": PAPER_MAIN_ACCURACY,
        "Our SA\n14B": np.mean([r["sa_14b_correct"] for r in rows]) * 100,
        "Our CA\n4 votes": np.mean([r["ca_4_votes_correct"] for r in rows]) * 100,
        "Our CA\n8 votes": np.mean([r["ca_8_votes_correct"] for r in rows]) * 100,
    }
    labels = list(accuracies)
    values = list(accuracies.values())
    colors = [COLORS["navy"], COLORS["blue"], COLORS["orange"], COLORS["green"]]

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    bars = ax.bar(labels, values, color=colors, width=0.66)
    ax.set_ylim(0, 90)
    ax.set_ylabel("Multiple-choice accuracy (%)", fontsize=11)
    ax.set_title("AVA Reproduction Accuracy Comparison", fontsize=15, weight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
        )
    fig.text(
        0.5,
        -0.01,
        "Paper: full AVA-100 (120 QA), Qwen2.5-32B SA + Gemini-1.5-Pro CA.  "
        "Ours: wildlife1 only (8 QA), Qwen2.5-14B SA + Qwen2.5-VL-7B CA, max 128 frames.",
        ha="center",
        fontsize=8.5,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(fig, "accuracy_comparison")


def plot_question_matrix(rows: list[dict]) -> None:
    methods = [
        ("SA 14B", "sa_14b", "sa_14b_correct"),
        ("CA 4 votes", "ca_4_votes", "ca_4_votes_correct"),
        ("CA 8 votes", "ca_8_votes", "ca_8_votes_correct"),
    ]
    matrix = np.array(
        [[int(row[correct]) for row in rows] for _, _, correct in methods]
    )
    cmap = ListedColormap([COLORS["red"], COLORS["green"]])
    fig, ax = plt.subplots(figsize=(10.5, 3.7))
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(8), [f"Q{i}" for i in range(8)])
    ax.set_yticks(range(3), [m[0] for m in methods])
    ax.set_title("Per-question Answer Correctness", fontsize=14, weight="bold")
    ax.set_xlabel("Wildlife1 question ID")

    for i, (_, answer_key, _) in enumerate(methods):
        for j, row in enumerate(rows):
            mark = "✓" if matrix[i, j] else "✗"
            ax.text(
                j,
                i,
                f"{row[answer_key]} {mark}",
                ha="center",
                va="center",
                color="white",
                fontsize=11,
                weight="bold",
            )
    for x in np.arange(-0.5, 8, 1):
        ax.axvline(x, color="white", linewidth=1.5)
    for y in np.arange(-0.5, 3, 1):
        ax.axhline(y, color="white", linewidth=1.5)
    fig.tight_layout()
    save_figure(fig, "per_question_correctness")


def plot_vote_ablation(rows: list[dict]) -> None:
    labels = ["4 votes", "8 votes"]
    correct = [
        sum(r["ca_4_votes_correct"] for r in rows),
        sum(r["ca_8_votes_correct"] for r in rows),
    ]
    values = [x / len(rows) * 100 for x in correct]
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    bars = ax.bar(labels, values, color=[COLORS["orange"], COLORS["green"]], width=0.56)
    ax.set_ylim(0, 70)
    ax.set_ylabel("CA accuracy (%)")
    ax.set_title("CA Self-consistency Vote Ablation", fontsize=14, weight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    for bar, value, count in zip(bars, values, correct):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"{value:.1f}% ({count}/8)",
            ha="center",
            weight="bold",
        )
    ax.text(
        0.5,
        -0.15,
        "Increasing self-consistency corrected Q6 and improved accuracy by 12.5 points.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.tight_layout()
    save_figure(fig, "ca_vote_ablation")


def format_timestamp(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def event_metadata() -> dict[str, dict]:
    payload = json.loads(EVENT_VDB.read_text())
    return {
        item["id"]: {
            "start": float(item["duration"][0]),
            "end": float(item["duration"][1]),
            "description": item.get("description", ""),
        }
        for item in payload["data"]
    }


def adjacent_edges(graph: nx.Graph, node: str) -> list[dict]:
    result = []
    if graph.is_directed():
        for source, target, data in graph.in_edges(node, data=True):
            result.append(
                {"adjacent": source, "direction": "incoming", "relation": data.get("type")}
            )
        for source, target, data in graph.out_edges(node, data=True):
            result.append(
                {"adjacent": target, "direction": "outgoing", "relation": data.get("type")}
            )
    else:
        for source, target, data in graph.edges(node, data=True):
            other = target if source == node else source
            result.append(
                {"adjacent": other, "direction": "undirected", "relation": data.get("type")}
            )
    return result


def print_debug_metadata(graph: nx.Graph, events: dict[str, dict]) -> None:
    lines = []
    for display_name, node in DEBUG_NODE_IDS.items():
        data = graph.nodes[node]
        event = events.get(node, {})
        block = {
            "requested_display_label": display_name,
            "node_id": node,
            "node_type": data.get("type", "unknown"),
            "full_label_name": data.get("id", node),
            "full_description": data.get("description", ""),
            "start_timestamp": event.get("start"),
            "end_timestamp": event.get("end"),
            "start_time": format_timestamp(event["start"]) if event else None,
            "end_time": format_timestamp(event["end"]) if event else None,
            "degree": graph.degree(node),
            "adjacent_nodes": adjacent_edges(graph, node),
        }
        rendered = json.dumps(block, indent=2, ensure_ascii=False)
        print(f"\n===== DEBUG {display_name} =====\n{rendered}")
        lines.append(f"===== DEBUG {display_name} =====\n{rendered}")
    (OUTPUT / "knowledge_graph_debug_metadata.txt").write_text(
        "\n\n".join(lines), encoding="utf-8"
    )


def normalize_entity_label(description: str) -> tuple[str, bool, bool]:
    text = description.lower()
    counts = Counter()
    for term in ANIMAL_TERMS:
        counts[term] = text.count(term)
    present = [term for term, count in counts.most_common() if count > 0]

    ambiguous = False
    if len(present) >= 2:
        first, second = present[:2]
        ambiguity_markers = (
            f"{first}/{second}",
            f"{second}/{first}",
            f"{first} or {second}",
            f"{second} or {first}",
            "possibly",
            "likely",
        )
        if any(marker in text[:500] for marker in ambiguity_markers):
            ambiguous = True
            return f"Ambiguous:\n{first.title()}/{second.title()}", True, False
    if present:
        label = "Person" if present[0] == "human" else present[0].title()
        return label, False, True

    words = [w.strip(".,;:()[]") for w in text.split()]
    meaningful = [w for w in words if w and w not in {"a", "an", "the", "large", "small"}]
    head = " ".join(meaningful[:3]).title() or "Unknown entity"
    generic = any(term in set(words[:8]) for term in GENERIC_ENTITY_TERMS)
    return head, False, generic


def short_event_description(description: str) -> str:
    text = " ".join(description.replace("\n", " ").split())
    lower = text.lower()
    species = next((term for term in ANIMAL_TERMS if term in lower), None)
    if species:
        display = "Person" if species == "human" else species.title()
        actions = (
            ("approach", "approaches camera"),
            ("forag", "forages on forest floor"),
            ("feed", "feeds near the trees"),
            ("dig", "digs among rocks"),
            ("run", "runs through the scene"),
            ("leav", "leaves the observed area"),
            ("rest", "rests in the clearing"),
            ("sniff", "sniffs around the ground"),
            ("explor", "explores the forest floor"),
            ("walk", "walks across the clearing"),
            ("move", "moves through the scene"),
        )
        action = next((phrase for stem, phrase in actions if stem in lower), "appears in forest clearing")
        return f"{display} {action}"

    first_sentence = text.split(".")[0]
    prefixes = (
        "The video captures ",
        "The video depicts ",
        "The video shows ",
        "The video sequence showcases ",
        "The scene shows ",
    )
    for prefix in prefixes:
        if first_sentence.startswith(prefix):
            first_sentence = first_sentence[len(prefix) :]
            break
    words = first_sentence.split()
    return " ".join(words[:7]).strip(" ,;:") or "Quiet forest interval"


def select_temporal_subgraph(
    graph: nx.Graph, events: dict[str, dict], backbone_size: int = 12, max_nodes: int = 30
) -> tuple[nx.Graph, list[str], dict[str, dict], list[str]]:
    ordered_events = sorted(
        (node for node, data in graph.nodes(data=True) if data.get("type") == "event" and node in events),
        key=lambda node: events[node]["start"],
    )
    order_index = {node: index for index, node in enumerate(ordered_events)}

    relevant = set()
    for node in ordered_events:
        description = events[node]["description"].lower()
        if any(keyword in description for keyword in KEYWORDS):
            relevant.add(node)
    linked_animal_counts = Counter()
    for node, data in graph.nodes(data=True):
        if data.get("type") != "entity":
            continue
        label, ambiguous, generic = normalize_entity_label(str(data.get("description", "")))
        if generic:
            continue
        is_animal = any(term.title() in label for term in ANIMAL_TERMS)
        if is_animal or ambiguous:
            for neighbor in graph.neighbors(node):
                if neighbor in events:
                    linked_animal_counts[neighbor] += 1
                    if any(keyword in label.lower() for keyword in KEYWORDS):
                        relevant.add(neighbor)
    if not relevant:
        raise RuntimeError("No events matched the requested wildlife keywords.")

    # Select the densest consecutive temporal window containing keyword events
    # and meaningful linked wildlife entities.
    best_score = -1
    best_start = 0
    for start in range(0, len(ordered_events) - backbone_size + 1):
        window = ordered_events[start : start + backbone_size]
        keyword_hits = sum(node in relevant for node in window)
        if keyword_hits == 0:
            continue
        wildlife_links = sum(linked_animal_counts[node] for node in window)
        score = keyword_hits * 2 + wildlife_links * 5
        if score > best_score:
            best_score = score
            best_start = start
    backbone = ordered_events[best_start : best_start + backbone_size]
    selected = set(backbone)

    entity_candidates = []
    hidden_generic = []
    for node, data in graph.nodes(data=True):
        if data.get("type") != "entity":
            continue
        related = [neighbor for neighbor in graph.neighbors(node) if neighbor in selected]
        if not related:
            continue
        label, ambiguous, generic = normalize_entity_label(str(data.get("description", "")))
        animal = any(term.title() in label for term in ANIMAL_TERMS)
        # This report view is wildlife-focused. Environmental objects and OCR
        # artifacts are visualization-only exclusions even when they are not
        # explicitly covered by the generic-term blacklist.
        if generic or (not animal and not ambiguous) or graph.degree(node) > 80:
            hidden_generic.append(node)
            continue
        keyword_match = sum(keyword in label.lower() for keyword in KEYWORDS)
        entity_candidates.append(
            (keyword_match, int(animal), int(ambiguous), len(related), -graph.degree(node), node, label)
        )
    entity_candidates.sort(reverse=True)
    entity_labels = {}
    for *_, node, label in entity_candidates[: max_nodes - len(backbone)]:
        selected.add(node)
        entity_labels[node] = {
            "label": label,
            "ambiguous": label.startswith("Ambiguous:"),
            "animal": any(term.title() in label for term in ANIMAL_TERMS),
        }

    subgraph = graph.subgraph(selected).copy()
    return subgraph, backbone, entity_labels, hidden_generic


def temporal_positions(
    subgraph: nx.Graph, backbone: list[str], events: dict[str, dict], entities: dict[str, dict]
) -> dict[str, tuple[float, float]]:
    pos = {}
    for index, node in enumerate(backbone):
        # Use the actual event start timestamp (seconds from video start).
        x = events[node]["start"]
        pos[node] = (x, 0.08 if index % 2 == 0 else -0.08)

    entity_nodes = list(entities)
    entity_nodes.sort(
        key=lambda node: np.mean(
            [pos[n][0] for n in subgraph.neighbors(node) if n in pos] or [0.0]
        )
    )
    upper_count = lower_count = 0
    for index, node in enumerate(entity_nodes):
        related_x = [pos[n][0] for n in subgraph.neighbors(node) if n in pos]
        x = float(np.mean(related_x)) if related_x else 0.0
        if index % 2 == 0:
            level = 1.15 + 0.55 * (upper_count % 2)
            upper_count += 1
        else:
            level = -(1.15 + 0.55 * (lower_count % 2))
            lower_count += 1
        pos[node] = (x, level)
    return pos


def plot_local_graph() -> None:
    graph = nx.read_graphml(GRAPHML)
    events = event_metadata()
    print_debug_metadata(graph, events)
    subgraph, backbone, entity_info, hidden_generic = select_temporal_subgraph(graph, events)
    pos = temporal_positions(subgraph, backbone, events, entity_info)

    fig, ax = plt.subplots(figsize=(22, 9))
    fig.suptitle(
        "Wildlife1 Event Knowledge Graph",
        fontsize=19,
        weight="bold",
        y=0.98,
    )
    ax.set_title(
        "Keywords: fox, deer, badger | Temporal window: 12 consecutive events | Entity hop depth: 1",
        fontsize=11,
        color="#4B5563",
        pad=18,
    )

    temporal_edges = []
    belongs_edges = []
    relation_edges = []
    for source, target, data in subgraph.edges(data=True):
        edge_type = data.get("type")
        if edge_type == "event_time_relation" and source in backbone and target in backbone:
            if events[source]["start"] <= events[target]["start"]:
                temporal_edges.append((source, target))
            else:
                temporal_edges.append((target, source))
        elif edge_type == "belong_to":
            belongs_edges.append((source, target))
        elif edge_type == "relation":
            relation_edges.append((source, target))

    nx.draw_networkx_edges(
        subgraph,
        pos,
        edgelist=belongs_edges,
        edge_color="#B8C0C8",
        width=0.9,
        alpha=0.65,
        arrows=False,
        ax=ax,
    )
    nx.draw_networkx_edges(
        subgraph,
        pos,
        edgelist=relation_edges,
        edge_color=COLORS["orange"],
        style="dashed",
        width=1.5,
        alpha=0.85,
        arrows=False,
        ax=ax,
    )
    nx.draw_networkx_edges(
        subgraph,
        pos,
        edgelist=temporal_edges,
        edge_color=COLORS["navy"],
        width=3.0,
        alpha=0.95,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        connectionstyle="arc3,rad=0.02",
        ax=ax,
    )

    nx.draw_networkx_nodes(
        subgraph,
        pos,
        nodelist=backbone,
        node_color=COLORS["blue"],
        node_shape="s",
        node_size=2400,
        edgecolors="white",
        linewidths=1.4,
        ax=ax,
    )
    normal_entities = [node for node, info in entity_info.items() if not info["ambiguous"]]
    ambiguous_entities = [node for node, info in entity_info.items() if info["ambiguous"]]
    animal_sizes = [1750 if entity_info[node]["animal"] else 1300 for node in normal_entities]
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        nodelist=normal_entities,
        node_color=COLORS["green"],
        node_shape="o",
        node_size=animal_sizes,
        edgecolors="white",
        linewidths=1.2,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        nodelist=ambiguous_entities,
        node_color="#B77AC4",
        node_shape="o",
        node_size=1900,
        edgecolors="#6B2D75",
        linewidths=1.8,
        ax=ax,
    )

    event_order = {
        node: index + 1
        for index, node in enumerate(
            sorted(events, key=lambda item: events[item]["start"])
        )
    }
    labels = {}
    mapping_rows = []
    for node in backbone:
        meta = events[node]
        semantic = "\n".join(textwrap.wrap(short_event_description(meta["description"]), width=24))
        labels[node] = (
            f"E{event_order[node]}\n"
            f"{format_timestamp(meta['start'])}-{format_timestamp(meta['end'])}\n"
            f"{semantic}"
        )
    for node, info in entity_info.items():
        labels[node] = info["label"]
    nx.draw_networkx_labels(
        subgraph,
        pos,
        labels=labels,
        font_size=5.7,
        font_color="white",
        font_weight="semibold",
        ax=ax,
    )

    for node in subgraph.nodes:
        data = subgraph.nodes[node]
        meta = events.get(node, {})
        mapping_rows.append(
            {
                "plot_label": labels.get(node, ""),
                "node_id": node,
                "node_type": data.get("type", "unknown"),
                "start_seconds": meta.get("start", ""),
                "end_seconds": meta.get("end", ""),
                "display_entity_name": entity_info.get(node, {}).get("label", ""),
                "description": str(data.get("description", "")),
            }
        )

    legend = [
        Line2D([0], [0], marker="s", color="w", label="Event", markerfacecolor=COLORS["blue"], markersize=12),
        Line2D([0], [0], marker="o", color="w", label="Entity", markerfacecolor=COLORS["green"], markersize=12),
        Line2D([0], [0], marker="o", color="w", label="Ambiguous entity", markerfacecolor="#B77AC4", markersize=12),
        Line2D([0], [0], color=COLORS["navy"], lw=3, label="Temporal event link"),
        Line2D([0], [0], color="#B8C0C8", lw=1, label="Entity belongs to event"),
        Line2D([0], [0], color=COLORS["orange"], lw=2, linestyle="--", label="Entity relation"),
    ]
    ax.legend(handles=legend, loc="upper left", ncol=3, frameon=True, fontsize=8.5)

    start_time = min(events[node]["start"] for node in backbone)
    end_time = max(events[node]["end"] for node in backbone)
    ax.text(
        0.5,
        -0.04,
        f"Local subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges  |  "
        f"Full EKG: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges  |  "
        f"Time range: {format_timestamp(start_time)} - {format_timestamp(end_time)}",
        transform=ax.transAxes,
        ha="center",
        fontsize=9.5,
        color="#4B5563",
    )
    time_span = max(1.0, end_time - start_time)
    ax.set_xlim(start_time - 0.04 * time_span, end_time + 0.04 * time_span)
    ax.set_ylim(-2.5, 2.5)
    ax.axis("off")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(OUTPUT / "knowledge_graph_local.png", dpi=320, bbox_inches="tight")
    fig.savefig(OUTPUT / "knowledge_graph_local.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "knowledge_graph_local.svg", bbox_inches="tight")
    plt.close(fig)

    with (OUTPUT / "knowledge_graph_local_nodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=mapping_rows[0].keys())
        writer.writeheader()
        writer.writerows(mapping_rows)

    stats = {
        "full_graph_nodes": graph.number_of_nodes(),
        "full_graph_edges": graph.number_of_edges(),
        "full_node_types": Counter(
            str(data.get("type", "unknown")) for _, data in graph.nodes(data=True)
        ),
        "full_edge_types": Counter(
            str(data.get("type", "unknown")) for _, _, data in graph.edges(data=True)
        ),
        "local_graph_nodes": subgraph.number_of_nodes(),
        "local_graph_edges": subgraph.number_of_edges(),
        "keywords": list(KEYWORDS),
        "temporal_window_events": len(backbone),
        "entity_hop_depth": 1,
        "hidden_generic_entity_ids": hidden_generic,
        "time_range_seconds": [start_time, end_time],
        "time_range": [format_timestamp(start_time), format_timestamp(end_time)],
    }
    (OUTPUT / "knowledge_graph_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )


def write_readme(rows: list[dict]) -> None:
    sa = np.mean([r["sa_14b_correct"] for r in rows]) * 100
    ca4 = np.mean([r["ca_4_votes_correct"] for r in rows]) * 100
    ca8 = np.mean([r["ca_8_votes_correct"] for r in rows]) * 100
    text = f"""\
    # Wildlife1 reproduction figures

    Generated from the completed Project-Ava run.

    - Paper main AVA-100 result: {PAPER_MAIN_ACCURACY:.1f}% on 120 questions, using
      Qwen2.5-32B for SA and Gemini-1.5-Pro for CA.
    - Our wildlife1 SA result: {sa:.1f}% on 8 questions, using Qwen2.5-14B.
    - Our wildlife1 CA result with 4 votes: {ca4:.1f}%.
    - Our wildlife1 CA result with 8 votes: {ca8:.1f}%.
    - Our CA model: Qwen2.5-VL-7B, with at most 128 uniformly sampled frames.

    The paper and reproduction bars have different model configurations and sample
    sizes. The comparison is contextual, not a claim of exact statistical replication.

    Files:
    - accuracy_comparison.png/pdf
    - per_question_correctness.png/pdf
    - ca_vote_ablation.png/pdf
    - knowledge_graph_local.png/pdf
    - wildlife1_results.csv
    - knowledge_graph_local_nodes.csv
    - knowledge_graph_stats.json
    """
    (OUTPUT / "README.md").write_text(textwrap.dedent(text), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = load_results()
    save_results_csv(rows)
    plot_accuracy_comparison(rows)
    plot_question_matrix(rows)
    plot_vote_ablation(rows)
    plot_local_graph()
    write_readme(rows)
    print(f"Generated figures in: {OUTPUT}")
    for path in sorted(OUTPUT.iterdir()):
        print(f"{path.name}: {path.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
