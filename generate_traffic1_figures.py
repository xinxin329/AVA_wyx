#!/usr/bin/env python3
"""Generate report-ready figures for the Project-Ava Traffic1 reproduction."""

from __future__ import annotations

import csv
import json
import re
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
CACHE = Path("/root/gpufree-data/AVA_cache/AVA100/7")
QUESTIONS = CACHE / "questions"
GROUND_TRUTH = Path("/root/gpufree-data/AVA100/traffic.json")
GRAPHML = CACHE / "kg/graph_event_knowledge_graph.graphml"
EVENT_VDB = CACHE / "kg/vdb_events.json"
ORACLE = CACHE / "oracle_time_debug.json"
OUTPUT = PROJECT / "outputs/traffic1_figures"
PAPER_MAIN_ACCURACY = 75.8
TARGET_TIME = 12 * 3600 + 46 * 60 + 45  # Q10 official video position

COLORS = {
    "navy": "#24476B",
    "blue": "#3A78A8",
    "green": "#2E8B57",
    "red": "#D55E55",
    "orange": "#E69F45",
    "purple": "#8064A2",
    "gray": "#7A8793",
    "light": "#E8EDF2",
}
CHOICE_COLORS = {
    "A": COLORS["blue"],
    "B": COLORS["orange"],
    "C": COLORS["green"],
    "D": COLORS["purple"],
}
TRAFFIC_TERMS = (
    "bus", "truck", "taxi", "car", "vehicle", "van", "pedestrian",
    "person", "bicycle", "motorcycle", "traffic light", "crosswalk",
)
GENERIC_TERMS = {
    "intersection", "road", "street", "camera", "scene", "frame",
    "traffic", "sidewalk", "building", "tree", "trees", "sky",
}


def first_answer(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    data = json.loads(path.read_text())
    return next(iter(data[0]["final_score"])) if data else "EMPTY"


def load_rows() -> list[dict]:
    qas = json.loads(GROUND_TRUTH.read_text())[0]["qa"]
    rows = []
    for qid, qa in enumerate(qas):
        qdir = QUESTIONS / str(qid)
        sa = first_answer(qdir / "sorted_SA_score_result.json")
        ca = first_answer(qdir / "sorted_CA_score_result.json")
        ca_file = qdir / "sorted_CA_score_result.json"
        shares = {choice: 0.0 for choice in "ABCD"}
        if ca_file.exists():
            payload = json.loads(ca_file.read_text())
            if payload:
                scores = payload[0].get("scores", {})
                for choice in shares:
                    shares[choice] = float(scores.get(choice, [0.0])[0])
        rows.append({
            "question_id": qid,
            "question": qa["query"],
            "ground_truth": qa["answer"],
            "time_reference": qa.get("time_reference", ""),
            "sa_14b": sa,
            "ca_8_votes": ca,
            "sa_correct": sa == qa["answer"],
            "ca_correct": ca == qa["answer"],
            **{f"ca_{choice}_share": shares[choice] for choice in "ABCD"},
        })
    return rows


def save_figure(fig: plt.Figure, stem: str, svg: bool = False) -> None:
    fig.savefig(OUTPUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    if svg:
        fig.savefig(OUTPUT / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def save_csv(rows: list[dict]) -> None:
    with (OUTPUT / "traffic1_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_accuracy(rows: list[dict]) -> None:
    values = [
        PAPER_MAIN_ACCURACY,
        np.mean([row["sa_correct"] for row in rows]) * 100,
        np.mean([row["ca_correct"] for row in rows]) * 100,
    ]
    labels = ["Paper AVA-100\nmain result", "Traffic1 SA\nQwen2.5-14B", "Traffic1 CA\nQwen2.5-VL-7B"]
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    bars = ax.bar(labels, values, color=[COLORS["navy"], COLORS["blue"], COLORS["orange"]], width=0.62)
    ax.set_ylim(0, 90)
    ax.set_ylabel("Multiple-choice accuracy (%)")
    ax.set_title("Traffic1 Reproduction Accuracy", fontsize=15, weight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    counts = ["91/120", f"{sum(r['sa_correct'] for r in rows)}/12", f"{sum(r['ca_correct'] for r in rows)}/12"]
    for bar, value, count in zip(bars, values, counts):
        ax.text(bar.get_x() + bar.get_width()/2, max(value + 1.5, 1.5), f"{value:.1f}% ({count})", ha="center", weight="bold")
    fig.text(0.5, 0.01, "Paper value is the full AVA-100 result (120 QA) and is contextual only; Traffic1 contains 12 QA and uses modified local models.", ha="center", fontsize=8.4, color="#4B5563")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_figure(fig, "accuracy_comparison")


def plot_question_matrix(rows: list[dict]) -> None:
    matrix = np.array([[int(r["sa_correct"]) for r in rows], [int(r["ca_correct"]) for r in rows]])
    predictions = [[r["sa_14b"] for r in rows], [r["ca_8_votes"] for r in rows]]
    fig, ax = plt.subplots(figsize=(13, 3.3))
    ax.imshow(matrix, cmap=ListedColormap([COLORS["red"], COLORS["green"]]), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(12), [f"Q{i}" for i in range(12)])
    ax.set_yticks(range(2), ["SA 14B", "CA 8 votes"])
    ax.set_title("Traffic1 Per-question Correctness", fontsize=14, weight="bold")
    ax.set_xlabel("Question ID")
    for i in range(2):
        for j, row in enumerate(rows):
            mark = "✓" if matrix[i, j] else "✗"
            ax.text(j, i, f"{predictions[i][j]} {mark}\nGT {row['ground_truth']}", ha="center", va="center", color="white", fontsize=9, weight="bold")
    for x in np.arange(-0.5, 12, 1): ax.axvline(x, color="white", linewidth=1.2)
    for y in np.arange(-0.5, 2, 1): ax.axhline(y, color="white", linewidth=1.2)
    fig.tight_layout()
    save_figure(fig, "per_question_correctness")


def plot_ca_vote_distribution(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for choice in "ABCD":
        values = np.array([row[f"ca_{choice}_share"] * 100 for row in rows])
        ax.bar(x, values, bottom=bottom, label=f"Choice {choice}", color=CHOICE_COLORS[choice], width=0.72)
        bottom += values
    ax.set_xticks(x, [f"Q{i}" for i in range(12)])
    ax.set_ylim(0, 108)
    ax.set_ylabel("Vote share of top CA candidate (%)")
    ax.set_title("Traffic1 CA Vote Distribution", fontsize=14, weight="bold")
    ax.legend(ncol=4, loc="upper center")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    for i, row in enumerate(rows):
        ax.text(i, 102, f"GT {row['ground_truth']}", ha="center", fontsize=8, weight="bold")
    fig.text(0.5, 0.01, "CA predictions collapsed toward choice A (Q6 predicted B), producing 0/12 accuracy.", ha="center", fontsize=9, color="#4B5563")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_figure(fig, "ca_vote_distribution")


def plot_answer_bias(rows: list[dict]) -> None:
    labels = list("ABCD")
    gt = Counter(row["ground_truth"] for row in rows)
    sa = Counter(row["sa_14b"] for row in rows)
    ca = Counter(row["ca_8_votes"] for row in rows)
    x = np.arange(4)
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(x-width, [gt[c] for c in labels], width, label="Ground truth", color=COLORS["gray"])
    ax.bar(x, [sa[c] for c in labels], width, label="SA prediction", color=COLORS["blue"])
    ax.bar(x+width, [ca[c] for c in labels], width, label="CA prediction", color=COLORS["orange"])
    ax.set_xticks(x, labels)
    ax.set_ylabel("Number of questions")
    ax.set_title("Answer Distribution and CA Choice Bias", fontsize=14, weight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, "answer_distribution")


def format_time(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def load_events() -> dict[str, dict]:
    payload = json.loads(EVENT_VDB.read_text())
    return {
        item["id"]: {
            "start": float(item["duration"][0]),
            "end": float(item["duration"][1]),
            "description": item.get("description", ""),
        }
        for item in payload["data"]
    }


def short_event(text: str) -> str:
    clean = " ".join(text.split())
    lower = clean.lower()
    actor = next((term for term in TRAFFIC_TERMS if term in lower), "traffic scene")
    action_map = (
        ("turn right", "turns right"), ("turn left", "turns left"),
        ("straight", "moves straight"), ("cross", "crosses intersection"),
        ("stop", "stops"), ("block", "blocks traffic"),
        ("pass", "passes intersection"), ("wait", "waits"),
    )
    action = next((label for key, label in action_map if key in lower), "appears")
    return f"{actor.title()} {action}"


def entity_label(text: str) -> str | None:
    clean = " ".join(str(text).split())
    lower = clean.lower()
    matched = [term for term in TRAFFIC_TERMS if re.search(rf"\b{re.escape(term)}s?\b", lower)]
    specific = [term for term in matched if term not in {"vehicle", "traffic light"}]
    if specific:
        return specific[0].title()
    words = [word.strip(".,;:()[]").lower() for word in clean.split()[:8]]
    if any(word in GENERIC_TERMS for word in words):
        return None
    return " ".join(clean.split()[:3]).title() if clean else None


def plot_local_graph() -> dict:
    graph = nx.read_graphml(GRAPHML)
    events = load_events()
    ordered = sorted((node for node, data in graph.nodes(data=True) if data.get("type") == "event" and node in events), key=lambda node: events[node]["start"])
    center = min(range(len(ordered)), key=lambda idx: abs(events[ordered[idx]]["start"] - TARGET_TIME))
    start = max(0, min(center - 6, len(ordered) - 12))
    backbone = ordered[start:start+12]
    selected = set(backbone)
    entity_info = {}
    candidates = []
    for node, data in graph.nodes(data=True):
        if data.get("type") != "entity":
            continue
        linked = [neighbor for neighbor in graph.neighbors(node) if neighbor in selected]
        if not linked:
            continue
        label = entity_label(data.get("description", ""))
        if label is None or graph.degree(node) > 100:
            continue
        keyword_score = int(any(term in label.lower() for term in TRAFFIC_TERMS))
        candidates.append((keyword_score, len(linked), -graph.degree(node), node, label))
    candidates.sort(reverse=True)
    for _, _, _, node, label in candidates[:10]:
        selected.add(node)
        entity_info[node] = label
    subgraph = graph.subgraph(selected).copy()

    pos = {node: (idx, 0.08 if idx % 2 == 0 else -0.08) for idx, node in enumerate(backbone)}
    for idx, node in enumerate(entity_info):
        linked_x = [pos[n][0] for n in subgraph.neighbors(node) if n in pos]
        x = float(np.mean(linked_x)) if linked_x else 0
        y = (1.35 + 0.45*(idx % 2)) * (1 if idx % 2 == 0 else -1)
        pos[node] = (x, y)

    fig, ax = plt.subplots(figsize=(22, 9))
    fig.suptitle("Traffic1 Event Knowledge Graph - Local View", fontsize=19, weight="bold", y=0.98)
    ax.set_title("Q10 oracle-time neighborhood | 12 consecutive events | one-hop traffic entities", fontsize=11, color="#4B5563", pad=18)
    temporal, belongs, relations = [], [], []
    for source, target, data in subgraph.edges(data=True):
        kind = data.get("type")
        if kind == "event_time_relation" and source in backbone and target in backbone:
            temporal.append((source, target))
        elif kind == "belong_to": belongs.append((source, target))
        elif kind == "relation": relations.append((source, target))
    nx.draw_networkx_edges(subgraph, pos, edgelist=belongs, edge_color="#B8C0C8", width=1, alpha=0.7, arrows=False, ax=ax)
    nx.draw_networkx_edges(subgraph, pos, edgelist=relations, edge_color=COLORS["orange"], style="dashed", width=1.5, arrows=False, ax=ax)
    nx.draw_networkx_edges(subgraph, pos, edgelist=temporal, edge_color=COLORS["navy"], width=2.7, arrows=True, arrowsize=17, ax=ax)
    nx.draw_networkx_nodes(subgraph, pos, nodelist=backbone, node_color=COLORS["blue"], node_shape="s", node_size=2500, edgecolors="white", linewidths=1.3, ax=ax)
    nx.draw_networkx_nodes(subgraph, pos, nodelist=list(entity_info), node_color=COLORS["green"], node_shape="o", node_size=1600, edgecolors="white", linewidths=1.2, ax=ax)
    event_order = {node: idx+1 for idx, node in enumerate(ordered)}
    labels = {}
    mapping = []
    for node in backbone:
        meta = events[node]
        labels[node] = f"E{event_order[node]}\n{format_time(meta['start'])}\n" + "\n".join(textwrap.wrap(short_event(meta["description"]), 20))
        mapping.append({"display_id": f"E{event_order[node]}", "node_id": node, **meta})
    labels.update({node: "\n".join(textwrap.wrap(label, 16)) for node, label in entity_info.items()})
    nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=7.2, font_color="white", font_weight="bold", ax=ax)
    legend = [
        Line2D([0], [0], marker="s", color="w", label="Event", markerfacecolor=COLORS["blue"], markersize=12),
        Line2D([0], [0], marker="o", color="w", label="Traffic entity", markerfacecolor=COLORS["green"], markersize=11),
        Line2D([0], [0], color=COLORS["navy"], lw=3, label="Temporal relation"),
        Line2D([0], [0], color=COLORS["orange"], lw=2, linestyle="--", label="Entity relation"),
    ]
    ax.legend(handles=legend, loc="upper left", ncol=4, frameon=False)
    ax.text(0.5, -0.04, f"Local subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges | Full EKG: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges", transform=ax.transAxes, ha="center", fontsize=10, color="#4B5563")
    ax.axis("off")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save_figure(fig, "knowledge_graph_local", svg=True)
    with (OUTPUT / "knowledge_graph_local_nodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["display_id", "node_id", "start", "end", "description"])
        writer.writeheader(); writer.writerows(mapping)
    return {
        "full_nodes": graph.number_of_nodes(), "full_edges": graph.number_of_edges(),
        "event_nodes": sum(data.get("type") == "event" for _, data in graph.nodes(data=True)),
        "entity_nodes": sum(data.get("type") == "entity" for _, data in graph.nodes(data=True)),
        "local_nodes": subgraph.number_of_nodes(), "local_edges": subgraph.number_of_edges(),
        "local_start": events[backbone[0]]["start"], "local_end": events[backbone[-1]]["end"],
    }


def plot_oracle() -> None:
    if not ORACLE.exists(): return
    items = json.loads(ORACLE.read_text())
    qids = [item["question_id"] for item in items]
    qas = json.loads(GROUND_TRUTH.read_text())[0]["qa"]
    original = [first_answer(QUESTIONS/str(qid)/"sorted_SA_score_result.json") == qas[qid]["answer"] for qid in qids]
    oracle = [item["correct"] for item in items]
    matrix = np.array([original, oracle], dtype=int)
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    ax.imshow(matrix, cmap=ListedColormap([COLORS["red"], COLORS["green"]]), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(qids)), [f"Q{qid}" for qid in qids])
    ax.set_yticks(range(2), ["Original SA", "Oracle time CA"])
    ax.set_title("Oracle-time Diagnostic", fontsize=14, weight="bold")
    for i in range(2):
        for j in range(len(qids)):
            ax.text(j, i, "Correct" if matrix[i,j] else "Wrong", ha="center", va="center", color="white", weight="bold")
    fig.tight_layout()
    save_figure(fig, "oracle_time_diagnostic")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    save_csv(rows)
    plot_accuracy(rows)
    plot_question_matrix(rows)
    plot_ca_vote_distribution(rows)
    plot_answer_bias(rows)
    plot_oracle()
    stats = plot_local_graph()
    summary = {
        "traffic1_questions": len(rows),
        "sa_correct": sum(row["sa_correct"] for row in rows),
        "sa_accuracy": np.mean([row["sa_correct"] for row in rows]) * 100,
        "ca_correct": sum(row["ca_correct"] for row in rows),
        "ca_accuracy": np.mean([row["ca_correct"] for row in rows]) * 100,
        **stats,
    }
    (OUTPUT / "traffic1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Figures saved to {OUTPUT}")


if __name__ == "__main__":
    main()
