import csv
import io
import json
import os
import re
from collections import Counter
from datetime import datetime

from bert_score import BERTScorer


BASE = "/root/gpufree-data/AVA_cache/AVA100/5/kg/events"
DESCRIPTIONS_PATH = os.path.join(BASE, "descriptions.json")
EVENTS_PATH = os.path.join(BASE, "events.json")
JSON_OUTPUT = os.path.join(BASE, "debug_semantic_chunking_analysis.json")
CSV_OUTPUT = os.path.join(BASE, "debug_semantic_chunking_pairs.csv")

CHUNK_SECONDS = 3
MODEL = "microsoft/deberta-xlarge-mnli"
LOCAL_MODEL = (
    "/root/gpufree-data/huggingface/hub/"
    "models--microsoft--deberta-xlarge-mnli/snapshots/"
    "5b07a9086c1dbb79981ff7b05b4d1ad83b3af51c"
)
NUM_LAYERS = 40
ORIGINAL_THRESHOLD = 0.65
WINDOW_SIZE = 8
BATCH_SIZE = 8
THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80]

TARGETS = [
    {"label": "Event 18", "start": 11829, "end": 11835},
    {"label": "Event 29", "start": 11836, "end": 11843},
    {"label": "Event 10", "start": 11844, "end": 11851},
]

SPECIES_PATTERNS = {
    "raccoon": r"\braccoons?\b",
    "badger": r"\bbadgers?\b",
    "deer": r"\bdeer\b",
    "hyena": r"\bhyenas?\b",
    "bear": r"\bbears?\b",
    "fox": r"\bfox(?:es)?\b",
    "wolf": r"\b(?:wolf|wolves)\b",
    "predator": r"\bpredators?\b",
    "person/human": r"\b(?:person|people|human|humans)\b",
}

ACTION_PATTERNS = {
    "feeding": r"\b(?:feed|feeds|feeding)\b",
    "foraging": r"\b(?:forage|forages|foraging)\b",
    "digging": r"\b(?:dig|digs|digging)\b",
    "exploring": r"\b(?:explore|explores|exploring|exploration)\b",
    "searching": r"\b(?:search|searches|searching)\b",
    "sniffing": r"\b(?:sniff|sniffs|sniffing)\b",
    "running": r"\b(?:run|runs|running)\b",
    "walking/moving": r"\b(?:walk|walks|walking|move|moves|moving)\b",
    "lying/resting": r"\b(?:lie|lies|lying|rest|rests|resting)\b",
    "observing": r"\b(?:observe|observes|observing|watch|watches|watching)\b",
    "investigating": r"\b(?:investigate|investigates|investigating)\b",
    "interacting": r"\b(?:interact|interacts|interacting|interaction)\b",
}

BACKGROUND_CONCEPTS = {
    "night/nighttime": r"\b(?:night|nighttime|nocturnal)\b",
    "forest/wooded": r"\b(?:forest|forested|wooded|woodland|wilderness)\b",
    "tree": r"\b(?:tree|trees)\b",
    "rocks/rocky": r"\b(?:rock|rocks|rocky)\b",
    "ground": r"\bground\b",
    "camera": r"\b(?:camera|camera trap|night vision)\b",
    "stationary": r"\bstationary\b",
    "foliage": r"\bfoliage\b",
    "undergrowth/underbrush": r"\b(?:undergrowth|underbrush)\b",
    "natural setting/habitat": r"\b(?:natural setting|natural habitat|environment)\b",
    "branches/log/trunk": r"\b(?:branch|branches|log|logs|trunk)\b",
    "dark/dim": r"\b(?:dark|dim|dimly)\b",
}

FOREGROUND_OBJECT_PATTERNS = {
    "animal remains/carcass": r"\b(?:remains|carcass|dead animal|lifeless animal)\b",
    "fallen tree/log/trunk": r"\b(?:fallen tree|fallen log|tree trunk|fallen tree trunk)\b",
    "rocks/rock pile": r"\b(?:rocks|rocky|pile of rocks)\b",
    "tree": r"\b(?:tree|trees)\b",
    "food": r"\bfood\b",
    "light-colored object": r"\blight-colored object\b",
}

MASK_PATTERNS = [
    r"\b(?:night|nighttime|nocturnal|night-time)\b",
    r"\b(?:forest|forested|wooded|woodland|wilderness)\b",
    r"\b(?:tree|trees|tree trunk|fallen tree trunk)\b",
    r"\b(?:rock|rocks|rocky|pile of rocks)\b",
    r"\bground\b",
    r"\b(?:camera|camera trap|stationary camera|night vision camera)\b",
    r"\bbackground\b",
    r"\b(?:foliage|undergrowth|underbrush|vegetation|shrubs|bushes)\b",
    r"\b(?:natural setting|natural environment|natural habitat|environment)\b",
    r"\b(?:scene|area|surroundings)\b",
]

STOPWORDS = set(
    """the a an and or of to in on with is are was were be been being this that
    it its as at by from for into throughout while where which their they them
    video captures shows seen scene appears overall remains remains same likely
    possibly suggesting suggest providing provide various across around within
    first second third final few throughout primary focus footage setting""".split()
)


def duration(index):
    return [index * CHUNK_SECONDS, (index + 1) * CHUNK_SECONDS]


def hms(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def mentions(text, patterns):
    return [name for name, pattern in patterns.items() if re.search(pattern, text, re.I)]


def animal_count(text):
    checks = [
        ("group", r"\b(?:group|several|multiple)\b"),
        ("three", r"\b(?:three|third animal)\b"),
        ("two/pair", r"\b(?:two animals|two raccoons|pair of|both animals)\b"),
        ("single/one", r"\b(?:single|solitary|one animal|a raccoon|a badger|a deer|a hyena|a bear|a fox)\b"),
    ]
    found = [label for label, pattern in checks if re.search(pattern, text, re.I)]
    return found or ["not explicitly stated"]


def mask_background(text):
    masked = text
    for pattern in MASK_PATTERNS:
        masked = re.sub(pattern, " ", masked, flags=re.I)
    masked = re.sub(r"\s+", " ", masked)
    return masked.strip()


def cal_chunk_score(start, end, score_lookup):
    total = 0.0
    count = 0
    for left in range(start, end + 1):
        for right in range(left + 1, end + 1):
            total += score_lookup[(left, right)]
            count += 1
    return total / count


def local_partition(start, end, threshold, score_lookup):
    partitions = []
    decisions = []
    cursor = start
    while cursor <= end:
        candidate_end = cursor + 1
        while candidate_end <= end and candidate_end - cursor < WINDOW_SIZE:
            average = cal_chunk_score(cursor, candidate_end, score_lookup)
            adjacent = score_lookup[(candidate_end - 1, candidate_end)]
            merge = average > threshold and adjacent > threshold
            decisions.append(
                {
                    "candidate": [cursor, candidate_end],
                    "all_pair_average": average,
                    "new_adjacent_score": adjacent,
                    "threshold": threshold,
                    "average_condition": average > threshold,
                    "adjacent_condition": adjacent > threshold,
                    "decision": "MERGE" if merge else "DO NOT MERGE",
                    "reason": (
                        "both all-pair average and newest adjacent score exceed threshold"
                        if merge
                        else "one or both strict > threshold conditions failed"
                    ),
                }
            )
            if not merge:
                break
            candidate_end += 1
        final_end = min(candidate_end - 1, end)
        partitions.append([cursor, final_end])
        cursor = candidate_end
    return partitions, decisions


def score_pairs(scorer, descriptions, pairs):
    left = [descriptions[a] for a, _ in pairs]
    right = [descriptions[b] for _, b in pairs]
    _, recall, _ = scorer.score(left, right, batch_size=BATCH_SIZE)
    return {pair: float(value) for pair, value in zip(pairs, recall.tolist())}


def matrix_for(indices, score_lookup):
    matrix = []
    for left in indices:
        row = []
        for right in indices:
            if left == right:
                row.append(1.0)
            elif left < right:
                row.append(score_lookup[(left, right)])
            else:
                row.append(None)
        matrix.append(row)
    return matrix


def print_matrix(indices, matrix):
    print("Exact algorithm matrix: upper triangle = directional BERTScore Recall; lower triangle = not computed")
    print("index".ljust(8) + "".join(str(i).rjust(11) for i in indices))
    for index, row in zip(indices, matrix):
        cells = []
        for value in row:
            cells.append(("—" if value is None else f"{value:.6f}").rjust(11))
        print(str(index).ljust(8) + "".join(cells))


def main():
    for path in (JSON_OUTPUT, CSV_OUTPUT):
        if os.path.exists(path):
            raise FileExistsError(f"Refusing to overwrite existing diagnostic output: {path}")

    descriptions = json.load(open(DESCRIPTIONS_PATH, encoding="utf-8"))
    events = json.load(open(EVENTS_PATH, encoding="utf-8"))

    target_indices = sorted(
        {index for target in TARGETS for index in range(target["start"], target["end"] + 1)}
    )
    # Include Event 18's following chunk to explain its actual boundary.
    required_pairs = sorted(
        {
            (left, right)
            for target in TARGETS
            for left in range(target["start"], target["end"] + 1)
            for right in range(left + 1, min(target["start"] + WINDOW_SIZE, target["end"] + 1))
        }
    )
    # Event 18 contains seven chunks, so the original algorithm attempts to add
    # 11836 as the eighth chunk. Its candidate-average condition needs every
    # pair from 11829 through 11836, not just the boundary-adjacent pair.
    for left in range(11829, 11837):
        for right in range(left + 1, 11837):
            if (left, right) not in required_pairs:
                required_pairs.append((left, right))
    required_pairs.sort()

    print(
        f"Loading exact scorer checkpoint {MODEL!r} from local snapshot {LOCAL_MODEL!r}; "
        f"num_layers={NUM_LAYERS}, lang='en'"
    )
    scorer = BERTScorer(
        model_type=LOCAL_MODEL,
        num_layers=NUM_LAYERS,
        lang="en",
    )
    original_scores = score_pairs(scorer, descriptions, required_pairs)

    masked_descriptions = list(descriptions)
    for index in target_indices:
        masked_descriptions[index] = mask_background(descriptions[index])
    masked_scores = score_pairs(
        scorer,
        masked_descriptions,
        [pair for pair in required_pairs if pair[0] in target_indices and pair[1] in target_indices],
    )

    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "read_only_inputs": [DESCRIPTIONS_PATH, EVENTS_PATH],
        "implementation": {
            "source": "/root/gpufree-data/Project-Ava-main/AVA/events.py",
            "model": MODEL,
            "local_snapshot": LOCAL_MODEL,
            "num_layers": NUM_LAYERS,
            "bert_score_component": "Recall",
            "bert_scorer": (
                f"BERTScorer(model_type={LOCAL_MODEL!r}, "
                f"num_layers={NUM_LAYERS}, lang='en')"
            ),
            "score_call": f"scorer.score(description_list1, description_list2, batch_size={BATCH_SIZE})",
            "threshold": ORIGINAL_THRESHOLD,
            "comparison": "strictly greater than threshold",
            "window_size": WINDOW_SIZE,
            "conditions": [
                "mean BERTScore Recall over every i<j pair in candidate > threshold",
                "BERTScore Recall between newest chunk and immediately previous chunk > threshold",
                "candidate end minus start < window_size (maximum 8 chunks)",
            ],
        },
        "background_frequency": {},
        "events": [],
    }

    print("\nIMPLEMENTATION")
    print(json.dumps(result["implementation"], indent=2))

    concept_frequency = {
        concept: sum(
            bool(re.search(pattern, descriptions[index], re.I)) for index in target_indices
        )
        for concept, pattern in BACKGROUND_CONCEPTS.items()
    }
    word_frequency = Counter()
    for index in target_indices:
        words = re.findall(r"[a-z][a-z-]+", descriptions[index].lower())
        word_frequency.update(word for word in words if word not in STOPWORDS and len(word) > 3)
    result["background_frequency"] = {
        "number_of_chunks": len(target_indices),
        "concept_chunk_frequency": concept_frequency,
        "automatic_top_words": word_frequency.most_common(40),
    }
    print("\nREPEATED BACKGROUND LANGUAGE")
    print(json.dumps(result["background_frequency"], indent=2))

    csv_rows = []
    for target in TARGETS:
        indices = list(range(target["start"], target["end"] + 1))
        event_duration = [target["start"] * 3, (target["end"] + 1) * 3]
        old_event = next(event for event in events if event["duration"] == event_duration)
        chunks = []

        print("\n" + "=" * 120)
        print(target["label"])
        for index in indices:
            metadata = {
                "index": index,
                "duration": duration(index),
                "time_hms": [hms(duration(index)[0]), hms(duration(index)[1])],
                "description": descriptions[index],
                "species": mentions(descriptions[index], SPECIES_PATTERNS),
                "animal_count": animal_count(descriptions[index]),
                "actions": mentions(descriptions[index], ACTION_PATTERNS),
                "foreground_objects": mentions(descriptions[index], FOREGROUND_OBJECT_PATTERNS),
                "background": mentions(descriptions[index], BACKGROUND_CONCEPTS),
                "foreground_focused_description": masked_descriptions[index],
            }
            chunks.append(metadata)
            print(f"\nChunk index: {index}")
            print(f"Time: {metadata['time_hms'][0]}–{metadata['time_hms'][1]}")
            print("Description:")
            print(descriptions[index])
            print("Parsed foreground semantics:")
            print(json.dumps({k: metadata[k] for k in ("species", "animal_count", "actions", "foreground_objects", "background")}, indent=2))

        matrix = matrix_for(indices, original_scores)
        print("\nBERTSCORE MATRIX")
        print_matrix(indices, matrix)

        adjacent = []
        problematic = []
        print("\nADJACENT SCORES: ORIGINAL VS FOREGROUND-FOCUSED")
        for left, right in zip(indices, indices[1:]):
            original = original_scores[(left, right)]
            masked = masked_scores[(left, right)]
            left_meta = chunks[left - indices[0]]
            right_meta = chunks[right - indices[0]]
            record = {
                "chunk_a": left,
                "chunk_b": right,
                "original_recall": original,
                "foreground_focused_recall": masked,
                "difference": masked - original,
                "above_original_threshold": original > ORIGINAL_THRESHOLD,
            }
            adjacent.append(record)
            print(
                f"C{left} ↔ C{right}: original={original:.6f} "
                f"({'ABOVE' if original > ORIGINAL_THRESHOLD else 'BELOW'} 0.65), "
                f"foreground={masked:.6f}, delta={masked-original:+.6f}"
            )
            species_changed = set(left_meta["species"]) != set(right_meta["species"])
            actions_changed = set(left_meta["actions"]) != set(right_meta["actions"])
            if original > ORIGINAL_THRESHOLD and (species_changed or actions_changed):
                shared_background = sorted(set(left_meta["background"]) & set(right_meta["background"]))
                item = {
                    **record,
                    "species_a": left_meta["species"],
                    "species_b": right_meta["species"],
                    "actions_a": left_meta["actions"],
                    "actions_b": right_meta["actions"],
                    "species_changed": species_changed,
                    "actions_changed": actions_changed,
                    "shared_background": shared_background,
                }
                problematic.append(item)
                print("POTENTIAL PROBLEMATIC MERGE:")
                print(json.dumps(item, indent=2))
            csv_rows.append(
                {
                    "event_id": target["label"],
                    "chunk_a": left,
                    "chunk_b": right,
                    "time_a": f"{hms(left*3)}-{hms((left+1)*3)}",
                    "time_b": f"{hms(right*3)}-{hms((right+1)*3)}",
                    "species_a": "|".join(left_meta["species"]),
                    "species_b": "|".join(right_meta["species"]),
                    "actions_a": "|".join(left_meta["actions"]),
                    "actions_b": "|".join(right_meta["actions"]),
                    "original_bertscore_recall": original,
                    "foreground_bertscore_recall": masked,
                }
            )

        groupings = {}
        original_decisions = None
        for threshold in THRESHOLDS:
            partitions, decisions = local_partition(
                target["start"], target["end"], threshold, original_scores
            )
            groupings[str(threshold)] = {"partitions": partitions, "decisions": decisions}
            if threshold == ORIGINAL_THRESHOLD:
                original_decisions = decisions

        print("\nORIGINAL 0.65 STEP-BY-STEP DECISIONS")
        for decision in original_decisions:
            print(json.dumps(decision, indent=2))
        print("\nTHRESHOLD SENSITIVITY")
        for threshold, data in groupings.items():
            print(f"{threshold}: {data['partitions']}")

        boundary = {
            "mechanism": (
                "maximum 8-chunk window forces boundary"
                if len(indices) == WINDOW_SIZE
                else "similarity conditions fail when attempting to add next chunk"
            )
        }
        if len(indices) < WINDOW_SIZE:
            next_index = target["end"] + 1
            candidate_average = cal_chunk_score(target["start"], next_index, original_scores)
            adjacent_score = original_scores[(target["end"], next_index)]
            boundary.update(
                {
                    "attempted_next_chunk": next_index,
                    "candidate_all_pair_average": candidate_average,
                    "new_adjacent_score": adjacent_score,
                    "average_condition": candidate_average > ORIGINAL_THRESHOLD,
                    "adjacent_condition": adjacent_score > ORIGINAL_THRESHOLD,
                }
            )
        print("\nBOUNDARY DETERMINATION")
        print(json.dumps(boundary, indent=2))

        summary_species = mentions(old_event["description"], SPECIES_PATTERNS)
        chunk_species = sorted({species for chunk in chunks for species in chunk["species"]})
        species_union = sorted(set(summary_species) | set(chunk_species))
        species_provenance = [
            {
                "species": species,
                "appears_in_chunk_descriptions": species in chunk_species,
                "appears_in_final_summary": species in summary_species,
                "appears_only_in_final_summary": species in summary_species and species not in chunk_species,
            }
            for species in species_union
        ]
        print("\nFINAL SUMMARY SPECIES PROVENANCE")
        print(json.dumps(species_provenance, indent=2))

        result["events"].append(
            {
                **target,
                "duration": event_duration,
                "chunks": chunks,
                "exact_algorithm_matrix": matrix,
                "adjacent_comparison": adjacent,
                "potential_problematic_merges": problematic,
                "original_threshold_decisions": original_decisions,
                "threshold_sensitivity": groupings,
                "boundary_determination": boundary,
                "original_final_summary": old_event["description"],
                "species_provenance": species_provenance,
            }
        )

    with open(JSON_OUTPUT, "x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(CSV_OUTPUT, "x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nSaved JSON: {JSON_OUTPUT}")
    print(f"Saved CSV: {CSV_OUTPUT}")


if __name__ == "__main__":
    main()
