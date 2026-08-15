import json
import os
from datetime import datetime

from AVA.prompt import PROMPTS
from llms.init_model import init_model


BASE = "/root/gpufree-data/AVA_cache/AVA100/5/kg"
DESCRIPTIONS_PATH = os.path.join(BASE, "events", "descriptions.json")
EVENTS_PATH = os.path.join(BASE, "events", "events.json")
OUTPUT_PATH = os.path.join(BASE, "events", "debug_fixed_event_summaries.json")
CHUNK_SECONDS = 3

TARGETS = [
    {"label": "Event 18", "duration": [35487, 35508]},
    {"label": "Event 29", "duration": [35508, 35532]},
    {"label": "Event 10", "duration": [35532, 35556]},
]


def print_chunk(prefix, index, text):
    start = index * CHUNK_SECONDS
    end = start + CHUNK_SECONDS
    print(f"{prefix} chunk_index={index}, time={start}-{end}")
    print(text)


def main():
    if os.path.exists(OUTPUT_PATH):
        raise FileExistsError(
            f"Refusing to overwrite existing debug output: {OUTPUT_PATH}"
        )

    with open(DESCRIPTIONS_PATH, "r", encoding="utf-8") as f:
        descriptions = json.load(f)
    with open(EVENTS_PATH, "r", encoding="utf-8") as f:
        events = json.load(f)

    prepared = []
    batch_inputs = []

    for target in TARGETS:
        start_time, end_time = target["duration"]
        partition_start = start_time // CHUNK_SECONDS
        partition_end = end_time // CHUNK_SECONDS - 1
        all_indices = list(range(partition_start, partition_end + 1))
        old_indices = list(range(partition_start, partition_end))
        fixed_indices = list(range(partition_start, partition_end + 1))
        omitted_indices = sorted(set(fixed_indices) - set(old_indices))

        matching_events = [
            event for event in events if event["duration"] == target["duration"]
        ]
        if len(matching_events) != 1:
            raise RuntimeError(
                f"Expected exactly one event for {target['duration']}, "
                f"found {len(matching_events)}"
            )

        print("\n" + "=" * 100)
        print(f"{target['label']}: {start_time}-{end_time}")
        print(
            f"partition start/end indices (inclusive): "
            f"{partition_start}/{partition_end}"
        )

        print("\nALL ORIGINAL 3-SECOND CHUNKS")
        for index in all_indices:
            print_chunk("ALL", index, descriptions[index])

        print("\nOLD CODE SENDS TO QWEN")
        for index in old_indices:
            print_chunk("OLD", index, descriptions[index])

        print("\nFIXED CODE SENDS TO QWEN")
        for index in fixed_indices:
            print_chunk("FIXED", index, descriptions[index])

        print("\nPREVIOUSLY OMITTED")
        for index in omitted_indices:
            print_chunk("OMITTED", index, descriptions[index])

        fixed_inputs = [descriptions[index] for index in fixed_indices]
        prompt = PROMPTS["summarize_descriptions"].format(inputs=fixed_inputs)
        batch_inputs.append({"text": prompt})
        prepared.append(
            {
                **target,
                "partition_start": partition_start,
                "partition_end_inclusive": partition_end,
                "all_chunk_indices": all_indices,
                "old_code_chunk_indices": old_indices,
                "fixed_code_chunk_indices": fixed_indices,
                "previously_omitted_chunk_indices": omitted_indices,
                "chunks": [
                    {
                        "chunk_index": index,
                        "duration": [
                            index * CHUNK_SECONDS,
                            (index + 1) * CHUNK_SECONDS,
                        ],
                        "description": descriptions[index],
                    }
                    for index in all_indices
                ],
                "old_summary": matching_events[0]["description"],
                "summarization_prompt": prompt,
            }
        )

    print("\nLoading the same Qwen-VL model used by graph_construction.py...")
    llm = init_model("qwenvl", 1)
    fixed_summaries = llm.batch_generate_response(
        batch_inputs,
        max_new_tokens=1024,
        temperature=0.5,
        max_batch_size=64,
    )
    if len(fixed_summaries) != len(prepared):
        raise RuntimeError(
            f"Expected {len(prepared)} summaries, got {len(fixed_summaries)}"
        )

    for record, fixed_summary in zip(prepared, fixed_summaries):
        record["fixed_summary"] = fixed_summary
        print("\n" + "=" * 100)
        print(f"{record['label']} ({record['duration'][0]}-{record['duration'][1]})")
        print("\nOLD SUMMARY:\n" + record["old_summary"])
        print("\nFIXED SUMMARY:\n" + fixed_summary)

    payload = {
        "purpose": "Isolated A/B validation of AVA event summarization off-by-one bug",
        "created_at": datetime.now().astimezone().isoformat(),
        "source_descriptions": DESCRIPTIONS_PATH,
        "source_events_read_only": EVENTS_PATH,
        "model": "/root/gpufree-data/models/Qwen2.5-VL-7B-Instruct-AWQ",
        "model_alias": "qwenvl",
        "generation_config": {
            "max_new_tokens": 1024,
            "temperature": 0.5,
            "do_sample": True,
            "max_batch_size": 64,
        },
        "prompt_template": PROMPTS["summarize_descriptions"],
        "events": prepared,
    }
    with open(OUTPUT_PATH, "x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved isolated debug output to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
