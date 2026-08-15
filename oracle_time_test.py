import json
from collections import Counter
from pathlib import Path

import numpy as np

from AVA.prompt import PROMPTS
from dataset.init_dataset import init_dataset
from llms.init_model import init_model


VIDEO_ID = 7
QUESTION_IDS = [4, 10, 11]
WINDOW_SECONDS = 30
MAX_FRAMES = 128
VOTES = 4
OUTPUT = Path(
    "/root/gpufree-data/AVA_cache/AVA100/7/oracle_time_debug.json"
)


def parse_timestamp(value: str) -> int:
    parts = [int(item) for item in value.strip().split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    else:
        raise ValueError(f"Invalid timestamp: {value}")
    return hours * 3600 + minutes * 60 + seconds


def extract_answer(text: str):
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start:end])
    except json.JSONDecodeError:
        return None
    answer = str(payload.get("Answer", "")).strip().upper()
    return answer if answer in {"A", "B", "C", "D"} else None


def main():
    dataset = init_dataset("ava100")
    video = dataset.get_video(VIDEO_ID)
    video_info = dataset.get_video_info(VIDEO_ID)
    qas = video_info["qa"]
    model = init_model("qwenvl", 1)

    results = []
    if OUTPUT.exists():
        results = json.loads(OUTPUT.read_text(encoding="utf-8"))
    completed = {item["question_id"] for item in results}

    for question_id in QUESTION_IDS:
        if question_id in completed:
            print(f"Question {question_id} already exists; skipping.")
            continue

        qa = qas[question_id]
        reference = qa["time_reference"].split(",")[0].strip()
        center = parse_timestamp(reference)
        start = max(0, center - WINDOW_SECONDS)
        end = min(video.config["duration"], center + WINDOW_SECONDS)

        frames, timestamps, _ = video.get_frames_by_fps(
            fps=2, duration=(start, end)
        )
        if len(frames) > MAX_FRAMES:
            indices = np.linspace(0, len(frames) - 1, MAX_FRAMES, dtype=int)
            frames = [frames[index] for index in indices]
            timestamps = [timestamps[index] for index in indices]

        prompt = PROMPTS["checkframe_and_answer_COT"].format(
            user_query=qa["question"]
        )
        responses = []
        for vote in range(VOTES):
            print(
                f"Question {question_id}: vote {vote + 1}/{VOTES}, "
                f"frames={len(frames)}, window={start}-{end}"
            )
            response = model.generate_response(
                {"text": prompt, "video": frames},
                max_new_tokens=512,
                temperature=0.5,
            )
            responses.append(response)

        answers = [extract_answer(response) for response in responses]
        valid_answers = [answer for answer in answers if answer is not None]
        prediction = (
            Counter(valid_answers).most_common(1)[0][0]
            if valid_answers
            else None
        )
        result = {
            "question_id": question_id,
            "question": qa["question"],
            "ground_truth": qa["answer"],
            "time_reference": qa["time_reference"],
            "tested_window": [start, end],
            "sampled_timestamp_range": (
                [timestamps[0], timestamps[-1]] if timestamps else []
            ),
            "num_frames": len(frames),
            "votes": answers,
            "prediction": prediction,
            "correct": prediction == qa["answer"],
            "responses": responses,
        }
        results.append(result)
        OUTPUT.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"Question {question_id}: prediction={prediction}, "
            f"ground_truth={qa['answer']}, correct={result['correct']}"
        )

    print(f"Results saved to {OUTPUT}")


if __name__ == "__main__":
    main()
