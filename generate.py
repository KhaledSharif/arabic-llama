from datasets import load_dataset
from datetime import datetime
from json import dumps
import ollama

ollama_options = ollama.Options(
    seed=42,
    temperature=0.0,
    num_ctx=2_000,
    num_predict=1_000,
)

model_name = "mistral:7b-instruct-v0.2-q6_K"

dataset_n = 1_000

dataset_name = "qiaojin/PubMedQA"
dataset = load_dataset(dataset_name, "pqa_unlabeled", split="all")
dataset = dataset.shuffle().select(range(dataset_n))

dt_now = datetime.now().strftime("%Y%m%d%H%M")

with open(f"./results/{dt_now}.jsonl", "a") as output_json_lines:
    for i in range(dataset_n):
        question, answer = (
            dataset["question"][i].strip(),
            dataset["long_answer"][i].strip(),
        )
        context = "\n".join(dataset["context"][i]["contexts"]).strip()

        print("=" * 20, "Q", i + 1, "=" * 20, "\n")
        print(context, "\n---\n")
        print(
            f"❓-- Question --",
            question,
            "📖 -- Original Answer --",
            answer,
            sep="\n\n",
            end="\n\n\n",
        )

        answer_without_hint = ""
        answer_with_hint = ""

        # ========================

        prompt_no_hint = f"{context} Question: {question} Answer:"

        print("🕯️ -- LLM Answer w/out hint --")
        stream = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt_no_hint}],
            stream=True,
            options=ollama_options,
        )

        for chunk in stream:
            c = chunk["message"]["content"]
            answer_without_hint += c
            print(c, end="", flush=True)

        print("\n\n\n")

        # ========================

        prompt_hint = f"{context} Question: {question} (Hint: {answer}) Answer:"

        print("💡 -- LLM Answer with hint --")
        stream = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt_hint}],
            stream=True,
            options=ollama_options,
        )

        for chunk in stream:
            c = chunk["message"]["content"]
            answer_with_hint += c
            print(c, end="", flush=True)

        print("\n")

        answer_with_hint += "\n"

        stream = ollama.chat(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt_hint},
                {"role": "assistant", "content": answer_with_hint},
                {"role": "user", "content": "Explain Your Answer"},
            ],
            stream=True,
            options=ollama_options,
        )

        for chunk in stream:
            c = chunk["message"]["content"]
            answer_with_hint += c
            print(c, end="", flush=True)

        print("\n\n")

        # ========================

        output_json_lines.write(
            dumps(
                {
                    "context": context,
                    "question": question,
                    "answer_original": answer,
                    "answer_rejected": answer_without_hint,
                    "answer_chosen": answer_with_hint,
                }
            )
            + "\n"
        )
        output_json_lines.flush()

        print("=" * 50, "\n")
