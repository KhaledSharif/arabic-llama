from datasets import load_dataset
import ollama

ollama_options = ollama.Options(
    num_predict=1_000,
    seed=42,
    temperature=0.0,
    num_ctx=1_000,
)

model_name = "dolphin-llama3:latest"

dataset_name = "qiaojin/PubMedQA"
dataset = load_dataset(dataset_name, "pqa_artificial", split="all")
dataset = dataset.shuffle().select(range(10))

for i in range(10):
    question, answer = dataset["question"][i].strip(), dataset["long_answer"][i].strip()
    context = "\n".join(dataset["context"][i]["contexts"])

    print("=" * 20, "Q", i + 1, "=" * 20, "\n")
    print(context, "\n---\n")
    print(
        f"❓ Question:", question, "Original Answer:", answer, sep="\n\n", end="\n\n\n"
    )

    answer_without_hint = ""
    answer_with_hint = ""

    # ========================

    print("🕯️ -- LLM w/out hint --")
    stream = ollama.chat(
        model=model_name,
        messages=[
            {"role": "user", "content": f"{context} Question: {question} Answer:"}
        ],
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

    print("💡 -- LLM with hint --")
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
        print(chunk["message"]["content"], end="", flush=True)
        
    print("\n\n")

    # ========================

    print("=" * 40, "\n")
