from datasets import load_dataset
import ollama

ollama_options = ollama.Options(
    num_predict=1_000,
    seed=42,
    temperature=0.0,
    num_ctx=1_000,
)
model_name = "dolphin-mistral:7b-v2.8-q3_K_M"
dataset_name = "medalpaca/medical_meadow_medical_flashcards"
dataset = load_dataset(dataset_name, split="all")
dataset = dataset.shuffle().select(range(10))

for i in range(10):
    question, answer = dataset["input"][i].strip(), dataset["output"][i].strip()

    if len(question) < 10 or len(answer) < 10:
        continue

    print("=" * 40, "\n")
    print(
        f"Question {i+1}:", question, "Original Answer:", answer, sep="\n\n", end="\n\n"
    )

    print("LLM w/out hint:")
    stream = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": f"{question}"}],
        stream=True,
        options=ollama_options,
    )

    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)

    print("\n\n")
    print("LLM with hint:")
    stream = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": f"{question} (hint: {answer}) answer:"}],
        stream=True,
        options=ollama_options,
    )

    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)
    print("\n\n")

    print("=" * 40, "\n")
