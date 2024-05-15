from datasets import load_dataset
import dspy

lm_provider = dspy.OllamaLocal(
    model="dolphin-mistral:7b-v2.8-q3_K_M",
    max_tokens=1_000,
    num_ctx=4_000,
    temperature=0.8,
)

dspy.settings.configure(lm=lm_provider)

dspy_qa = dspy.ChainOfThought("question->answer")
dspy_hint = dspy.ChainOfThought("question,hint->answer")

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

    dspy_answer = dspy_qa(question=question).answer
    print("LLM w/out hint:")
    print(dspy_answer, "\n")

    dspy_answer_with_hint = dspy_hint(question=question, hint=answer).answer
    print("LLM with hint:")
    print(dspy_answer_with_hint, "\n")

    print("=" * 40, "\n")
