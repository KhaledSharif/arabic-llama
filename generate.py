from datasets import load_dataset
import dspy
from duckduckgo_search import DDGS

# --------------------------------------------------------------
#
#   FREE-FORM ANSWER
#   https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards
#   https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc
#
#   MULTIPLE CHOICE
#   https://huggingface.co/datasets/medalpaca/medical_meadow_medqa
#
# --------------------------------------------------------------


def duck_search(query: str) -> str:
    results = DDGS().text(query, max_results=10)
    results = [x["title"] + "\n" + x["body"] for x in results]
    return "\n\n".join(results)


lm_provider = dspy.OllamaLocal(
    model="dolphin-mistral:7b-v2.8-q3_K_M",
    max_tokens=1_000,
    num_ctx=4_000,
    temperature=0.8,
)

dspy.settings.configure(lm=lm_provider)


class QuestionAnswer(dspy.Signature):
    """Use the context to form an answer to the question and explain your answer"""

    question = dspy.InputField(desc="The question we are trying to answer")
    context = dspy.InputField(
        desc="List of potentially related information to the question"
    )
    answer = dspy.OutputField(
        desc="""
        Answer to the question using context and internal knowledge
        """
    )


dspy_qa = dspy.ChainOfThought(QuestionAnswer)
dspy_hint = dspy.ChainOfThoughtWithHint(QuestionAnswer)

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

    search_results = duck_search(query=question)
    # print("-" * 20)
    # print(search_results)
    # print("-" * 20, "\n")

    dspy_answer = dspy_qa(question=question, context=search_results).answer
    print("LLM w/out hint:")
    print(dspy_answer, "\n")

    dspy_answer_with_hint = dspy_hint(
        question=question, context=search_results, hint=answer
    ).answer
    print("LLM with hint:")
    print(dspy_answer_with_hint, "\n")

    print("=" * 40, "\n")
