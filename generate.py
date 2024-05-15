from datasets import load_dataset
import dspy

# --------------------------------------------------------------
#
# https://huggingface.co/medalpaca
#   
#   FREE-FORM ANSWER
#   https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards
#   https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc
#   
#   MULTIPLE CHOICE
#   https://huggingface.co/datasets/medalpaca/medical_meadow_medqa   
#
# --------------------------------------------------------------
dataset_name = "medical_meadow_medical_flashcards"
dataset = load_dataset(dataset_name, split="all")
dataset = dataset.shuffle(seed=42).select(range(1000))

