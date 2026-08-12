"""Prompt templates, verbatim from the RGB repo config/instruction.yaml (paper Figure 3).
Extracted from RGB_Evaluation_v2_CP4_Batch26.ipynb, cell 6.
"""

# CELL 6: Prompt Templates — VERBATIM from RGB repo config/instruction.yaml (paper Figure 3).
# Mentor instruction: "Use the prompt in Figure 3 of the research paper" — this is the exact string,
# including its original quote characters. Do NOT reword (lesson from the 7.4 judge prompt).
SYSTEM_PROMPT_RGB = (
"""You are an accurate and reliable AI assistant that can answer questions with the help of external
documents. Please note that external documents may contain noisy or factually incorrect information.
If the information in the document contains the correct answer, you will give an accurate answer. If
the information in the document does not contain the answer, you will generate 'I can not answer the
question because of the insufficient information in documents.' If there are inconsistencies with the facts
in some of the documents, please generate the response 'There are factual errors in the provided documents.'
and provide the correct answer.""")

USER_TEMPLATE = "Document:\n{DOCS} \n\nQuestion:\n{QUERY}"

SYSTEM_PROMPT_NODOCS = "You are a helpful assistant. Answer the question based on your own knowledge."


def format_docs(docs):
    return "\n".join(doc[:1500] for doc in docs)  # clip to prevent 413s


def build_user_prompt(query, docs, ability):
    if ability == "counterfactual_nodocs":
        return f"Question:\n{query}"
    return USER_TEMPLATE.replace("{DOCS}", format_docs(docs)).replace("{QUERY}", query)


def get_system_prompt(ability):
    return SYSTEM_PROMPT_NODOCS if ability == "counterfactual_nodocs" else SYSTEM_PROMPT_RGB


print("Verbatim RGB prompt ready.")
print(SYSTEM_PROMPT_RGB[:120], "...")
