def build_prompt(
    user_message,
    style_examples
):
    style_block = "\n".join(f"- {ex}" for ex in style_examples)

    prompt = f"""You are analyzing how someone typically writes.

Based on the examples below, infer the writing style.
Do NOT reuse topics, facts, names, or specific phrases from the examples.

Examples:
{style_block}

First, summarize the writing style in a few bullet points.
Then write a SHORT reply (1–2 sentences maximum) to the message below,
using ONLY the inferred style.

Do not add extra explanation or context.

Message:
{user_message}

Reply:
"""
    return prompt
