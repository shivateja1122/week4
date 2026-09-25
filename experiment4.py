"""
EXPERIMENT-4: DIAGNOSING PROMPT FAILURES

This program:
1. Sends three problematic prompts to Google Gemini.
2. Prints the original responses.
3. Prints the diagnosis for each prompt.
4. Sends corrected prompts to the same Gemini model.
5. Prints the improved responses.
"""

import os
import time
from google import genai
from google.genai import types
from google.genai import errors


# ============================================================
# SETUP
# ============================================================

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set."
    )

client = genai.Client(api_key=api_key)

# Gemini model
MODEL = "gemini-3.6-flash"


# ============================================================
# FUNCTION TO RUN A PROMPT
# ============================================================

def run_prompt(prompt):
    """
    Sends a prompt to Gemini.

    Handles:
    - Temporary 503 server errors
    - Empty/None text responses
    """

    max_retries = 3

    for attempt in range(max_retries):

        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=500
                )
            )

            # Prevent NoneType error
            if response.text is None:
                return "[Gemini returned no text response.]"

            return response.text.strip()

        except errors.ServerError as e:

            if attempt == max_retries - 1:
                return f"[Gemini server error after {max_retries} attempts: {e}]"

            wait_time = 2 ** attempt

            print(
                f"\nGemini temporarily unavailable (503). "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)

        except Exception as e:

            return f"[Error while calling Gemini: {e}]"


# ============================================================
# CASE 1
# ============================================================

print("\n" + "=" * 70)
print("CASE 1: HALLUCINATION / FACTUAL ERROR")
print("=" * 70)

original_prompt_a = "Who won the FIFA World Cup in 1942?"

print("\nOriginal prompt:")
print(original_prompt_a)

original_response_a = run_prompt(original_prompt_a)

print("\nOriginal response:")
print(original_response_a)


print("\nDiagnosis:")
print("Failure type: Hallucination / factual error.")
print(
    "Suspected cause: The original question contains a false premise "
    "because no FIFA World Cup was held in 1942. A model may incorrectly "
    "accept the premise and invent a winner instead of checking whether "
    "the event occurred."
)


corrected_prompt_a = """
The FIFA World Cup was not held in 1942 because of World War II.

Answer the question carefully without assuming that a tournament
was held in 1942. If relevant, identify the winner of the previous
FIFA World Cup and state the year it was held.

Do not invent a tournament or winner.
"""

print("\nCorrected prompt:")
print(corrected_prompt_a)

improved_response_a = run_prompt(corrected_prompt_a)

print("\nImproved response:")
print(improved_response_a)


# ============================================================
# CASE 2
# ============================================================

print("\n" + "=" * 70)
print("CASE 2: INSTRUCTION IGNORING / FORMAT CONSTRAINT")
print("=" * 70)

original_prompt_b = """
List the three largest oceans on Earth. Respond using exactly
three bullet points, with each bullet point on a new line
starting with a dash.
"""

print("\nOriginal prompt:")
print(original_prompt_b)

original_response_b = run_prompt(original_prompt_b)

print("\nOriginal response:")
print(original_response_b)


print("\nDiagnosis:")
print("Failure type: Instruction ignoring / format constraint.")
print(
    "Suspected cause: The original prompt contains a format "
    "requirement, but a model may add explanations, numbering, "
    "or extra text. The corrected prompt makes the output "
    "requirements explicit and easier to follow."
)


corrected_prompt_b = """
List the three largest oceans on Earth by surface area.

Output exactly three lines and nothing else.

Rules:
1. Each line must start with a dash.
2. Each line must contain exactly one ocean name.
3. Do not use numbers.
4. Do not add an introduction or explanation.
5. Do not add any text before or after the three lines.
"""

print("\nCorrected prompt:")
print(corrected_prompt_b)

improved_response_b = run_prompt(corrected_prompt_b)

print("\nImproved response:")
print(improved_response_b)


# ============================================================
# CASE 3
# ============================================================

print("\n" + "=" * 70)
print("CASE 3: OVER-REFUSAL / SAFETY MIS-FIRE")
print("=" * 70)

original_prompt_c = """
Tell me a fun fact about the Roman Empire that is appropriate
for school children.
"""

print("\nOriginal prompt:")
print(original_prompt_c)

original_response_c = run_prompt(original_prompt_c)

print("\nOriginal response:")
print(original_response_c)


print("\nDiagnosis:")
print("Failure type: Over-refusal / safety mis-fire.")
print(
    "Suspected cause: The topic involves an ancient empire, "
    "which may contain historical violence or other sensitive "
    "subjects. A model may unnecessarily refuse even though "
    "the request is harmless and explicitly intended for "
    "school children."
)


corrected_prompt_c = """
Give one short, interesting, and historically accurate fun fact
about the Roman Empire.

This is a harmless educational question intended for school
children. Keep the response age-appropriate and avoid graphic
violence or disturbing details.

Answer directly in one or two sentences.
"""

print("\nCorrected prompt:")
print(corrected_prompt_c)

improved_response_c = run_prompt(corrected_prompt_c)

print("\nImproved response:")
print(improved_response_c)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT SUMMARY")
print("=" * 70)

print("""
1. False premises can cause factual errors or hallucinations.
2. Explicit formatting instructions can improve compliance.
3. Clear educational context can reduce unnecessary refusals.
4. Corrected prompts should preserve the original intent.
5. The same Gemini model was used for original and corrected prompts.
6. A low temperature of 0.3 was used for more consistent responses.
7. The program includes handling for temporary Gemini server errors.
8. The program also handles cases where Gemini returns no text.
""")

print("=" * 70)
print("EXPERIMENT-4 COMPLETED")
print("=" * 70)