import json
import time
import os
import logging
import sys

def final_answer(question, responses, api_handler):

    prompt = f"""You are an expert AI assistant tasked with responding to a question from a user. Multiple expertes already considered the problem. Your goal is to provide a concise yet comprehensive response that captures the key points from their answer. Please structure your response starting with a clear, single, well defined conclusive answer. AS A SINGLE PARAGRAPH. Then proceed with explanations on a separated single paragraph. MAKE THE RESPONSE YOUR OWN: USE I AND NOT WE OR THEY.

Here is the original question:

{question}

Here are the responses from other experts:

{responses}

Please provide your response in a clear, well-structured format. Do not respond in JSON, respond in plain text. Single paragraph. The experts are part of an internal process, do not mention them.

If the question provides specific instructions, follow them thouroughly.
"""

    messages = [
        {
            "role": "system",
            "content": "You are an expert AI assistant expert in providing useful responses the user questions.",
        },
        {
            "role": "user",
            "content": prompt
        },
    ]

    logging.debug("MESSAGES")
    logging.debug(messages)
    logging.debug("")

    summary_data = api_handler.make_api_call(messages, 2000, is_final_answer=False, is_json_content=False, temperature=0)
    logging.debug("SUMMARY DATA")
    logging.debug(summary_data)
    logging.debug("")

    return summary_data

def generate_response(prompt, api_handler):
    messages = [
        {
            "role": "system",
            "content": """You are playing 2 expert AI assistants that explain their reasoning step by step. For each step, they provide a title that describes what they are doing in that step, along with the content formatted as a discussion between Expert 1 and Expert 2, as pure text in a single line. Then you decide if they need another step or if they are ready to give the final answer. Final answer have to be well detailled and formatted as pure text on a single line. Final answer should also provide a quick summary of important steps and assumptions taken. Respond in JSON format with 'title', 'content', and 'next_action' (either 'continue' or 'final_answer') keys. BE CAREFUL TO PROPERLY ESCAPE JSON. USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES. YOUR FINAL RESPONSE SHOULD REFLECT BOTH OPINIONS. EXPERTS ARE CRITICAL OF EACH OTHER ANSWERS. BE CRITICAL OF ANY UNPROVEN ASSUMPTIONS. JSON SHOULD BE VALID.""",
        },
        {"role": "user", "content": prompt},
        {
            "role": "assistant",
            "content": "Thank you! We will now debate following your instructions, starting at the beginning after decomposing the problem.",
        },
    ]

    steps = []
    step_count = 1
    total_thinking_time = 0

    while True:
        start_time = time.time()
        step_data = api_handler.make_api_call(messages, 300)
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time

        # Add error handling for missing 'title' key
        step_title = step_data.get('title', 'Untitled Step')
        step_content = step_data.get('content', 'No content provided')
        next_action = step_data.get('next_action', 'continue').lower().strip()

        steps.append(
            (
                f"Step {step_count}: {step_title}",
                step_content,
                thinking_time,
            )
        )

        messages.append({"role": "assistant", "content": json.dumps(step_data)})
        logging.debug("Next reasoning step: %s", next_action)
        if next_action == "final_answer":
            break

        step_count += 1

        yield steps, None

    messages.append(
        {
            "role": "user",
            "content": "Please provide the final answer based on your reasoning above.",
        }
    )

    start_time = time.time()
    final_data = api_handler.make_api_call(messages, 2000, is_final_answer=True)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time

    steps.append(("Final Answer", final_data["content"], thinking_time))

    yield steps, total_thinking_time


def load_env_vars():
    return {
        "OLLAMA_URL": os.getenv("OLLAMA_URL", "http://localhost:11434"),
        "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL", "llama3.1:70b"),
        "GROQ_MODEL": os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"),
        "PERPLEXITY_API_KEY": os.getenv("PERPLEXITY_API_KEY"),
        "PERPLEXITY_MODEL": os.getenv(
            "PERPLEXITY_MODEL", "llama-3.1-sonar-small-128k-online"
        ),
    }

def setup_logging(level=logging.INFO, stream=sys.stderr):
    logging.basicConfig(level=level, format='%(message)s', stream=stream)

