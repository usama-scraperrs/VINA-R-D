# create tools
import os

# create Agent
import openai
from dotenv import load_dotenv
from ddgs import DDGS
import json
import re

load_dotenv()

llm = openai.OpenAI(
    api_key= os.environ.get('TOGETHERAI_API_KEY'),
    base_url='https://api.together.xyz/v1/',
)


# Real search tool using DuckDuckGo
def search_tool(query):
    with DDGS() as duck_search:
        results = list(duck_search.text(query, max_results=3))
    if results:
        return results[0]["body"]
    return "No relevant information found."


def run_calculator(expression: str) -> float:
    allowed_chars = "0123456789+-*/().% "
    if all(c in allowed_chars for c in expression):
        return eval(expression)
    else:
        raise ValueError("Unsafe characters in expression.")

# ReAct Agent Loop
def react_loop(question: str):
    system_prompt = (
        "[INST] You are a tool-using assistant.\n"
        "If a tool is needed, respond with:\n"
        '{\n  "tool": "<tool_name>",\n  "input": "<input string>"\n}\n'
        "Available tools:\n"
        "- SEARCH\n- CALCULATOR\n"
        "Otherwise, just reply normally. [/INST]"
    )

    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}"}
    ]

    response = llm.chat.completions.create(
        model="meta-llama/Llama-3-70b-chat-hf",
        messages=conversation,
        temperature=0.5
    )

    reply = response.choices[0].message.content
    print("LLM reply:", reply)

    try:
        tool_call = json.loads(reply)

        if tool_call.get("tool") == "SEARCH":
            result = search_tool(tool_call["input"])
            print("\n[Tool Output]:", result)

            # Send result back to LLM
            conversation.append({"role": "assistant", "content": reply})
            conversation.append({"role": "user", "content": f"[Search result]: {result}"})

        if tool_call.get("tool") == "CALCULATOR":
            result = run_calculator(tool_call["input"])
            print("\n[Tool Output]:", result)

            # Send result back to LLM
            conversation.append({"role": "assistant", "content": reply})
            conversation.append({"role": "user", "content": f"[calculation result]: {result}"})

        followup = llm.chat.completions.create(
            model="meta-llama/Llama-3-70b-chat-hf",
            messages=conversation,
            temperature=0.5
        )
        print("\nFinal answer:", followup.choices[0].message.content)
    except json.JSONDecodeError:
        print("Direct answer:", reply)


def run(query: str):
    # Run the ReAct agent
    react_loop(query)
