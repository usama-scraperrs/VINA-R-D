# create tools
import os

# create Agent
import openai
from dotenv import load_dotenv
from ddgs import DDGS
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


# ReAct Agent Loop
def react_loop(question, max_steps=5):
    system_prompt = (
        "You are a reasoning agent. Use Thought and Action steps to solve problems. "
        "Respond in the format:\n"
        "Thought: <your reasoning>\n"
        "Action: <tool>[<input>]\n"
        "Only use the tool 'Search' for now.\n"
        "After an action, wait for an Observation before continuing.\n"
        "Answer: <your final answer> when ready."
    )

    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}"}
    ]

    for step in range(max_steps):
        # LLM responds with Thought and optional Action
        response = llm.chat.completions.create(
            model="meta-llama/Llama-3-70b-chat-hf",
            messages=conversation,
            temperature=0.5
        )
        reply = response.choices[0].message.content
        print(reply)
        conversation.append({"role": "assistant", "content": reply})

        # Check if the model has answered
        if "Answer:" in reply:
            break

        # Look for an action like Action: Search[...]
        if "Action:" in reply:
            import re
            match = re.search(r'Action:\s*(\w+)\[(.+?)\]', reply)
            if match:
                tool, query = match.groups()
                if tool.lower() == "search":
                    observation = search_tool(query.strip())
                    obs_text = f"Observation: {observation}"
                    print(obs_text)
                    conversation.append({"role": "user", "content": obs_text})
                else:
                    print(f"Unknown tool: {tool}")
                    break
            else:
                print("No valid action found.")
                break
        else:
            print("No action requested. Ending.")
            break


def run():
    # Run the ReAct agent
    react_loop("What is the capital of the country with the largest population?")
