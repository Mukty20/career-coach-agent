"""
Career Coach Agent - core agent loop (Groq-backed).

Includes token-budget trimming since Groq's free tier caps at 8000 TPM:
- CV text truncated to a reasonable length
- RAG context limited to top 2 chunks
- Conversation history limited to last 6 messages
- max_tokens for responses reduced
"""

import json
from datetime import date
from groq import Groq
from tools.web_search import search_web
from tools.rag import CareerKnowledgeBase

MODEL = "openai/gpt-oss-120b"

MAX_CV_CHARS = 1500          # ~375 tokens
MAX_RAG_CHUNKS = 2           # fewer knowledge base chunks per turn
MAX_HISTORY_MESSAGES = 6     # only send the last few turns, not the whole conversation
MAX_RESPONSE_TOKENS = 700    # cap the response size too

SYSTEM_PROMPT = f"""You are a Career Coach Agent for Nigerian tech students and early-career \
professionals. Today's date is {date.today().strftime('%B %d, %Y')}. You help users with career \
guidance, CV/resume feedback, skill-gap analysis, and job search strategy, grounded in the \
Nigerian job market context.

Guidelines:
- Be specific and actionable, not generic.
- When the user has uploaded a CV, reference specific details from it in your advice.
- When you have retrieved knowledge base context, use it to ground your answer, but \
don't just repeat it verbatim.
- Use the web_search tool when the user asks about current job openings, recent salary \
data, or anything time-sensitive.
- Keep responses concise - a few short paragraphs or bullet points, not walls of text.
- Never state a year for "current" salary or job market data unless it comes directly \
from a web_search result - describe figures as general estimates otherwise.

CRITICAL - GROUNDING RULE FOR WEB SEARCH RESULTS:
- NEVER invent specific company names, job titles, salary figures, contact emails, or \
application links that do not literally appear in the web_search tool results.
- If results are generic (e.g. "17 jobs available on Indeed"), say so plainly rather than \
inventing specifics.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information such as job listings, salary "
                "benchmarks, in-demand skills, or company info."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, e.g. 'entry level React developer jobs Lagos 2026'",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


class CareerCoachAgent:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.kb = CareerKnowledgeBase()

    def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        if tool_name == "web_search":
            return search_web(tool_args["query"])
        return f"Unknown tool: {tool_name}"

    def respond(self, conversation_history: list[dict], cv_context: str = "") -> str:
        # Trim conversation history to keep token usage bounded
        trimmed_history = conversation_history[-MAX_HISTORY_MESSAGES:]

        last_user_msg = next(
            (m["content"] for m in reversed(trimmed_history) if m["role"] == "user"),
            "",
        )
        rag_context = self.kb.format_context(last_user_msg, top_k=MAX_RAG_CHUNKS)

        # Truncate CV text so it doesn't dominate the token budget
        truncated_cv = cv_context[:MAX_CV_CHARS] if cv_context else ""

        system_prompt = SYSTEM_PROMPT
        if truncated_cv:
            system_prompt += f"\n\n--- USER'S CV CONTENT (truncated) ---\n{truncated_cv}\n--- END CV ---"
        if rag_context:
            system_prompt += f"\n\n--- RELEVANT KNOWLEDGE BASE CONTEXT ---\n{rag_context}\n--- END CONTEXT ---"

        messages = [{"role": "system", "content": system_prompt}]
        messages += [{"role": m["role"], "content": m["content"]} for m in trimmed_history]

        while True:
            response = self.client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_RESPONSE_TOKENS,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )

            choice = response.choices[0]
            message = choice.message

            if choice.finish_reason == "tool_calls" and message.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in message.tool_calls
                    ],
                })

                for tc in message.tool_calls:
                    tool_args = json.loads(tc.function.arguments)
                    result = self._execute_tool(tc.function.name, tool_args)
                    # Truncate tool results too, web search can return a lot of text
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result[:2000],
                    })
                continue

            return message.content or ""
