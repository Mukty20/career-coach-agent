"""Web search tool using ddgs (free, no API key required)."""

from ddgs import DDGS


def search_web(query: str, max_results: int = 5) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return f"No search results found for: {query}"

        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            body = r.get("body", "")
            url = r.get("href", "")
            formatted.append(f"{i}. {title}\n   {body}\n   Source: {url}")

        return "\n\n".join(formatted)

    except Exception as e:
        return f"ERROR: Web search failed - {str(e)}"
