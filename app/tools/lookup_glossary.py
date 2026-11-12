from pathlib import Path

from langchain_core.tools import BaseTool, tool


def create_lookup_glossary_tool(glossary_path: Path) -> BaseTool:
    @tool
    def lookup_glossary(term: str) -> str:
        """Look up a pharmacology term in the local corpus glossary."""
        if not glossary_path.is_file():
            raise FileNotFoundError(f"Glossary file does not exist: {glossary_path}")

        lines = glossary_path.read_text(encoding="utf-8").splitlines()
        matches: list[str] = []
        current: list[str] = []
        for line in lines:
            if line.startswith("- **"):
                if current and term.casefold() in " ".join(current).casefold():
                    matches.append(" ".join(current))
                current = [line.strip()]
            elif current and line.strip() and not line.startswith("#"):
                current.append(line.strip())

        if current and term.casefold() in " ".join(current).casefold():
            matches.append(" ".join(current))
        if not matches:
            return f"No glossary entry found for '{term}'."
        return f"[{glossary_path.name}]\n" + "\n".join(matches)

    return lookup_glossary
