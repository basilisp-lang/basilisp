import linecache
import os
from typing import List, Optional

try:
    import pygments.formatters
    import pygments.lexers
    import pygments.styles
except ImportError:  # pragma: no cover

    def _format_source(s: str) -> str:
        return f"{s}{os.linesep}"

else:

    def _get_formatter_name() -> Optional[str]:  # pragma: no cover
        """Get the Pygments formatter name for formatting the source code by
        inspecting various environment variables set by terminals.

        If `BASILISP_NO_COLOR` is set to a truthy value, use no formatting."""
        if os.environ.get("BASILISP_NO_COLOR", "false").lower() in {"1", "true"}:
            return None
        elif os.environ.get("COLORTERM", "") in {"truecolor", "24bit"}:
            return "terminal16m"
        elif "256" in os.environ.get("TERM", ""):
            return "terminal256"
        else:
            return "terminal"

    def _format_source(s: str) -> str:  # pragma: no cover
        """Format source code for terminal output."""
        if (formatter_name := _get_formatter_name()) is None:
            return f"{s}{os.linesep}"
        return pygments.highlight(
            s,
            lexer=pygments.lexers.get_lexer_by_name("clojure"),
            formatter=pygments.formatters.get_formatter_by_name(
                formatter_name, style=pygments.styles.get_style_by_name("emacs")
            ),
        )


def format_source_context(
    filename: str,
    line: int,
    end_line: Optional[int] = None,
    num_context_lines: int = 5,
    show_cause_marker: bool = True,
) -> List[str]:
    """Format source code context with line numbers and identifiers for the affected
    line(s)."""
    assert num_context_lines >= 0

    lines = []

    if not filename.startswith("<") and not filename.endswith(">"):
        cause_range: Optional[range]
        if not show_cause_marker:
            cause_range = None
        elif end_line is not None and end_line != line:
            cause_range = range(line, end_line)
        else:
            cause_range = range(line, line + 1)

        if source_lines := linecache.getlines(filename):
            start = max(0, line - num_context_lines)
            end = min((end_line or line) + num_context_lines, len(source_lines))
            num_justify = max(len(str(start)), len(str(end))) + 1
            for n, source_line in zip(range(start, end), source_lines[start:end]):
                if cause_range is None:
                    line_marker = " "
                elif n + 1 in cause_range:
                    line_marker = " > "
                else:
                    line_marker = "   "

                line_num = str(n + 1).rjust(num_justify)
                lines.append(
                    f"{line_num}{line_marker}| {_format_source(source_line.rstrip())}"
                )

    return lines
