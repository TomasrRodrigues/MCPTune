"""Local lookup rules for offline semantic argument generation.

Each rule matches a parameter by name substring and/or schema `format`
value, and produces a plausible value. Rules are evaluated in order;
the first match wins. Format-based rules come first since they're
more specific than name-based heuristics.

The `examples` field in a schema overrides all rule matching - if a
schema author provided examples, we use them.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LookupRule:
    name_keywords: tuple[str, ...]
    format_keywords: tuple[str, ...]
    value: Any

    def matches(self, name_lower: str, fmt: str) -> bool:
        fmt_lower = fmt.lower() if fmt else ""
        if self.format_keywords and any(kw in fmt_lower for kw in self.format_keywords):
            return True
        if self.name_keywords and any(kw in name_lower for kw in self.name_keywords):
            return True
        return False


DEFAULT_RULES: tuple[LookupRule, ...] = (
    # Format-driven (most specific first)
    LookupRule(("email",), ("email",), "test@example.com"),
    LookupRule(("url", "uri", "link"), ("uri", "url"), "https://example.com"),
    LookupRule(("datetime", "timestamp"), ("date-time",), "2024-01-15T10:30:00Z"),
    LookupRule(("date",), ("date",), "2024-01-15"),
    LookupRule(("time",), ("time",), "10:30:00"),
    LookupRule(("uuid",), ("uuid",), "550e8400-e29b-41d4-a716-446655440000"),
    LookupRule(("ipv4", "ip_address"), ("ipv4",), "192.168.1.1"),
    LookupRule(("ipv6",), ("ipv6",), "2001:db8::1"),
    LookupRule(("hostname",), ("hostname",), "example.com"),
    # Location / address
    LookupRule(("city", "location"), (), "Lisbon"),
    LookupRule(("country",), (), "Portugal"),
    LookupRule(("region", "state", "province"), (), "Lisbon District"),
    LookupRule(("address",), (), "Rua Augusta 100"),
    LookupRule(("zipcode", "postal_code", "postcode"), (), "1100-053"),
    LookupRule(("timezone", "tz"), (), "Europe/Lisbon"),
    # People
    LookupRule(("first_name", "given_name"), (), "Alex"),
    LookupRule(("last_name", "surname", "family_name"), (), "Silva"),
    LookupRule(("username",), (), "alex_silva"),
    LookupRule(("name",), (), "Alex Silva"),
    LookupRule(("phone", "mobile", "telephone"), (), "+351912345678"),
    # Identifiers / tokens
    LookupRule(("token", "api_key", "apikey"), (), "sk-example-1234567890"),
    LookupRule(("id",), (), "abc-123"),
    # Files / paths
    LookupRule(("filepath", "file_path", "path"), (), "documents/report.txt"),
    LookupRule(("filename",), (), "report.txt"),
    LookupRule(("extension", "filetype"), (), "txt"),
    # Misc strings
    LookupRule(("query", "search"), (), "example query"),
    LookupRule(("language", "lang", "locale"), (), "en"),
    LookupRule(("currency",), (), "EUR"),
    LookupRule(("description", "summary", "comment"), (), "A short example description."),
    LookupRule(("title", "subject"), (), "Example Title"),
    LookupRule(("message", "content", "body"), (), "Example content."),
)


def lookup_value(
    name: str,
    schema: dict,
    rules: tuple[LookupRule, ...] = DEFAULT_RULES,
) -> Any | None:
    """Find a plausible value for a parameter based on its name and schema.

    Returns None if no rule matches; callers should omit such parameters
    from the result so the structural sampler can handle them.
    """
    examples = schema.get("examples")
    if isinstance(examples, list) and examples:
        return examples[0]

    name_lower = name.lower()
    fmt = schema.get("format", "")

    for rule in rules:
        if rule.matches(name_lower, fmt):
            return rule.value

    return None
