from __future__ import annotations

DOI_TRAILING_CHARS = ")]}>,.;:\"'"


def normalize_doi(raw: str) -> str:
    value = raw.strip()
    lowered = value.casefold()
    if lowered.startswith("https://doi.org/"):
        value = value[len("https://doi.org/") :]
    elif lowered.startswith("http://doi.org/"):
        value = value[len("http://doi.org/") :]
    elif lowered.startswith("https://dx.doi.org/"):
        value = value[len("https://dx.doi.org/") :]
    elif lowered.startswith("http://dx.doi.org/"):
        value = value[len("http://dx.doi.org/") :]

    if value.casefold().startswith("doi:"):
        value = value.split(":", 1)[1]

    value = value.strip()
    while value and value[-1] in DOI_TRAILING_CHARS:
        value = value[:-1]
    return value.casefold().strip()
