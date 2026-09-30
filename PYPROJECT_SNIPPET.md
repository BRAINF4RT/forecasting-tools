# pyproject.toml changes (poetry layout)

Open pyproject.toml and mirror however the existing `stats` / `agents` extras are declared.
For a Poetry-style file that looks like this:

```toml
[tool.poetry.dependencies]
# ...existing...
ddgs = { version = "^9.0.0", optional = true }
trafilatura = { version = "^2.0.0", optional = true }
beautifulsoup4 = { version = "^4.14.0", optional = true }

[tool.poetry.extras]
# ...existing extras...
free-search = ["ddgs", "trafilatura", "beautifulsoup4"]
# and append "ddgs", "trafilatura", "beautifulsoup4" to the existing `all` extra
```

If the file uses PEP 621 instead, add a `free-search` list under `[project.optional-dependencies]`
with the same three packages and append them to `all`. Then run `poetry lock` (or `poetry lock --no-update`)
and `poetry install --all-extras`.

`requests` is already pulled in by litellm; no change needed for it.
