<h1>
<p align="center">
  <img src="assets/logo.svg" alt="Phantom" width="80">
  <br>phantom
</h1>
  <p align="center">
    Sandboxed data analysis with LLMs (powered by DuckDB).
    <br><br>
    <a href="https://github.com/James-Wirth/phantom/actions/workflows/ci.yml"><img src="https://github.com/James-Wirth/phantom/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  </p>
</p>

Phantom is a Python framework for LLM-assisted data analysis. The LLM doesn't need to see the actual data. Phantom reasons with opaque **semantic references** (`@a3f2`), writes SQL, and executes the queries locally in a sandboxed [DuckDB](https://duckdb.org/) engine.

## Quick Start

```bash
pip install git+https://github.com/James-Wirth/phantom.git
```

```python
import phantom

session = phantom.Session(allowed_dirs=["./data"])

chat = phantom.Chat(
    session,
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    system="You are an astrophysicist. Data files are in ./data/.",
)
response = chat.ask(
    "Which habitable-zone exoplanets are closest to Earth, "
    "and what kind of stars do they orbit?"
)
```

## How It Works

Here's an example trace where Claude analyses two CSV files (20 planets, 18 host stars) to answer: *"Which habitable-zone exoplanets are closest to Earth, and what kind of stars do they orbit?"*

| Turn | Tool call | Result |
|:----:|:----------|:-------|
| 1 | `read_csv("planets.csv")` | `@6a97` |
| | `read_csv("stars.csv")` | `@cc35` |
| 2 | `peek(@6a97)` | columns: `[name, host_star, distance_ly, ...]` |
| | `peek(@cc35)` | columns: `[name, spectral_type, mass_solar, ...]` |
| 3 | `query("SELECT p.name, p.distance_ly, s.spectral_type, ... FROM planets p JOIN stars s ON p.host_star = s.name WHERE p.in_habitable_zone ORDER BY p.distance_ly", refs={planets: @6a97, stars: @cc35})` | `@f127` |
| 4 | `export(@f127)` | `[{name: "Proxima Cen b", distance_ly: 4.2, spectral_type: "M"}, ...]` |


Refs compose into a **lazy DAG**: shared subgraphs are resolved once and cached. The SQL engine is DuckDB, so JOINs, window functions, CTEs, and aggregations all work out of the box.

## Example Output

From the same exoplanet example, Claude's answer (abridged):

> **Closest habitable-zone exoplanets:**
>
> | Planet | Distance | Star | Spectral type |
> |:-------|:---------|:-----|:--------------|
> | Proxima Cen b | 4.2 ly | Proxima Cen | M-dwarf (3,042 K) |
> | Ross 128 b | 11 ly | Ross 128 | M-dwarf (3,192 K) |
> | Teegarden b | 12 ly | Teegarden | M-dwarf (2,904 K) |
> | TRAPPIST-1 e/f/g | 40 ly | TRAPPIST-1 | M-dwarf (2,566 K) |
>
> The nearest habitable-zone candidates overwhelmingly orbit **M-dwarf** stars — small, cool, and the most common type in the galaxy.

## LLM Providers

Built-in support for **Anthropic**, **OpenAI**, and **Google Gemini**:

```bash
pip install "phantom[anthropic]"
pip install "phantom[openai]"
pip install "phantom[google]"
```

```python
chat = phantom.Chat(
    session, 
    provider="anthropic", 
    model="claude-sonnet-4-20250514"
)
chat = phantom.Chat(
    session, 
    provider="openai", 
    model="gpt-4o"
)
chat = phantom.Chat(
    session, 
    provider="google", 
    model="gemini-2.0-flash"
)
```

Any **OpenAI-compatible** API (Groq, Together, Fireworks, Ollama, vLLM, ...) works via `base_url`:

```python
chat = phantom.Chat(
    session,
    provider=phantom.OpenAIProvider(
        api_key="...",
        base_url="https://api.groq.com/openai/v1",
    ),
    model="llama-3.1-70b-versatile",
)
```

## The Chat Interface

`phantom.Chat` manages the tool-call loop, message history, and ref tracking:

```python
r1 = chat.ask(
    "Which habitable-zone exoplanets have a mass under 5 Earth masses?"
)
print(r1.text)
print(r1.tool_calls_made)
print(r1.usage.total_tokens)

r2 = chat.ask("Of those, which orbit M-dwarf stars?")
...
```

## Customization

### Custom Operations

Register domain-specific operations alongside the built-in ones:

```python
session = phantom.Session(allowed_dirs=["./data"])

@session.op
def fetch_lightcurve(target: str) -> dict:
    """Fetch a lightcurve from the MAST archive."""
    return mast_api.query(target)

@session.op
def blackbody_flux(temperature_k: float, wavelength_nm: float) -> float:
    """Compute spectral radiance via Planck's law."""
    ...
```

### Inspectors

Define what the LLM sees when it peeks at custom data types:

```python
import pandas as pd

@session.inspector(pd.DataFrame)
def inspect_dataframe(df: pd.DataFrame) -> dict:
    return {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "sample": df.head(3).to_dict("records"),
    }
```

### Operation Sets

Group related operations into reusable modules:

```python
from phantom import OperationSet

spectral_ops = OperationSet()

@spectral_ops.op
def classify_star(temperature_k: float, luminosity_solar: float) -> str:
    """Classify a star on the Hertzsprung-Russell diagram."""
    ...

session.register(spectral_ops)
```

### Manual Tool-Call Loop

For full control over the LLM interaction:

```python
response = client.chat.completions.create(
    model="gpt-4o", 
    tools=session.get_tools(), 
    messages=[
        {
            "role": "user", 
            "content": "Find the closest habitable-zone planets"
        }
    ]
)
for tool_call in response.choices[0].message.tool_calls:
    result = session.handle_tool_call(
        tool_call.function.name,
        tool_call.function.arguments,
    )
    # result.to_json() → send back to LLM
```

## Session

```python
session = phantom.Session(
    allowed_dirs=["./data"],    # restrict file access
    max_file_bytes=10_000_000,  # limit individual file sizes
    output_format="dicts",      # default export format
)
```

Resolve asynchronously with parallel execution:

```python
result = await session.aresolve(habitable, parallel=True)
```

Save and reload pipelines:

```python
session.save_graph(habitable, "pipeline.json")
loaded = session.load_graph("pipeline.json")
```