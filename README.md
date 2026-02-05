<h1>
<p align="center">
  <img src="assets/logo.svg" alt="Phantom" width="80">
  <br>phantom
</h1>
  <p align="center">
    Semantic references and lazy DAGs for LLM-assisted workflows.
    <br><br>
    <a href="https://github.com/James-Wirth/phantom/actions/workflows/ci.yml"><img src="https://github.com/James-Wirth/phantom/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  </p>
</p>

Phantom is a Python framework for building LLM-assisted data workflows. The LLMs don't need to see your data. They reason with opaque semantic references (`@a3f2`), from which Phantom constructs a lazy computation graph backed by DuckDB. Graph execution is deterministic and runs locally on your machine.

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
    "Which potentially hazardous near-Earth asteroids "
    "have an orbital period under 200 days?"
)
```

## How It Works

Phantom creates a **symbolic layer** between the LLM and your data. Each operation returns an opaque reference like `@a3f2` instead of real data. SQL is the primary query language, powered by DuckDB.

| Step | LLM Tool Call | Response |
|:----:|-----------|----------|
| 1 | `read_csv(path="neo_survey.csv")` | `@a3f2` |
| 2 | `peek(ref="@a3f2")` | `columns: {name: VARCHAR, hazardous: BOOLEAN, orbital_period: DOUBLE, ...}, sample: [...]` |
| 3 | `query(sql="SELECT * FROM neo WHERE hazardous AND orbital_period < 200", refs={"neo": "@a3f2"})` | `@b4c3` |
| 4 | `peek(ref="@b4c3")` | `columns: {name: VARCHAR, ...}, sample: [{name: "2015 FP345", ...}]` |

The LLM loaded a CSV, wrote SQL to filter hazardous asteroids with short orbital periods, and inspected the results. Since refs form a DAG, shared subgraphs are resolved once and cached.

## LLM Providers

Phantom has built-in support for **Anthropic**, **OpenAI**, and **Google Gemini**. Install the one you need:

```bash
pip install "phantom[anthropic]"
pip install "phantom[openai]"
pip install "phantom[google]"
```

Pass API keys directly or set the standard environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`):

```python
chat = phantom.Chat(
    session,
    provider=phantom.AnthropicProvider(api_key="sk-ant-..."),
    model="claude-sonnet-4-20250514",
)

chat = phantom.Chat(
    session,
    provider=phantom.OpenAIProvider(api_key="sk-..."),
    model="gpt-4o",
)

chat = phantom.Chat(
    session,
    provider=phantom.GoogleProvider(api_key="..."),
    model="gemini-2.0-flash",
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

`phantom.Chat` is the high-level interface for multi-turn LLM conversations. It handles the tool-call loop, message history, and ref tracking:

```python
r1 = chat.ask(
    "Which exoplanets in the habitable zone "
    "have a mass under 5 Earth masses?"
)
print(r1.text)
print(r1.tool_calls_made)
print(r1.usage.total_tokens)

r2 = chat.ask("Of those, which orbit K-type stars?")
```

## Customization

### Custom Operations

You can add your own operations alongside the built-in ones:

```python
session = phantom.Session(allowed_dirs=["./data"])

@session.op
def fetch_lightcurve(target: str) -> dict:
    """Fetch a lightcurve from the MAST archive."""
    return mast_api.query(target)
```

### Inspectors

**Inspectors** define what the LLM sees when it peeks at data.

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

analytics_ops = OperationSet()

@analytics_ops.op
def rolling_average(data: phantom.Ref[list], window: int) -> list:
    """Compute a rolling average."""
    ...

session.register(analytics_ops)
```

### Manual Tool-Call Loop

Phantom is designed to be hackable. If you need full control over the LLM interaction:

```python
tools = session.get_tools()

messages = [{"role": "user", "content": "Find the closest hazardous asteroids"}]

response = client.chat.completions.create(
    model="gpt-4o",
    tools=tools,
    messages=messages
)

for tool_call in response.choices[0].message.tool_calls:
    result = session.handle_tool_call(
        tool_call.function.name,
        tool_call.function.arguments
    )
    # result.to_json() → send back to LLM
```

### Session Features

```python
session = phantom.Session(
    allowed_dirs=["./data"],   # restrict file access
    max_file_bytes=10_000_000, # limit file sizes
    output_format="dicts",     # default export format
)

# Lazy evaluation
ref = session.ref("read_csv", path="neo_survey.csv")
result = session.resolve(ref)

# SQL queries with refs as virtual tables
neo = session.ref("read_csv", path="neo_survey.csv")
missions = session.ref("read_csv", path="missions.csv")
targeted = session.ref(
    "query",
    sql="SELECT n.name, m.mission FROM neo n "
        "JOIN missions m ON n.name = m.target "
        "WHERE n.hazardous",
    refs={"neo": neo, "missions": missions},
)

# Async resolution with parallel execution
result = await session.aresolve(ref, parallel=True)

# Save and replay pipelines
session.save_graph(ref, "pipeline.json")
loaded = session.load_graph("pipeline.json")

# Rich error context
try:
    session.resolve(bad_ref)
except phantom.ResolutionError as e:
    print(e.chain)  # ['@a3f2', '@b4c3'] — trace the failure path
```