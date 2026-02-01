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

Phantom is a Python framework for building LLM-assisted workflows. The LLMs don't need to see your data. They can reason with opaque semantic references (`@a3f2`), from which Phantom constructs a lazy computation graph using your registered operations. The graph execution is deterministic and runs locally on your machine.

## Quick Start

```bash
pip install git+https://github.com/James-Wirth/phantom.git
```

```python
import phantom
import pandas as pd

session = phantom.Session()

@session.op
def load(name: str) -> pd.DataFrame:
    """Load a dataset. Available: neo_survey, missions, launches."""
    return pd.read_parquet(DATA_DIR / f"{name}.parquet")

@session.op
def query(df: phantom.Ref[pd.DataFrame], expr: str) -> pd.DataFrame:
    """Filter rows with a pandas query expression."""
    return df.query(expr)

chat = phantom.Chat(
    session,
    provider=phantom.AnthropicProvider(api_key="sk-ant-..."),
    model="claude-sonnet-4-20250514",
)
response = chat.ask("Which potentially hazardous near-Earth asteroids have an orbital period under 200 days?")
```

## How It Works

Phantom creates a **symbolic layer** between the LLM and your data. Each operation returns an opaque reference like `@a3f2` instead of real data. 

| Step | LLM Tool Call | Response |
|:----:|-----------|----------|
| 1 | `load(name="neo_survey")` | `@a3f2` |
| 2 | `peek(ref="@a3f2")` | `shape: [32000, 9], columns: [name, est_diameter_km, hazardous, ...]` |
| 3 | `query(df="@a3f2", expr="hazardous == True & orbital_period < 200")` | `@b4c3` |
| 4 | `peek(ref="@b4c3")` | `shape: [47, 9], sample: [{name: "2015 FP345", ...}]` |

The LLM orchestrated a load, filter, and inspection across 32,000 asteroid records. Since refs form a DAG, shared subgraphs are resolved once and cached (i.e. no redundant computation).

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
r1 = chat.ask("Which exoplanets in the habitable zone have a mass under 5 Earth masses?")
print(r1.text)
print(r1.tool_calls_made)
print(r1.usage.total_tokens)

r2 = chat.ask("Of those, which orbit K-type stars?")
```

## Contrib Modules

Pre-built operation sets for common data libraries:

```bash
pip install "phantom[pandas]"
pip install "phantom[polars]"
pip install "phantom[duckdb]"
pip install "phantom[all]"
```

```python
from phantom.contrib.pandas import pandas_ops

session = phantom.Session()
session.register(pandas_ops) # adds common Pandas operations

chat = phantom.Chat(session, system="You are an astrophysicist.")
response = chat.ask("What's the median orbital eccentricity of confirmed exoplanets discovered by Kepler?")
```

## Customization

### Operations and Inspectors

**Operations** are Python functions decorated with `@session.op`. Docstrings become LLM tool descriptions. Parameters typed as `Ref[T]` are resolved automatically (i.e. you can just write normal Python).

```python
@session.op
def merge(left: phantom.Ref[pd.DataFrame], right: phantom.Ref[pd.DataFrame], on: str) -> pd.DataFrame:
    """Merge two DataFrames on a shared column."""
    return pd.merge(left, right, on=on)
```

**Inspectors** define what the LLM sees when it peeks at data. 

```python
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

messages = [{"role": "user", "content": "Analyze sales by region"}]

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
session = phantom.Session()

# Lazy evaluation
ref = session.ref("load_csv", path="data.csv")
result = session.resolve(ref)

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