# Phantom

[![CI](https://github.com/James-Wirth/phantom/actions/workflows/ci.yml/badge.svg)](https://github.com/James-Wirth/phantom/actions/workflows/ci.yml)

**Let LLMs orchestrate data pipelines without seeing the data.**

The LLM reasons with opaque references (`@a3f2`), while your code works with actual values in memory. A lazy computation graph connects them.

## The Problem

When an LLM needs to analyze data, you have two bad options:

1. **Send the data to the LLM**: wastes tokens, hits context limits, leaks sensitive information
2. **Hard-code the pipeline**: no flexibility, defeats the purpose of using an LLM

Phantom offers a third way: let the LLM build the pipeline symbolically, then execute it.

## How It Works

| LLM calls tool | You return |
|----------------|------------|
| `load("sales.parquet")` | `{"ref": "@a3f2"}` |
| `load("products.parquet")` | `{"ref": "@b4c3"}` |
| `join("@a3f2", "@b4c3", on="product_id")` | `{"ref": "@c5d6"}` |
| `peek("@c5d6")` | `{"shape": [8420, 5], "columns": ["region", "product", ...]}` |
| `compute("@c5d6", "quantity * price")` | `{"ref": "@d7e8"}` |
| `group_by("@d7e8", by="region")` | `{"ref": "@e9f0"}` |
| `phantom.resolve("@e9f0")` | DataFrame (executes full graph) |

The LLM never sees the data itself, just refs. `peek` resolves immediately so the LLM can inspect structure and decide what to do next.

## Usage

### 1. Define operations

```python
import phantom
import pandas as pd

@phantom.op
def load(path: str) -> pd.DataFrame:
    """Load a dataset from disk."""
    return pd.read_parquet(path)

@phantom.op
def query(data: phantom.Ref[pd.DataFrame], condition: str) -> pd.DataFrame:
    """Filter rows matching a condition."""
    return data.query(condition)

@phantom.op
def group_by(data: phantom.Ref[pd.DataFrame], columns: list[str], agg: dict) -> pd.DataFrame:
    """Group by columns and aggregate."""
    return data.groupby(columns).agg(agg).reset_index()
```

When a parameter is typed as `Ref[T]`, Phantom resolves it before calling your function. You receive the actual `DataFrame`, not the semantic ref.

### 2. Wire up your LLM

```python
tools = phantom.get_openai_tools()

while not done:
    response = client.chat.completions.create(model="gpt-4o", messages=messages, tools=tools)

    for tool_call in response.choices[0].message.tool_calls:
        result = phantom.handle_tool_call(
            tool_call.function.name,
            json.loads(tool_call.function.arguments)
        )
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result.to_dict())
        })

# When the LLM is done, execute the final pipeline
data = phantom.resolve(result.ref)
```

`handle_tool_call` handles everything uniformly. Lazy operations return a ref, while `peek` resolves immediately so the LLM can inspect column names, types, and sample rows.

## Key Features

**Lazy evaluation**: Nothing runs until you call `resolve()`.

**Session isolation**: Run multiple conversations without ref collisions.
```python
session = phantom.Session("user_123")
ref = session.ref("load", path="data.parquet")
result = session.resolve(ref)
```

**Data inspection**: Let the LLM peek at structure without loading everything.
```python
info = phantom.peek(ref)
# {"ref": "@a3f2", "type": "dataframe", "shape": [10000, 5], "columns": {...}, "sample": [...]}
```

**Graph serialization**: Save pipelines and replay them later.
```python
phantom.save_graph(ref, "analysis.json")
loaded = phantom.load_graph("analysis.json")
```

**Rich errors**: When resolution fails, see the full ref chain.
```python
except ResolutionError as e:
    print(e.chain)  # ['@a3f2', '@b4c3', '@d5e4']
```
