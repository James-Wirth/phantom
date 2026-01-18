<p align="center">
  <img src="assets/logo.svg" alt="Phantom" width="400">
</p>

---

<h3 align="center">Let LLMs orchestrate data pipelines without seeing the data.</h3>

<p align="center">
  <a href="https://github.com/James-Wirth/phantom/actions/workflows/ci.yml"><img src="https://github.com/James-Wirth/phantom/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
</p>

Phantom creates a symbolic layer between your LLM and your data. The LLM reasons with opaque references (`@a3f2`) and builds a computation graph, while your code holds the actual data in memory. When you're ready, `resolve()` executes the graph and returns the result.

## The Problem

When building LLM-powered data analysis tools, you face a dilemma:

1. **Send the data to the LLM**. Wastes tokens, hits context limits, exposes sensitive information
2. **Hard-code the pipeline**. No flexibility, defeats the purpose of using an LLM

Phantom offers a third way: the LLM builds the pipeline symbolically and your code executes it.

## Installation

Install directly from GitHub:
```bash
pip install git+https://github.com/James-Wirth/phantom.git
```

Or clone and install in development mode:
```bash
git clone https://github.com/James-Wirth/phantom.git
cd phantom
pip install -e .
```

## Defining Operations

Operations are regular Python functions decorated with `@phantom.op`. Your function's docstring becomes the tool description for the LLM.

```python
import pandas as pd
import phantom

@phantom.op
def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file as a DataFrame."""
    return pd.read_csv(path)

@phantom.op
def merge(left: phantom.Ref[pd.DataFrame], right: phantom.Ref[pd.DataFrame], on: str) -> pd.DataFrame:
    """Merge two DataFrames on a key column."""
    return left.merge(right, on=on)

@phantom.op
def groupby_agg(df: phantom.Ref[pd.DataFrame], by: list[str], agg: dict) -> pd.DataFrame:
    """Group by columns and aggregate. Example: agg={"profit": "sum"}"""
    return df.groupby(by, as_index=False).agg(agg)

@phantom.op
def sort_values(df: phantom.Ref[pd.DataFrame], by: str, ascending: bool = True) -> pd.DataFrame:
    """Sort DataFrame by column."""
    return df.sort_values(by, ascending=ascending)
```

The key insight: **parameters typed as `Ref[T]` are automatically resolved** before your function runs. You write normal pandas code that receives normal DataFrames.

## Wiring Up Your LLM

Phantom auto-generates tool definitions from your operations and handles tool calls uniformly:

```python
import json
import openai
import phantom

client = openai.OpenAI()
tools = phantom.get_tools()  # or format="anthropic" for Claude  

messages = [{"role": "user", "content": "What's our most profitable segment by region?"}]

while True:
    response = client.chat.completions.create(
        model="gpt-4o",
        tools=tools,
        messages=messages,
    )
    message = response.choices[0].message

    if message.tool_calls:
        messages.append(message)
        for tool_call in message.tool_calls:
            result = phantom.handle_tool_call(
                tool_call.function.name,
                json.loads(tool_call.function.arguments)
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result.to_dict()),
            })
    else:
        break

# Execute the pipeline the LLM built
final_data = phantom.resolve(result.ref)
```

`handle_tool_call` returns refs for lazy operations and immediate results for `peek`. The LLM sees just enough to make decisions.

## Example: AI Data Analyst

Imagine a user asks: *"What's our most profitable customer segment?"*

The LLM doesn't need to see 10,000 rows of data to answer this. It needs to know the schema, understand the relationships, and compose the right transformations. Here's how it works with Phantom:

**ANALYST:** I'll load the datasets and explore their structure.

| Tool Call | Response |
|-----------|----------|
| `load_csv(path="orders.csv")` | `{"ref": "@a3f2", "op": "load_csv", ...}` |
| `load_csv(path="customers.csv")` | `{"ref": "@b4c3", "op": "load_csv", ...}` |
| `peek(ref="@a3f2")` | `{"type": "dataframe", "shape": [10000, 4], "columns": ["order_id", "customer_id", "quantity", "profit"]}` |
| `peek(ref="@b4c3")` | `{"type": "dataframe", "shape": [500, 3], "columns": ["customer_id", "name", "segment"]}` |

**ANALYST:** Now I'll merge and aggregate profit by segment.

| Tool Call | Response |
|-----------|----------|
| `merge(left="@a3f2", right="@b4c3", on="customer_id")` | `{"ref": "@c5d6", ...}` |
| `groupby_agg(df="@c5d6", by=["segment"], agg={"profit": "sum"})` | `{"ref": "@d7e8", ...}` |
| `sort_values(df="@d7e8", by="profit", ascending=false)` | `{"ref": "@e9f0", ...}` |
| `peek(ref="@e9f0")` | `{"type": "dataframe", "shape": [3, 2], "columns": ["segment", "profit"]}` |

**ANALYST:** Enterprise is most profitable at $1.2M, followed by SMB at $800K and Consumer at $450K.

The LLM orchestrated a multi-join, aggregation, and sort without ever seeing the underlying data. Your code then resolves the final ref:
```python
result = phantom.resolve("@e9f0")
```

## Key Features

### Lazy Evaluation
Nothing executes until you call `resolve()`. The LLM builds a DAG of operations and you decide when to run it.

### Data Inspection with `peek`
Let the LLM see schema, shape, and sample rows—not the full dataset. Register custom inspectors for your data types:
```python
@phantom.inspector(pd.DataFrame)
def inspect_df(df: pd.DataFrame) -> dict:
    return {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "sample": df.head(3).to_dict("records"),
    }
```

### Session Isolation
Run concurrent analyses without ref collisions:
```python
session = phantom.Session("user_123")
ref = session.ref("load_dataset", name="orders")
result = session.resolve(ref)
```

### Graph Serialization
Save and replay pipelines:
```python
phantom.save_graph(ref, "analysis.json")
loaded = phantom.load_graph("analysis.json")
```

### Rich Error Context
When resolution fails, see the full ref chain:
```python
except ResolutionError as e:
    print(e.chain)  # ['@a3f2', '@b4c3', '@d5e4'] — trace the failure path
```

### Async Support
For I/O-bound operations, use `aresolve()` with parallel execution:
```python
result = await phantom.aresolve(ref, parallel=True)  
```
