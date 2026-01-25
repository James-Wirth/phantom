<h1>
<p align="center">
  <img src="assets/logo.svg" alt="Phantom" width="80">
  <br>phantom
</h1>
  <p align="center">
    Let LLMs orchestrate data pipelines without seeing the data.
    <br><br>
    <a href="https://github.com/James-Wirth/phantom/actions/workflows/ci.yml"><img src="https://github.com/James-Wirth/phantom/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  </p>
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

## Defining Operations and Inspectors

**Operations** are Python functions that become LLM tools. The docstring becomes the tool description.

```python
import pandas as pd
import phantom

session = phantom.Session()

@session.op
def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path)

@session.op
def filter_rows(df: phantom.Ref[pd.DataFrame], column: str, value: str) -> pd.DataFrame:
    """Filter rows where column equals value."""
    return df[df[column] == value]

@session.op
def get_mean(df: phantom.Ref[pd.DataFrame], column: str) -> float:
    """Calculate the mean of a column."""
    return df[column].mean()
```

Parameters typed as `Ref[T]` are resolved automatically. You write normal code and Phantom handles the wiring.

**Inspectors** define how data types appear when the LLM calls `peek()`. They return a summary dict instead of raw data:

```python
@session.inspector(pd.DataFrame)
def inspect_dataframe(df: pd.DataFrame) -> dict:
    return {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "sample": df.head(3).to_dict("records"),
    }
```

Now when the LLM peeks at a DataFrame ref, it sees shape and schema (not 10,000 rows!).

## Wiring Up Your LLM

Phantom auto-generates tool definitions from your operations and handles tool calls elegantly:

```python
import openai

client = openai.OpenAI()

# using the Session instance defined above...
tools = session.get_tools()

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
            result = session.handle_tool_call(
                tool_call.function.name,
                tool_call.function.arguments,  
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result.to_json(),
            })
    else:
        break

# Execute the pipeline the LLM built
final_data = session.resolve(result.ref)
```

`handle_tool_call` accepts JSON strings or dicts and returns refs for lazy operations, immediate results for `peek`. The LLM sees just enough to make decisions.

## Contrib Modules

Phantom provides pre-built operation sets for common data libraries. Install the optional dependencies and register them with your session:

```bash
pip install phantom[pandas]    # Pandas
pip install phantom[polars]    # Polars
pip install phantom[duckdb]    # SQL (via DuckDB)
pip install phantom[all]       # All contrib modules
```

```python
from phantom import Session
from phantom.contrib.pandas import pandas_ops

session = Session()
session.register(pandas_ops)  # registers Pandas operations 
```

Each contrib module provides domain-specific operations (`read_csv`, `filter_rows`, `groupby_agg`, etc.) and inspectors that help the LLM understand data shapes and schemas.

## Example: AI Data Scientist

This example uses the **pandas contrib** module to let an LLM analyze data without seeing it.

> **User:** What's our most profitable customer segment?

| Step | Tool Call | Response |
|:----:|-----------|----------|
| 1 | `read_csv(path="orders.csv")` | `@a3f2` |
| 2 | `read_csv(path="customers.csv")` | `@b4c3` |
| 3 | `peek(ref="@a3f2")` | `shape: [10000, 4], columns: [order_id, customer_id, quantity, profit]` |
| 4 | `peek(ref="@b4c3")` | `shape: [500, 3], columns: [customer_id, name, segment]` |
| 5 | `merge(left="@a3f2", right="@b4c3", on="customer_id")` | `@c5d6` |
| 6 | `groupby_agg(df="@c5d6", by=["segment"], aggregations={"profit": "sum"})` | `@d7e8` |
| 7 | `sort_values(df="@d7e8", by="profit", ascending=false)` | `@e9f0` |
| 8 | `peek(ref="@e9f0")` | `shape: [3, 2], columns: [segment, profit]` |

The LLM orchestrated a multi-join, aggregation, and sort without ever seeing the underlying data. Your code then resolves the final ref:

```python
result = session.resolve("@e9f0")
```

## Key Features

### Lazy Evaluation
Nothing executes until you call `resolve()`. The LLM builds a DAG of operations and you decide when to run it.

### Data Inspection with `peek`
The built-in `peek` operation lets the LLM see data summaries without the full content. Contrib modules register inspectors automatically (but you can also define your own for custom types - see above).

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
session.save_graph(ref, "analysis.json")
loaded = session.load_graph("analysis.json")
```

### Rich Error Context
When resolution fails, see the full ref chain:
```python
except ResolutionError as e:
    print(e.chain)  # ['@a3f2', '@b4c3', '@d5e4'] - trace the failure path
```

### Async Support
For I/O-bound operations, use `aresolve()` with parallel execution:
```python
result = await session.aresolve(ref, parallel=True)
```
