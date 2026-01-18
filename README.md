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

Imagine a user asks: *"What's our most profitable customer segment by region?"*

The LLM doesn't need to see 10,000 rows of data to answer this. It needs to know the schema, understand the relationships, and compose the right transformations. Here's how it works with Phantom:

```yaml
ANALYST: I'll analyze profitability by customer segment. Let me load the datasets and
explore their structure.

[load_csv] {"path": "orders.csv"}
  → @a3f2

[peek] {"ref": "@a3f2"}
  → 10000 rows, columns: ['order_id', 'customer_id', 'product_id', 'quantity', 'order_date']

[load_csv] {"path": "customers.csv"}
  → @b4c3

[peek] {"ref": "@b4c3"}
  → 500 rows, columns: ['customer_id', 'name', 'region', 'segment']

[load_csv] {"path": "products.csv"}
  → @c5d6

[peek] {"ref": "@c5d6"}
  → 50 rows, columns: ['product_id', 'name', 'category', 'price', 'cost']

ANALYST: Now I'll merge the datasets and aggregate profit by segment and region.

[merge] {"left": "@a3f2", "right": "@c5d6", "on": "product_id"}
  → @d7e8

[merge] {"left": "@d7e8", "right": "@b4c3", "on": "customer_id"}
  → @e9f0

[groupby_agg] {"df": "@e9f0", "by": ["segment", "region"], "agg": {"profit": "sum"}}
  → @g3b4

[sort_values] {"df": "@g3b4", "by": "profit", "ascending": false}
  → @h5c6

[peek] {"ref": "@h5c6"}
  → 12 rows, columns: ['segment', 'region', 'profit']

ANALYST: Enterprise customers in the West region are most profitable at $1.2M, followed
by Enterprise East at $980K. SMB segments show consistent mid-tier performance across
all regions.
```

The LLM orchestrated a multi-join, aggregation, and sort without ever seeing the underlying data. Your code then resolves the final ref:
```python
result = phantom.resolve("@h5c6")
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
