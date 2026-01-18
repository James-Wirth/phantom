# Phantom

[![CI](https://github.com/James-Wirth/phantom/actions/workflows/ci.yml/badge.svg)](https://github.com/James-Wirth/phantom/actions/workflows/ci.yml)

**Let LLMs orchestrate data pipelines without seeing the data.**

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

## Example: AI Data Analyst

Imagine a user asks: *"What's our most profitable customer segment by region?"*

The LLM doesn't need to see 10,000 rows of data to answer this. It needs to know the schema, understand the relationships, and compose the right transformations. Here's how it works with Phantom:

**Step 1: LLM loads datasets and receives opaque refs**
```python
LLM calls: load_dataset("orders")     → returns {"ref": "@a3f2"}
LLM calls: load_dataset("customers")  → returns {"ref": "@b4c3"}
LLM calls: load_dataset("products")   → returns {"ref": "@c5d6"}
```

**Step 2: LLM peeks at structure (not the full data)**
```python
LLM calls: peek("@a3f2")
→ {
    "ref": "@a3f2",
    "type": "list",
    "length": 12,
    "keys": ["order_id", "customer_id", "product_id", "quantity", "order_date"],
    "sample": [{"order_id": 1, "customer_id": "C001", "product_id": "P001", ...}]
  }
```

**Step 3: LLM builds the pipeline using refs**
```python
LLM calls: join("@a3f2", "@c5d6", left_on="product_id", right_on="product_id")  → {"ref": "@d7e8"}
LLM calls: join("@d7e8", "@b4c3", left_on="customer_id", right_on="customer_id") → {"ref": "@e9f0"}
LLM calls: compute("@e9f0", column="profit", expression="quantity * (unit_price - cost)")  → {"ref": "@f1a2"}
LLM calls: group_by("@f1a2", keys=["segment", "region"], aggregations={"total": "sum:profit"})  → {"ref": "@g3b4"}
LLM calls: sort("@g3b4", by="total", descending=True)  → {"ref": "@h5c6"}
```

**Step 4: Your code resolves the final ref into actual data**
```python
result = phantom.resolve("@h5c6")  
```

The LLM orchestrated a multi-join, aggregation, and sort without ever seeing the underlying data.

## Defining Operations

Operations are regular Python functions decorated with `@phantom.op`. Your function's docstring becomes the tool description for the LLM.

```python
import phantom

@phantom.op
def load_dataset(name: str) -> list[dict]:
    """Load a dataset by name. Available: 'orders', 'customers', 'products'."""
    return db.query(f"SELECT * FROM {name}")

@phantom.op
def join(left: phantom.Ref[list], right: phantom.Ref[list], left_on: str, right_on: str) -> list[dict]:
    """Join two datasets on matching keys."""
    right_lookup = {row[right_on]: row for row in right}
    return [{**l, **right_lookup[l[left_on]]} for l in left if l[left_on] in right_lookup]

@phantom.op
def group_by(data: phantom.Ref[list], keys: list[str], aggregations: dict) -> list[dict]:
    """Group by keys and aggregate. Use format {"output_col": "func:source_col"}."""
    # Your aggregation logic here...
    ...
```

The key insight: **parameters typed as `Ref[T]` are automatically resolved** before your function runs. You write normal code that receives normal data.

## Wiring Up Your LLM

Phantom auto-generates tool definitions from your operations and handles tool calls uniformly:

```python
import json
import openai
import phantom

client = openai.OpenAI()
tools = phantom.get_openai_tools()

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

## Key Features

### Lazy Evaluation
Nothing executes until you call `resolve()`. The LLM builds a DAG of operations and you decide when to run it.

### Data Inspection with `peek`
Let the LLM see schema, shape, and sample rows, not the full dataset. Register custom inspectors for your data types:
```python
@phantom.inspector(pd.DataFrame)
def inspect_df(df):
    return {"shape": df.shape, "columns": list(df.columns), "sample": df.head(3).to_dict("records")}
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
