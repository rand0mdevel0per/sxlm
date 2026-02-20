# Python Bindings

## Installation

```bash
pip install -r requirements.txt
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

## Usage

```python
import quila_core

# Initialize model
model = quila_core.QuilaModel(num_neurons=32768, hidden_dim=256)

# Run inference
response = model.inference("Your prompt here")
print(response)

# Get status
status = model.get_status()
print(f"Neurons: {status['neurons']}")
print(f"Phase: {status['phase']}")
```

## API Reference

### QuilaModel

**Constructor:**
```python
QuilaModel(num_neurons=32768, hidden_dim=256)
```

**Methods:**

- `inference(prompt: str) -> str`: Run inference on prompt
- `get_status() -> dict`: Get model status

## Example

```python
import quila_core

model = quila_core.QuilaModel()
result = model.inference("Explain neural networks")
print(result)
```
