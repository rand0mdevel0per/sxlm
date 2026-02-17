# Qualia

AI assistant with 128M context and Plan-Think-Execute reasoning.

## Features

- **128M Context**: Access up to 128 million tokens through Semantic Context Tree
- **PTE Architecture**: Plan-Think-Execute flow for systematic problem-solving
- **Dynamic Replanning**: Automatic correction when logical drift is detected
- **MCP Integration**: Dynamic tool discovery and execution
- **Tiered Storage**: Efficient knowledge retrieval across GPU, RAM, and disk

## Quick Start

```python
from sxlm.client import QualiaClient

client = QualiaClient(api_key="your-api-key")

async for chunk in client.chat_stream([
    {"role": "user", "content": "Explain quantum computing"}
]):
    print(chunk)
```

## Architecture

Qualia is built on the QualiaTrace architecture featuring:

- **Planner-Port**: Generates implicit instruction sequences
- **Ring Buffer**: Monitors attention entropy for drift detection
- **el-trace**: Parameter-level credit assignment for RL
- **KFE**: Key-Feature Encoding with tiered storage

[Learn more →](/architecture/)
