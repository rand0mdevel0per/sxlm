"""MCP (Model Context Protocol) handler"""

from typing import Dict, Any, Callable, List
import asyncio

class MCPHandler:
    def __init__(self, model=None):
        self.tools: Dict[str, Callable] = {}
        self.model = model
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default MCP tools"""
        self.register_tool("inference", self._tool_inference)
        self.register_tool("status", self._tool_status)

    async def _tool_inference(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Inference tool"""
        prompt = params.get("prompt", "")
        if self.model:
            response = self.model.inference(prompt)
        else:
            response = f"Response to: {prompt}"
        return {"result": response}

    async def _tool_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Status tool"""
        if self.model:
            return self.model.get_status()
        return {"status": "mock"}

    def register_tool(self, name: str, handler: Callable):
        """Register MCP tool"""
        self.tools[name] = handler

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP request"""
        tool_name = request.get("tool")
        if tool_name in self.tools:
            return await self.tools[tool_name](request.get("params", {}))
        return {"error": "Tool not found"}

    def list_tools(self) -> List[str]:
        """List available tools"""
        return list(self.tools.keys())
