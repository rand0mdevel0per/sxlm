"""MCP (Model Context Protocol) handler"""

from typing import Dict, Any

class MCPHandler:
    def __init__(self):
        self.tools = {}

    def register_tool(self, name: str, handler: callable):
        """Register MCP tool"""
        self.tools[name] = handler

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP request"""
        tool_name = request.get("tool")
        if tool_name in self.tools:
            return await self.tools[tool_name](request.get("params", {}))
        return {"error": "Tool not found"}
