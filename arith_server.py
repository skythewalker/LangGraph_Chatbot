# arith_server.py
# C:\Users\akashinde\Langgraph_chatbot

from fastmcp import FastMCP

mcp = FastMCP("Arithmetic Server")


# ---------------- ADDITION ----------------
@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


# ---------------- SUBTRACTION ----------------
@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


# ---------------- MULTIPLICATION ----------------
@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


# ---------------- DIVISION ----------------
@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b


if __name__ == "__main__":
    mcp.run()