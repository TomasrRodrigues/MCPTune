from abc import ABC, abstractmethod


"""Backend execution interface.

This module defines the abstract `Backend` base class that concrete
execution backends must implement. A backend is responsible for executing
tool calls against an MCP (Model Control Plane) server or providing a
local/mock implementation used for dataset generation and testing.

Concrete backends should return a serializable dictionary describing the
result of the tool invocation. Implementations are free to choose the
exact shape of that dictionary but common keys include:

- ``content``: raw response blocks or messages
- ``structured_content``: an optional parsed payload
- ``is_error``: boolean flag indicating execution failure

The interface is intentionally minimal to allow different transport and
server implementations while keeping downstream code transport-agnostic.
"""


class Backend(ABC):
    """Abstract base for MCP execution backends.

    Subclasses must implement ``call_tool`` which executes a named tool
    with the provided arguments and returns a normalized response dict.
    The returned value must be JSON-serializable.
    """

    @abstractmethod
    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Execute a tool call.

        Parameters
        ----------
        tool_name:
            The name of the tool to invoke.

        arguments:
            A mapping of parameter names to values that matches the tool's
            declared input schema.

        Returns
        -------
        dict
            A transport-agnostic result object. Typical implementations
            return a dict containing execution "content", optional
            "structured_content", and an "is_error" flag, but backends
            may include additional backend-specific fields.
        """
        raise NotImplementedError()
