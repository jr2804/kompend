"""Expose an ACP ``Connection`` as the ``Client`` the harness adapter streams through.

The harness adapter talks to the client through an ``acp.Client``: it sends
``session/update`` notifications while a turn runs, asks ``session/request_permission``
before an approval-gated tool call, and the client-backed toolsets read and write files
and drive terminals. The SDK's own ``AgentSideConnection`` supplies that surface, but it
also builds the router internally — leaving no place to register the ``providers/*``
routes (see :mod:`kompend.acp.provider_wire`).

So kompend builds the ``Connection`` itself and hands the adapter this adapter object,
which is the same thin translation layer the SDK's connection performs, written against
public API only: ``Connection.send_request`` / ``send_notification`` plus the
``acp.utils`` serializers.
"""

from __future__ import annotations

from typing import Any

import acp
from acp import CLIENT_METHODS, schema
from acp.connection import Connection
from acp.utils import notify_model, request_model, request_optional_model, serialize_params
from pydantic import TypeAdapter

#: Every ``session/update`` payload the adapter may send. Owned here rather
#: than in a typing-only guard: the union is the parameter type of a public
#: method, and the SDK's own ``acp.helpers.SessionUpdate`` omits five kinds
#: kompend forwards (config option, plan content/removal, usage).
type SessionUpdate = (
    schema.UserMessageChunk
    | schema.AgentMessageChunk
    | schema.AgentThoughtChunk
    | schema.ToolCallStart
    | schema.ToolCallProgress
    | schema.AgentPlanUpdate
    | schema.AgentPlanContentUpdate
    | schema.AgentPlanRemovedUpdate
    | schema.AvailableCommandsUpdate
    | schema.CurrentModeUpdate
    | schema.ConfigOptionUpdate
    | schema.SessionInfoUpdate
    | schema.UsageUpdate
)


_CREATE_ELICITATION_RESPONSE = TypeAdapter(
    schema.AcceptElicitationResponse | schema.DeclineElicitationResponse | schema.CancelElicitationResponse,
)


class ConnectionClient:
    """The ``acp.Client`` surface, backed by an open :class:`acp.Connection`.

    Args:
        conn: The connection that carries both the incoming requests and the
            outgoing client calls.
    """

    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    async def session_update(self, session_id: str, update: SessionUpdate, **kwargs: Any) -> None:
        """Stream one ``session/update`` notification to the client."""
        await notify_model(
            self._conn,
            CLIENT_METHODS["session_update"],
            schema.SessionNotification(session_id=session_id, update=update, field_meta=kwargs or None),
        )

    async def request_permission(
        self,
        session_id: str,
        tool_call: schema.ToolCallUpdate,
        options: list[schema.PermissionOption],
        **kwargs: Any,
    ) -> schema.RequestPermissionResponse:
        """Ask the client to approve one tool call."""
        return await request_model(
            self._conn,
            CLIENT_METHODS["session_request_permission"],
            schema.RequestPermissionRequest(
                options=options,
                session_id=session_id,
                tool_call=tool_call,
                field_meta=kwargs or None,
            ),
            schema.RequestPermissionResponse,
        )

    async def read_text_file(
        self,
        session_id: str,
        path: str,
        line: int | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> schema.ReadTextFileResponse:
        """Read a file through the client."""
        return await request_model(
            self._conn,
            CLIENT_METHODS["fs_read_text_file"],
            schema.ReadTextFileRequest(path=path, session_id=session_id, limit=limit, line=line, field_meta=kwargs or None),
            schema.ReadTextFileResponse,
        )

    async def write_text_file(
        self,
        session_id: str,
        path: str,
        content: str,
        **kwargs: Any,
    ) -> schema.WriteTextFileResponse | None:
        """Write a file through the client."""
        return await request_optional_model(
            self._conn,
            CLIENT_METHODS["fs_write_text_file"],
            schema.WriteTextFileRequest(content=content, path=path, session_id=session_id, field_meta=kwargs or None),
            schema.WriteTextFileResponse,
        )

    async def create_terminal(
        self,
        session_id: str,
        command: str,
        args: list[str] | None = None,
        env: list[schema.EnvVariable] | None = None,
        cwd: str | None = None,
        output_byte_limit: int | None = None,
        **kwargs: Any,
    ) -> schema.CreateTerminalResponse:
        """Start a command in a client terminal."""
        return await request_model(
            self._conn,
            CLIENT_METHODS["terminal_create"],
            schema.CreateTerminalRequest(
                command=command,
                session_id=session_id,
                args=args,
                cwd=cwd,
                env=env,
                output_byte_limit=output_byte_limit,
                field_meta=kwargs or None,
            ),
            schema.CreateTerminalResponse,
        )

    async def terminal_output(self, session_id: str, terminal_id: str, **kwargs: Any) -> schema.TerminalOutputResponse:
        """Read the output of a client terminal."""
        return await request_model(
            self._conn,
            CLIENT_METHODS["terminal_output"],
            schema.TerminalOutputRequest(session_id=session_id, terminal_id=terminal_id, field_meta=kwargs or None),
            schema.TerminalOutputResponse,
        )

    async def release_terminal(
        self,
        session_id: str,
        terminal_id: str,
        **kwargs: Any,
    ) -> schema.ReleaseTerminalResponse | None:
        """Release a client terminal."""
        return await request_optional_model(
            self._conn,
            CLIENT_METHODS["terminal_release"],
            schema.ReleaseTerminalRequest(session_id=session_id, terminal_id=terminal_id, field_meta=kwargs or None),
            schema.ReleaseTerminalResponse,
        )

    async def wait_for_terminal_exit(
        self,
        session_id: str,
        terminal_id: str,
        **kwargs: Any,
    ) -> schema.WaitForTerminalExitResponse:
        """Wait for a client terminal's command to exit."""
        return await request_model(
            self._conn,
            CLIENT_METHODS["terminal_wait_for_exit"],
            schema.WaitForTerminalExitRequest(session_id=session_id, terminal_id=terminal_id, field_meta=kwargs or None),
            schema.WaitForTerminalExitResponse,
        )

    async def kill_terminal(
        self,
        session_id: str,
        terminal_id: str,
        **kwargs: Any,
    ) -> schema.KillTerminalResponse | None:
        """Kill a client terminal's command."""
        return await request_optional_model(
            self._conn,
            CLIENT_METHODS["terminal_kill"],
            schema.KillTerminalRequest(session_id=session_id, terminal_id=terminal_id, field_meta=kwargs or None),
            schema.KillTerminalResponse,
        )

    async def create_elicitation(
        self,
        message: str,
        mode: schema.ElicitationMode,
        **kwargs: Any,
    ) -> schema.CreateElicitationResponse:
        """Ask the client for additional user input."""
        request = _elicitation_request(message, mode, kwargs or None)
        response = await self._conn.send_request(CLIENT_METHODS["elicitation_create"], serialize_params(request))
        return _CREATE_ELICITATION_RESPONSE.validate_python(response)

    async def complete_elicitation(self, elicitation_id: str, **kwargs: Any) -> None:
        """Tell the client an elicitation has finished."""
        await notify_model(
            self._conn,
            CLIENT_METHODS["elicitation_complete"],
            schema.CompleteElicitationNotification(elicitation_id=elicitation_id, field_meta=kwargs or None),
        )

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send an extension request to the client."""
        return await self._conn.send_request(f"_{method}", params)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        """Send an extension notification to the client."""
        await self._conn.send_notification(f"_{method}", params)

    def on_connect(self, conn: acp.Agent) -> None:
        """Nothing to hand back: this object is the client the agent already has."""

    async def close(self) -> None:
        """Close the underlying connection."""
        await self._conn.close()


def _elicitation_request(
    message: str,
    mode: schema.ElicitationMode,
    field_meta: dict[str, Any] | None,
) -> Any:
    """Build the wire request for one elicitation mode."""
    mode_fields = mode.model_dump(mode="json", exclude_none=True)
    if isinstance(mode, schema.ElicitationFormSessionMode):
        return schema.CreateFormSessionElicitationRequest(message=message, mode="form", field_meta=field_meta, **mode_fields)
    if isinstance(mode, schema.ElicitationFormRequestMode):
        return schema.CreateFormRequestElicitationRequest(message=message, mode="form", field_meta=field_meta, **mode_fields)
    if isinstance(mode, schema.ElicitationUrlSessionMode):
        return schema.CreateUrlSessionElicitationRequest(message=message, mode="url", field_meta=field_meta, **mode_fields)
    if isinstance(mode, schema.ElicitationUrlRequestMode):
        return schema.CreateUrlRequestElicitationRequest(message=message, mode="url", field_meta=field_meta, **mode_fields)
    raise TypeError(f"Unsupported elicitation mode: {type(mode).__name__}")


__all__ = ["ConnectionClient"]
