"""Optional format-only reminder; never selects an action or edits a response."""
from __future__ import annotations


ACTION_PROTOCOL_REMINDER = '''Action output contract (format only, not a next-action suggestion):
The executable functions are described in the tools block above; this runner
accepts their calls via XML text. Choose an action from the current task and
observation. Return exactly one complete <tool_call>...</tool_call> containing
one valid JSON object with a string name and an object arguments. Use an exact
provided function name and its documented argument names and types. For GUI
actions the function name is computer_use and arguments.action is its action.
For a point coordinate, supply exactly two numeric entries; no extra button
entry, variable, identifier, comment, undefined value or trailing comma is valid
JSON. Do not substitute action_type/parameters, fn/args or explanatory prose for
name/arguments. Do not claim an action ran. If the current observation does not
support the intended operation, select a documented observation/wait action or
report failure through the documented termination call. A valid format alone
does not justify declaring success. Keep all decisions grounded in the current
task and observations, not this format reminder.'''


def with_action_protocol(messages):
    """Append a fresh text message without mutating caller-owned multimodal data."""
    return list(messages) + [dict(role='user', content=ACTION_PROTOCOL_REMINDER)]
