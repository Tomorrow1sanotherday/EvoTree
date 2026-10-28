"""Verify the installed runner's real hotkey assembler without GUI imports."""
import ast
from pathlib import Path
import sys

path = Path(sys.argv[1])
module = ast.parse(path.read_text(encoding='utf-8'))
fn = next(n for n in module.body if isinstance(n, ast.FunctionDef)
          and n.name == 'parsing_response_to_pyautogui_code')
ns = {}
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(path), 'exec'), ns)
assemble = ns[fn.name]
for hotkey, expected in [('CTRL ALT T', ['ctrl', 'alt', 't']), ('ESC', ['esc']), ('F2', ['f2'])]:
    code = assemble({'action_type': 'hotkey', 'action_inputs': {'hotkey': hotkey}}, 1080, 1920, False)
    call = next(n for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == 'hotkey')
    assert [ast.literal_eval(a) for a in call.args] == expected
print('HOTKEY_ASSEMBLY_PASS: CTRL ALT T, ESC, F2')
