"""Generate observation fixes for an isolated OSWorld runner copy.

Refuses unfamiliar sources and never edits an experiment snapshot in place.
"""
import argparse
import difflib
from pathlib import Path


def patch_desktop(source, office_schema=False):
    replacements = [
        ('        tool_name = None\n        if cur_id in app_list:',
         '        read_state = None\n        tool_name = None\n        if cur_id in app_list:'),
        ('                app_info = read_python_output(self.controller, command)',
         '                def read_state():\n'
         '                    return read_python_output(self.controller, command)'),
        ('        return {\n            "screenshot": self.controller.get_screenshot(),',
         '        import json\n'
         '        from evotree.current_observation import capture_current_observation\n'
         '        captured = capture_current_observation(\n'
         '            read_state, self.controller.get_screenshot,\n'
         '            action_sequence=self._step_no,\n'
         '            source=(tool_name + ".env_info()" if read_state else "no application state reader"))\n'
         '        if captured["app_info"] is not None:\n'
         '            captured["app_info"] = json.dumps(captured["app_info"], ensure_ascii=False)\n'
         '        return {\n            **captured,'),
        ('            "app_info": app_info,\n', ''),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('Unrecognized/already patched DesktopEnv observation source')
        source = source.replace(old, new, 1)
    if office_schema:
        old = '                from evotree.observation_io import read_python_output\n'
        if source.count(old) != 1:
            raise ValueError('Application observation query point not found')
        source = source.replace(old, old +
            '                from evotree.office_observation import office_observation_command\n'
            '                command = office_observation_command(tool_name) or command\n', 1)
        source = source.replace('tool_name + ".env_info()"', 'tool_name + ".current_object()"')
    return source


def patch_agent(source):
    old = '        messages = user_prompt\n        # print(messages)'
    new = ('        from evotree.current_observation import append_current_observation\n'
           '        messages = append_current_observation(user_prompt, obs, use_dashscope)\n'
           '        # print(messages)')
    if source.count(old) != 1:
        raise ValueError('Unrecognized/already patched OwlAgent prompt source')
    return source.replace(old, new, 1)


def prepare(runner, output, office_schema=False):
    changes = []
    for relative, transform in [('desktop_env/desktop_env.py', patch_desktop),
                                ('mm_agents/owl_agent.py', patch_agent)]:
        original = (runner / relative).read_text(encoding='utf-8')
        updated = transform(original, office_schema) if transform is patch_desktop else transform(original)
        compile(updated, relative, 'exec')
        target = output / relative
        if target.exists():
            raise FileExistsError(target)
        changes.append((relative, original, updated, target))
    diff = []
    for relative, original, updated, target in changes:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(updated, encoding='utf-8')
        diff.extend(difflib.unified_diff(original.splitlines(True), updated.splitlines(True),
                    fromfile='a/'+relative, tofile='b/'+relative))
    return ''.join(diff)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--patch-file', type=Path)
    parser.add_argument('--office-schema', action='store_true')
    args = parser.parse_args()
    diff = prepare(args.runner, args.output, args.office_schema)
    if args.patch_file:
        args.patch_file.write_text(diff, encoding='utf-8')
    print('Prepared two isolated observation/prompt fixes.')
