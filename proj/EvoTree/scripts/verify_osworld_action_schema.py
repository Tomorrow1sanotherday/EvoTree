"""Exercise the installed runner's real parser/assembler for all GUI actions."""
import ast
import json
from pathlib import Path
import re
import sys


def verify(path: Path, failed_reply: Path | None = None) -> dict:
    module = ast.parse(path.read_text(encoding='utf-8'))
    names = {'parse_action_fncall_id', '_parse_single_action_fncall_id',
             'parsing_response_to_pyautogui_code', 'escape_single_quotes'}
    functions = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {'json': json, 're': re, 'ast': ast,
          'update_image_size_': lambda value: value,
          'convert_point_format': lambda point, *args, **kwargs: point}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), ns)
    parse = ns['parse_action_fncall_id']
    assemble = ns['parsing_response_to_pyautogui_code']

    def call(name, args):
        return '<tool_call>' + json.dumps({'name': name, 'arguments': args}) + '</tool_call>'

    cases = [
        {'action':'key', 'keys':['CTRL','ALT','T']},
        {'action':'type', 'text':'hello', 'clear':1, 'enter':1},
        *[{'action':action,'coordinate':[100,200]} for action in
          ('mouse_move','click','right_click','middle_click','double_click')],
        {'action':'drag','coordinate':[100,200],'coordinate2':[300,400]},
        {'action':'scroll','pixels':-3}, {'action':'wait','time':1},
        {'action':'terminate','status':'success'},
        {'action':'terminate','status':'failure'},
    ]
    for args in cases:
        parsed = parse(call('computer_use',args),None,1080,1920,'gpt-5.6-terra')
        assert len(parsed)==1
        code=assemble(parsed,1080,1920,False)
        if args['action']=='middle_click':
            assert "button='middle'" in code
        if args['action']=='key':
            assert "'ctrl', 'alt', 't'" in code
    try:
        parse(call('computer_use',{'action':'unsupported'}),None,1080,1920,'gpt-5.6-terra')
    except ValueError as exc:
        assert 'Unsupported computer_use action' in str(exc)
    else:
        raise AssertionError('Unknown action must trigger format correction')
    # A tool's parameter text is not its name.
    parsed=parse(call('libreoffice_impress.set_text',{'text':'computer_use'}),None,1080,1920,'gpt-5.6-terra')
    assert parsed[0]['action_type']=='libreoffice_impress.set_text'
    multi=call('computer_use',{'action':'middle_click','coordinate':[10,20]})+call('computer_use',{'action':'terminate','status':'success'})
    assert [row['action_type'] for row in parse(multi,None,1080,1920,'gpt-5.6-terra')]==['middle_single','finished']
    count=len(cases)+3
    if failed_reply is not None:
        value=json.loads(failed_reply.read_text(encoding='utf-8'))
        prediction=value['prediction']
        try:
            parse(prediction,None,1080,1920,'gpt-5.6-terra')
        except ValueError as exc:
            assert "'TAB'" in str(exc)
        else:
            raise AssertionError('Original bare TAB action should request correction')
        corrected=[]
        for tool in value['calls']:
            if tool['name']=='computer_use' and tool['arguments'].get('action')=='TAB':
                tool={**tool,'arguments':{'action':'key','keys':['TAB']}}
            corrected.append(call(tool['name'],tool['arguments']))
        parsed=parse(''.join(corrected),None,1080,1920,'gpt-5.6-terra')
        assert len(parsed)==len(corrected)
        for row in parsed:
            assemble(row,1080,1920,False)
        count+=2
    return {'passed':True,'cases':count,'source':str(path)}


if __name__=='__main__':
    print(json.dumps(verify(Path(sys.argv[1]),Path(sys.argv[2]) if len(sys.argv)>2 else None)))
