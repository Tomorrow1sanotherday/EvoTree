"""Read-only application schemas, independent of the task or next action.

The returned Python code runs in the existing VM controller. No source/target
roles, expected fonts, historical recommendations or test labels are supplied.
"""

CONNECTION = '''import sys, json, io, contextlib
sys.path.insert(0, '/home/user/mcp_server/tools/package')
sys.path.append('/usr/lib/python3/dist-packages')
'''

CALC = '''from libreoffice_calc import CalcTools
with contextlib.redirect_stdout(io.StringIO()):
    state = CalcTools.env_info()
if not isinstance(state, dict):
    raise RuntimeError('Calc state reader did not return a mapping')
state['active_sheet'] = CalcTools.doc.CurrentController.ActiveSheet.getName()
state['schema'] = 'calc.active_worksheet.v1'
state['scope'] = 'current active worksheet and its measured cell contents'
state['document_url'] = CalcTools.doc.getURL()
print(json.dumps(state, ensure_ascii=False))
'''

IMPRESS = '''from libreoffice_impress import ImpressTools
with contextlib.redirect_stdout(io.StringIO()):
    ImpressTools.get_current_slide_index()
doc = ImpressTools.doc
pages = doc.getDrawPages()
active = doc.getCurrentController().getCurrentPage()
state = {'schema':'impress.presentation.v1', 'scope':'all slides with measured object text and font sizes',
         'document_url':doc.getURL(),
         'current_slide_index':None, 'slides':[]}
for page_index in range(pages.getCount()):
    page = pages.getByIndex(page_index)
    if page == active:
        state['current_slide_index'] = page_index + 1
    objects = []
    for box_index in range(page.getCount()):
        shape = page.getByIndex(box_index)
        if not hasattr(shape, 'getText'):
            continue
        text = shape.getText()
        content = text.getString()
        if not content.strip():
            continue
        sizes = set()
        paragraphs = text.createEnumeration()
        while paragraphs.hasMoreElements():
            paragraph = paragraphs.nextElement()
            if not hasattr(paragraph, 'createEnumeration'):
                continue
            portions = paragraph.createEnumeration()
            while portions.hasMoreElements():
                portion = portions.nextElement()
                if hasattr(portion, 'getString') and portion.getString():
                    try:
                        sizes.add(round(float(portion.getPropertyValue('CharHeight')), 3))
                    except Exception:
                        pass
        objects.append({'box_index':box_index, 'text':content,
                        'font_sizes_pt':sorted(sizes),
                        'position':{'x':shape.Position.X, 'y':shape.Position.Y},
                        'extent':{'width':shape.Size.Width, 'height':shape.Size.Height}})
    state['slides'].append({'slide_index':page_index+1, 'text_objects':objects})
print(json.dumps(state, ensure_ascii=False))
'''


def office_observation_command(tool_name):
    if tool_name == 'libreoffice_calc':
        return CONNECTION + CALC
    if tool_name == 'libreoffice_impress':
        return CONNECTION + IMPRESS
    return None
