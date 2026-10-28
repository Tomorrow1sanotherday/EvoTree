"""Test real external action assembly without importing model/VM dependencies."""
from pathlib import Path
import ast
import json
import re
import sys


def verify(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OwlAgent")
    predict = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "predict")
    start = next(i for i,n in enumerate(predict.body)
                 if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id == "actions" for t in n.targets))
    body = predict.body[start:]
    fn = ast.FunctionDef(name="assembly", args=ast.arguments(posonlyargs=[],
                         args=[ast.arg(arg=n) for n in ("self","parsed_responses","prediction","messages")],
                         kwonlyargs=[],kw_defaults=[],defaults=[]),body=body,decorator_list=[])
    namespace = {"parsing_response_to_pyautogui_code":lambda r,*a:r["action_type"],
                 "remove_base64_images":lambda m:m}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(path),"exec"),namespace)
    class Agent:
        def __init__(self, turns=1,limit=15):
            self.actions=[];self.runtime_conf={"screen_height":1080,"screen_width":1920}
            self.input_swap=False;self.history_responses=["x"]*turns;self.max_trajectory_length=limit
    def run(types, expected, turns=1, limit=15):
        agent=Agent(turns,limit)
        rows=[{"action_type":t,"action_inputs":{}} for t in types]
        output=namespace["assembly"](agent,rows,"prediction",[])
        assert output[1]==expected,(types,output[1],expected)
        assert agent.actions==[expected],agent.actions
    cases=[(["tool.fill","finished"],["tool.fill","DONE"]),
           (["tool.open","tool.fill","finished"],["tool.open","tool.fill","DONE"]),
           (["tool.open","fail"],["tool.open","FAIL"]),
           (["finished"],["DONE"]),(["fail"],["FAIL"]),
           (["tool.fill","finished","tool.after"],["tool.fill","DONE"]),
           (["tool.one","tool.two"],["tool.one","tool.two"])]
    for types,expected in cases:run(types,expected)
    run(["tool.fill"],["tool.fill"],turns=15,limit=15)
    run(["tool.fill","finished"],["tool.fill","DONE"],turns=1,limit=1)
    run(["tool.fill","fail","tool.after"],["tool.fill","FAIL"])
    # Parse an actual multi-action response, then pass it through actual assembly.
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in
               ("parse_action_fncall_id","_parse_single_action_fncall_id")]
    namespace.update(json=json,re=re,update_image_size_=lambda x:x)
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),"exec"),namespace)
    reply=''.join('<tool_call>'+json.dumps(c)+'</tool_call>' for c in [
        {"name":"libreoffice_calc2.fill_blank_down","arguments":{"columns":["B","C"]}},
        {"name":"computer_use","arguments":{"action":"terminate","status":"success"}}])
    parsed=namespace["parse_action_fncall_id"](reply,None,1080,1920,"gpt-5.6-terra")
    assert namespace["assembly"](Agent(),parsed,reply,[])[1]==["libreoffice_calc2.fill_blank_down","DONE"]
    return {"cases":11,"passed":True,"source":str(path)}


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1]))))
