import sys
from . import Small_LLM_Model, Processor, Parser
import json
from pydantic import ValidationError
from .PydanticModels import FunctionDef, PromptEntry, FunctionCall

if '__main__' == __name__:
    argc: int = len(sys.argv)

    arguments: dict[str, str] = {
        'functions_definition': 'data/input/functions_definition.json',
        'input': 'data/input/function_calling_tests.json',
        'output': 'data/output/output.json'
    }
    for i in range(1, argc, 2):
        if (sys.argv[i] == '--functions_definition'):
            arguments.update({'functions_definition': sys.argv[i + 1]})
        elif (sys.argv[i] == '--input'):
            arguments.update({'input': sys.argv[i + 1]})
        elif (sys.argv[i] == '--output'):
            arguments.update({'output': sys.argv[i + 1]})
    try:
        parser: Parser = Parser(**arguments)
    except (OSError, ValidationError, ValueError) as e:
        print(f"error while validating pydantic: {e}")
        exit(1)
    llm: Small_LLM_Model = Small_LLM_Model()
    processor: Processor = Processor(llm)
    prompts: list[PromptEntry] = parser.get_input()
    functions: list[FunctionDef] = parser.get_functions_definition()
    output: list[FunctionCall] = []
    for prompt in prompts:
        try:
            result: FunctionCall = processor.process_prompt(prompt, functions)
            output.append(json.loads(result))
        except ValueError as e:
            print("there was an error while processing prompt: " + e.args[0])
    print(f"output: {output}")
    parser.load_in_output(output)
