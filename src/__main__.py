import sys
from . import Small_LLM_Model, Processor, Parser
from pydantic import ValidationError
from .PydanticModels import FunctionDef, PromptEntry, FunctionCall

if '__main__' == __name__:
    argc: int = len(sys.argv)

    arguments: dict[str, str] = {
        'functions_definition': 'data/input/functions_definition.json',
        'input': 'data/input/function_calling_tests.json',
        'output': 'data/output/output.json'
    }
    try:
        for i in range(1, argc, 2):
            if (sys.argv[i] == '--functions_definition'):
                arguments.update({'functions_definition': sys.argv[i + 1]})
            elif (sys.argv[i] == '--input'):
                arguments.update({'input': sys.argv[i + 1]})
            elif (sys.argv[i] == '--output'):
                arguments.update({'output': sys.argv[i + 1]})
    except Exception as e:
        print(f"error while parsing: {e}")
        exit(1)
    prompts: list[PromptEntry]
    functions: list[FunctionDef]
    try:
        parser: Parser = Parser(**arguments)
        prompts = parser.get_input()
        functions = parser.get_functions_definition()
    except (OSError, ValidationError, ValueError) as e:
        print(f"error while validating pydantic: {e}")
        exit(1)
    llm: Small_LLM_Model = Small_LLM_Model()
    processor: Processor = Processor(llm)
    output: list[FunctionCall] = []
    for prompt in prompts:
        try:
            result: FunctionCall = processor.process_prompt(prompt, functions)
            output.append(result)
        except ValueError as e:
            print("there was an error while processing prompt: " + e.args[0])
    parser.load_in_output(output)
