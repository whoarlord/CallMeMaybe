import sys
from . import Small_LLM_Model, Processor, Parser


def check_validation():
    """ function for checking directories and files """
    print('validate')


if '__main__' == __name__:
    argc: int = len(sys.argv)

    arguments: dict = {
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
    parser: Parser = Parser(**arguments)
    llm: Small_LLM_Model = Small_LLM_Model()
    processor: Processor = Processor(llm)
    prompts: list[dict] = parser.get_input_json()
    for prompt in prompts:
        processor.process_prompt(prompt)
        exit(0)
