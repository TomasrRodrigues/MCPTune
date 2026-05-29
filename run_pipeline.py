# run_pipeline.py
import json
from mcptune.generate.generate import Generator

# 1. Define a dummy tool object matching your pipeline expectation
class MockTool:
    def __init__(self, name, description):
        self.name = name
        self.description = description

# Instantiate sample inputs
sample_tool = MockTool(
    name="fetch_webpage",
    description="Retrieves the clean markdown or text contents from a given web URL endpoint."
)
sample_arguments = {
    "url": "https://example.com/api/v1/report",
    "format": "markdown"
}

# 2. Fire up the Generator engine without test mocks
# (Remove your 'raise Exception' debug line from generate.py before running this!)
generator = Generator(model_name="Qwen/Qwen2.5-1.5B-Instruct")

# 3. Print the raw prompt structure by calling your generation method
try:
    print("\n" + "="*40 + " RUNNING REAL GENERATION " + "="*40)
    result = generator.generate_intent(tool=sample_tool, arguments=sample_arguments)
    print(f"\nFinal Generated Output Result:\n{result}")
except Exception as e:
    # If your 'raise Exception' line is still there, this will catch it and print it anyway!
    print(f"\nCaught Execution String:\n{e}")