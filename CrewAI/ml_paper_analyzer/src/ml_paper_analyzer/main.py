#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from ml_paper_analyzer.crew import MlPaperAnalyzerCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    # This input is injected into {topic} if you use it in your YAML descriptions
    inputs = {
        'topic': 'The "Attention is All You Need" paper and Transformer architecture'
    }
    
    MlPaperAnalyzerCrew().crew().kickoff(inputs=inputs)


def train():
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year)
    }
    try:
        # FIX: Use the correct class name
        MlPaperAnalyzerCrew().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    try:
        # FIX: Use the correct class name
        MlPaperAnalyzerCrew().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }
    try:
        # FIX: Use the correct class name
        MlPaperAnalyzerCrew().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")



