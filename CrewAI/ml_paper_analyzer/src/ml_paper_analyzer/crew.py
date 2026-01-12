from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.tools import tool
from typing import List
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class MlPaperAnalyzerCrew():
    """MlPaperAnalyzer crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # --- Agents ---
    @agent
    def researcher(self) -> Agent:
        return Agent(config=self.agents_config['researcher'], verbose=True)

    @agent
    def critical_analyst(self) -> Agent:
        return Agent(config=self.agents_config['critical_analyst'], verbose=True)

    @agent
    def visionary_strategist(self) -> Agent:
        return Agent(config=self.agents_config['visionary_strategist'], verbose=True)

    # --- Tasks ---
    @task
    def extraction_task(self) -> Task:
        return Task(config=self.tasks_config['extraction_task'], agent=self.researcher())

    @task
    def critique_task(self) -> Task:
        return Task(config=self.tasks_config['critique_task'], agent=self.critical_analyst())

    @task
    def hypothesis_generation_task(self) -> Task:
        return Task(config=self.tasks_config['hypothesis_generation_task'], agent=self.visionary_strategist())

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents, # Automatically collects the 3 agents above
            tasks=self.tasks,   # Automatically collects the 3 tasks above
            process=Process.sequential,
            verbose=True,
        )
