from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.tools import tool
from typing import List
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class MlPaperAnalyzer():
    """MlPaperAnalyzer crew"""

    # Load the YAML files automatically
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def visionary_strategist(self) -> Agent:
        return Agent(
            config=self.agents_config['visionary_strategist'],
            verbose=True
        )

    @task
    def hypothesis_generation_task(self) -> Task:
        return Task(
            config=self.tasks_config['hypothesis_generation_task'],
            # This is crucial: we map the task to the agent here
            agent=self.visionary_strategist() 
        )

    @crew
    def crew(self) -> Crew:
        """Creates the MlPaperAnalyzer crew"""
        return Crew(
            # The decorator automatically collects all @agent and @task methods
            agents=self.agents, 
            tasks=self.tasks, 
            process=Process.sequential, # Or hierarchical if you add a manager
            verbose=True,
        )
