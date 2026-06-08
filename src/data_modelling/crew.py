from crewai import Agent, Crew, Process, Task, LLM
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from data_modelling.tools.custom_tool import (
    describe_dataset_tool,
    correlation_tool,
    one_hot_encoding_tool,
    drop_columns_tool,
    prediction_tool
    
)
from dotenv import load_dotenv
import os
from pyprojroot import here
from crewai_tools import  FileReadTool,FileWriterTool


load_dotenv(here(".env"))

api = os.getenv("DEEPSEEK_API_KEY")
api2 = os.getenv("e2b")

#sandbox = E2BPythonTool()

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators
load_dotenv(here(".env"))



@CrewBase
class ContentCrew:
    """Content Crew — sequential cleaning → modelling pipeline"""

    agents: list[BaseAgent]
    tasks: list[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # If you would like to add tools to your crew, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools

    @staticmethod
    def _llm() -> LLM:
        """Shared LLM factory — all agents use the same DeepSeek configuration."""
        return LLM(
            model="deepseek-chat",
            api_key=api,
            base_url="https://api.deepseek.com/v1",
            temperature=0.7,
        )

    def manager_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['manager_agent'],
            llm=self._llm(),
            allow_delegation=True,
            verbose=True,
        )
    
    @agent
    def data_transformation_a(self) -> Agent:
        return Agent(
            config=self.agents_config["data_transformation_a"],
            llm=self._llm(),
            max_iter=12,
            tools=[describe_dataset_tool(),correlation_tool(),one_hot_encoding_tool(),drop_columns_tool(), FileReadTool(), FileWriterTool()],
        )
    
    @agent
    def machine_learning_a(self) -> Agent:
        return Agent(
            config=self.agents_config["machine_learning_a"],
            llm=self._llm(),
            max_iter=12,
            tools=[prediction_tool() ,FileReadTool(), FileWriterTool()],
        )

    """
    @task
    def data_transformation_t(self) -> Task:
        
        return Task(
            config=self.tasks_config["correlation_analysis"],  # type: ignore[index]
            human_input=True,) # modelling receives cleaning output

    """
    
    
    @task
    def machine_learning_t(self) -> Task:
        
        return Task(
            config=self.tasks_config["machine_learning"],
            human_input=True,)  # type: ignore[index]) # modelling receives cleaning output

    @crew
    def crew(self) -> Crew:
        """Creates the Content Crew — sequential execution (no manager needed)."""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically populated by @agent decorators
            tasks=self.tasks,    # Automatically populated by @task decorators
            process=Process.sequential,
            verbose=True,
            output_log_file = True
        )
