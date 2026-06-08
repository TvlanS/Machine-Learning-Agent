#!/usr/bin/env python
from pathlib import Path
from pyprojroot import here

from crewai.flow import Flow, listen, start


import sys

import os
import sys


sys.path.insert(0, str(here()))      # project root — makes toolbox/, Utils/ importable
sys.path.insert(0, str(here("src")))  # src root

from data_modelling.crew import ContentCrew
from pydantic import BaseModel
from datetime import datetime

now = datetime.now()
format_ = now.strftime("%Y_%m_%d_%H_%M_%S")


class ContentState(BaseModel):

    dataset_path: str = "C:/Users/tvlan/Documents/1.0 Python/data_modelling/datafolder/early_sepsis_full_simulated_dataset_dropped_encoded_20260607_195121.csv"
    goal_crew: str = "Target feature = sepsis_risk , trasformation summmary = C:/Users/tvlan/Documents/1.0 Python/data_modelling/datafolder/transformation_summary_2026_06_07_19_50_49.md "
    datadesc: str = "C:/Users/tvlan/Documents/1.0 Python/data_modelling/datafolder/dataset_description.txt"
    hard_rules: str = "If the tools don't run stop and report back to manager. If the user explicitly mentioned to skip  a task you must obliged and skip to the next task. Never use emojis, generate any document based on ISO/IEEE 82079-1. Never read CSV filess"
    date_time : str = f"current date and time {format_} , format = %Y_%m_%d_%H_%M_%S "
    result: str = ""

class ContentFlow(Flow[ContentState]):
    """Two-step ETL flow: clean → model."""

    @start()
    def run_cleaning_crew(self):
        """Step 1: run the cleaning + modelling crew (sequential tasks)."""
        crew_input = {"dataset_path" :self.state.dataset_path ,
                 "goal_crew" :self.state.goal_crew ,
                 "hard_rules" :self.state.hard_rules,
                  "dataset_description" : self.state.datadesc,
                   "date_time" : self.state.date_time }
        
        result = ContentCrew().crew().kickoff(inputs = crew_input)
        self.state.result = str(result)
        print("Cleaning & Modelling result:", result)
        return result

    @listen(run_cleaning_crew)
    def on_crew_complete(self, crew_result):
        """Step 2: post-process or pass-through the crew result."""
        print("Flow complete. Crew output:", crew_result)
        return crew_result


def kickoff():
    flow = ContentFlow()
    flow.plot()
    flow.kickoff()


if __name__ == "__main__":
    kickoff()
