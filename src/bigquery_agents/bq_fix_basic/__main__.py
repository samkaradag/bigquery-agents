# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script defines a basic agent that can fix BigQuery queries by retrieving bigquery job. This script defines tools by extending BaseTool and Pydantic schemas.
# Please check bq_fix for a more comprehensive version that also retrieves table schemas and uses decorators for tool definitions
    
import os
import sys
from typing import Annotated, Optional, Type
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import BaseMessage, HumanMessage
from google.cloud import bigquery
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver


# Configuration
PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = "us-central1"  # Or your preferred location
MODEL_NAME = "gemini-2.0-flash-exp"  # Or your preferred model

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# Define Input Schema for the BigQuery Job Details tool
class BigQueryJobDetailsInput(BaseModel):
    job_id: str = Field(..., description="The ID of the BigQuery job to retrieve details for.")

# Define the Custom Tool for retrieving job details
class BigQueryJobDetailsTool(BaseTool):
    name: str = "bigquery_job_details"
    description: str = "Retrieves details of a BigQuery job, including query and errors."
    args_schema: Type[BigQueryJobDetailsInput] = BigQueryJobDetailsInput

    def _run(self, job_id: str) -> str:
        """Retrieves details of a BigQuery job."""
        bigquery_client = bigquery.Client(project=PROJECT_ID)
        try:
            job = bigquery_client.get_job(job_id)
            query = job.query
            errors = job.error_result
            if errors:
                error_message = f"Error: {errors['message']}"
            else:
                error_message = "No errors."
            return f"Query: {query}\n{error_message}"
        except Exception as e:
            return f"Error getting job details: {e}"

def suggest_fixes(state: State):
    """
    Takes the tool's output (BigQuery job details) and prompts the LLM to suggest fixes.
    """
    messages = state["messages"]
    tool_result = messages[-1].content  # Assuming last message is the ToolMessage
    # Create a new prompt for the LLM
    fix_prompt = [
        HumanMessage(
            content=f"Based on the following BigQuery job details, suggest possible fixes, especially if there are errors:\n\n{tool_result}"
        )
    ]

    response = llm.invoke(fix_prompt)
    return {"messages": [response]}

llm = ChatVertexAI(model=MODEL_NAME, google_project=PROJECT_ID, location=LOCATION)

# Instantiate the tool
bigquery_job_lookup_tool = BigQueryJobDetailsTool()

tools = [bigquery_job_lookup_tool]
llm_with_tools = llm.bind_tools(tools)

memory = MemorySaver()

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("suggest_fixes", suggest_fixes)
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "suggest_fixes")
graph_builder.add_edge("suggest_fixes", "chatbot") # Loop back to chatbot for further interaction
graph_builder.set_entry_point("chatbot")


graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

def stream_graph_updates(user_input: str):
    for event in graph.stream(
         {"messages": [HumanMessage(content=user_input)]},
         config=config,
         stream_mode="values",
         ):
        if "messages" in event:
            event["messages"][-1].pretty_print()


def main():
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(main())
