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


import os
import sys
import re
from typing import Annotated, Optional, Type, List, Dict, Any
from langchain_google_vertexai import ChatVertexAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from google.cloud import bigquery
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
# from PIL import Image, ImageDraw
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

# Configuration
PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.0-flash-exp"

class State(TypedDict):
    messages: Annotated[list, add_messages]
    # tables: Optional[List[Dict[str, Any]]] = None
    # query: Optional[str] = None
    # error: Optional[str] = None
    # job_id: Optional[str] = None

graph_builder = StateGraph(State)
memory = MemorySaver()
        
@tool
def bigquery_job_details_tool(job_id: str) -> str:
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
        
@tool   
def bigquery_table_schema_tool(dataset_id: str, table_id: str) -> str:
        """Retrieves the schema of a BigQuery table."""
        bigquery_client = bigquery.Client(project=PROJECT_ID)
        try:
            table_ref = bigquery_client.dataset(dataset_id).table(table_id)
            table = bigquery_client.get_table(table_ref)
            schema_info = []
            for field in table.schema:
                schema_info.append(f"{field.name} ({field.field_type})")
            print(schema_info)
            # return "Table Schema:\n" + "\n".join(schema_info)
            return {"dataset:": dataset_id, "table": table_id, "schema": schema_info}
        except Exception as e:
            return f"Error getting table schema: {e}"
        
def suggest_fixes(state: State):
    messages = state["messages"]
    tool_result_messages = [msg.content for msg in messages if isinstance(msg, ToolMessage)]

    print(f"Previous tool messages: {tool_result_messages}")

    if tool_result_messages:
        # tool_result = tool_result_message.content
        fix_prompt = [
            HumanMessage(
                content=f"""Based on the following BigQuery job details, suggest possible fixes, especially if there are errors:
                \n\n{tool_result_messages}
                \n\nIf the error is related to table schemas or data types and column data types are not included above, consider using the `bigquery_table_schema` tool to get more information about the relevant tables.
                \n\nIf the schema is already provided do not use the `bigquery_table_schema` tool.
                """
            )
        ]
        response = llm_with_tools.invoke(fix_prompt)
        return {"messages": [response]}
    else:
        # Handle cases where no ToolMessage is found (e.g., initial chatbot turn)
        return {"messages": [HumanMessage(content="No tool result yet. What can I help you with?")]}
    

def should_continue(state):
    messages = state['messages']
    last_message = messages[-1]

    if 'function_call' not in last_message.additional_kwargs:
        return 'END'
    else:
        return 'TOOLS'  # More descriptive name

llm = ChatVertexAI(model=MODEL_NAME, google_project=PROJECT_ID, location=LOCATION)

# Bind LangChain tools.
tools = [bigquery_job_details_tool, bigquery_table_schema_tool]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools)
graph_builder.add_node("TOOLS", tool_node)  # Use consistent naming
graph_builder.add_node("suggest_fixes", suggest_fixes)

graph_builder.add_conditional_edges(
    "chatbot",
    should_continue,
    {
        "TOOLS": "TOOLS",  # Correct edge name
        "END": END,  # End the graph when no tool is needed
    },
)

graph_builder.add_conditional_edges(
    "suggest_fixes",
    should_continue,
    {
        "TOOLS": "TOOLS",  # Correct edge name
        "END": END,  # End the graph when no tool is needed
    },
)

graph_builder.add_edge("TOOLS", "suggest_fixes") # Continue to suggest fixes after tools are used
graph_builder.add_edge("suggest_fixes", "chatbot")  # Loop back to chatbot

graph_builder.set_entry_point("chatbot")

graph = graph_builder.compile(checkpointer=memory)


# Optional print the graph

# def save_img(img_bytes, filename='graph.jpg'):
#     try:
#         with open(filename, 'wb') as f:
#             f.write(img_bytes)
#         print(f"Bytes saved as {filename}")
#     except Exception as e:
#         print(f"Error saving bytes: {e}")

# img = graph.get_graph().draw_mermaid_png()
# save_img( img)


config = {"configurable": {"thread_id": "1"}}

def stream_graph_updates(user_input: str):
    for event in graph.stream(
         {"messages": [HumanMessage(content=user_input)]},
         config=config,
         stream_mode="values",
         ):
        state = event # update the state with the returned state.
        if "messages" in event:
            event["messages"][-1].pretty_print()
        elif "BigQueryJobDetailsToolOutput" in event:
            event["messages"][-1].pretty_print()


def main():
    print(f"Project ID: {PROJECT_ID} (OS env PROJECT_ID)")
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            stream_graph_updates(user_input)
        except Exception as e: 
            print(f"An error occurred: {e}") 
            break

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(main())