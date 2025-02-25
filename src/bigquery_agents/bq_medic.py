import os
import sys
from typing import Annotated, Optional, Type, List, Dict, Any
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from google.cloud import bigquery
from typing_extensions import TypedDict
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
# from PIL import Image, ImageDraw
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles


# Configuration
PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.0-flash-exp"

class State(TypedDict):
    messages: Annotated[list, add_messages]
    tables: Optional[List[Dict[str, Any]]] = None
    query: Optional[str] = None
    error: Optional[str] = None
    job_id: Optional[str] = None

graph_builder = StateGraph(State)
memory = MemorySaver()


# Define Input Schema for the tool
class BigQueryJobDetailsInput(BaseModel):
    job_id: str = Field(..., description="The ID of the BigQuery job to retrieve details for.")

# Define the Custom Tool
class BigQueryJobDetailsTool(BaseTool):
    name: str = "bigquery_job_details"  # Add type annotation :str
    description: str = "Retrieves details of a BigQuery job, including query and errors."  # Add type annotation :str
    args_schema: type[BigQueryJobDetailsInput] = BigQueryJobDetailsInput

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
            # return f"Query: {query}\n{error_message}"
            # return {"BigQueryJobDetailsToolOutput":{"job_id": job_id, "query": query, "errors": error_message, "status": job.state}}
            return Command(
                update={
                    "job_id": job_id, "query": query, "errors": error_message,
                    "messages": [ ToolMessage(content=f"Successfully looked up BigQuery Job. Query: {query}\n{error_message}", tool_call_id=self.tool_call_id)]
                }
            )
        except Exception as e:
            return f"Error getting job details: {e}"
        
# Define Input Schema for the table schema tool
class BigQueryTableSchemaInput(BaseModel):
    dataset_id: str = Field(..., description="The ID of the dataset containing the table.")
    table_id: str = Field(..., description="The ID of the table to retrieve the schema for.")

# Define the Custom Tool for retrieving table schema
class BigQueryTableSchemaTool(BaseTool):
    name: str = "bigquery_table_schema"
    description: str = "Retrieves the schema of a BigQuery table, including column names and data types."
    args_schema: Type[BigQueryTableSchemaInput] = BigQueryTableSchemaInput

    def _run(self, dataset_id: str, table_id: str) -> str:
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
            # return {"dataset:": dataset_id, "table": table_id, "schema": schema_info}
            return Command(
                update={
                    "tables":{"dataset:": dataset_id, "table": table_id, "schema": schema_info},
                    "messages": [ ToolMessage(content=f"Tables: {'dataset:': dataset_id, 'table': table_id, 'schema': schema_info}", tool_call_id=self.tool_call_id)]
                }
            )
        except Exception as e:
            return f"Error getting table schema: {e}"
        
def suggest_fixes(state: State):
    messages = state["messages"]
    tool_result_message = next((msg for msg in reversed(messages) if isinstance(msg, ToolMessage)), None)

    if tool_result_message:
        tool_result = tool_result_message.content
        fix_prompt = [
            HumanMessage(
                content=f"""Based on the following BigQuery job details, suggest possible fixes, especially if there are errors:
                \n\n{tool_result}
                \n\nIf the error is related to table schemas or data types, consider using the `bigquery_table_schema` tool to get more information about the relevant tables such as double checking data type of a column.
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

# Instantiate the tools
bigquery_job_details_tool = BigQueryJobDetailsTool()
bigquery_table_schema_tool = BigQueryTableSchemaTool()

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



# def stream_graph_updates(user_input: str):
#     state = State(messages=[HumanMessage(content=user_input)]) #initialize the state object
    
#     for event in graph.stream(
#          state,
#          config=config,
#          stream_mode="yield_state",
#          ):
#         state = event # update the state with the returned state.

#         # if "messages" in state.__dict__: #access the state with __dict__
#         state.messages[-1].pretty_print()
#             #  if len(state.messages) > 0:
#             #     if isinstance(state.messages[-1], HumanMessage) == False:
#             #         print(state.messages[-1])
#             #     else:
#             #         state.messages[-1].pretty_print()

#         if "tool_invocation" in state.__dict__ and state.tool_invocation is not None:
#             if state.tool_invocation.tool == "BigQueryJobDetailsTool": #if the tool matches
#                 try:
#                     state.job_id = state.tool_invocation.tool_input["job_id"] #get the job_id
#                     print(f"Job ID set to: {state.job_id}") #print the job id.
#                 except KeyError:
#                     print("job id not found in tool_input")


def main():
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            stream_graph_updates(user_input)
        except Exception as e: # Catch exceptions for debugging
            print(f"An error occurred: {e}") # Print the error message
            break

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(main())
