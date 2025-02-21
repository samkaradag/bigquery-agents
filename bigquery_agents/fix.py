import os
from typing import Annotated, Optional, Type, List, Any, Dict
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
PROJECT_ID = "samets-ai-playground"  #  Good practice is to get it from env, but this works
LOCATION = "us-central1"  # Or your preferred location
MODEL_NAME = "gemini-2.0-flash-exp"  # Or your preferred model

class State(TypedDict):
    messages: Annotated[list, add_messages]
    tables: Optional[List[Dict[str, Any]]] = None
    query: Optional[str] = None  # Make query optional
    error: Optional[str] = None # Make error optional
    job_id: Optional[str] = None  # Make job_id optional
    text: Optional[str] = None #add text

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
            return "Table Schema:\n" + "\n".join(schema_info)
        except Exception as e:
            return f"Error getting table schema: {e}"
        
def retrieve_table_schema(state: State):
    """
    Takes the tool's output (BigQuery job details) and prompts the LLM to suggest fixes.
    """
    query = state["query"]
    # Create a new prompt for the LLM
    retrieve_table_schema_prompt = [
        HumanMessage(
            content=f"Based on the following BigQuery query, provide all the table schema information:\n\n{tool_result}"
        )
    ]

    response = llm.invoke_with_tools(retrieve_table_schema_prompt)
    return {"messages": [response]}


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

    response = llm.invoke_with_tools(fix_prompt)
    return {"messages": [response]}

llm = ChatVertexAI(model=MODEL_NAME, google_project=PROJECT_ID, location=LOCATION)


# Instantiate the tools
bigquery_job_details_tool = BigQueryJobDetailsTool()
bigquery_table_schema_tool = BigQueryTableSchemaTool()

# Bind LangChain tools.
tools = [bigquery_job_details_tool, bigquery_table_schema_tool]
llm_with_tools = llm.bind_tools(tools)


memory = MemorySaver()

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def process_tool_result(state: State):
    """Processes the tool result and updates the state."""
    messages = state['messages']
    last_message = messages[-1]

    if isinstance(last_message, ToolMessage):
        try:
            # Parse the tool output to extract relevant information
            tool_output = last_message.content
            job_id = None
            query = None
            error = None
            text = tool_output

            # Check if the tool is bigquery_job_details and extract details
            if last_message.name == "bigquery_job_details":
                # Extract job_id from the tool call
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage) and msg.content.startswith("bigquery_job_details"):
                        try:
                            job_id = msg.content.split('"job_id": "')[1].split('"')[0]
                            break  # Exit loop once job_id is found

                        except IndexError:
                            pass

                if "Query:" in tool_output:
                    query_start = tool_output.find("Query:") + len("Query:")
                    query_end = tool_output.find("\n", query_start)
                    query = tool_output[query_start:query_end].strip()

                if "Error:" in tool_output:
                    error_start = tool_output.find("Error:") + len("Error:")
                    error = tool_output[error_start:].strip()

            return {
                "messages": [last_message],
                "job_id": job_id,
                "query": query,
                "error": error,
                "text": text
            }

        except Exception as e:
            return {"messages": [last_message], "error": str(e)}  # Catch any parsing errors

    return {"messages": [last_message]}  # Default return if not a ToolMessage



graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("process_tool_result", process_tool_result)  # Add the new node
graph_builder.add_node("suggest_fixes", suggest_fixes)
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

graph_builder.add_edge("tools", "process_tool_result")
graph_builder.add_edge("process_tool_result", "suggest_fixes")  # Connect to suggest_fixes

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
        if "job_id" in event and event["job_id"] is not None:
            print(f"Job ID: {event['job_id']}")
        if "query" in event and event["query"] is not None:
            print(f"Query: {event['query']}")
        if "error" in event and event["error"] is not None:
            print(f"Error: {event['error']}")
        if "text" in event and event["text"] is not None:
            print(f"Text: {event['text']}")

        # for value in event.values():
        #     print("Assistant:", value["messages"][-1].content)

def main():
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)

if __name__ == "__main__":
    main()