import streamlit as st
import json
import warnings
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_cohere.embeddings import CohereEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import ShellTool
from langchain.tools import Tool
from langchain.agents import create_react_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor

# Initialize the Groq LLM
groq_api_key = "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"
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-groq-70b-8192-tool-use-preview", temperature=0.1)

# Embeddings and Vector Store Setup
embeddings_obj = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key="cdu2whMVXxqmnCTG8wAIqd35ghLPgCnCKG7b8liN")
db = FAISS.load_local("Deep_learning_Java_Codes", embeddings_obj, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

# Define Tools
retriever_tool = create_retriever_tool(retriever, "Vector_database", description="This tool retrieves documents file from a vector database based on the user's query.")

shell_tool = ShellTool(
    name="Intermediate Answer",
    description="This tool is used for commanding into windows os terminal and getting back the results of all types (eg.'grep','cls')",
    handle_tool_error=True,
    handle_validation_error=True,
    response_format="content",
)

def find_files_by_name(filename: str):
    root_dir = "C:\\Users\\Rudra\\Desktop"
    matches = []
    def search_directory(directory):
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file() and entry.name == filename:
                    matches.append(entry.path)
                elif entry.is_dir():
                    search_directory(entry.path)
    search_directory(root_dir)
    return matches

def refine_prompt(prompt: str, method: str = 'summarization'):
    if method == 'summarization':
        refined_prompt = llm.invoke(f"Summarize the following query for a vector database search: {prompt}")
    elif method == 'keyword_extraction':
        refined_prompt = llm.invoke(f"Extract the most important keywords for vector database search from this query: {prompt}")
    elif method == 'stopword_removal':
        refined_prompt = llm.invoke(f"Remove stopwords and simplify the query for a more efficient search: {prompt}")
    else:
        raise ValueError(f"Unknown refinement method: {method}")
    return refined_prompt

file_finder = Tool.from_function(
    func=find_files_by_name,
    name="file_finder",
    description="A tool for finding the file by their filename in 'C:\\Users\\Rudra\\Desktop' directory"
)

# prompt_refiner = Tool.from_function(
#     func=refine_prompt,
#     name="Prompt Refiner",
#     description="Refines the prompt for appropriate vector database queries using different methods."
# )

tools = [ retriever_tool, file_finder]

template = """Answer the following questions as best you can. You have access to the following tools:
{tools}
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!
Question: {input}
Thought:{agent_scratchpad}"""
prompt_react = PromptTemplate.from_template(template)

memory = ConversationBufferMemory(memory_key="history", return_messages=True)
react_agent = create_react_agent(llm, tools, prompt_react)

agent_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
    memory=memory,
    max_execution_time=300
)

# Streamlit GUI
st.title("LangChain Document Retriever with Prompt Refinement")
st.subheader("Enter your prompt and refine it with advanced AI tools.")

# User Inputs
prompt_input = st.text_area("Enter your prompt:")

# Button to Run
if st.button("Run"):
        refined_prompt = refine_prompt(prompt_input)
        refined_prompt_str = str(refined_prompt)  # Convert refined prompt to a string
        
        # Get response from agent
        warnings.filterwarnings("ignore")
        response = agent_executor.invoke({"input": refined_prompt_str})["output"]

        # Ensure the response is also a string before saving it to memory
        response_str = str(response)

        # Save the refined prompt and response context to memory
        agent_executor.memory.save_context({"input": refined_prompt_str}, {"output": response_str})

        st.subheader("Response")
        st.write(response_str)
