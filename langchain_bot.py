import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from src.bot_session import get_session, save_session, delete_session
from src.bot_logging import logger
from src.tools import retrieve, recommend_blog

import logging

from langchain.chat_models import init_chat_model
from langchain import hub
from typing_extensions import List, TypedDict, Annotated
import operator
from langgraph.graph import MessagesState, StateGraph
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
import json
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages import messages_to_dict, messages_from_dict

from pydantic import BaseModel
from typing import List
from langchain_core.messages import BaseMessage

from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
import uuid
import time
# from IPython.display import Image, display
import numpy as np

import redis
# import boto3


app = Flask(__name__)
CORS(app)


# Constants
MAX_SESSION_TIME = 30 * 60      # 30 mins in seconds
INACTIVITY_TIMEOUT = 5 * 60     # 5 mins in seconds
MAX_SESSIONS = 100               # max concurrent sessions


llm = init_chat_model("gpt-4o-mini", model_provider="openai")

class State(TypedDict):
    rec_prod: bool
    reci_form: bool 
    messages: Annotated[list[BaseMessage], operator.add]


# ------------------------------------------------------------------------------------------------------------------------------------------------
# Sub graph for product recommendation function
catalog_json = {'placeholder'}

@tool
def ask_or_recommend(state: State):
    """Recommend a product if condent or ask follow up questions if not confident."""
    messages = state["messages"]
    print(state["rec_prod"],  state["reci_form"])
    # System prompt
    system_prompt = f"""
    You are a helpful assistant trained to recommend pet care products.

    Based on the prior conversation, do the following:
    1. Summarize what the user is looking for
    2. Try to match a product from the catalog below
    3. If confident, recommend a product
    4. If not confident, ask a concise follow-up question to clarify the user's needs without recommending any product.


    Product catalog:
    cat spray:remove cat allergens, dog spray:remove dog allergens, dust spray:remove dust allergens
    """
    #{"cat spray":"remove cat allergens", "dog spray":"remove dog allergens", "dust spray":"remove dust allergens",}
    print("=== ask_or_recommend called ===")
    response = llm.invoke([SystemMessage(content=system_prompt)] + [HumanMessage(messages[0].content)])
    print("=== ask_or_recommend called ===")
    print(response.content)
    print(state["rec_prod"],  state["reci_form"])
    return {"messages": [response], "rec_prod": state["rec_prod"], "reci_form": state["reci_form"]}

def ask_more(state: State):
    """Ask follow-up questions."""
    messages = state["messages"]

    # System prompt
    system_prompt = f"""
    You are a helpful assistant trained to recommend pet care products.

    Based on the prior conversation, do the following:
    1. Summurize previous messages.
    2. Ask a concise follow-up question to clarify the user's needs
    3. Do not recommend products at this step.

    Product catalog:
    cat spray:remove cat allergens, dog spray:remove dog allergens, dust spray:remove dust allergens
    """
    #{"cat spray":"remove cat allergens", "dog spray":"remove dog allergens", "dust spray":"remove dust allergens",}

    response = llm.invoke([SystemMessage(content=system_prompt)] + messages)
    return {"messages": [response], "rec_prod": state["rec_prod"], "reci_form": state["reci_form"]}

def check_if_confident(state: State):
    last_ai = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
    if last_ai:
        content = last_ai.content.lower()
        if any(x in content for x in ["we recommend", "you should try", "i suggest", "i recommend"]):
            print('--- route to form ----')
            state['rec_prod'] = True
            return "store_product_response"
    print('--- route to ask more ---')
    return "ask_more"

# ------------------------------------------------------------------------------------------------------------------------------------------------
# Main graph of the chat bot
graph_builder = StateGraph(State)

def store_product_response(state: State):
    if state['rec_prod'] == True and state['reci_form'] == False:
        print(' -- FORM -- ')
        return {"messages": [AIMessage(content="__FORM__")], "rec_prod": state["rec_prod"], "reci_form": state["reci_form"]}
    elif state['rec_prod'] == True and state['reci_form'] == True:
        state['rec_prod'] = False
        state['reci_form'] = False
        print(' -- give recommendation -- ')
        return state
    else:
        return {"messages": [AIMessage(content="Something went wrong! Please retry later")], "rec_prod": state["rec_prod"], "reci_form": state["reci_form"]}

# Routing functions
def route_query(state: State) -> str:
    last_message = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
    if not last_message:
        return END

    if last_message.tool_calls:
        tool_names = [t["name"] for t in last_message.tool_calls]
        if "ask_or_recommend" in tool_names:
            print(' *************************** recommend ***************************')
            return "tools_recommend"
        else:
            print(' *************************** Other ***************************')
            return "tools"
        
    return END


# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: State):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve, ask_or_recommend, recommend_blog])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    print(' =-=-=-=-=-=-= INSIDE query_or_respond =-=-=-=-=-=-=-=-=')
    print(state["rec_prod"], state["reci_form"])
    print(response)
    print(' =-=-=-=-=-=-= INSIDE query_or_respond =-=-=-=-=-=-=-=-=')
    return {"messages": [response], "rec_prod": state["rec_prod"], "reci_form": state["reci_form"]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve, recommend_blog])
tools_recommend = ToolNode([ask_or_recommend])

# Step 3: Generate a response using the retrieved content.
def generate(state: State):
    """Generate answer."""
    # If product recommendation exists and we just returned from form

    if "product_recommendation_result" in state:
        return {"messages": [response], "rec_prod": state["rec_prod"], "reci_form": state["reci_form"]}

    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are a warmhearted and patient customer service representative. "
        "Use the 3 tools [retrieve, ask_or_recommend, recommend_blog], "
        "to answer questions, recommend products and recommend blogs to read. "
        "When recommend products, if you get follow up questions, ask the user the questions"
        "If you don't know the answer, send '__FORM__'. "
        "Respond using HTML formatting (e.g., <b>, <br>, etc)."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response], "rec_prod": state["rec_prod"], "reci_form": state["reci_form"]}


graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.add_node("tools_recommend", tools_recommend)
graph_builder.add_node("ask_more", ask_more)
graph_builder.add_node("store_product_response", store_product_response)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    route_query,
    {
        "tools_recommend": "tools_recommend",
        "tools": "tools",
        END: END
    }
)

graph_builder.add_conditional_edges(
    "tools_recommend",
    check_if_confident,
    {
        "ask_more": "ask_more", 
        "store_product_response": "store_product_response"
    }
)

#graph_builder.add_edge("product_recommendation", "store_product_response")
graph_builder.add_edge("store_product_response", END)

# And allow regular use of generate
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)


memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

rec_prod = False 
reci_form = False 

@app.route('/api/chat', methods=['POST'])
def chat():
    # Specify an ID for the thread
    now = time.time()

    # Try to get session_id or create a new one
    session_id = request.json.get('session_id')
    session = get_session(session_id)
    logging.info(f"Received session_id: {session_id}")
    logging.info(f"Loaded session: {session}")

    new_session = False

    if not session:
        # Clean up not needed — Redis handles TTL
        if session_id:
            delete_session(session_id)  # Clean up possibly expired ID

        # Specify an ID for the thread
        session_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": session_id}}
        session = {
                "messages": [SystemMessage("# Identity \n You are a helpful, polite, and patient assistant that answers customers' questions and tries to convert them to actual buyers. \n # Instructions \n After you receive the question from the customer, use template answer to response with a nice tone. Please format your response with HTML tags (<b>, <br>, etc, instead of Markdown) for better readability. Divide the response into separate paragraphs when it is more than 50 words. \n IMPORTANT POINT: 1. Our products DO NOT cure allergies; 2. topper refers to 'cat food topper'; 3. spray possibly refers to 'cat allergen neutralizing spray', 'dog allerge neutralizing spray' or 'dust allergen spray'. 4. Don't include links or reference unless asked to.")],
            "start_time": now,
            "last_active": now,
            "rec_prod": False, 
            "reci_form": False,
        }
        new_session = True
        logging.info(f"New session created: {session_id}")
    else:
    
        # Check expiration
        if now - session["last_active"] > INACTIVITY_TIMEOUT or now - session["start_time"] > MAX_SESSION_TIME:
            delete_session(session_id)
            logging.info(f"Session expired: {session_id}")
            return jsonify({"error": "Session expired. Please refresh to start a new chat."}), 440
        # Specify an ID for the thread
        config = {"configurable": {"thread_id": session_id}}
        session['messages'] = messages_from_dict(session['messages'])

    input_message = request.json.get("message")
    logging.info(f"Session {session_id} received message: {input_message}")
    session["last_active"] = now
    session['messages'].append(HumanMessage(input_message))
    # For debugging
    print(' --- LOOPING --- ')
    print(session['rec_prod'] )
    print(session['reci_form'] )
    for step in graph.stream(
        {"rec_prod": rec_prod,
        "reci_form": reci_form,
        "messages": session['messages']},
        stream_mode="values",
        config=config,
    ):
        
        # print(step)
        step['messages'][-1].pretty_print()
        response = step['messages'][-1].content
        print(' ())()()()()()()()()()()()()()()()')
        print(step['rec_prod'])
        print(step['reci_form'])
        print(' ())()()()()()()()()()()()()()()()')
        rec_bool = step['rec_prod'] 
        form_bool = step['reci_form'] 

    session['messages'].append(AIMessage(response))
    session['rec_prod'] = rec_bool
    session['reci_form'] = form_bool
    # output = graph.invoke({"messages": [{"role": "user", "content": input_message}]}, config=config)
    # # Find the last AIMessage with content
    # for msg in reversed(output['messages']):
    #     if isinstance(msg, AIMessage) and msg.content.strip():
    #         response = msg.content.strip()
    #         break
    session['messages'] = messages_to_dict(session['messages'])
    save_session(session_id, session)
    return jsonify({
        "response": response,
        "session_id": session_id,
        "new_session": new_session,
    })
                
@app.route('/api/form', methods=['POST'])
def handle_form():
    data = request.json
    name = data.get("name")
    email = data.get("email")
    message = data.get("message")
    session_id = data.get("session_id", "unknown")
    
    session = get_session(session_id)

    output_dict = {"session_id": session_id,"name": name, "email": email, "message": message, "conversation": session['messages']}
    
    # send out email to support@pacagen.com
    # send_email(output_dict)

    # Record the entries
    f = open(f"form_submission/{session_id}.json", 'w')
    f.write(json.dumps(output_dict, indent=4))
    f.close()
    
    return jsonify({"message": "Your response has been recorded!"})


# def send_email(msg_dict):
#     # initiate a client    
#     client = boto3.client('ses', region_name='us-east-1')  # Change region as needed
    
#     # send emails
#     response = client.send_email(
#         Source='bot@pacagen.com',
#         Destination={'ToAddresses': ['support@pacagen.com']},
#         Message={
#             'Subject': {'Data': 'Form submission from chatbot'},
#             'Body': {'Text': {'Data': json.dumps(msg_dict, indent=2)}}
#         }
#     )
#     return None


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001)
