import streamlit as st
from huggingface_hub import InferenceClient

# Initialize Hugging Face Inference client with the API key
client = InferenceClient(api_key="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")

# Set the title for your Streamlit app
st.title("Mistral Chatbot")

# Session state management
if "mistral_model" not in st.session_state:
    st.session_state["mistral_model"] = "mistralai/Mistral-7B-Instruct-v0.2"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the conversation so far
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("What is up?"):
    # Add user input to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in the chat interface
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate the response using the Mistral model
    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]
    
    # Send the request to Hugging Face API to get model response
    stream = client.chat.completions.create(
        model=st.session_state["mistral_model"], 
        messages=messages, 
        max_tokens=100,
        stream=True
    )
    
    # Collect the response in a single string
    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content

    # Display the entire response in one go
    with st.chat_message("assistant"):
        st.markdown(response)  # Display the full response at once
    
    # Add assistant response to session state
    st.session_state.messages.append({"role": "assistant", "content": response})
