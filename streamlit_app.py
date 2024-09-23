import streamlit as st
import requests

st.title("AI RESPONSE GENERATOR")

user_input = st.text_input("Prompt")

sites_required=int(st.slider("Number of sites to scrape from.", min_value=3, max_value=30, value=10, step=1, disabled=False, label_visibility="visible"))

click=st.button("Enter", on_click=None, type="secondary", disabled=False, use_container_width=False)

message_placeholder=st.empty()

if click and user_input:
    with st.spinner('Generating response. Please wait...'):
        #send POST request to Flask API
        try:
            response = requests.post('http://localhost:5000/process', json={"input": user_input,"sites_required": sites_required}, timeout=250)
            response.raise_for_status()
            if response.status_code == 200:
                result = response.json()  #parse JSON response from Flask

                message_placeholder.empty()
                st.write("Result:")
                st.write(result.get("result", "No result returned"))
            else:
                st.error(f"API returned an error: {response.status_code}")
                st.write(response.text)

        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")
