import requests
import streamlit as st


def server_active() -> bool:
    try:
        res = requests.get("http://127.0.0.1:8001/health")
        return res.status_code == 200
    except:
        return False


def server_endpoint() -> list | bool:
    try:
        res = requests.get("http://127.0.0.1:8001/mcp/list_functions")
        return res.json()
    except:
        return False


st.title("Is the server running?")
st.text(f"Server active: {server_active()}")

st.title("Available Endpoints:")
data = server_endpoint()

if data and isinstance(data, dict) and "functions" in data:
    for entry in data["functions"]:
        st.subheader(entry.get("name", "Unnamed"))
        st.write(entry.get("doc", "No documentation available."))
else:
    st.write("Failed to fetch endpoint list.")
