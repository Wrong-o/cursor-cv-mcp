with left_col:
    st.header("Available Endpoints")
    data = server_endpoint()

    if not data:
        st.error("Could not connect to MCP server. Make sure it's running.")
    elif isinstance(data, dict) and "functions" in data:
        # Group functions by category for better organization
        functions = data["functions"] 