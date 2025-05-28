import os

import requests
import streamlit as st

st.write(os.getcwd())

st.title("MCP Interface")
if "latest_screenshot" not in st.session_state:
    st.session_state.latest_screenshot = None
if "monitors" not in st.session_state:
    st.session_state.monitors = None
if "primary_monitor" not in st.session_state:
    st.session_state.primary_monitor = None

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))


def server_endpoint() -> list | bool:
    try:
        res = requests.get("http://127.0.0.1:8001/mcp/list_functions")
        return res.json()
    except:
        return False


# Create sticky container for the image (sidebar approach)
st.sidebar.header("Latest Screenshot")
screenshot_container = st.sidebar.container()


# Display the screenshot function - called once at startup
def show_screenshot():
    if st.session_state.latest_screenshot:
        img_path = os.path.join(parent_dir, st.session_state.latest_screenshot)
        screenshot_container.image(
            img_path,
            caption=st.session_state.latest_screenshot,
            use_container_width=True,
        )
        if screenshot_container.button("Refresh Image"):
            st.rerun()
    else:
        screenshot_container.info(
            "No screenshot available yet. Run a screenshot capture endpoint."
        )


# Create container for monitors
st.sidebar.header("Monitors")
monitors_container = st.sidebar.container()


# Display monitors function
def show_monitors():
    if st.session_state.monitors:
        monitors_container.subheader("Available Monitors")
        for monitor in st.session_state.monitors:
            is_primary = ""
            if st.session_state.primary_monitor and monitor.get("id") == st.session_state.primary_monitor:
                is_primary = " (PRIMARY)"
            
            monitors_container.write(f"• Monitor {monitor.get('id')}{is_primary}")
            monitors_container.write(f"  - Resolution: {monitor.get('width')}x{monitor.get('height')}")
            monitors_container.write(f"  - Position: ({monitor.get('left')}, {monitor.get('top')})")
            monitors_container.write("---")
    else:
        monitors_container.info("No monitors available yet.")


# Main content area
st.header("Available Functions")
data = server_endpoint()

if data and isinstance(data, dict) and "functions" in data:
    for entry in data["functions"]:
        name = entry.get("name", "Unnamed")
        st.subheader(name)
        if st.button(f"Run: {name}"):
            try:
                res = requests.post(
                    "http://127.0.0.1:8001/mcp/call_function",
                    headers={"Content-Type": "application/json"},
                    json={"function_name": name, "params": {}},
                )
                result = res.json()

                # Check if this is a screenshot-related result
                if "screenshot_path" in result and result["screenshot_path"]:
                    st.session_state.latest_screenshot = result["screenshot_path"]
                    st.success(
                        f"Updated screenshot: {st.session_state.latest_screenshot}"
                    )
                    st.rerun()

                # Check for monitors in the result
                if "monitors" in result and result["monitors"]:
                    st.session_state.monitors = result["monitors"]
                    if "primary" in result and result["primary"]:
                        st.session_state.primary_monitor = result["primary"]
                    st.success(f"Found {len(st.session_state.monitors)} monitors")
                    st.rerun()

                # Display the response JSON
                st.json(result)

            except Exception as e:
                st.error(f"Request failed: {e}")

# Display the screenshot and monitors initially
show_screenshot()
show_monitors()
