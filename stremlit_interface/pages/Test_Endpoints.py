import os
import json
from datetime import datetime
import time

import requests
import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

st.write(os.getcwd())

st.title("MCP Interface Testing Dashboard")
if "latest_screenshot" not in st.session_state:
    st.session_state.latest_screenshot = None
if "monitors" not in st.session_state:
    st.session_state.monitors = None
if "primary_monitor" not in st.session_state:
    st.session_state.primary_monitor = None
if "test_results" not in st.session_state:
    st.session_state.test_results = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))


def server_endpoint() -> list | bool:
    try:
        res = requests.get("http://127.0.0.1:8001/mcp/list_functions")
        return res.json()
    except:
        return False


# Create main layout
left_col, right_col = st.columns([3, 2])

# Right sidebar for results and screenshots
with right_col:
    # Test Results Section
    st.header("Test Results")
    results_container = st.container()
    
    # Latest Screenshot Section
    st.header("Latest Screenshot")
    screenshot_container = st.container()
    
    # Monitor Information Section
    st.header("Monitor Information")
    monitors_container = st.container()

# Display the screenshot function
def show_screenshot():
    with screenshot_container:
        if st.session_state.latest_screenshot:
            img_path = os.path.join(parent_dir, st.session_state.latest_screenshot)
            st.image(
                img_path,
                caption=st.session_state.latest_screenshot,
                use_container_width=True,
            )
            if st.button("Refresh Image", key="refresh_screenshot"):
                st.rerun()
        else:
            st.info(
                "No screenshot available yet. Run a screenshot capture endpoint."
            )

# Display monitors function
def show_monitors():
    with monitors_container:
        if st.session_state.monitors:
            for monitor in st.session_state.monitors:
                is_primary = ""
                if st.session_state.primary_monitor and monitor.get("id") == st.session_state.primary_monitor:
                    is_primary = " (PRIMARY)"
                
                st.markdown(f"### Monitor {monitor.get('id')}{is_primary}")
                st.markdown(f"**Resolution:** {monitor.get('width')}x{monitor.get('height')}")
                st.markdown(f"**Position:** ({monitor.get('left')}, {monitor.get('top')})")
                st.divider()
        else:
            st.info("No monitors available yet.")

# Display test results
def show_test_results():
    with results_container:
        if st.session_state.test_results:
            # Create a DataFrame from the test results
            df = pd.DataFrame(st.session_state.test_results)
            st.dataframe(df, use_container_width=True)
            
            if st.button("Clear Results", key="clear_results"):
                st.session_state.test_results = []
                st.rerun()
        else:
            st.info("No test results yet. Run some endpoints to see results here.")
        
        # Show last result details if available
        if st.session_state.last_result:
            st.subheader("Last Result Details")
            st.json(st.session_state.last_result)

# Add a test result
def add_test_result(endpoint, status, details=None):
    timestamp = datetime.now().strftime("%H:%M:%S")
    result = {
        "timestamp": timestamp,
        "endpoint": endpoint,
        "status": status
    }
    
    st.session_state.test_results.append(result)
    st.session_state.last_result = details

# Main content area with endpoints
with left_col:
    st.header("Available Endpoints")
    data = server_endpoint()

    if not data:
        st.error("Could not connect to MCP server. Make sure it's running.")
    elif isinstance(data, dict) and "functions" in data:
        # Group functions by category for better organization
        functions = data["functions"]
        
        # Define categories
        categories = {
            "Screenshot": ["mcp_screenshot_capture", "mcp_screenshot_analyze_image", "mcp_screenshot_list_monitors"],
            "Automation": ["mcp_find_element", "mcp_mouse_click", "mcp_type_text", "mcp_press_key"],
            "Other": []
        }
        
        # Categorize functions
        categorized_functions = {cat: [] for cat in categories}
        for func in functions:
            name = func.get("name", "")
            categorized = False
            for cat, func_list in categories.items():
                if any(name.startswith(prefix) for prefix in func_list):
                    categorized_functions[cat].append(func)
                    categorized = True
                    break
            if not categorized:
                categorized_functions["Other"].append(func)
        
        # Display functions by category
        for category, funcs in categorized_functions.items():
            if funcs:  # Only show category if it has functions
                st.subheader(category)
                for entry in funcs:
                    name = entry.get("name", "Unnamed")
                    with st.expander(f"{name}", expanded=False):
                        st.markdown(f"**Description:** {entry.get('description', 'No description')}")
                        
                        # Special case for functions that need the latest screenshot
                        if name in ["mcp_screenshot_analyze_image", "mcp_find_element"] and st.session_state.latest_screenshot:
                            # For find_element, we need additional parameters
                            if name == "mcp_find_element":
                                # Add link to debug tool
                                st.info("Need help understanding how element finding works? Try the [Element Finder Debug Tool](/Element_Finder_Debug) for a detailed visualization of the process.")
                                
                                # Create tabs for different search methods
                                search_tabs = st.tabs(["Text Search", "OCR/Visual Search", "Advanced Options"])
                                
                                with search_tabs[0]:
                                    # Text-based search options
                                    element_type = st.selectbox(
                                        "Element Type", 
                                        ["text", "button", "input", "image", "icon", "checkbox", "radio", "dropdown", "link", "any"],
                                        key=f"{name}_element_type"
                                    )
                                    search_text = st.text_input("Search Text (leave empty to find any element of the selected type)", "", key=f"{name}_search_text")
                                    match_type = st.radio(
                                        "Match Type", 
                                        ["exact", "contains", "starts_with", "ends_with", "regex"],
                                        horizontal=True,
                                        key=f"{name}_match_type"
                                    )
                                
                                with search_tabs[1]:
                                    # Visual search options
                                    st.markdown("**OCR Settings**")
                                    ocr_enabled = st.checkbox("Enable OCR for text detection", value=True, key=f"{name}_ocr")
                                    detect_ui = st.checkbox("Detect UI elements", value=True, key=f"{name}_detect_ui")
                                    detect_text = st.checkbox("Detect visible text", value=True, key=f"{name}_detect_text")
                                    detect_images = st.checkbox("Detect images", value=True, key=f"{name}_detect_images")
                                
                                with search_tabs[2]:
                                    # Advanced options
                                    st.markdown("**Search Parameters**")
                                    confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05, key=f"{name}_confidence")
                                    max_results = st.number_input("Max Results", 1, 100, 5, key=f"{name}_max_results")
                                    search_area = st.text_input(
                                        "Search Area (x,y,width,height - leave empty for full image)", 
                                        "", 
                                        help="Format: x,y,width,height (e.g., 100,100,500,300)",
                                        key=f"{name}_area"
                                    )
                                
                                # Button to run with parameters
                                if st.button(f"Find Element", key=f"run_{name}"):
                                    try:
                                        with st.spinner(f"Running {name}..."):
                                            # Create absolute path to the screenshot file
                                            full_img_path = os.path.join(parent_dir, st.session_state.latest_screenshot)
                                            
                                            # Parse search area if provided
                                            area = None
                                            if search_area:
                                                try:
                                                    coords = [int(x.strip()) for x in search_area.split(',')]
                                                    if len(coords) == 4:
                                                        area = {
                                                            "x": coords[0],
                                                            "y": coords[1],
                                                            "width": coords[2],
                                                            "height": coords[3]
                                                        }
                                                except:
                                                    st.warning("Invalid search area format. Using full image.")
                                            
                                            # Prepare parameters
                                            params = {
                                                "image_path": full_img_path,
                                                "element_type": element_type,
                                                "confidence": confidence,
                                                "max_results": max_results,
                                                "ocr_enabled": ocr_enabled,
                                                "detect_ui": detect_ui,
                                                "detect_text": detect_text,
                                                "detect_images": detect_images
                                            }
                                            
                                            # Add optional parameters
                                            if search_text:
                                                params["search_text"] = search_text
                                                params["match_type"] = match_type
                                            
                                            if area:
                                                params["search_area"] = area
                                            
                                            # Show what we're searching for
                                            search_desc = f"{element_type}"
                                            if search_text:
                                                search_desc += f" with text '{search_text}' ({match_type})"
                                            else:
                                                search_desc += " (any text)"
                                                
                                            st.info(f"Finding {search_desc} with confidence {confidence} in: {full_img_path}")
                                            
                                            # Make API call
                                            res = requests.post(
                                                "http://127.0.0.1:8001/mcp/call_function",
                                                headers={"Content-Type": "application/json"},
                                                json={
                                                    "function_name": name, 
                                                    "params": params
                                                },
                                            )
                                            result = res.json()
                                            
                                            # Display success/failure and add to test results
                                            if result.get("success") and result.get("element"):
                                                st.success(f"Successfully found element: {search_desc}")
                                                
                                                # Display element details
                                                element = result.get("element", {})
                                                st.markdown("### Element Found")
                                                
                                                # Create columns for element details
                                                col1, col2 = st.columns(2)
                                                with col1:
                                                    st.markdown(f"**Type:** {element.get('type', 'N/A')}")
                                                    st.markdown(f"**Text:** {element.get('text', 'N/A')}")
                                                    st.markdown(f"**Confidence:** {element.get('confidence', 'N/A')}")
                                                
                                                with col2:
                                                    pos = element.get("position", {})
                                                    if pos:
                                                        st.markdown(f"**Position:** ({pos.get('x', 0)}, {pos.get('y', 0)})")
                                                        st.markdown(f"**Size:** {pos.get('width', 0)}×{pos.get('height', 0)}")
                                                    
                                                # Option to click on the found element
                                                if st.button("Click This Element", key=f"click_found_element"):
                                                    if pos:
                                                        # Calculate center of element
                                                        center_x = pos.get('x', 0) + pos.get('width', 0) // 2
                                                        center_y = pos.get('y', 0) + pos.get('height', 0) // 2
                                                        
                                                        click_res = requests.post(
                                                            "http://127.0.0.1:8001/mcp/call_function",
                                                            headers={"Content-Type": "application/json"},
                                                            json={
                                                                "function_name": "mcp_mouse_click", 
                                                                "params": {
                                                                    "x": center_x,
                                                                    "y": center_y
                                                                }
                                                            },
                                                        )
                                                        click_result = click_res.json()
                                                        
                                                        if click_result.get("success"):
                                                            st.success(f"Clicked element at ({center_x}, {center_y})")
                                                        else:
                                                            st.error(f"Failed to click: {click_result.get('error')}")
                                                
                                                add_test_result(name, "✅ Success", result)
                                            else:
                                                st.error(f"Failed to find element: {result.get('error', 'No matching elements found')}")
                                                
                                                # Provide suggestions to improve search
                                                st.markdown("### Troubleshooting Suggestions")
                                                st.markdown("- Try lowering the confidence threshold")
                                                st.markdown("- Try a different element type or 'any' type")
                                                st.markdown("- Make search text less specific or empty")
                                                st.markdown("- Check if OCR is enabled for text detection")
                                                st.markdown("- Ensure the element is visible in the screenshot")
                                                
                                                add_test_result(name, "❌ Failed", result)
                                    except Exception as e:
                                        st.error(f"Request failed: {e}")
                                        add_test_result(name, "❌ Error", {"error": str(e)})
                            else:
                                # For mcp_screenshot_analyze_image
                                if st.button(f"Run Test", key=f"run_{name}"):
                                    try:
                                        with st.spinner(f"Running {name}..."):
                                            # Create absolute path to the screenshot file
                                            full_img_path = os.path.join(parent_dir, st.session_state.latest_screenshot)
                                            
                                            # Show what path we're sending for debugging
                                            st.info(f"Analyzing image: {full_img_path}")
                                            
                                            res = requests.post(
                                                "http://127.0.0.1:8001/mcp/call_function",
                                                headers={"Content-Type": "application/json"},
                                                json={
                                                    "function_name": name, 
                                                    "params": {"image_path": full_img_path}
                                                },
                                            )
                                            result = res.json()
                                            
                                            # Display success/failure and add to test results
                                            if result.get("success"):
                                                st.success(f"Successfully analyzed image")
                                                add_test_result(name, "✅ Success", result)
                                            else:
                                                st.error(f"Failed to analyze image: {result.get('error', 'Unknown error')}")
                                                add_test_result(name, "❌ Failed", result)
                                    except Exception as e:
                                        st.error(f"Request failed: {e}")
                                        add_test_result(name, "❌ Error", {"error": str(e)})
                        # Special case for automation functions
                        elif name == "mcp_mouse_click":
                            # Mouse click requires x and y coordinates
                            col1, col2 = st.columns(2)
                            with col1:
                                x_coord = st.number_input("X Coordinate", value=500, key=f"{name}_x")
                            with col2:
                                y_coord = st.number_input("Y Coordinate", value=500, key=f"{name}_y")
                            
                            button_type = st.selectbox(
                                "Button", 
                                ["left", "right", "middle"],
                                key=f"{name}_button"
                            )
                            
                            clicks = st.number_input("Clicks", min_value=1, value=1, key=f"{name}_clicks")
                            
                            if st.button(f"Run Test", key=f"run_{name}"):
                                try:
                                    with st.spinner(f"Running {name}..."):
                                        st.info(f"Clicking at coordinates ({x_coord}, {y_coord}) with {button_type} button, {clicks} times")
                                        
                                    res = requests.post(
                                        "http://127.0.0.1:8001/mcp/call_function",
                                        headers={"Content-Type": "application/json"},
                                        json={
                                            "function_name": name, 
                                            "params": {
                                                "x": x_coord,
                                                "y": y_coord,
                                                "button": button_type,
                                                "clicks": clicks
                                            }
                                        },
                                    )
                                    result = res.json()
                                        
                                    if result.get("success"):
                                        st.success(f"Successfully clicked at ({x_coord}, {y_coord})")
                                        add_test_result(name, "✅ Success", result)
                                    else:
                                        st.error(f"Failed to click: {result.get('error', 'Unknown error')}")
                                        add_test_result(name, "❌ Failed", result)
                                except Exception as e:
                                    st.error(f"Request failed: {e}")
                                    add_test_result(name, "❌ Error", {"error": str(e)})
                        
                        elif name == "mcp_type_text":
                            # Type text requires text input
                            text_to_type = st.text_area("Text to Type", value="Hello, world!", key=f"{name}_text")
                            delay = st.slider("Delay between keystrokes (ms)", min_value=0, max_value=500, value=10, key=f"{name}_delay")
                            
                            # Add keyboard layout selection with improved help text
                            st.markdown("""
                            ### Keyboard Layout Settings
                            For international characters like `ö`, `å`, `ä`, select your keyboard layout 
                            and check the option to use xdotool on Linux.
                            """)
                            
                            layout_col1, layout_col2 = st.columns(2)
                            
                            with layout_col1:
                                keyboard_layout = st.selectbox(
                                    "Keyboard Layout",
                                    ["auto", "us", "sv", "de", "fr", "es", "uk", "it"],
                                    index=0,
                                    help="Select your keyboard layout for better character support",
                                    key=f"{name}_layout"
                                )
                            
                            with layout_col2:
                                # Add option to use xdotool on Linux
                                use_xdotool = st.checkbox(
                                    "Use xdotool (Linux only)", 
                                    value=True,
                                    help="Recommended for international characters on Linux",
                                    key=f"{name}_xdotool"
                                )
                            
                            # Add special characters test box
                            st.markdown("### Test with Special Characters")
                            special_chars = st.text_input(
                                "Try typing these characters",
                                value="åäö ÅÄÖ ;:[]{}@$€\\|~",
                                help="Copy these to test international character support",
                                key=f"{name}_special_chars"
                            )
                            
                            if st.button("Use Special Characters", key=f"{name}_use_special"):
                                text_to_type = special_chars
                                st.session_state[f"{name}_text"] = special_chars
                                st.experimental_rerun()
                            
                            # Split into two actions: prepare and execute
                            preparation_tab, execution_tab = st.tabs(["Prepare", "Execute"])
                            
                            with preparation_tab:
                                st.info("Configure your text input here, then switch to the Execute tab to perform the action.")
                                st.info("This two-step process prevents accidental activation when testing.")
                            
                            with execution_tab:
                                st.warning("⚠️ Make sure your cursor is positioned where you want to type BEFORE clicking Run.")
                                
                                # Add delay option to give user time to position cursor
                                pre_delay = st.slider("Pre-execution delay (seconds)", 0, 10, 3, 1, 
                                                     help="Wait this many seconds before typing to give you time to position your cursor")
                                
                                if st.button(f"Run Text Input", key=f"run_{name}"):
                                    try:
                                        # Show countdown if pre-delay is set
                                        if pre_delay > 0:
                                            progress_text = "Preparing to type..."
                                            progress_bar = st.progress(0, text=progress_text)
                                            
                                            for i in range(pre_delay):
                                                # Update progress bar
                                                progress_bar.progress((i+1)/pre_delay, 
                                                                     text=f"{progress_text} Starting in {pre_delay-i} seconds...")
                                                time.sleep(1)
                                                
                                            progress_bar.progress(1.0, text="Typing now!")
                                        
                                        st.info(f"Typing text: '{text_to_type}' with {delay}ms delay, keyboard layout: {keyboard_layout}")
                                        
                                        # Prepare parameters with new options
                                        params = {
                                            "text": text_to_type,
                                            "delay": delay,
                                            "keyboard_layout": keyboard_layout
                                        }
                                        
                                        # Add xdotool option if selected
                                        if use_xdotool:
                                            params["use_xdotool"] = True
                                        
                                        res = requests.post(
                                            "http://127.0.0.1:8001/mcp/call_function",
                                            headers={"Content-Type": "application/json"},
                                            json={
                                                "function_name": name, 
                                                "params": params
                                            },
                                        )
                                        result = res.json()
                                        
                                        if result.get("success"):
                                            st.success(f"Successfully typed text")
                                            add_test_result(name, "✅ Success", result)
                                        else:
                                            st.error(f"Failed to type text: {result.get('error', 'Unknown error')}")
                                            add_test_result(name, "❌ Failed", result)
                                    except Exception as e:
                                        st.error(f"Request failed: {e}")
                                        add_test_result(name, "❌ Error", {"error": str(e)})
                        
                        elif name == "mcp_press_key":
                            # Split into two actions: prepare and execute
                            preparation_tab, execution_tab = st.tabs(["Prepare", "Execute"])
                            
                            with preparation_tab:
                                # Press key requires key name
                                common_keys = [
                                    "enter", "tab", "space", "backspace", "escape", "up", "down", "left", "right",
                                    "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
                                    "ctrl", "alt", "shift", "win", "cmd"
                                ]
                                key_to_press = st.selectbox("Key to Press", common_keys, key=f"{name}_key")
                                
                                # Allow specifying modifiers
                                st.markdown("**Modifiers**")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    ctrl = st.checkbox("Ctrl", key=f"{name}_ctrl")
                                with col2:
                                    alt = st.checkbox("Alt", key=f"{name}_alt")
                                with col3:
                                    shift = st.checkbox("Shift", key=f"{name}_shift") 
                                with col4:
                                    meta = st.checkbox("Win/Cmd", key=f"{name}_meta")
                                
                                st.info("Configure your key press here, then switch to the Execute tab to perform the action.")
                            
                            with execution_tab:
                                st.warning("⚠️ Make sure you're ready BEFORE clicking Run. The key will be pressed immediately.")
                                
                                # Build modifiers list
                                modifiers = []
                                if ctrl:
                                    modifiers.append("ctrl")
                                if alt:
                                    modifiers.append("alt")
                                if shift:
                                    modifiers.append("shift")
                                if meta:
                                    modifiers.append("meta")
                                
                                modifier_text = ", ".join(modifiers) if modifiers else "no modifiers"
                                st.info(f"Will press key: '{key_to_press}' with {modifier_text}")
                                
                                # Add delay option to give user time to prepare
                                pre_delay = st.slider("Pre-execution delay (seconds)", 0, 10, 3, 1, 
                                                     help="Wait this many seconds before pressing the key")
                                
                                if st.button(f"Press Key Now", key=f"run_{name}"):
                                    try:
                                        # Show countdown if pre-delay is set
                                        if pre_delay > 0:
                                            progress_text = "Preparing to press key..."
                                            progress_bar = st.progress(0, text=progress_text)
                                            
                                            for i in range(pre_delay):
                                                # Update progress bar
                                                progress_bar.progress((i+1)/pre_delay, 
                                                                     text=f"{progress_text} Starting in {pre_delay-i} seconds...")
                                                time.sleep(1)
                                                
                                            progress_bar.progress(1.0, text="Pressing key now!")
                                        
                                        res = requests.post(
                                            "http://127.0.0.1:8001/mcp/call_function",
                                            headers={"Content-Type": "application/json"},
                                            json={
                                                "function_name": name, 
                                                "params": {
                                                    "key": key_to_press,
                                                    "modifiers": modifiers
                                                }
                                            },
                                        )
                                        result = res.json()
                                        
                                        if result.get("success"):
                                            st.success(f"Successfully pressed key")
                                            add_test_result(name, "✅ Success", result)
                                        else:
                                            st.error(f"Failed to press key: {result.get('error', 'Unknown error')}")
                                            add_test_result(name, "❌ Failed", result)
                                    except Exception as e:
                                        st.error(f"Request failed: {e}")
                                        add_test_result(name, "❌ Error", {"error": str(e)})
                        else:
                            # Standard case for all other functions
                            params = {}
                            # For screenshot capture, add monitor selection
                            if name == "mcp_screenshot_capture":
                                monitor = st.number_input("Monitor ID", min_value=0, value=1, key=f"{name}_monitor")
                                params["monitor"] = monitor
                            
                            if st.button(f"Run Test", key=f"run_{name}"):
                                try:
                                    with st.spinner(f"Running {name}..."):
                                        res = requests.post(
                                            "http://127.0.0.1:8001/mcp/call_function",
                                            headers={"Content-Type": "application/json"},
                                            json={"function_name": name, "params": params},
                                        )
                                        result = res.json()

                                        # Handle specific result types
                                        if "screenshot_path" in result and result["screenshot_path"]:
                                            st.session_state.latest_screenshot = result["screenshot_path"]
                                            st.success(f"Screenshot captured: {st.session_state.latest_screenshot}")
                                            add_test_result(name, "✅ Success", result)
                                            st.rerun()

                                        # Check for monitors in the result
                                        elif "monitors" in result and result["monitors"]:
                                            st.session_state.monitors = result["monitors"]
                                            if "primary" in result and result["primary"]:
                                                st.session_state.primary_monitor = result["primary"]
                                            st.success(f"Found {len(st.session_state.monitors)} monitors")
                                            add_test_result(name, "✅ Success", result)
                                            st.rerun()
                                        
                                        # Generic success/failure handling
                                        elif result.get("success"):
                                            st.success(f"Successfully executed {name}")
                                            add_test_result(name, "✅ Success", result)
                                        else:
                                            st.error(f"Failed: {result.get('error', 'Unknown error')}")
                                            add_test_result(name, "❌ Failed", result)

                                except Exception as e:
                                    st.error(f"Request failed: {e}")
                                    add_test_result(name, "❌ Error", {"error": str(e)})
    else:
        st.error("Invalid response from MCP server.")

# Display the components in the right sidebar
show_screenshot()
show_monitors()
show_test_results()
