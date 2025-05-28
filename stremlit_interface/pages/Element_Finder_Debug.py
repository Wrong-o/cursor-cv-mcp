import os
import json
import base64
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import io

import requests
import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches

st.set_page_config(layout="wide", page_title="Element Finder Debug")

st.title("Element Finder Debugging Tool")

# Initialize session state
if "latest_screenshot" not in st.session_state:
    st.session_state.latest_screenshot = None
if "debug_results" not in st.session_state:
    st.session_state.debug_results = None
if "annotated_image" not in st.session_state:
    st.session_state.annotated_image = None
if "cv_debug_images" not in st.session_state:
    st.session_state.cv_debug_images = None

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))

# Create main layout
left_col, right_col = st.columns([1, 1])

# Function to draw bounding boxes on the image
def annotate_elements(image_path, elements, colors=None):
    if not elements:
        return None
    
    # Default colors for different element types
    if colors is None:
        colors = {
            "text": "blue",
            "button": "red",
            "input": "green",
            "image": "purple",
            "icon": "orange",
            "checkbox": "brown",
            "radio": "pink",
            "dropdown": "cyan",
            "link": "magenta",
            "any": "yellow",
            "default": "white"
        }
    
    # Open the image
    try:
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, use default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Draw each element
        for i, element in enumerate(elements):
            # Handle both new format (position as dict) and old format (x, y directly in element)
            if "position" in element:
                pos = element["position"]
                x = pos.get("x", 0)
                y = pos.get("y", 0)
                width = pos.get("width", 0)
                height = pos.get("height", 0)
            else:
                # Legacy format
                x = element.get("x", 0)
                y = element.get("y", 0)
                width = element.get("width", 0)
                height = element.get("height", 0)
                
            element_type = element.get("type", "default")
            color = colors.get(element_type, colors["default"])
            
            # Draw rectangle
            draw.rectangle(
                [
                    (x, y),
                    (x + width, y + height)
                ],
                outline=color,
                width=3
            )
            
            # Draw label with confidence
            confidence = element.get("confidence", 0)
            text = element.get("text", "")
            label = f"{i+1}: {element_type} ({confidence:.2f})"
            
            # Draw text background
            text_width, text_height = draw.textsize(label, font=font) if hasattr(draw, "textsize") else font.getsize(label)
            draw.rectangle(
                [
                    (x, y - text_height - 5),
                    (x + text_width + 10, y)
                ],
                fill=color
            )
            
            # Draw text
            draw.text(
                (x + 5, y - text_height - 3),
                label,
                fill="white",
                font=font
            )
        
        # Convert to bytes for Streamlit
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()
        
    except Exception as e:
        st.error(f"Error annotating image: {e}")
        import traceback
        traceback.print_exc()
        return None

# UI for element finder debug
with left_col:
    st.header("Element Finder Settings")
    
    # Step 1: Select or capture a screenshot
    st.subheader("Step 1: Select Screenshot")
    
    # Option to use latest screenshot or capture new one
    screenshot_option = st.radio(
        "Screenshot Source",
        ["Use Latest Screenshot", "Capture New Screenshot"],
        key="screenshot_source"
    )
    
    if screenshot_option == "Capture New Screenshot":
        monitor = st.number_input("Monitor ID", min_value=0, value=1, key="capture_monitor")
        if st.button("Capture Screenshot", key="capture_btn"):
            with st.spinner("Capturing screenshot..."):
                res = requests.post(
                    "http://127.0.0.1:8001/mcp/call_function",
                    headers={"Content-Type": "application/json"},
                    json={
                        "function_name": "mcp_screenshot_capture", 
                        "params": {"monitor": monitor}
                    },
                )
                result = res.json()
                
                if result.get("success") and result.get("screenshot_path"):
                    st.session_state.latest_screenshot = result["screenshot_path"]
                    st.success(f"Screenshot captured: {st.session_state.latest_screenshot}")
                else:
                    st.error(f"Failed to capture screenshot: {result.get('error', 'Unknown error')}")
    else:
        # Check for existing screenshot
        if not st.session_state.latest_screenshot:
            st.warning("No screenshot available. Please capture a new screenshot.")
    
    # Step 2: Configure element finder
    if st.session_state.latest_screenshot:
        st.subheader("Step 2: Configure Element Search")
        
        # Element type selection
        element_type = st.selectbox(
            "Element Type", 
            ["text", "button", "input", "image", "icon", "checkbox", "radio", "dropdown", "link", "any"],
            key="debug_element_type"
        )
        
        # Text search
        search_text = st.text_input(
            "Search Text (leave empty to find any element of the selected type)", 
            "",
            key="debug_search_text"
        )
        
        match_type = st.radio(
            "Match Type", 
            ["exact", "contains", "starts_with", "ends_with", "regex"],
            horizontal=True,
            key="debug_match_type"
        )
        
        # Advanced options in expander
        with st.expander("Advanced Options"):
            # OCR and detection settings
            st.markdown("**Detection Settings**")
            col1, col2 = st.columns(2)
            with col1:
                ocr_enabled = st.checkbox("Enable OCR", value=True, key="debug_ocr")
                detect_ui = st.checkbox("Detect UI elements", value=True, key="debug_ui")
            with col2:
                detect_text = st.checkbox("Detect visible text", value=True, key="debug_text")
                detect_images = st.checkbox("Detect images", value=True, key="debug_images")
            
            # Confidence and result settings
            confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05, key="debug_confidence")
            max_results = st.number_input("Max Results", 1, 100, 10, key="debug_max_results")
            
            # Search area
            st.markdown("**Search Area**")
            use_area = st.checkbox("Limit search area", value=False, key="debug_use_area")
            if use_area:
                col1, col2 = st.columns(2)
                with col1:
                    area_x = st.number_input("X", value=0, key="debug_area_x")
                    area_width = st.number_input("Width", value=500, key="debug_area_width")
                with col2:
                    area_y = st.number_input("Y", value=0, key="debug_area_y")
                    area_height = st.number_input("Height", value=500, key="debug_area_height")
            
            # Computer Vision debug options
            st.markdown("**Computer Vision Debug**")
            enable_cv_debug = st.checkbox("Show CV Processing Steps", value=True, key="cv_debug")
            debug_level = st.slider("Debug Detail Level", 1, 3, 2, key="debug_level")
        
        # Step 3: Run the element finder with debug options
        st.subheader("Step 3: Find Elements")
        
        debug_mode = st.checkbox("Enable Detailed Debug Output", value=True, key="detailed_debug")
        
        if st.button("Find Elements", key="find_debug"):
            with st.spinner("Finding elements..."):
                # Create absolute path to the screenshot file
                full_img_path = os.path.join(parent_dir, st.session_state.latest_screenshot)
                
                # Prepare parameters
                params = {
                    "image_path": full_img_path,
                    "element_type": element_type,
                    "confidence": confidence,
                    "max_results": max_results,
                    "ocr_enabled": ocr_enabled,
                    "detect_ui": detect_ui,
                    "detect_text": detect_text,
                    "detect_images": detect_images,
                    "debug": debug_mode,
                    "cv_debug": enable_cv_debug,
                    "debug_level": debug_level
                }
                
                # Add optional parameters
                if search_text:
                    params["search_text"] = search_text
                    params["match_type"] = match_type
                
                if use_area:
                    params["search_area"] = {
                        "x": area_x,
                        "y": area_y,
                        "width": area_width,
                        "height": area_height
                    }
                
                # Show search description
                search_desc = f"{element_type}"
                if search_text:
                    search_desc += f" with text '{search_text}' ({match_type})"
                else:
                    search_desc += " (any text)"
                
                st.info(f"Finding {search_desc} with confidence {confidence}")
                
                # Make API call
                res = requests.post(
                    "http://127.0.0.1:8001/mcp/call_function",
                    headers={"Content-Type": "application/json"},
                    json={
                        "function_name": "mcp_find_element", 
                        "params": params
                    },
                )
                result = res.json()
                
                # Save debug results
                st.session_state.debug_results = result
                
                # Extract CV debug images if available
                if "debug_info" in result and "cv_debug_images" in result["debug_info"]:
                    st.session_state.cv_debug_images = result["debug_info"]["cv_debug_images"]
                    print(f"CV debug images received: {list(result['debug_info']['cv_debug_images'].keys())}")
                    
                    # Check if we have preprocessing images
                    if "preprocessing" in result["debug_info"]["cv_debug_images"]:
                        preprocessing_images = result["debug_info"]["cv_debug_images"]["preprocessing"]
                        print(f"Preprocessing images: {len(preprocessing_images)}")
                        if preprocessing_images:
                            print(f"First preprocessing image keys: {list(preprocessing_images[0].keys())}")
                            # Check the length of the base64 image
                            if "image" in preprocessing_images[0]:
                                base64_length = len(preprocessing_images[0]["image"])
                                print(f"Base64 image length: {base64_length}")
                                # Try decoding it to verify it's valid
                                try:
                                    img_bytes = base64.b64decode(preprocessing_images[0]["image"])
                                    print(f"Successfully decoded base64 image, size: {len(img_bytes)} bytes")
                                except Exception as e:
                                    print(f"Error decoding base64 image: {e}")
                else:
                    print("No CV debug images found in response")
                    print(f"Debug info keys: {list(result.get('debug_info', {}).keys())}")
                    
                # Print full debug info structure (truncated to avoid huge output)
                if "debug_info" in result:
                    debug_info_str = str(result["debug_info"])
                    print(f"Debug info structure length: {len(debug_info_str)}")
                    if len(debug_info_str) > 1000:
                        debug_info_str = debug_info_str[:1000] + "... [truncated]"
                    print(f"Debug info structure (truncated): {debug_info_str}")
                
                # Annotate image if elements found
                elements = []
                if result.get("success") and result.get("element"):
                    elements = [result.get("element")]
                if "all_elements" in result:
                    elements = result.get("all_elements", [])
                
                if elements:
                    annotated = annotate_elements(full_img_path, elements)
                    st.session_state.annotated_image = annotated
                    st.success(f"Found {len(elements)} elements")
                else:
                    st.error(f"No elements found: {result.get('error', 'Unknown error')}")

# Display the results
with right_col:
    st.header("Results Visualization")
    
    # Show the screenshot with annotations
    if st.session_state.latest_screenshot:
        st.subheader("Screenshot")
        full_img_path = os.path.join(parent_dir, st.session_state.latest_screenshot)
        
        # Show annotated image if available, otherwise show original
        if st.session_state.annotated_image:
            st.image(st.session_state.annotated_image, caption="Annotated Screenshot", use_column_width=True)
        else:
            st.image(full_img_path, caption="Original Screenshot", use_column_width=True)
    
    # Show CV debug images if available
    if st.session_state.cv_debug_images:
        st.subheader("Computer Vision Processing Steps")
        
        # Create tabs for different CV visualization categories
        cv_tabs = st.tabs(["Preprocessing", "Edge Detection", "Contour Analysis", "OCR Steps", "UI Detection"])
        
        # Group debug images by category
        cv_images = st.session_state.cv_debug_images
        
        # Preprocessing tab
        with cv_tabs[0]:
            if "preprocessing" in cv_images:
                for i, img_data in enumerate(cv_images["preprocessing"]):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        # Decode base64 image
                        try:
                            img_bytes = base64.b64decode(img_data["image"])
                            st.image(img_bytes, caption=img_data.get("caption", f"Preprocessing Step {i+1}"))
                        except Exception as e:
                            st.error(f"Error displaying image: {e}")
                    with col2:
                        st.markdown(f"**Step {i+1}**")
                        st.markdown(img_data.get("description", "No description"))
            else:
                st.info("No preprocessing debug images available")
        
        # Edge Detection tab
        with cv_tabs[1]:
            if "edge_detection" in cv_images:
                for i, img_data in enumerate(cv_images["edge_detection"]):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        try:
                            img_bytes = base64.b64decode(img_data["image"])
                            st.image(img_bytes, caption=img_data.get("caption", f"Edge Detection Step {i+1}"))
                        except Exception as e:
                            st.error(f"Error displaying image: {e}")
                    with col2:
                        st.markdown(f"**Step {i+1}**")
                        st.markdown(img_data.get("description", "No description"))
                        if "parameters" in img_data:
                            st.markdown("**Parameters:**")
                            for param, value in img_data["parameters"].items():
                                st.markdown(f"- {param}: {value}")
            else:
                st.info("No edge detection debug images available")
        
        # Contour Analysis tab
        with cv_tabs[2]:
            if "contours" in cv_images:
                for i, img_data in enumerate(cv_images["contours"]):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        try:
                            img_bytes = base64.b64decode(img_data["image"])
                            st.image(img_bytes, caption=img_data.get("caption", f"Contour Analysis Step {i+1}"))
                        except Exception as e:
                            st.error(f"Error displaying image: {e}")
                    with col2:
                        st.markdown(f"**Step {i+1}**")
                        st.markdown(img_data.get("description", "No description"))
                        if "contour_count" in img_data:
                            st.metric("Contours Found", img_data["contour_count"])
            else:
                st.info("No contour analysis debug images available")
        
        # OCR Steps tab
        with cv_tabs[3]:
            if "ocr" in cv_images:
                for i, img_data in enumerate(cv_images["ocr"]):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        try:
                            img_bytes = base64.b64decode(img_data["image"])
                            st.image(img_bytes, caption=img_data.get("caption", f"OCR Step {i+1}"))
                        except Exception as e:
                            st.error(f"Error displaying image: {e}")
                    with col2:
                        st.markdown(f"**Step {i+1}**")
                        st.markdown(img_data.get("description", "No description"))
                        if "text_found" in img_data:
                            st.markdown("**Text Detected:**")
                            st.code(img_data["text_found"])
            else:
                st.info("No OCR debug images available")
        
        # UI Detection tab
        with cv_tabs[4]:
            if "ui_detection" in cv_images:
                for i, img_data in enumerate(cv_images["ui_detection"]):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        try:
                            img_bytes = base64.b64decode(img_data["image"])
                            st.image(img_bytes, caption=img_data.get("caption", f"UI Detection Step {i+1}"))
                        except Exception as e:
                            st.error(f"Error displaying image: {e}")
                    with col2:
                        st.markdown(f"**Step {i+1}**")
                        st.markdown(img_data.get("description", "No description"))
                        if "elements_found" in img_data:
                            st.metric("Elements Detected", img_data["elements_found"])
            else:
                st.info("No UI detection debug images available")
    
    # Show debug results
    if st.session_state.debug_results:
        st.subheader("Element Finder Process")
        
        result = st.session_state.debug_results
        
        # First show overview
        if result.get("success"):
            st.success("Element found successfully!")
        else:
            st.error(f"Failed to find element: {result.get('error', 'Unknown error')}")
        
        # Show debug information if available
        if "debug_info" in result:
            debug_info = result["debug_info"]
            
            # Detection steps
            if "detection_steps" in debug_info:
                st.markdown("### Detection Process")
                steps = debug_info["detection_steps"]
                
                for i, step in enumerate(steps):
                    with st.expander(f"Step {i+1}: {step.get('name', 'Unnamed Step')}"):
                        st.markdown(f"**Description:** {step.get('description', 'No description')}")
                        
                        if "detected_count" in step:
                            st.markdown(f"**Elements detected:** {step['detected_count']}")
                        
                        if "details" in step:
                            st.markdown("**Details:**")
                            st.code(json.dumps(step["details"], indent=2))
            
            # Statistics
            if "stats" in debug_info:
                st.markdown("### Performance Statistics")
                stats = debug_info["stats"]
                
                cols = st.columns(3)
                cols[0].metric("Total Processing Time", f"{stats.get('total_time', 0):.2f}s")
                cols[1].metric("OCR Processing Time", f"{stats.get('ocr_time', 0):.2f}s")
                cols[2].metric("UI Detection Time", f"{stats.get('ui_detection_time', 0):.2f}s")
        
        # Show all detected elements
        if "all_elements" in result:
            elements = result["all_elements"]
            
            st.markdown("### All Detected Elements")
            st.markdown(f"Found {len(elements)} potential elements")
            
            # Create a DataFrame for better visualization
            if elements:
                element_data = []
                for i, elem in enumerate(elements):
                    # Handle both new format (position as dict) and old format (x, y directly in element)
                    if "position" in elem:
                        pos = elem["position"]
                        x = pos.get("x", 0)
                        y = pos.get("y", 0)
                        width = pos.get("width", 0)
                        height = pos.get("height", 0)
                    else:
                        # Legacy format
                        x = elem.get("x", 0)
                        y = elem.get("y", 0)
                        width = elem.get("width", 0)
                        height = elem.get("height", 0)
                        
                    element_data.append({
                        "Index": i + 1,
                        "Type": elem.get("type", "unknown"),
                        "Text": elem.get("text", ""),
                        "Confidence": f"{elem.get('confidence', 0):.2f}",
                        "Position": f"({x}, {y})",
                        "Size": f"{width}×{height}"
                    })
                
                st.dataframe(element_data)
        
        # Show best match
        if result.get("element"):
            element = result["element"]
            
            st.markdown("### Best Match")
            st.json(element)
            
            # Option to click on the found element
            if "position" in element:
                pos = element["position"]
                center_x = pos.get('center_x') or (pos.get('x', 0) + pos.get('width', 0) // 2)
                center_y = pos.get('center_y') or (pos.get('y', 0) + pos.get('height', 0) // 2)
            else:
                # Legacy format
                center_x = element.get('center_x') or (element.get('x', 0) + element.get('width', 0) // 2)
                center_y = element.get('center_y') or (element.get('y', 0) + element.get('height', 0) // 2)
            
            if st.button("Click This Element", key="click_debug_element"):
                with st.spinner("Clicking element..."):
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
        
        # Troubleshooting suggestions if no element found
        if not result.get("success"):
            st.markdown("### Troubleshooting Suggestions")
            
            suggestions = [
                "Lower the confidence threshold (try 0.2-0.3 for more permissive matching)",
                "Use 'any' as the element type to see all detectable elements",
                "Leave the search text empty to find elements based on type only",
                "Try 'contains' match type instead of 'exact' for more flexible text matching",
                "Make sure OCR is enabled for text detection",
                "Check if the element is clearly visible in the screenshot",
                "Try specifying a more focused search area around the element"
            ]
            
            for suggestion in suggestions:
                st.markdown(f"- {suggestion}") 