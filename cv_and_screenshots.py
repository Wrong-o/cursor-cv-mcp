import mss
import platform
import io
import cv2
import numpy as np
import pytesseract
from PIL import Image
from typing import Any, Dict, Optional, Tuple

# Configuration
debug = False  # Debug mode flag

def get_available_monitors():
        try:
            with mss.mss() as sct:
                monitors_info = {
                    "monitors": [],
                    "primary": 0
                }
                
                # Get primary monitor (monitor 0 is a special case - it's the entire virtual screen)
                primary_idx = 0  # Default to first physical monitor
                
                # Collect monitor information
                for idx, monitor in enumerate(sct.monitors[1:], 1):  # Skip monitor 0 (entire virtual screen)
                    monitor_info = {
                        "id": idx,
                        "width": monitor["width"],
                        "height": monitor["height"],
                        "left": monitor["left"],
                        "top": monitor["top"],
                    }
                    monitors_info["monitors"].append(monitor_info)
                    
                    # Check if this is the primary monitor (typically at position 0,0)
                    if monitor["left"] == 0 and monitor["top"] == 0:
                        primary_idx = idx
                
                monitors_info["primary"] = primary_idx
                return monitors_info
        except Exception as e:
            print(f"Error getting monitor info: {e}")
            return None

def get_screenshot(monitor_id: int = 1) -> bytes:
    with mss.mss() as sct:
        monitor_dict = sct.monitors[monitor_id]
        screenshot = sct.grab(monitor_dict)
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)

        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        return buf.getvalue()

def analyze_image(image_data: bytes) -> Dict[str, Any]:
    """
    Analyze the image content.
    
    Args:
        image_data: Raw image data
    
    Returns:
        Dict containing analysis results
    """
    results = {
        "basic_info": {},
        "text_content": None,
        "analysis": None
    }
    
    try:
        img = Image.open(io.BytesIO(image_data))
        
        results["basic_info"] = {
            "format": img.format,
            "size": img.size,
            "mode": img.mode,
        }
        
        
        # Try to extract text using OCR if available
        print("Performing OCR to extract text...")
        # Convert PIL Image to OpenCV format
        if image_data:
            nparr = np.frombuffer(image_data, np.uint8)
            cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            print("No image data provided")
            return results
        
        # Convert to grayscale for better OCR
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # Perform OCR
        text = pytesseract.image_to_string(gray)
        results["text_content"] = text
        
        # Print a sample of the extracted text
        text_sample = text[:200] + "..." if len(text) > 200 else text
        print(f"\nExtracted text sample:\n{text_sample}")
    except Exception as e:
        print(f"Error analyzing image: {e}")
        results["analysis"] = f"Error: {str(e)}"
    
    return results 

def get_screenshot_with_analysis(
    monitor_id: int = 0,
) -> Tuple[Optional[bytes], Optional[Dict[str, Any]], Optional[Any], Optional[Any]]:
    try:
        # Get monitor information
        with mss.mss() as sct:
            # Monitor 0 is the entire virtual screen
            # Physical monitors start at index 1
            if monitor_id == 0:
                # Use the primary monitor instead of the entire virtual screen
                monitors_info = get_available_monitors()
                if monitors_info and "primary" in monitors_info:
                    monitor_id = monitors_info["primary"]
                else:
                    monitor_id = 1  # Default to first physical monitor
            
            # Ensure the monitor exists
            if monitor_id >= len(sct.monitors):
                print(f"Error: Monitor {monitor_id} not found. Using monitor 1.")
                monitor_id = 1
            
            # Capture the screen
            monitor_dict = sct.monitors[monitor_id]
            
            # Get screenshot using the existing function
            image_data = get_screenshot(monitor_id)
            
            # Analyze the image
            print("\nAnalyzing screenshot...")
            analysis_results = analyze_image(image_data)
            ui_buttons = cv_find_all_buttons(image_data)
            ui_checkboxes = cv_find_checkboxes(image_data)
            
            return image_data, analysis_results, ui_buttons, ui_checkboxes
    except Exception as e:
        print(f"Error capturing screenshot: {e}")
        return None, None, None, None

def cv_find_all_buttons(image_data: bytes):
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    buttons = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 30 or h < 15 or w / h < 1.5:
            continue

        roi = img[y:y+h, x:x+w]
        text = pytesseract.image_to_string(roi, config='--psm 7').strip()
        if text:
            buttons.append((text, (x, y, w, h)))

    return buttons

def cv_find_checkboxes(image_data: bytes):
    """
    Find checkbox-like elements in an image.
    
    Args:
        image_data: Raw image data in bytes
        
    Returns:
        List of tuples with format (checkbox_type, (x, y, width, height))
        where checkbox_type is "unchecked", "checked", or "indeterminate"
    """
    try:
        print("Looking for checkboxes in image...")
        
        if not image_data:
            print("Error: No image data provided")
            return []
            
        # Convert bytes to OpenCV image
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            print("Error: Failed to decode image data")
            return []

        # Create a copy of the original image for visualization
        original_img = img.copy()
        height, width = img.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply GaussianBlur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours in the edge map
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to find potential checkboxes
        checkboxes = []
        for contour in contours:
            # Calculate bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter based on aspect ratio and size
            aspect_ratio = float(w) / h
            min_size = min(height, width) * 0.01  # Min 1% of image dimension
            max_size = min(height, width) * 0.1   # Max 10% of image dimension
            
            # Checkboxes are roughly square (aspect ratio close to 1)
            if (0.7 <= aspect_ratio <= 1.3 and 
                min_size <= w <= max_size and 
                min_size <= h <= max_size):
                
                # Further analyze the candidate checkbox
                checkbox_roi = gray[y:y+h, x:x+w]
                
                # Check if it's a checkbox by analyzing the contour
                # Calculate approximated contour
                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Checkboxes often have 4 corners (rectangle/square)
                if 4 <= len(approx) <= 8:
                    # Calculate average intensity inside and outside the contour
                    mask = np.zeros_like(checkbox_roi)
                    cv2.drawContours(mask, [contour - np.array([[x, y]])], 0, 255, -1)
                    inner_mean = cv2.mean(checkbox_roi, mask=mask)[0]
                    outer_mean = cv2.mean(checkbox_roi, mask=cv2.bitwise_not(mask))[0]
                    
                    # Determine if it's checked based on inner content
                    # Create a mask for the inner area (smaller than the checkbox)
                    inner_mask = np.zeros_like(checkbox_roi)
                    shrink_factor = 0.25
                    inner_x = int(w * shrink_factor)
                    inner_y = int(h * shrink_factor)
                    inner_w = int(w * (1 - 2 * shrink_factor))
                    inner_h = int(h * (1 - 2 * shrink_factor))
                    
                    if inner_w > 0 and inner_h > 0:
                        inner_roi = checkbox_roi[inner_y:inner_y+inner_h, inner_x:inner_x+inner_w]
                        
                        if inner_roi.size > 0:
                            # Calculate statistics for the inner region
                            inner_mean = np.mean(inner_roi)
                            inner_std = np.std(inner_roi)
                            
                            # Threshold the inner region to detect marks
                            _, inner_thresh = cv2.threshold(inner_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                            white_pixel_ratio = np.sum(inner_thresh == 255) / inner_roi.size
                            
                            # Determine checkbox state
                            checkbox_type = "unchecked"
                            
                            # If inner area has significant dark pixels, it might be checked
                            if white_pixel_ratio < 0.7:
                                # Check for checkmark pattern (diagonal lines)
                                # Apply Hough line transform to detect lines
                                edges_inner = cv2.Canny(inner_roi, 50, 150)
                                lines = cv2.HoughLinesP(edges_inner, 1, np.pi/180, 
                                                       threshold=10, 
                                                       minLineLength=min(inner_w, inner_h)*0.3, 
                                                       maxLineGap=5)
                                
                                if lines is not None and len(lines) > 0:
                                    checkbox_type = "checked"
                                else:
                                    # If no lines but still dark, might be indeterminate
                                    checkbox_type = "indeterminate"
                            
                            checkboxes.append((checkbox_type, (x, y, w, h)))
                            print(f"Found {checkbox_type} checkbox at ({x}, {y}, {w}, {h})")
        
        print(f"Found {len(checkboxes)} checkboxes in total")
        return checkboxes
    except Exception as e:
        import traceback
        print(f"Error finding checkboxes: {str(e)}")
        traceback.print_exc()
        return []

def find_text_in_image(image_data: bytes, target: str):
    try:
        print(f"Finding text '{target}' in image using enhanced detection...")
        
        if not image_data:
            print("Error: No image data provided to find_text_in_image")
            return []
            
        # Convert bytes to OpenCV image
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            print("Error: Failed to decode image data")
            return []

        # Create a copy of the original image for visualization
        original_img = img.copy()
        height, width = img.shape[:2]
        
        # Apply multiple preprocessing techniques to improve OCR
        preprocessed_images = []
        
        # 1. Basic grayscale conversion
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        preprocessed_images.append(("basic_gray", gray))
        
        # 2. Grayscale with threshold
        _, thresh1 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        preprocessed_images.append(("threshold_150", thresh1))
        
        # 3. Grayscale with adaptive threshold
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        preprocessed_images.append(("adaptive_thresh", adaptive_thresh))
        
        # 4. Grayscale with Otsu's threshold
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(("otsu_thresh", otsu))
        
        # 5. Add some blurring to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh_blurred = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(("blurred_otsu", thresh_blurred))
        
        # Store all matches
        all_matches = []
        target_lower = target.lower()
        
        # Try each preprocessing method
        for name, processed_img in preprocessed_images:
            print(f"Trying OCR with {name} preprocessing...")
            
            # Get both standard string and detailed data with positions
            text = pytesseract.image_to_string(processed_img).lower()
            data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)
            
            # Simple string matching for debugging
            if target_lower in text:
                print(f"Found target '{target}' in text with {name} preprocessing")
            
            # Extract matches from detailed data
            for i, word in enumerate(data["text"]):
                if not word:
                    continue
                    
                word_lower = word.lower()
                # Try different matching strategies
                if (target_lower in word_lower or  # Substring match
                    word_lower in target_lower or  # Partial match
                    # Fuzzy match (if words are similar enough)
                    (len(target) > 3 and len(word) > 3 and 
                     (target_lower[:3] == word_lower[:3] or 
                      target_lower[-3:] == word_lower[-3:]))):
                    
                    confidence = data["conf"][i]
                    # Filter low-confidence matches
                    if confidence < 30:
                        continue
                        
                    x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                    match_info = {
                        "word": word,
                        "bounds": (x, y, w, h),
                        "confidence": confidence,
                        "method": name
                    }
                    all_matches.append(match_info)
                    print(f"Found match '{word}' at ({x}, {y}, {w}, {h}) with confidence {confidence}")
        
        # Filter duplicates (matches at very similar positions)
        filtered_matches = []
        for match in all_matches:
            x1, y1, w1, h1 = match["bounds"]
            center1 = (x1 + w1//2, y1 + h1//2)
            
            # Check if this match is too close to any existing filtered match
            is_duplicate = False
            for filtered in filtered_matches:
                x2, y2, w2, h2 = filtered["bounds"]
                center2 = (x2 + w2//2, y2 + h2//2)
                
                # Calculate distance between centers
                distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
                
                # If centers are close, consider it a duplicate
                if distance < max(w1, h1, w2, h2) * 0.5:
                    is_duplicate = True
                    # Keep the match with higher confidence
                    if match["confidence"] > filtered["confidence"]:
                        filtered_matches.remove(filtered)
                        filtered_matches.append(match)
                    break
            
            if not is_duplicate:
                filtered_matches.append(match)
        
        # Prepare final result
        result = []
        for match in filtered_matches:
            result.append((match["word"], match["bounds"]))
            
        print(f"Found {len(result)} unique matches for '{target}'")
        return result
    except Exception as e:
        import traceback
        print(f"Error in find_text_in_image: {str(e)}")
        traceback.print_exc()
        return []
 