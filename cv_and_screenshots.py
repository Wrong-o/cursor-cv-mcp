import platform
import io
from typing import Dict, Any, List, Tuple, Optional
import mss
from PIL import Image
import numpy as np
import subprocess
import tempfile
import traceback
import os
import time

# Ensure required packages are available
try:
    import mss
except ImportError:
    print("mss package not found, attempting to install...")
    try:
        import subprocess
        subprocess.check_call(["pip", "install", "mss"])
        import mss
        print("mss package installed successfully")
    except Exception as e:
        print(f"Failed to install mss package: {e}")
        # Create a minimal mss substitute to prevent crashes
        class DummyMSS:
            class DummyShot:
                def __init__(self):
                    self.size = (1, 1)
                    self.rgb = b'\x00\x00\x00'
            
            def __enter__(self):
                return self
                
            def __exit__(self, *args):
                pass
                
            def grab(self, *args, **kwargs):
                print("WARNING: Using dummy screenshot - mss package not available")
                return self.DummyShot()
        
        mss = type('', (), {'mss': DummyMSS})

# Try importing PIL for image processing
try:
    from PIL import Image
except ImportError:
    print("PIL/Pillow not found, attempting to install...")
    try:
        import subprocess
        subprocess.check_call(["pip", "install", "Pillow"])
        from PIL import Image
        print("Pillow package installed successfully")
    except Exception as e:
        print(f"Failed to install Pillow package: {e}")

# Try importing OpenCV and numpy for image processing
try:
    import cv2
    import numpy as np
except ImportError:
    print("OpenCV/numpy not found, attempting to install...")
    try:
        import subprocess
        subprocess.check_call(["pip", "install", "opencv-python", "numpy"])
        import cv2
        import numpy as np
        print("OpenCV and numpy packages installed successfully")
    except Exception as e:
        print(f"Failed to install OpenCV/numpy packages: {e}")
        # Create minimal numpy functionality
        np = type('', (), {'array': lambda x: x})

# Try importing PyTesseract for OCR
try:
    import pytesseract
except ImportError:
    print("PyTesseract not found, attempting to install...")
    try:
        import subprocess
        subprocess.check_call(["pip", "install", "pytesseract"])
        import pytesseract
        print("PyTesseract package installed successfully")
    except Exception as e:
        print(f"Failed to install PyTesseract package: {e}")

# Try importing torch and transformers for BLIP model
try:
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
except ImportError:
    print("torch/transformers not found - some AI vision features will be limited")

# Configuration
BLIP_MODEL_LOADED = False
BLIP_MODEL = None

# Configuration
debug = False  # Debug mode flag
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# BLIP model configuration
blip_config = {
    "model_name": "Salesforce/blip-image-captioning-base",
    "max_new_tokens": 200,
    "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32  # Use half precision if CUDA is available
}
print(f"Using device: {device}")

# Initialize BLIP model and processor globally to avoid reloading
blip_processor = None
blip_model = None

def load_blip_model():
    """
    Load BLIP model and processor only once and cache them.
    """
    global blip_processor, blip_model
    if blip_processor is None or blip_model is None:
        print("Loading BLIP model...")
        blip_processor = BlipProcessor.from_pretrained(blip_config["model_name"])
        blip_model = BlipForConditionalGeneration.from_pretrained(
            blip_config["model_name"],
            torch_dtype=blip_config["torch_dtype"]
        )
        blip_model = blip_model.to(device)
        print("BLIP model loaded successfully.")
    return blip_processor, blip_model

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
    # Get cached or load BLIP model
    processor, model = load_blip_model()
    
    image = Image.open(io.BytesIO(image_data))
    inputs = processor(image, return_tensors="pt")
    
    # Move inputs to the same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():  # Disable gradient calculation for inference
        out = model.generate(**inputs, max_new_tokens=blip_config["max_new_tokens"])
    caption = processor.decode(out[0], skip_special_tokens=True)
    print(f"Caption: {caption}")
    
    # Clear CUDA cache to prevent memory leaks if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {"caption": caption}

def cv_detect_and_analyze_regions(image_data: bytes) -> List[Dict[str, Any]]:
    """
    Detect image-like regions in a screenshot and analyze them with BLIP.
    
    Args:
        image_data: Raw image data in bytes
        
    Returns:
        List of dictionaries containing region information and analysis
    """
    try:
        print("Detecting and analyzing image-like regions...")
        
        if not image_data:
            print("Error: No image data provided")
            return []
            
        # Convert bytes to OpenCV image
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            print("Error: Failed to decode image data")
            return []

        # Create a copy of the original image
        original_img = img.copy()
        height, width = img.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply GaussianBlur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding to find edges
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Minimum region size - ignore very small regions
        min_region_size = max(64, min(height, width) * 0.05)
        # Maximum region size - ignore very large regions (like the entire screen)
        max_region_size = min(height, width) * 0.5
        
        # Maximum number of regions to analyze to avoid performance issues
        max_regions = 10
        
        # Store potential UI regions
        regions = []
        
        # Get cached or load BLIP model
        processor, model = load_blip_model()
        
        # Process each contour
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip if too small or too large
            if w < min_region_size or h < min_region_size or w > max_region_size or h > max_region_size:
                continue
            
            # Calculate edge density within the region
            roi = binary[y:y+h, x:x+w]
            edge_pixels = np.count_nonzero(roi)
            total_pixels = roi.size
            edge_density = edge_pixels / total_pixels
            
            # Skip regions with very low or very high edge density
            # Low density = mostly blank, high density = likely noise or text
            if edge_density < 0.05 or edge_density > 0.5:
                continue
            
            # Extract the region from the original image
            region_img = original_img[y:y+h, x:x+w]
            
            # Convert region to PIL Image for BLIP analysis
            region_pil = Image.fromarray(cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB))
            
            # Create a buffer for the region image
            region_buffer = io.BytesIO()
            region_pil.save(region_buffer, format='JPEG')
            region_bytes = region_buffer.getvalue()
            
            # Analyze the region with BLIP
            try:
                inputs = processor(region_pil, return_tensors="pt")
                
                # Move inputs to the same device as model
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():  # Disable gradient calculation for inference
                    out = model.generate(**inputs, max_new_tokens=blip_config["max_new_tokens"])
                
                caption = processor.decode(out[0], skip_special_tokens=True)
                
                # Store the region information
                region_info = {
                    "position": {
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "center_x": x + w // 2,
                        "center_y": y + h // 2
                    },
                    "caption": caption,
                    "edge_density": edge_density,
                    "image_data": region_bytes
                }
                
                regions.append(region_info)
                print(f"Analyzed region at ({x}, {y}, {w}, {h}): {caption}")
                
                # Limit the number of regions to analyze
                if len(regions) >= max_regions:
                    break
            
            except Exception as e:
                print(f"Error analyzing region: {e}")
                continue
        
        # Clear CUDA cache to prevent memory leaks if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"Found and analyzed {len(regions)} image-like regions")
        return regions
    
    except Exception as e:
        import traceback
        print(f"Error in region detection and analysis: {str(e)}")
        traceback.print_exc()
        return []

def get_screenshot_with_analysis(
    monitor_id: int = 0,
) -> Tuple[Optional[bytes], Optional[Dict[str, Any]], Optional[Any], Optional[Any], Optional[List[Dict[str, Any]]]]:
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
            
            # Detect and analyze image-like regions
            ui_regions = cv_detect_and_analyze_regions(image_data)
            
            return image_data, analysis_results, ui_buttons, ui_checkboxes, ui_regions
    except Exception as e:
        print(f"Error capturing screenshot: {e}")
        return None, None, None, None, None

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
        
        # Apply Canny edge detection with tighter thresholds
        edges = cv2.Canny(blurred, 75, 200)
        
        # Find contours in the edge map
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to find potential checkboxes
        checkboxes = []
        
        # More strict minimum size - at least 16x16 pixels or 2% of the screen dimension
        min_size = max(16, min(height, width) * 0.02)  
        # More reasonable maximum size - not more than 5% of the screen dimension
        max_size = min(height, width) * 0.05
        
        # Maximum number of checkboxes to return
        max_checkboxes = 20
        
        for contour in contours:
            # Calculate bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip if too small or too large
            if w < min_size or h < min_size or w > max_size or h > max_size:
                continue
                
            # Filter based on aspect ratio - stricter range for squareness
            aspect_ratio = float(w) / h
            if not (0.8 <= aspect_ratio <= 1.2):
                continue
                
            # Calculate approximated contour
            epsilon = 0.05 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Checkboxes must have 4 corners (roughly rectangular)
            if len(approx) != 4:
                continue
                
            # Check for convexity (checkboxes are convex shapes)
            if not cv2.isContourConvex(approx):
                continue
                
            # Further analyze the candidate checkbox
            checkbox_roi = gray[y:y+h, x:x+w]
            
            # Check for border characteristics
            # Create edge map of just the checkbox region
            checkbox_edges = cv2.Canny(checkbox_roi, 75, 200)
            edge_pixels = np.count_nonzero(checkbox_edges)
            
            # Calculate perimeter-to-area ratio
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            if area == 0:
                continue
                
            perimeter_area_ratio = perimeter / area
            
            # Real checkboxes have a specific edge density
            # Too few edges might be a solid shape, too many might be a complex icon
            edge_density = edge_pixels / (w * h)
            if not (0.1 <= edge_density <= 0.5):
                continue
                
            # Threshold the inner region
            _, checkbox_thresh = cv2.threshold(checkbox_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Create a mask for the inner area (smaller than the checkbox)
            shrink_factor = 0.25
            inner_x = int(w * shrink_factor)
            inner_y = int(h * shrink_factor)
            inner_w = int(w * (1 - 2 * shrink_factor))
            inner_h = int(h * (1 - 2 * shrink_factor))
            
            if inner_w <= 0 or inner_h <= 0:
                continue
                
            inner_roi = checkbox_roi[inner_y:inner_y+inner_h, inner_x:inner_x+inner_w]
            
            if inner_roi.size <= 0:
                continue
                
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
            
            # Format the position with center points for easier clicking
            center_x = x + w // 2
            center_y = y + h // 2
            position = {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "center_x": center_x,
                "center_y": center_y
            }
            
            checkboxes.append({"type": checkbox_type, "position": position})
            print(f"Found {checkbox_type} checkbox at ({x}, {y}, {w}, {h})")
            
            # Limit the number of checkboxes returned
            if len(checkboxes) >= max_checkboxes:
                break
        
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

def analyze_window(window_title: str = None, window_id: str = None) -> Dict[str, Any]:
    """
    Capture and analyze a specific window's content across platforms.
    
    Args:
        window_title: Title of the window to analyze
        window_id: ID of the window to analyze (platform-specific, optional)
    
    Returns:
        Dict containing analysis results with:
        - caption: Description of the window content
        - ui_buttons: Detected buttons in the window
        - ui_checkboxes: Detected checkboxes in the window
        - ui_regions: Detected image regions with captions
        - window_info: Metadata about the window
    """
    system = platform.system()
    window_image = None
    window_info = None
    
    # Find window info based on title or ID
    windows = []
    try:
        # Import locally to avoid module-level import issues
        from window_control import get_open_windows
        windows = get_open_windows()
    except Exception as e:
        print(f"Error getting window list: {e}")
        print(f"Window control module not available or failed, falling back to basic window detection")
        import traceback
        traceback.print_exc()
    
    # Find the target window
    target_window = None
    if windows:
        for window in windows:
            if window_id and "id" in window and str(window["id"]) == str(window_id):
                target_window = window
                break
            elif window_title and "title" in window and window_title.lower() in window["title"].lower():
                target_window = window
                break
    
    if target_window is None:
        print(f"Warning: No window found with ID {window_id} or title {window_title}")
        # Create a minimal target window with the provided information
        target_window = {"id": window_id, "title": window_title}
    
    # Capture the window content
    try:
        if system == "Windows":
            window_image = _capture_window_windows(target_window, window_title, window_id)
        elif system == "Darwin":  # macOS
            window_image = _capture_window_macos(target_window, window_title, window_id)
        elif system == "Linux":
            window_image = _capture_window_linux(target_window, window_title, window_id)
        else:
            print(f"Unsupported platform: {system}")
            return {"error": f"Unsupported platform: {system}"}
    except Exception as e:
        import traceback
        print(f"Error capturing window: {e}")
        traceback.print_exc()
        return {"error": f"Error capturing window: {str(e)}"}
    
    # If window capture failed
    if window_image is None:
        # Fall back to full screen capture
        try:
            print(f"Failed to capture window: {window_title or window_id}, falling back to full screen capture")
            window_image = get_screenshot(1)
        except Exception as e:
            import traceback
            print(f"Error during fallback screenshot: {e}")
            traceback.print_exc()
            return {"error": f"Failed to capture window: {window_title or window_id}"}
    
    # Analyze the window content
    try:
        # Analyze image content
        analysis_results = analyze_image(window_image)
        
        # Detect UI elements
        ui_buttons = cv_find_all_buttons(window_image)
        ui_checkboxes = cv_find_checkboxes(window_image)
        
        # Detect and analyze image regions
        ui_regions = cv_detect_and_analyze_regions(window_image)
        
        # Combine results
        result = {
            "caption": analysis_results.get("caption", ""),
            "ui_buttons": ui_buttons,
            "ui_checkboxes": ui_checkboxes,
            "ui_regions": ui_regions,
            "window_info": target_window
        }
        
        return result
    except Exception as e:
        import traceback
        print(f"Error analyzing window content: {e}")
        traceback.print_exc()
        return {"error": f"Error analyzing window content: {str(e)}"}

def _capture_window_windows(window_info=None, window_title=None, window_id=None) -> Optional[bytes]:
    """Capture a window on Windows"""
    try:
        # If window_info contains position and size, use that for screenshot
        if window_info and "position" in window_info and "size" in window_info:
            pos = window_info["position"]
            size = window_info["size"]
            
            with mss.mss() as sct:
                monitor = {
                    "left": pos["x"],
                    "top": pos["y"],
                    "width": size["width"],
                    "height": size["height"]
                }
                screenshot = sct.grab(monitor)
                img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
                
                buf = io.BytesIO()
                img.save(buf, format='JPEG')
                return buf.getvalue()
        
        # Try alternative method if window_info doesn't have position/size
        # or if window capture failed
        try:
            import win32gui
            import win32ui
            from ctypes import windll
            from PIL import Image
            
            if window_id:
                hwnd = int(window_id)
            elif window_title:
                hwnd = win32gui.FindWindow(None, window_title)
            else:
                return None
                
            if not hwnd:
                return None
                
            # Get window dimensions
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            width = right - left
            height = bottom - top
            
            # Create device context
            hwndDC = win32gui.GetWindowDC(hwnd)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()
            
            # Create bitmap
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
            saveDC.SelectObject(saveBitMap)
            
            # Copy screen to bitmap
            result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)
            
            # Convert to PIL Image
            bmpinfo = saveBitMap.GetInfo()
            bmpstr = saveBitMap.GetBitmapBits(True)
            img = Image.frombuffer(
                'RGB',
                (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                bmpstr, 'raw', 'BGRX', 0, 1)
            
            # Clean up
            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwndDC)
            
            buf = io.BytesIO()
            img.save(buf, format='JPEG')
            return buf.getvalue()
        except ImportError:
            print("win32gui/win32ui not available, falling back to full screen capture")
            return get_screenshot(1)  # Capture primary monitor as fallback
    except Exception as e:
        print(f"Error capturing window on Windows: {e}")
        return None

def _capture_window_macos(window_info=None, window_title=None, window_id=None) -> Optional[bytes]:
    """Capture a window on macOS"""
    try:
        # If window_info contains position and size, use that for screenshot
        if window_info and "position" in window_info and "size" in window_info:
            pos = window_info["position"]
            size = window_info["size"]
            
            # If position is available and valid
            if pos["x"] > 0 or pos["y"] > 0:
                with mss.mss() as sct:
                    monitor = {
                        "left": pos["x"],
                        "top": pos["y"],
                        "width": size["width"],
                        "height": size["height"]
                    }
                    screenshot = sct.grab(monitor)
                    img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
                    
                    buf = io.BytesIO()
                    img.save(buf, format='JPEG')
                    return buf.getvalue()
        
        # Use screencapture command-line tool as fallback
        if window_title:
            # Create a temporary file to save the screenshot
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Use AppleScript to capture the specific window
            script = f"""
            tell application "System Events"
                set frontApp to first application process whose frontmost is true
                set frontAppName to name of frontApp
                
                set targetApp to null
                set targetWindow to null
                
                set appList to every application process whose visible is true
                repeat with oneApp in appList
                    set appName to name of oneApp
                    
                    repeat with oneWindow in windows of oneApp
                        if name of oneWindow contains "{window_title}" then
                            set targetApp to oneApp
                            set targetWindow to oneWindow
                            exit repeat
                        end if
                    end repeat
                    
                    if targetApp is not null then
                        exit repeat
                    end if
                end repeat
                
                if targetApp is not null then
                    set frontmost of targetApp to true
                    delay 0.5
                    do shell script "screencapture -l$(osascript -e 'tell application \\"System Events\\" to id of window 1 of process \\"" & name of targetApp & "\\"') -o {temp_path}"
                    return true
                else
                    return false
                end if
            end tell
            """
            
            result = subprocess.run(['osascript', '-e', script], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0 and "true" in result.stdout.lower() and os.path.exists(temp_path):
                # Read the captured screenshot
                with open(temp_path, 'rb') as img_file:
                    img_data = img_file.read()
                
                # Clean up temp file
                os.unlink(temp_path)
                return img_data
            else:
                # Clean up temp file if it exists
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                
                # Fall back to full screen capture
                return get_screenshot(1)
        else:
            # If no window title provided, capture the active window
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Capture the active window
            result = subprocess.run(['screencapture', '-JW', temp_path], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(temp_path):
                # Read the captured screenshot
                with open(temp_path, 'rb') as img_file:
                    img_data = img_file.read()
                
                # Clean up temp file
                os.unlink(temp_path)
                return img_data
            else:
                # Clean up temp file if it exists
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                
                # Fall back to full screen capture
                return get_screenshot(1)
    except Exception as e:
        print(f"Error capturing window on macOS: {e}")
        return None

def _capture_window_linux(window_info=None, window_title=None, window_id=None) -> Optional[bytes]:
    """Capture a window on Linux"""
    try:
        # If window_info contains position and size, use that for screenshot
        if window_info and "position" in window_info and "size" in window_info and isinstance(window_info["position"], dict) and isinstance(window_info["size"], dict):
            pos = window_info["position"]
            size = window_info["size"]
            
            # If position is available and valid
            if pos.get("x", 0) > 0 or pos.get("y", 0) > 0:
                try:
                    with mss.mss() as sct:
                        monitor = {
                            "left": pos.get("x", 0),
                            "top": pos.get("y", 0),
                            "width": size.get("width", 800),
                            "height": size.get("height", 600)
                        }
                        screenshot = sct.grab(monitor)
                        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
                        
                        buf = io.BytesIO()
                        img.save(buf, format='JPEG')
                        return buf.getvalue()
                except Exception as e:
                    print(f"Error using mss for window capture: {e}")
                    # Continue to alternative methods
        
        # Try using xwd or import to capture the window
        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
            
            capture_cmd = None
            
            if window_id:
                # Try xwd with window ID
                capture_cmd = ['xwd', '-id', str(window_id), '-out', temp_path]
            elif window_title:
                # Try to get window ID first using xdotool
                try:
                    win_id_result = subprocess.run(
                        ['xdotool', 'search', '--name', window_title], 
                        capture_output=True, text=True
                    )
                    
                    if win_id_result.returncode == 0 and win_id_result.stdout.strip():
                        window_id = win_id_result.stdout.strip().split('\n')[0]
                        # Now use import to capture the window
                        capture_cmd = ['import', '-window', window_id, temp_path]
                    else:
                        # Activate the window first, then capture
                        subprocess.run(['xdotool', 'search', '--name', window_title, 'windowactivate'], 
                                      capture_output=True, text=True)
                        time.sleep(0.5)  # Allow time for window to activate
                        capture_cmd = ['import', '-window', 'root', temp_path]
                except FileNotFoundError:
                    print("xdotool not found, trying alternative methods")
                    # Try using scrot instead
                    capture_cmd = ['scrot', temp_path]
            else:
                # Capture active window or full screen
                capture_cmd = ['scrot', '-u', temp_path]
            
            if capture_cmd:
                try:
                    result = subprocess.run(capture_cmd, capture_output=True, text=True, timeout=5)
                    
                    if result.returncode == 0 and os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                        # Read the captured screenshot
                        with open(temp_path, 'rb') as img_file:
                            img_data = img_file.read()
                        
                        # Clean up temp file
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
                            
                        return img_data
                except subprocess.TimeoutExpired:
                    print("Screenshot command timed out")
                except Exception as e:
                    print(f"Error running screenshot command: {e}")
            
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
        except Exception as e:
            print(f"Error with command-line screenshot tools: {e}")
        
        # Fall back to full screen capture if all else fails
        print("Falling back to full screen capture")
        return get_screenshot(1)
    except Exception as e:
        import traceback
        print(f"Error capturing window on Linux: {e}")
        traceback.print_exc()
        # Return None and let the caller handle the fallback
        return None
 