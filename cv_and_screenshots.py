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
import PIL
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import pyautogui

# Initialize globals for BLIP model
global blip_model, blip_processor
BLIP_MODEL_LOADED = False
blip_model = None
blip_processor = None
blip_config = {"max_new_tokens": 100}

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

# Try importing pyautogui for mouse position
try:
    import pyautogui
except ImportError:
    print("pyautogui not found, attempting to install...")
    try:
        import subprocess
        subprocess.check_call(["pip", "install", "pyautogui"])
        import pyautogui
        print("pyautogui package installed successfully")
    except Exception as e:
        print(f"Failed to install pyautogui package: {e}")

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_blip_model():
    global BLIP_MODEL_LOADED, blip_model, blip_processor
    
    print("Starting BLIP model loading process...")
    
    # Check if already loaded successfully
    if BLIP_MODEL_LOADED and blip_model is not None and blip_processor is not None:
        print("BLIP model already loaded, reusing existing model")
        return blip_processor, blip_model
    
    try:
        # First check if torch and transformers are available
        try:
            import torch
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            print(f"Using torch {torch.__version__} on device: {device}")
        except ImportError as imp_err:
            print(f"Required libraries not available: {imp_err}")
            print("Please install with: pip install torch transformers")
            BLIP_MODEL_LOADED = False
            return None, None
        
        # Use a smaller model if available
        model_options = [
            "Salesforce/blip2-opt-2.7b",  # First choice - larger but better
        ]
        
        # Try each model until one works
        for model_name in model_options:
            try:
                print(f"Attempting to load model: {model_name}")
                
                # Try with less memory usage for systems with limited GPU/RAM
                processor = Blip2Processor.from_pretrained(model_name)
                
                # Set lower precision and offload to CPU if memory is limited
                model = Blip2ForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                    offload_folder="offload",  # Offload to disk if needed
                    revision="main"
                )
                
                print(f"Successfully loaded model: {model_name}")
                BLIP_MODEL_LOADED = True
                blip_processor = processor
                blip_model = model
                return processor, model
            except Exception as model_err:
                print(f"Failed to load model {model_name}: {model_err}")
                continue
        
        # If we get here, all models failed
        print("All model options failed to load")
        BLIP_MODEL_LOADED = False
        return None, None
    except Exception as e:
        print(f"Unexpected error loading BLIP model: {e}")
        import traceback
        traceback.print_exc()
        BLIP_MODEL_LOADED = False
        return None, None

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


def caption_image(image: np.ndarray) -> Dict[str, str]:
    global blip_model, blip_processor, BLIP_MODEL_LOADED
    
    # Check if model is loaded, if not, load it
    if not BLIP_MODEL_LOADED or blip_model is None or blip_processor is None:
        try:
            print("Loading BLIP model for image captioning...")
            blip_processor, blip_model = load_blip_model()
            if not blip_processor or not blip_model:
                print("Failed to load BLIP model properly")
                return {"caption": "Unable to load image captioning model", 
                        "error": "Model initialization failed"}
            print("BLIP model loaded successfully")
        except Exception as e:
            print(f"Error loading BLIP model: {e}")
            return {"caption": "Failed to load image captioning model", "error": str(e)}
    
    try:
        # Convert bytes to PIL Image if image is in bytes format
        if isinstance(image, bytes):
            image_pil = Image.open(io.BytesIO(image))
            print(f"Loaded image from bytes, size: {image_pil.size}")
        else:
            image_pil = Image.fromarray(image)
            print(f"Converted numpy array to PIL Image, size: {image_pil.size}")
            
        # Use no-prompt approach directly since it's what worked
        print("Using no-prompt approach for image captioning")
        inputs = blip_processor(images=image_pil, text="", return_tensors="pt").to(device)
        
        # Generate caption with more parameters for better quality
        out = blip_model.generate(
            **inputs, 
            max_new_tokens=100,
            num_beams=5,
            no_repeat_ngram_size=3,
            temperature=0.7
        )
        caption = blip_processor.tokenizer.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        print(f"Generated caption with no prompt: '{caption}'")
        return {"caption": caption}
    except Exception as e:
        print(f"Error generating image caption: {e}")
        import traceback
        traceback.print_exc()
        return {"caption": "Failed to generate caption", "error": str(e)}

def cv_detect_and_analyze_regions(image_data: bytes) -> List[Dict[str, Any]]:
    """
    Detect image-like regions in a screenshot and analyze them with BLIP.
    
    Args:
        image_data: Raw image data in bytes
        
    Returns:
        List of dictionaries containing region information and analysis
    """
    global blip_model, blip_processor, BLIP_MODEL_LOADED
    
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
        
        # MODIFIED: More permissive minimum region size for browser content
        min_region_size = max(40, min(height, width) * 0.03)  # Reduced from 0.05 to 0.03
        
        # Maximum region size - increased to handle larger browser images
        max_region_size = min(height, width) * 0.7  # Increased from 0.5 to 0.7
        
        # MODIFIED: Reduced max regions to prevent memory issues
        max_regions = 8  # Reduced from 20 to 8 to prevent CUDA OOM
        
        # Store potential UI regions
        regions = []
        candidate_regions = []  # Store region coordinates before processing
        
        # Get cached or load BLIP model - using same pattern as caption_image function
        if not BLIP_MODEL_LOADED or blip_processor is None or blip_model is None:
            try:
                print("Loading BLIP model for region analysis...")
                blip_processor, blip_model = load_blip_model()
                if blip_processor is None or blip_model is None:
                    print("Failed to load BLIP model, skipping region analysis")
                    return []
            except Exception as e:
                print(f"Error loading BLIP model: {e}")
                return []
        
        processor, model = blip_processor, blip_model
        
        # Check GPU memory availability
        try:
            is_gpu_low_memory = False
            if torch.cuda.is_available():
                total_mem = torch.cuda.get_device_properties(0).total_memory
                reserved_mem = torch.cuda.memory_reserved(0)
                allocated_mem = torch.cuda.memory_allocated(0)
                free_mem = total_mem - reserved_mem
                print(f"GPU memory - Total: {total_mem/1e9:.2f}GB, Free: {free_mem/1e9:.2f}GB, Used: {allocated_mem/1e9:.2f}GB")
                
                # If less than 1GB free, consider low memory
                if free_mem < 1e9:
                    print("GPU memory is low, will use reduced processing or CPU fallback")
                    is_gpu_low_memory = True
        except Exception as e:
            print(f"Error checking GPU memory: {e}")
            is_gpu_low_memory = True
        
        # ADDED: Browser-specific detection for rectangular images
        # This helps detect images in search results and web content
        browser_images = []
        try:
            # Apply Canny edge detection with parameters tuned for web content
            edges = cv2.Canny(blurred, 50, 150)
            # Dilate edges to connect broken ones
            kernel = np.ones((3,3), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Find rectangular contours (common for web images)
            web_contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in web_contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Minimum size for browser images (smaller than general detection)
                if w < 40 or h < 40:  # More permissive minimum size
                    continue
                
                # Skip if too large
                if w > width * 0.9 or h > height * 0.9:
                    continue
                
                # Calculate aspect ratio - web images often have standard aspect ratios
                aspect_ratio = float(w) / h
                
                # Most web images have reasonable aspect ratios between 0.5 and 2.5
                if aspect_ratio < 0.5 or aspect_ratio > 2.5:
                    continue
                
                # Extract the region
                region_img = original_img[y:y+h, x:x+w]
                
                # Check for content diversity - images should have varied pixel values
                region_gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
                _, region_std = cv2.meanStdDev(region_gray)
                
                # Skip regions with very low variance (likely solid colors or UI elements)
                if region_std[0][0] < 20:
                    continue
                
                # Add to browser images list
                browser_images.append((x, y, w, h))
        except Exception as e:
            print(f"Error in browser-specific detection: {e}")

        # Process each contour from the original detection
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
            
            # MODIFIED: More permissive edge density range for browser content
            # Low density = mostly blank, high density = likely noise or text
            if edge_density < 0.03 or edge_density > 0.7:  # Changed from 0.05-0.5 to 0.03-0.7
                continue
            
            # Instead of processing immediately, add to candidate list
            candidate_regions.append((x, y, w, h))
        
        # Combine both standard and browser-specific regions
        all_candidate_regions = candidate_regions + browser_images
        
        # Prioritize regions by size (prefer larger regions) - helps focus on main content
        all_candidate_regions.sort(key=lambda r: r[2] * r[3], reverse=True)
        
        # Limit number of regions to process
        all_candidate_regions = all_candidate_regions[:12]  # Process at most 12 candidates
        
        # ADDED: Function to free GPU memory
        def free_gpu_memory():
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    print("Cleared GPU cache")
                except Exception as e:
                    print(f"Error clearing GPU cache: {e}")
        
        # MODIFIED: Process regions with memory management
        processed_count = 0
        for x, y, w, h in all_candidate_regions:
            # Stop if we've reached max regions
            if len(regions) >= max_regions:
                break
                
            # Skip if this region overlaps significantly with already detected regions
            skip = False
            for region in regions:
                rx = region["position"]["x"]
                ry = region["position"]["y"]
                rw = region["position"]["width"]
                rh = region["position"]["height"]
                
                # Calculate overlap
                overlap_x = max(0, min(x + w, rx + rw) - max(x, rx))
                overlap_y = max(0, min(y + h, ry + rh) - max(y, ry))
                overlap_area = overlap_x * overlap_y
                
                # Skip if overlap is significant (>50% of the smaller region)
                if overlap_area > 0.5 * min(w * h, rw * rh):
                    skip = True
                    break
            
            if skip:
                continue
            
            # Extract the region from the original image
            region_img = original_img[y:y+h, x:x+w]
            
            # Convert region to PIL Image for BLIP analysis
            region_pil = Image.fromarray(cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB))
            
            # ADDED: Downsize large images to conserve memory
            max_image_size = 512 if is_gpu_low_memory else 1024
            if region_pil.width > max_image_size or region_pil.height > max_image_size:
                region_pil.thumbnail((max_image_size, max_image_size), Image.LANCZOS)
                print(f"Resized large region to {region_pil.size} to save memory")
            
            # Create a buffer for the region image
            region_buffer = io.BytesIO()
            region_pil.save(region_buffer, format='JPEG')
            region_bytes = region_buffer.getvalue()
            
            # ADDED: CPU fallback if GPU memory is low
            device_to_use = "cpu" if is_gpu_low_memory else device
            
            # Analyze the region with BLIP
            try:
                # Free memory before processing
                if processed_count > 0 and processed_count % 2 == 0:  # Every 2 images
                    free_gpu_memory()
                
                inputs = processor(images=region_pil, text="What can you see in this image?", return_tensors="pt")
                
                # Move inputs to the selected device
                inputs = {k: v.to(device_to_use) for k, v in inputs.items()}
                
                # Move model to same device if using CPU fallback
                if device_to_use == "cpu" and str(next(model.parameters()).device) != "cpu":
                    # Use a smaller subsection of the model on CPU to save memory
                    print("Using CPU for inference due to GPU memory constraints")
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=50)  # Reduced tokens for CPU
                else:
                    with torch.no_grad():  # Disable gradient calculation for inference
                        out = model.generate(**inputs, max_new_tokens=blip_config["max_new_tokens"])
                
                caption = processor.decode(out[0], skip_special_tokens=True)
                
                # Only add if caption suggests this is an actual image
                if not any(keyword in caption.lower() for keyword in ["website", "webpage", "interface"]):
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
                        "edge_density": edge_density if 'edge_density' in locals() else None,
                        "image_data": region_bytes
                    }
                    
                    regions.append(region_info)
                    print(f"Analyzed region at ({x}, {y}, {w}, {h}): {caption}")
                    processed_count += 1
                else:
                    print(f"Skipped non-image region at ({x}, {y}, {w}, {h}): {caption}")
            
            except Exception as e:
                print(f"Error analyzing region: {e}")
                # If CUDA OOM, try to recover by clearing cache and reducing future processing
                if "CUDA out of memory" in str(e):
                    print("CUDA OOM detected, trying to recover...")
                    free_gpu_memory()
                    is_gpu_low_memory = True  # Switch to low memory mode
                    
                    # Try one more time with CPU
                    try:
                        print("Retrying with CPU...")
                        # Move to CPU
                        cpu_inputs = {k: v.to("cpu") for k, v in inputs.items()}
                        
                        with torch.no_grad():
                            # Use a smaller text generation to save memory
                            out = model.to("cpu").generate(**cpu_inputs, max_new_tokens=30)
                            caption = processor.decode(out[0], skip_special_tokens=True)
                            
                            region_info = {
                                "position": {
                                    "x": x,
                                    "y": y,
                                    "width": w,
                                    "height": h,
                                    "center_x": x + w // 2,
                                    "center_y": y + h // 2
                                },
                                "caption": caption + " (CPU processed)",
                                "edge_density": edge_density if 'edge_density' in locals() else None,
                                "image_data": region_bytes
                            }
                            
                            regions.append(region_info)
                            print(f"Successfully processed with CPU fallback: {caption}")
                            
                            # Move model back to original device
                            if torch.cuda.is_available():
                                model.to(device)
                    except Exception as cpu_e:
                        print(f"CPU fallback also failed: {cpu_e}")
                continue
        
        # Final attempt to clean up memory
        free_gpu_memory()
        
        print(f"Found and analyzed {len(regions)} image-like regions")
        return regions
    
    except Exception as e:
        import traceback
        print(f"Error in region detection and analysis: {str(e)}")
        traceback.print_exc()
        return []

def get_screenshot_with_analysis(
    monitor_id: int = 0,
) -> Tuple[Optional[bytes], Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]:
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
            analysis_results = caption_image(image_data)
            ui_buttons = cv_find_all_buttons(image_data)
            ui_checkboxes = cv_find_checkboxes(image_data)
            
            # Detect and analyze image-like regions
            ui_regions = cv_detect_and_analyze_regions(image_data)
            
            # ADDED: Filter out buttons with text "Button"
            if ui_buttons:
                filtered_buttons = []
                for button in ui_buttons:
                    if button.get("text") != "Button":
                        filtered_buttons.append(button)
                print(f"Filtered out {len(ui_buttons) - len(filtered_buttons)} buttons with generic 'Button' text")
                ui_buttons = filtered_buttons
            
            return image_data, analysis_results, ui_buttons, ui_checkboxes, ui_regions
    except Exception as e:
        print(f"Error capturing screenshot: {e}")
        return None, None, None, None, None

def cv_find_all_buttons(image_data: bytes):
    """
    Find button-like elements in an image with improved detection for styled UI buttons.
    
    Args:
        image_data: Raw image data in bytes
        
    Returns:
        List of tuples with format (button_text, (x, y, w, h))
    """
    try:
        print("Looking for buttons in image with enhanced detection...")
        
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
        
        # Convert to HSV for color-based detection (good for colorful buttons)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Convert to grayscale for shape-based detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply thresholding - using multiple approaches for better detection
        _, thresh1 = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)
        
        # Also try adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY_INV, 11, 2)
        
        # Combine the thresholds
        combined_thresh = cv2.bitwise_or(thresh1, adaptive_thresh)
        
        # Find contours in both thresholded images
        contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours = contours1 + contours2
        
        # More reasonable minimum size - at least 40 pixels wide or 3% of the screen width
        min_width = max(40, width * 0.03)
        min_height = max(15, height * 0.02)
        
        # Max size - not more than 30% of the screen dimension
        max_width = width * 0.3
        max_height = height * 0.15
        
        # Store detected buttons
        buttons = []
        button_regions = []  # To avoid duplicates
        
        # First, apply direct OCR to look for prominent text that might be button labels
        # This helps identify UI text without relying solely on contour detection
        direct_text_results = pytesseract.image_to_data(original_img, output_type=pytesseract.Output.DICT)
        
        # Look for common UI button texts - BE MORE SPECIFIC to avoid false positives
        common_button_texts = ["play", "start game", "single player", "multiplayer", "options", "settings", 
                              "quit", "exit", "cancel", "ok", "yes", "no", "next", "back", "continue"]
        
        direct_button_regions = []
        for i, text in enumerate(direct_text_results["text"]):
            if not text or len(text) < 5:  # CHANGED: Increased minimum text length from 4 to 5 characters
                continue
                
            # Check confidence
            confidence = int(direct_text_results["conf"][i])
            if confidence < 80:  # INCREASED confidence threshold (was 70)
                continue
                
            text_lower = text.lower()
            x, y, w, h = (direct_text_results["left"][i], direct_text_results["top"][i],
                        direct_text_results["width"][i], direct_text_results["height"][i])
            
            # Skip if the width-to-height ratio is too extreme
            aspect_ratio = float(w) / h if h > 0 else 0
            if aspect_ratio > 8.0 or aspect_ratio < 1.0:  # ADDED aspect ratio check
                continue
            
            # Expand region slightly to capture button borders
            expanded_x = max(0, x - int(w * 0.2))
            expanded_y = max(0, y - int(h * 0.2))
            expanded_w = min(width - expanded_x, int(w * 1.4))
            expanded_h = min(height - expanded_y, int(h * 1.4))
            
            # STRICTER button text matching - must be an exact match to common button texts or
            # have specific visual characteristics of buttons
            is_button_text = text_lower in common_button_texts
            
            # Additional checks for "PLAY" to avoid false positives
            if text.upper() == "PLAY" or text_lower == "play":
                # Check if this is inside a button-like UI element
                # Extract the region around the text to check for button-like appearance
                region_x = max(0, x - w)
                region_y = max(0, y - h)
                region_w = min(width - region_x, w * 3)
                region_h = min(height - region_y, h * 3)
                
                if region_x < width and region_y < height and region_w > 0 and region_h > 0:
                    region = original_img[region_y:region_y+region_h, region_x:region_x+region_w]
                    
                    # Check for visual button-like features
                    region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(region_gray, 50, 150)
                    edge_density = np.count_nonzero(edges) / (region_w * region_h)
                    
                    # Buttons typically have edges/borders
                    if edge_density < 0.1:  # Not enough edges to be a button
                        continue
            
            if is_button_text:
                print(f"Found potential button text '{text}' at ({x}, {y}, {w}, {h}) with confidence {confidence}")
                direct_button_regions.append((expanded_x, expanded_y, expanded_w, expanded_h, text, confidence))
        
        # Process each contour
        for cnt in all_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Skip if too small
            if w < min_width or h < min_height:
                continue
                
            # Skip if too large
            if w > max_width or h > max_height:
                continue

            # Calculate aspect ratio
            aspect_ratio = float(w) / h
            
            # Most buttons are wider than tall with reasonable limits
            # This is more flexible than previous criterion (w/h < 1.5)
            if not (1.2 <= aspect_ratio <= 6.0):
                continue
                
            # Check if this region overlaps with already detected buttons
            is_duplicate = False
            for bx, by, bw, bh in button_regions:
                # Calculate overlap
                overlap_x = max(0, min(x + w, bx + bw) - max(x, bx))
                overlap_y = max(0, min(y + h, by + bh) - max(y, by))
                overlap_area = overlap_x * overlap_y
                current_area = w * h
                
                # If significant overlap, consider it a duplicate
                if overlap_area > 0.5 * current_area:
                    is_duplicate = True
                    break
                    
            if is_duplicate:
                continue
                
            # Extract region from original image
            roi = original_img[y:y+h, x:x+w]
            
            # ADDED: Check for visual button-like features
            # Buttons typically have distinct edges or borders
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(roi_gray, 50, 150)
            edge_density = np.count_nonzero(edges) / (w * h)
            
            # If edge density is too low, this is likely not a button (just a plain region)
            if edge_density < 0.05:
                continue
            
            # Check color properties for button-like appearance
            hsv_roi = hsv[y:y+h, x:x+w]
            
            # Calculate color histogram
            hist = cv2.calcHist([hsv_roi], [0, 1], None, [30, 30], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            
            # Calculate color concentration (higher values indicate more solid color regions)
            color_concentration = np.max(hist)
            
            # Enhanced OCR preprocessing for better text extraction
            # Try multiple preprocessing approaches
            ocr_texts = []
            confidences = []
            
            # 1. Original ROI
            ocr_result = pytesseract.image_to_data(roi, config='--psm 7 --oem 1', output_type=pytesseract.Output.DICT)
            for i, txt in enumerate(ocr_result["text"]):
                if txt and len(txt.strip()) > 3:
                    ocr_texts.append(txt.strip())
                    confidences.append(int(ocr_result["conf"][i]))
            
            # 2. Threshold version
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ocr_result = pytesseract.image_to_data(roi_thresh, config='--psm 7 --oem 1', output_type=pytesseract.Output.DICT)
            for i, txt in enumerate(ocr_result["text"]):
                if txt and len(txt.strip()) > 3:
                    ocr_texts.append(txt.strip())
                    confidences.append(int(ocr_result["conf"][i]))
                    
            # 3. Contrast enhanced version
            roi_contrast = cv2.convertScaleAbs(roi, alpha=1.5, beta=0)
            ocr_result = pytesseract.image_to_data(roi_contrast, config='--psm 7 --oem 1', output_type=pytesseract.Output.DICT)
            for i, txt in enumerate(ocr_result["text"]):
                if txt and len(txt.strip()) > 3:
                    ocr_texts.append(txt.strip())
                    confidences.append(int(ocr_result["conf"][i]))
                    
            # Choose the best text based on confidence
            text = ""
            best_confidence = 0
            if ocr_texts:
                best_idx = np.argmax(confidences) if confidences else 0
                text = ocr_texts[best_idx]
                best_confidence = confidences[best_idx] if confidences else 0
                print(f"Best OCR text: '{text}' with confidence {best_confidence}")
            
            # INCREASED confidence threshold for text
            if best_confidence < 75:
                text = ""
            
            # Special case for stylized text that OCR might miss
            if not text:
                # Check for the green PLAY button in Minecraft - BE MORE SPECIFIC
                # Green color detection in HSV
                lower_green = np.array([40, 40, 40])
                upper_green = np.array([80, 255, 255])
                green_mask = cv2.inRange(hsv_roi, lower_green, upper_green)
                green_ratio = np.count_nonzero(green_mask) / (w * h)
                
                # ADDED MORE SPECIFIC CRITERIA for a Minecraft PLAY button
                # If a significant portion is green, shape is button-like, and has button-like edges
                if (green_ratio > 0.4 and 2.0 <= aspect_ratio <= 3.5 and 
                    0.1 <= edge_density <= 0.5):  # Buttons have reasonable edge density
                    # Check for rounded corners (typical of Minecraft buttons)
                    approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
                    if len(approx) > 4:  # More than 4 points suggests rounded corners
                        text = "PLAY"
                        print(f"Detected stylized PLAY button based on color at ({x}, {y}, {w}, {h})")
                    
                # Check for white text on dark background (common in games)
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                white_pixels = np.sum(roi_gray > 200)
                dark_pixels = np.sum(roi_gray < 50)
                white_ratio = white_pixels / (w * h)
                dark_ratio = dark_pixels / (w * h)
                
                if white_ratio > 0.1 and dark_ratio > 0.4:
                    # Apply inverse binary threshold to isolate white text
                    _, white_text = cv2.threshold(roi_gray, 180, 255, cv2.THRESH_BINARY)
                    text_result = pytesseract.image_to_string(white_text, config='--psm 7 --oem 1').strip()
                    if text_result and len(text_result) > 3:
                        text = text_result
                        print(f"Detected white-on-dark text: '{text}' at ({x}, {y}, {w}, {h})")
            
            # ADDED: More precise checks for adding buttons
            should_add_button = False
            
            # 1. If we found text matching common button labels
            common_button_keywords = ["play", "start", "options", "quit", "exit", "settings", 
                                      "ok", "cancel", "yes", "no", "continue", "back", "next",
                                      "single player", "multiplayer", "menu", "create"]
            if text and any(keyword == text.lower() for keyword in common_button_keywords):
                should_add_button = True
            
            # 2. If it has strong button-like visual appearance
            elif color_concentration > 0.3 and 0.1 <= edge_density <= 0.5:
                # Calculate the variance of colors to determine if it looks like a UI element
                roi_std = np.std(roi_gray)
                if roi_std > 20:  # Not a uniform region
                    should_add_button = True
            
            # If button criteria satisfied, add it
            if should_add_button or (text and len(text) > 3):
                button_text = text if text else "Button"
                button_info = {
                    "text": button_text,
                    "position": {
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "center_x": x + w // 2,
                        "center_y": y + h // 2
                    }
                }
                buttons.append(button_info)
                button_regions.append((x, y, w, h))
                print(f"Found button '{button_text}' at ({x}, {y}, {w}, {h})")
        
        # Add the direct text buttons if they don't overlap with existing buttons
        for ex, ey, ew, eh, text, conf in direct_button_regions:
            is_duplicate = False
            for bx, by, bw, bh in button_regions:
                # Calculate overlap
                overlap_x = max(0, min(ex + ew, bx + bw) - max(ex, bx))
                overlap_y = max(0, min(ey + eh, by + bh) - max(ey, by))
                overlap_area = overlap_x * overlap_y
                direct_area = ew * eh
                
                # If significant overlap, consider it a duplicate
                if overlap_area > 0.3 * direct_area:
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                # ADDED: Extract the region to check if it has button-like appearance
                if (ex >= 0 and ey >= 0 and ex + ew <= width and ey + eh <= height and 
                    ew > 0 and eh > 0):
                    region = original_img[ey:ey+eh, ex:ex+ew]
                    region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(region_gray, 50, 150)
                    edge_density = np.count_nonzero(edges) / (ew * eh)
                    
                    # Only add if it has visual button-like characteristics
                    if 0.05 <= edge_density <= 0.5:
                        button_info = {
                            "text": text,
                            "position": {
                                "x": ex,
                                "y": ey,
                                "width": ew,
                                "height": eh,
                                "center_x": ex + ew // 2,
                                "center_y": ey + eh // 2
                            }
                        }
                        buttons.append(button_info)
                        button_regions.append((ex, ey, ew, eh))
                        print(f"Found text button '{text}' at ({ex}, {ey}, {ew}, {eh})")
        
        print(f"Found {len(buttons)} buttons")
        return buttons
    except Exception as e:
        import traceback
        print(f"Error finding buttons: {str(e)}")
        traceback.print_exc()
        return []

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
        print(f"Finding text '{target}' in image using enhanced text detection...")
        
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
        
        # Pre-process target text for better matching
        target_lower = target.lower().strip()
        print(f"Searching for target text: '{target_lower}'")
        
        # Check if this is a long string search (more than 15 characters)
        is_long_string = len(target_lower) > 15
        print(f"Searching for {'long string' if is_long_string else 'short string'}")
        
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
        
        # 6. High contrast enhancement (good for game UIs)
        contrast_enhanced = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
        contrast_gray = cv2.cvtColor(contrast_enhanced, cv2.COLOR_BGR2GRAY)
        _, contrast_thresh = cv2.threshold(contrast_gray, 150, 255, cv2.THRESH_BINARY)
        preprocessed_images.append(("high_contrast", contrast_thresh))
        
        # 7. Color filtering for white text (common in games)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Create a mask that captures white/light text
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        preprocessed_images.append(("white_text_mask", white_mask))
        
        # 8. Inverse for dark text on light background
        inverse_mask = cv2.bitwise_not(white_mask)
        preprocessed_images.append(("dark_text_mask", inverse_mask))
        
        # 9. Edge detection to highlight text boundaries
        edges = cv2.Canny(gray, 100, 200)
        dilated_edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        preprocessed_images.append(("edge_enhance", dilated_edges))
        
        # Store all matches
        all_matches = []
        
        # For long strings, focus on paragraph detection
        if is_long_string:
            print("Using paragraph detection mode for long string")
            # Use PSM modes optimized for paragraph text
            psm_modes = [
                ('--psm 3', 'Full page'),     # Fully automatic page segmentation
                ('--psm 4', 'Single column'), # Assume a single column of variable-sized text
                ('--psm 6', 'Block of text')  # Assume a uniform block of text
            ]
            
            # For each preprocessing method, try full page OCR
            for name, processed_img in preprocessed_images:
                for psm_config, psm_desc in psm_modes:
                    print(f"Trying paragraph OCR with {name} preprocessing, {psm_desc}...")
                    
                    # Extract full text using image_to_string for better paragraph handling
                    full_text = pytesseract.image_to_string(processed_img, config=f'{psm_config} --oem 1').lower().strip()
                    
                    if not full_text:
                        continue
                        
                    # Look for the target text in the extracted text using fuzzy matching
                    if target_lower in full_text:
                        # Exact match found
                        print(f"Found exact match in paragraph: '{full_text[:50]}...'")
                        match_score = 100
                    else:
                        # Try fuzzy matching
                        import difflib
                        similarity = difflib.SequenceMatcher(None, target_lower, full_text).ratio()
                        
                        # Only consider reasonable matches
                        if similarity < 0.6:
                            continue
                            
                        match_score = int(similarity * 100)
                        print(f"Found fuzzy match (score: {match_score}): '{full_text[:50]}...'")
                    
                    # Create a match for the entire image as we don't have precise bounds
                    match_info = {
                        "word": full_text,
                        "bounds": (0, 0, width, height),  # Entire image
                        "confidence": 70,  # Moderate confidence for paragraph matches
                        "method": f"{name}_{psm_desc}_paragraph",
                        "match_score": match_score,
                        "is_paragraph": True
                    }
                    all_matches.append(match_info)
            
            # If we found paragraph matches, also try to find more precise locations
            if all_matches:
                # Try to narrow down the location by splitting the image
                regions = []
                # Split into a 3x3 grid for more precise location
                h_step, w_step = height // 3, width // 3
                for y in range(0, height, h_step):
                    for x in range(0, width, w_step):
                        # Ensure we don't go out of bounds
                        region_w = min(w_step, width - x)
                        region_h = min(h_step, height - y)
                        regions.append((x, y, region_w, region_h))
                
                for x, y, w, h in regions:
                    for name, processed_img in [("basic_gray", gray), ("threshold_150", thresh1)]:
                        # Extract just this region
                        region = processed_img[y:y+h, x:x+w]
                        
                        # Skip very small regions
                        if w < 50 or h < 30:
                            continue
                            
                        # Extract text from this region
                        region_text = pytesseract.image_to_string(region, config='--psm 6 --oem 1').lower().strip()
                        
                        if not region_text or len(region_text) < 10:
                            continue
                            
                        # Check if target text is in this region
                        if target_lower in region_text:
                            match_info = {
                                "word": region_text,
                                "bounds": (x, y, w, h),
                                "confidence": 80,  # Higher confidence for region match
                                "method": f"{name}_region",
                                "match_score": 90,
                                "is_paragraph": True
                            }
                            all_matches.append(match_info)
                            print(f"Found region match at ({x},{y}): '{region_text[:30]}...'")
        
        # Try different PSM modes for different text layouts
        psm_modes = [
            ('--psm 6', 'Block of text'),  # Assume a single uniform block of text
            ('--psm 7', 'Single line'),    # Treat the image as a single text line
            ('--psm 8', 'Word'),           # Treat the image as a single word
            ('--psm 11', 'Sparse text'),   # Sparse text. Find as much text as possible in no particular order
            ('--psm 12', 'Sparse text with OSD'),  # Sparse text with OSD
            ('--psm 13', 'Raw line')       # Raw line. Treat the image as a single text line
        ]
        
        # Try each preprocessing method with different PSM modes
        for name, processed_img in preprocessed_images:
            for psm_config, psm_desc in psm_modes:
                print(f"Trying OCR with {name} preprocessing, {psm_desc}...")
                
                # Get both standard string and detailed data with positions
                data = pytesseract.image_to_data(processed_img, config=f'{psm_config} --oem 1', output_type=pytesseract.Output.DICT)
                
                # Extract matches from detailed data
                for i, word in enumerate(data["text"]):
                    if not word or len(word.strip()) == 0:
                        continue
                        
                    word_lower = word.lower().strip()
                    confidence = float(data["conf"][i])
                    
                    # Skip very low confidence results but use lower threshold for longer strings
                    min_confidence = 20 if len(word_lower) > 10 else 30
                    if confidence < min_confidence:
                        continue
                    
                    # Various matching strategies
                    exact_match = target_lower == word_lower
                    contains_match = target_lower in word_lower or word_lower in target_lower
                    
                    # Special case for UI elements: Check if first few chars match (for partially detected text)
                    prefix_match = False
                    suffix_match = False
                    if len(target_lower) >= 3 and len(word_lower) >= 2:
                        # Check if first 60% of characters match
                        prefix_len = max(2, int(len(target_lower) * 0.6))
                        if len(word_lower) >= prefix_len:
                            prefix_match = target_lower[:prefix_len] == word_lower[:prefix_len]
                        
                        # Check if last few characters match
                        suffix_len = max(2, int(len(target_lower) * 0.4))
                        if len(word_lower) >= suffix_len:
                            suffix_match = target_lower[-suffix_len:] == word_lower[-suffix_len:]
                    
                    # Fuzzy matching for longer text
                    fuzzy_match = False
                    fuzzy_score = 0
                    if len(word_lower) > 5 and len(target_lower) > 5:
                        import difflib
                        similarity = difflib.SequenceMatcher(None, target_lower, word_lower).ratio()
                        if similarity > 0.7:  # 70% similarity threshold
                            fuzzy_match = True
                            fuzzy_score = int(similarity * 100)
                    
                    # Calculate match score
                    match_score = 0
                    if exact_match:
                        match_score = 100
                    elif contains_match:
                        match_score = 80
                    elif fuzzy_match:
                        match_score = fuzzy_score
                    elif prefix_match:
                        match_score = 60
                    elif suffix_match:
                        match_score = 50
                    
                    # Only consider reasonable matches
                    if match_score > 0:
                        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                        
                        # Skip if region is too small
                        if w < 5 or h < 5:
                            continue
                            
                        match_info = {
                            "word": word,
                            "bounds": (x, y, w, h),
                            "confidence": confidence,
                            "method": f"{name}_{psm_desc}",
                            "match_score": match_score,
                            "is_paragraph": False
                        }
                        all_matches.append(match_info)
                        print(f"Found match '{word}' for '{target}' at ({x}, {y}, {w}, {h}) with confidence {confidence}, score {match_score}")
        
        # Special case: Try direct extraction of common game UI elements
        common_ui_texts = ["play", "single player", "multiplayer", "options", "quit", "settings", 
                          "start", "continue", "exit", "main menu", "save", "load"]
                          
        if target_lower in common_ui_texts:
            print("Target is a common UI element, trying specialized detection...")
            
            # Try detecting high-contrast regions that might contain buttons
            for processed_img in [thresh1, adaptive_thresh, contrast_thresh, white_mask]:
                # Find contours in the processed image
                contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # Skip very small regions
                    if w < 40 or h < 15:
                        continue
                        
                    # Skip very large regions
                    if w > width * 0.3 or h > height * 0.15:
                        continue
                    
                    # Extract the region
                    roi = original_img[y:y+h, x:x+w]
                    
                    # Try different preprocessing on this region
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    _, roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # Enlarge image for better OCR (gaming fonts are often stylized)
                    enlarged_roi = cv2.resize(roi_thresh, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
                    
                    # Apply OCR with gaming-optimized settings
                    text = pytesseract.image_to_string(enlarged_roi, config='--psm 7 --oem 1').lower().strip()
                    
                    # Check if it matches our target
                    if target_lower in text or text in target_lower:
                        match_score = 70  # Good score for UI element detection
                        match_info = {
                            "word": text,
                            "bounds": (x, y, w, h),
                            "confidence": 70,  # Reasonable confidence for UI element
                            "method": "specialized_ui_detection",
                            "match_score": match_score,
                            "is_paragraph": False
                        }
                        all_matches.append(match_info)
                        print(f"Found UI element match '{text}' at ({x}, {y}, {w}, {h})")
        
        # Sort matches by match score and confidence
        all_matches.sort(key=lambda x: (x["match_score"], x["confidence"]), reverse=True)
        
        # Group matches by method to avoid duplicates
        grouped_matches = {}
        for match in all_matches:
            method = match["method"]
            if method not in grouped_matches or match["match_score"] > grouped_matches[method]["match_score"]:
                grouped_matches[method] = match
                
        # Return the best matches
        best_matches = list(grouped_matches.values())
        best_matches.sort(key=lambda x: (x["match_score"], x["confidence"]), reverse=True)
        
        # Limit to top 5 best matches
        return best_matches[:5]
    except Exception as e:
        import traceback
        print(f"Error in find_text_in_image: {str(e)}")
        traceback.print_exc()
        return []

def caption_window(window_title: str = None, window_id: str = None) -> Dict[str, Any]:
    """
    Capture and analyze a specific window's content across platforms.
    
    Args:
        window_title: Title of the window to analyze
        window_id: ID of the window to analyze (platform-specific, optional)
    
    Returns:
        Dict containing analysis results with:
        - caption: Description of the window content
    """
    system = platform.system()
    window_image = None
    window_info = None
    
    print(f"Analyzing window with ID {window_id} and title {window_title}")
    
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
    
    
    try:
        if not window_image or len(window_image) == 0:
            print("Error: Empty window image data")
            return {"error": "Empty window image data"}
            
        # Try the primary BLIP captioning method first
        try:
            # Get window image description
            print("Attempting primary caption method with BLIP...")
            image_analysis = caption_image(window_image)
            
            # Check if we got a valid caption
            caption_text = image_analysis.get("caption", "")
            if caption_text and "error" not in image_analysis and not caption_text.startswith("Failed to"):
                print(f"Successfully generated caption: {caption_text}")
                return {
                    "caption": caption_text,
                }
            else:
                print(f"BLIP captioning failed or returned an error: {image_analysis}")
                # Continue to fallback methods
        except Exception as caption_error:
            print(f"Error in primary caption method: {caption_error}")
            import traceback
            traceback.print_exc()
            # Continue to fallback methods

        # Fallback 1: Try OCR to extract text from the window
        print("Attempting fallback: OCR text extraction...")
        try:
            extracted_text = extract_text_from_window(window_image)
            if extracted_text and len(extracted_text) > 0:
                # Collect top text entries based on confidence
                text_items = [item["text"] for item in extracted_text[:10]]
                top_text = ", ".join(text_items)
                fallback_caption = f"Window containing text: {top_text}"
                print(f"Generated fallback caption from OCR: {fallback_caption}")
                return {
                    "caption": fallback_caption,
                    "note": "Caption generated using OCR fallback method"
                }
        except Exception as ocr_error:
            print(f"OCR fallback failed: {ocr_error}")
            
        # Fallback 2: Basic window information
        print("Using basic window information as final fallback...")
        window_title = target_window.get("title", "Unknown window")
        basic_caption = f"Window titled '{window_title}'"
        
        return {
            "caption": basic_caption,
            "note": "Caption generated using basic window information fallback"
        }
    except Exception as e:
        import traceback
        print(f"Error analyzing window content: {e}")
        traceback.print_exc()
        return {"error": f"Error analyzing window content: {str(e)}"}

def extract_text_from_window(image_data: bytes) -> List[Dict[str, Any]]:
    """
    Extract text from a window image with high-quality OCR optimization for gaming UIs.
    
    Args:
        image_data: Raw image data in bytes
        
    Returns:
        List of dictionaries containing text and position information
    """
    try:
        print("Extracting text from window...")
        
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
        
        # Apply multiple preprocessing techniques for optimal OCR
        preprocessed_images = []
        
        # 1. Original image
        preprocessed_images.append(("original", original_img))
        
        # 2. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        preprocessed_images.append(("gray", gray))
        
        # 3. High contrast enhancement
        contrast_img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
        preprocessed_images.append(("contrast", contrast_img))
        
        # 4. Binary threshold on grayscale
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(("binary", binary))
        
        # 5. Enhanced for white text on dark background (common in games)
        _, white_text = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        preprocessed_images.append(("white_text", white_text))
        
        # Store all text detections
        all_text = []
        
        # Different PSM modes for different text layouts
        psm_modes = [
            "--psm 6",  # Assume a single uniform block of text
            "--psm 11",  # Sparse text
            "--psm 12",  # Sparse text with OSD
            "--psm 3",   # Fully automatic page segmentation (good for paragraphs)
            "--psm 4"    # Assume a single column of text of variable sizes
        ]
        
        # Extract text from each preprocessed image with different PSM modes
        for name, processed_img in preprocessed_images:
            for psm_mode in psm_modes:
                data = pytesseract.image_to_data(processed_img, config=f"{psm_mode} --oem 1", 
                                               output_type=pytesseract.Output.DICT)
                
                for i, text in enumerate(data["text"]):
                    if not text or len(text.strip()) < 4:  # CHANGED: Increased minimum character requirement from 1 to 4
                        continue
                        
                    confidence = float(data["conf"][i])
                    
                    # Lower confidence threshold for longer strings
                    min_confidence = 55 if len(text.strip()) < 10 else 35
                    
                    # Filter low-confidence results
                    if confidence < min_confidence:
                        continue
                        
                    x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                    
                    # Adjust filter for very small regions - lower requirements for longer text
                    min_width = 5 if len(text.strip()) > 10 else 10
                    min_height = 5 if len(text.strip()) > 10 else 10
                    
                    if w < min_width or h < min_height:
                        continue
                        
                    # Check if this text could be a button label
                    is_potential_button = False
                    text_lower = text.lower().strip()
                    
                    # Common button text detection
                    button_keywords = ["play", "start", "options", "quit", "exit", "settings", 
                                      "ok", "cancel", "yes", "no", "continue", "back", "next",
                                      "single player", "multiplayer", "menu", "create"]
                                      
                    # IMPROVED BUTTON TEXT DETECTION: More selective criteria
                    # Only mark as a button if it exactly matches a button keyword
                    # or if it has specific button-like characteristics
                    if text_lower in button_keywords:
                        is_potential_button = True
                    # Check for common button patterns like "OK" or "PLAY"
                    elif (text.isupper() and len(text) <= 8 and confidence > 80):
                        # Only consider uppercase text as buttons if they're high confidence and match known patterns
                        if text_lower in ["ok", "play", "yes", "no", "exit", "back", "next"]:
                            is_potential_button = True
                    # Don't consider text that's too long to be a button
                    elif len(text) > 15:
                        is_potential_button = False
                    # Short capitalized words might be buttons (like "Play", "Start")
                    elif (len(text) <= 12 and text[0].isupper() and confidence > 75 and
                          any(keyword in text_lower for keyword in button_keywords)):
                        is_potential_button = True
                    
                    text_info = {
                        "text": text.strip(),
                        "position": {
                            "x": x,
                            "y": y,
                            "width": w,
                            "height": h,
                            "center_x": x + w // 2,
                            "center_y": y + h // 2
                        },
                        "confidence": confidence,
                        "method": f"{name}_{psm_mode.replace('--psm ', '')}",  # Simplified method name
                        "is_potential_button": is_potential_button
                    }
                    all_text.append(text_info)
                    print(f"Found text '{text}' at ({x}, {y}, {w}, {h}) with confidence {confidence}")
        
        # Additional step: try to detect larger text blocks using paragraph mode
        for name, processed_img in [("original", original_img), ("binary", binary)]:
            # Use PSM 3 (fully automatic page segmentation) to detect paragraphs
            paragraph_text = pytesseract.image_to_string(processed_img, config="--psm 3 --oem 1")
            
            if paragraph_text and len(paragraph_text.strip()) > 15:  # Only consider substantial paragraphs
                # Get rough bounding box for the paragraph (using the entire image is a fallback)
                paragraph_info = {
                    "text": paragraph_text.strip(),
                    "position": {
                        "x": 0,
                        "y": 0,
                        "width": width,
                        "height": height,
                        "center_x": width // 2,
                        "center_y": height // 2
                    },
                    "confidence": 60,  # Moderate confidence for paragraph detection
                    "method": f"{name}_paragraph_mode",
                    "is_potential_button": False
                }
                all_text.append(paragraph_info)
                print(f"Found paragraph: '{paragraph_text[:50]}...' (length: {len(paragraph_text)})")
        
        # Filter duplicates
        filtered_text = []
        for text_item in all_text:
            x1 = text_item["position"]["x"]
            y1 = text_item["position"]["y"]
            w1 = text_item["position"]["width"]
            h1 = text_item["position"]["height"]
            center1 = (x1 + w1//2, y1 + h1//2)
            text1 = text_item["text"].lower()
            
            # Check if this text is too close to existing filtered text
            is_duplicate = False
            for filtered in filtered_text[:]:  # Use a copy to allow modification during iteration
                x2 = filtered["position"]["x"]
                y2 = filtered["position"]["y"]
                w2 = filtered["position"]["width"]
                h2 = filtered["position"]["height"]
                center2 = (x2 + w2//2, y2 + h2//2)
                text2 = filtered["text"].lower()
                
                # Calculate distance between centers
                distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
                
                # Check for text similarity
                if text1 in text2 or text2 in text1:
                    # If one text is contained in the other, it's likely a duplicate or substring
                    if distance < max(w1, h1, w2, h2) * 0.7:  # Increased overlap threshold
                        is_duplicate = True
                        # Keep the longer text with higher confidence
                        if len(text1) > len(text2) or (len(text1) == len(text2) and text_item["confidence"] > filtered["confidence"]):
                            filtered_text.remove(filtered)
                            filtered_text.append(text_item)
                        break
                # If centers are close, consider it a duplicate
                elif distance < max(w1, h1, w2, h2) * 0.5:
                    is_duplicate = True
                    # Keep the detection with higher confidence
                    if text_item["confidence"] > filtered["confidence"]:
                        filtered_text.remove(filtered)
                        filtered_text.append(text_item)
                    break
            
            if not is_duplicate:
                filtered_text.append(text_item)
        
        print(f"Found {len(filtered_text)} unique text elements")
        return filtered_text
    except Exception as e:
        import traceback
        print(f"Error extracting text from window: {str(e)}")
        traceback.print_exc()
        return []

def filter_button_candidates(buttons: List[Dict[str, Any]], extracted_text: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter button candidates to reduce false positives, using extracted text for validation.
    
    Args:
        buttons: List of button candidates from cv_find_all_buttons
        extracted_text: List of text elements from extract_text_from_window
        
    Returns:
        Filtered list of buttons
    """
    try:
        if not buttons:
            return []
            
        print(f"Filtering {len(buttons)} button candidates...")
        
        # Create a mapping of extracted text positions
        text_positions = []
        button_keywords = ["play", "start", "options", "quit", "exit", "settings", 
                          "ok", "cancel", "yes", "no", "continue", "back", "next",
                          "single player", "multiplayer", "menu", "create"]
                          
        for text_item in extracted_text:
            if text_item["is_potential_button"]:
                pos = text_item["position"]
                text_positions.append({
                    "text": text_item["text"],
                    "x": pos["x"],
                    "y": pos["y"],
                    "width": pos["width"],
                    "height": pos["height"],
                    "confidence": text_item.get("confidence", 0)
                })
        
        # Filter buttons
        filtered_buttons = []
        
        for button in buttons:
            # Get button position
            pos = button["position"]
            x, y, w, h = pos["x"], pos["y"], pos["width"], pos["height"]
            button_text = button["text"]
            
            # Skip buttons with unreasonable dimensions
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio > 8.0 or aspect_ratio < 1.0:
                print(f"Skipping button with unreasonable aspect ratio: {aspect_ratio}")
                continue
                
            # CHANGED: Completely reject ALL generic "Button" text without any exceptions
            if button_text == "Button":
                print(f"Skipping generic 'Button' text completely")
                continue
            
            # Additional validation for buttons with specific text
            if button_text.upper() == "PLAY" or button_text.lower() == "play":
                # We need extra validation for "PLAY" text to avoid false positives
                # Check if this text has confirmation from extracted_text
                confirmed_by_text = False
                for text_pos in text_positions:
                    tx = text_pos["x"]
                    ty = text_pos["y"]
                    tw = text_pos["width"]
                    th = text_pos["height"]
                    
                    # Calculate overlap
                    overlap_x = max(0, min(x + w, tx + tw) - max(x, tx))
                    overlap_y = max(0, min(y + h, ty + th) - max(y, ty))
                    
                    if overlap_x > 0 and overlap_y > 0:
                        text_lower = text_pos["text"].lower()
                        if "play" in text_lower and text_pos.get("confidence", 0) > 75:
                            confirmed_by_text = True
                            break
                
                # If we can't confirm this "PLAY" text through other means, be very selective
                if not confirmed_by_text:
                    # Extract the region to check button-like appearance
                    import cv2
                    import numpy as np
                    
                    # Get the image data
                    try:
                        with mss.mss() as sct:
                            monitor = sct.monitors[1]  # Use primary monitor
                            screenshot = sct.grab(monitor)
                            img_np = np.array(screenshot)
                            
                            # Make sure the coordinates are within bounds
                            if (x >= 0 and y >= 0 and 
                                x + w <= img_np.shape[1] and 
                                y + h <= img_np.shape[0]):
                                
                                # Extract the button region
                                roi = img_np[y:y+h, x:x+w]
                                
                                # Convert to grayscale
                                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                                
                                # Check for edges (buttons typically have edges/borders)
                                edges = cv2.Canny(roi_gray, 50, 150)
                                edge_density = np.count_nonzero(edges) / (w * h)
                                
                                # Check color variance (buttons often have distinct colors)
                                color_variance = np.std(roi_gray)
                                
                                # Only keep if strong visual button indicators are present
                                if edge_density < 0.1 or color_variance < 15:
                                    print(f"Rejecting 'PLAY' button without strong visual indicators")
                                    continue
                    except Exception as e:
                        print(f"Error validating PLAY button: {e}")
                        # Be conservative - if we can't verify, reject it
                        continue
            
            # Accept this button
            filtered_buttons.append(button)
        
        print(f"Filtered to {len(filtered_buttons)} valid buttons")
        return filtered_buttons
    except Exception as e:
        import traceback
        print(f"Error filtering button candidates: {str(e)}")
        traceback.print_exc()
        return buttons  # Return original list if filtering fails

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
        # 1. First try: If we have window info with position and size, try direct capture
        if window_info and "position" in window_info and "size" in window_info:
            pos = window_info.get("position", {})
            size = window_info.get("size", {})
            
            # Ensure we have numeric values for position and size
            if (isinstance(pos, dict) and isinstance(size, dict) and
                all(k in pos and isinstance(pos[k], int) for k in ["x", "y"]) and
                all(k in size and isinstance(size[k], int) for k in ["width", "height"])):
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
                        img.save(buf, format='PNG')  # Use PNG for better quality
                        img_data = buf.getvalue()
                        
                        # Validate the image data
                        try:
                            test_img = Image.open(io.BytesIO(img_data))
                            test_img.verify()  # Verify image data is valid
                            print("Successfully captured window using mss with position/size")
                            return img_data
                        except Exception as e:
                            print(f"Invalid image data from mss: {e}")
                except Exception as e:
                    print(f"Error using mss for window capture: {e}")
        
        # 2. Second try: Use xdotool to activate window then take screenshot
        try:
            import subprocess
            
            # Create a temporary file for the screenshot
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                
            # First, activate the window to bring it to front
            if window_id:
                print(f"Activating window with ID {window_id}")
                subprocess.run(['xdotool', 'windowactivate', str(window_id)], capture_output=True, check=False)
                # Short delay to allow window to become active
                time.sleep(0.5)
                
                # Try using xwd to capture the active window
                try:
                    subprocess.run(['xwd', '-id', str(window_id), '-out', f"{temp_path}.xwd"], capture_output=True, check=False)
                    # Convert xwd to PNG
                    subprocess.run(['convert', f"{temp_path}.xwd", temp_path], capture_output=True, check=False)
                    
                    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 100:
                        with open(temp_path, 'rb') as img_file:
                            img_data = img_file.read()
                        # Clean up temp files
                        os.unlink(temp_path)
                        if os.path.exists(f"{temp_path}.xwd"):
                            os.unlink(f"{temp_path}.xwd")
                        print("Successfully captured window using xwd")
                        return img_data
                except Exception as e:
                    print(f"Error with xwd: {e}")
                    
                # If xwd failed, try import
                try:
                    subprocess.run(['import', '-window', str(window_id), temp_path], capture_output=True, check=False)
                    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 100:
                        with open(temp_path, 'rb') as img_file:
                            img_data = img_file.read()
                        os.unlink(temp_path)
                        print("Successfully captured window using import")
                        return img_data
                except Exception as e:
                    print(f"Error with import: {e}")
            
            # If we get here, try using general window capture approach
            if window_title:
                try:
                    # First get window ID using wmctrl if we don't have it
                    if not window_id:
                        wmctrl_output = subprocess.run(['wmctrl', '-l'], capture_output=True, text=True, check=False)
                        if wmctrl_output.returncode == 0:
                            for line in wmctrl_output.stdout.split('\n'):
                                if window_title in line:
                                    parts = line.split(None, 1)
                                    if parts:
                                        window_id = parts[0]
                                        break
                    
                    # If we found window_id, try capturing it
                    if window_id:
                        subprocess.run(['xdotool', 'windowactivate', str(window_id)], capture_output=True, check=False)
                        time.sleep(0.5)
                        subprocess.run(['import', '-window', str(window_id), temp_path], capture_output=True, check=False)
                        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 100:
                            with open(temp_path, 'rb') as img_file:
                                img_data = img_file.read()
                            os.unlink(temp_path)
                            print("Successfully captured window using import with window title")
                            return img_data
                except Exception as e:
                    print(f"Error with window title capture: {e}")
            
            # If we get here, try capturing active window
            try:
                subprocess.run(['import', '-window', 'root', temp_path], capture_output=True, check=False)
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 100:
                    with open(temp_path, 'rb') as img_file:
                        img_data = img_file.read()
                    os.unlink(temp_path)
                    print("Captured active window/root")
                    return img_data
            except Exception as e:
                print(f"Error capturing active window: {e}")
                
            # Clean up temp file if it exists and we haven't used it yet
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception as e:
            import traceback
            print(f"Error with command-line screenshot tools: {e}")
            traceback.print_exc()
        
        # 3. Last resort: Fall back to full screen capture
        print("Falling back to full screen capture")
        try:
            img_data = get_screenshot(1)
            print("Successfully captured full screen as fallback")
            return img_data
        except Exception as e:
            print(f"Error during fallback screenshot: {e}")
            return None
    except Exception as e:
        import traceback
        print(f"Error capturing window on Linux: {e}")
        traceback.print_exc()
        return None

def extract_gaming_ui_text(image_data: bytes) -> List[Dict[str, Any]]:
    """
    Specialized function for extracting text from gaming interfaces with stylized fonts and high contrast.
    
    This function is optimized for text commonly found in gaming UIs including:
    - Stylized fonts with unusual shapes
    - High contrast text (white on dark or dark on light backgrounds)
    - Button labels and menu options
    - Uppercase text that's common in game UI elements
    - Longer text such as instructions, descriptions, and dialogue
    
    Args:
        image_data: Raw image data in bytes
        
    Returns:
        List of dictionaries containing text and position information
    """
    try:
        print("Extracting gaming UI text...")
        
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
        
        # Store all text detections
        all_text = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. SPECIALIZED APPROACH FOR WHITE TEXT ON DARK BACKGROUNDS (common in games)
        # ------------------------------------------------------
        # Create binary image optimized for white text
        _, white_text_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        # Dilate to connect nearby text components
        kernel = np.ones((2,2), np.uint8)
        dilated_white = cv2.dilate(white_text_mask, kernel, iterations=1)
        
        # Find contours around potential text areas
        white_contours, _ = cv2.findContours(dilated_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract text from white regions
        for cnt in white_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Filter out regions that are too small, but use more permissive thresholds
            if w < 10 or h < 8:
                continue
                
            # Also filter out regions that are too large, but allow for larger text blocks
            # Allow up to 80% of width for things like paragraph text in dialogue boxes
            if w > width * 0.8 or h > height * 0.4:
                continue
                
            # Extract ROI
            roi = white_text_mask[y:y+h, x:x+w]
            
            # Calculate white pixel density (white text should have reasonable density)
            white_pixel_count = np.sum(roi > 0)
            white_pixel_density = white_pixel_count / (w * h)
            
            # Skip regions with too few or too many white pixels, adjusted for larger text areas
            min_density = 0.05 if w > 100 else 0.1  # Lower density threshold for larger areas
            max_density = 0.95 if w > 100 else 0.9  # Higher density threshold for larger areas
            
            if white_pixel_density < min_density or white_pixel_density > max_density:
                continue
                
            # Calculate edges in the region to check for text-like patterns
            edges = cv2.Canny(roi, 100, 200)
            edge_pixel_count = np.sum(edges > 0)
            edge_density = edge_pixel_count / (w * h)
            
            # Skip regions without enough edges (text has edges), with adjustment for size
            min_edge_density = 0.03 if w > 100 else 0.05  # Lower edge density requirement for larger blocks
            
            if edge_density < min_edge_density:
                continue
                
            # Resize for better OCR
            enlarged_roi = cv2.resize(roi, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
            
            # Choose PSM mode based on region size
            if w > 150:  # Likely a paragraph or multi-line text
                ocr_config = '--psm 3 --oem 1'  # Automatic page segmentation
            else:
                ocr_config = '--psm 7 --oem 1'  # Single line of text
                
            text = pytesseract.image_to_string(enlarged_roi, config=ocr_config).strip()
            
            # Skip empty results, but allow shorter text for buttons
            if not text:
                continue
                
            # Different threshold based on region size
            if len(text) < 4 and w < 50:  # CHANGED: Increased from 2 to 4 chars minimum for small regions
                continue
                
            # Calculate if this is likely a button label
            is_button = False
            text_lower = text.lower()
            button_keywords = ["play", "start", "options", "quit", "exit", "settings", 
                             "ok", "cancel", "yes", "no", "continue", "back", "next", 
                             "single player", "multiplayer", "menu", "create"]
                             
            if (text_lower in button_keywords or 
                any(keyword in text_lower for keyword in button_keywords) or
                text.isupper() or
                (len(text) <= 12 and text[0].isupper())):
                is_button = True
            
            # Add to results
            text_info = {
                "text": text,
                "position": {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "center_x": x + w // 2,
                    "center_y": y + h // 2
                },
                "confidence": 85,  # Higher confidence for white text in games
                "method": "white_text_extraction",
                "is_potential_button": is_button
            }
            all_text.append(text_info)
            print(f"Found white text '{text}' at ({x}, {y}, {w}, {h})")
        
        # 2. CONTRAST ENHANCEMENT APPROACH FOR HARD-TO-READ TEXT
        # ------------------------------------------------------
        # Create high contrast version of the image
        contrast_enhanced = cv2.convertScaleAbs(img, alpha=2.0, beta=0)
        contrast_gray = cv2.cvtColor(contrast_enhanced, cv2.COLOR_BGR2GRAY)
        
        # Create an adaptive threshold
        contrast_binary = cv2.adaptiveThreshold(
            contrast_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
        # Try both sparse text and paragraph modes for contrast-enhanced image
        psm_modes = [
            ('--psm 11 --oem 1', 'sparse_text'),  # Sparse text detection
            ('--psm 3 --oem 1', 'paragraph'),     # Full page text detection (for longer texts)
            ('--psm 4 --oem 1', 'column_text')    # Single column detection (for dialogue boxes)
        ]
        
        for ocr_config, mode_name in psm_modes:
            if mode_name == 'paragraph' or mode_name == 'column_text':
                # For paragraph mode, extract full string first
                full_text = pytesseract.image_to_string(contrast_binary, config=ocr_config).strip()
                
                if full_text and len(full_text) > 15:  # Only consider substantial paragraphs
                    # Create a rough paragraph entry for the full text
                    paragraph_info = {
                        "text": full_text,
                        "position": {
                            "x": 0,
                            "y": 0,
                            "width": width,
                            "height": height,
                            "center_x": width // 2,
                            "center_y": height // 2
                        },
                        "confidence": 75,  # Decent confidence for paragraph in gaming UIs
                        "method": f"contrast_{mode_name}",
                        "is_potential_button": False
                    }
                    all_text.append(paragraph_info)
                    print(f"Found paragraph text: '{full_text[:50]}...' (length: {len(full_text)})")
            else:
                # For sparse text mode, use image_to_data to get detailed info
                contrast_data = pytesseract.image_to_data(
                    contrast_binary, config=ocr_config, output_type=pytesseract.Output.DICT)
                    
                for i, text in enumerate(contrast_data["text"]):
                    if not text:
                        continue
                        
                    # Different minimum length based on whether it might be a button/UI element
                    if len(text.strip()) < 4:  # CHANGED: Increased from 1 to 4 chars minimum
                        continue
                        
                    confidence = float(contrast_data["conf"][i])
                    # Lower confidence threshold for longer texts
                    min_confidence = 60 if len(text.strip()) < 10 else 40
                    
                    if confidence < min_confidence:
                        continue
                        
                    x, y, w, h = (contrast_data["left"][i], contrast_data["top"][i],
                                contrast_data["width"][i], contrast_data["height"][i])
                                
                    # Skip very small regions with more permissive thresholds for longer text
                    min_width = 5 if len(text.strip()) > 10 else 10
                    min_height = 5 if len(text.strip()) > 10 else 10
                    
                    if w < min_width or h < min_height:
                        continue
                        
                    # Check if this is likely a button/menu label
                    is_button = False
                    text_lower = text.lower()
                    if (text_lower in button_keywords or 
                        any(keyword in text_lower for keyword in button_keywords) or
                        text.isupper() or
                        (len(text) <= 12 and text[0].isupper())):
                        is_button = True
                        
                    text_info = {
                        "text": text.strip(),
                        "position": {
                            "x": x,
                            "y": y,
                            "width": w,
                            "height": h,
                            "center_x": x + w // 2,
                            "center_y": y + h // 2
                        },
                        "confidence": confidence,
                        "method": f"contrast_{mode_name}",
                        "is_potential_button": is_button
                    }
                    all_text.append(text_info)
                    print(f"Found contrast-enhanced text '{text}' at ({x}, {y}, {w}, {h}) with confidence {confidence}")
        
        # 3. ADVANCED TEXT BLOCK DETECTION (for longer text)
        # ------------------------------------------------------
        # Create MSER detector for text region detection (good for paragraphs)
        mser = cv2.MSER_create()
        
        # Convert to grayscale if not already
        if len(img.shape) > 2:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img.copy()
            
        # Detect regions
        regions, _ = mser.detectRegions(gray_img)
        
        # Filter and merge regions to detect text blocks
        if regions:
            # Convert regions to bounding boxes
            boxes = []
            for region in regions:
                x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
                # Filter very small regions
                if w > 10 and h > 10:
                    boxes.append((x, y, w, h))
                    
            # Merge overlapping boxes to form text blocks
            merged_boxes = []
            while boxes:
                current_box = boxes.pop(0)
                
                # Check if current box overlaps with any merged box
                overlaps = False
                for i, merged_box in enumerate(merged_boxes):
                    if _boxes_overlap(current_box, merged_box):
                        # Merge the boxes
                        merged_boxes[i] = _merge_boxes(current_box, merged_box)
                        overlaps = True
                        break
                        
                if not overlaps:
                    merged_boxes.append(current_box)
            
            # Process each merged box as a potential text block
            for x, y, w, h in merged_boxes:
                # Skip boxes that are too small or too large
                if w < 30 or h < 15 or w > width * 0.9 or h > height * 0.5:
                    continue
                    
                # Extract the region
                roi = gray_img[y:y+h, x:x+w]
                
                # Apply OCR with settings suitable for text blocks
                ocr_config = '--psm 3 --oem 1'  # Automatic page segmentation
                text = pytesseract.image_to_string(roi, config=ocr_config).strip()
                
                # Skip empty or very short results
                if not text or len(text) < 10:  # Text blocks should have reasonable length
                    continue
                    
                # Add to results
                text_info = {
                    "text": text,
                    "position": {
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "center_x": x + w // 2,
                        "center_y": y + h // 2
                    },
                    "confidence": 70,  # Moderate confidence for MSER text block detection
                    "method": "mser_text_block",
                    "is_potential_button": False
                }
                all_text.append(text_info)
                print(f"Found text block: '{text[:50]}...' (length: {len(text)})")
        
        # Filter out duplicates and keep better quality detections
        filtered_text = []
        for text_item in all_text:
            x1 = text_item["position"]["x"]
            y1 = text_item["position"]["y"]
            w1 = text_item["position"]["width"]
            h1 = text_item["position"]["height"]
            center1 = (x1 + w1//2, y1 + h1//2)
            text1 = text_item["text"].lower()
            
            is_duplicate = False
            for filtered in filtered_text[:]:  # Use a copy to allow modification during iteration
                x2 = filtered["position"]["x"]
                y2 = filtered["position"]["y"]
                w2 = filtered["position"]["width"]
                h2 = filtered["position"]["height"]
                center2 = (x2 + w2//2, y2 + h2//2)
                text2 = filtered["text"].lower()
                
                # Check for text similarity
                if text1 in text2 or text2 in text1:
                    # If one text is contained in the other, it's likely a duplicate or substring
                    if distance_between_boxes((x1, y1, w1, h1), (x2, y2, w2, h2)) < max(w1, h1, w2, h2) * 0.7:
                        is_duplicate = True
                        # Keep the longer text with higher confidence
                        if len(text1) > len(text2) or (len(text1) == len(text2) and text_item["confidence"] > filtered["confidence"]):
                            filtered_text.remove(filtered)
                            filtered_text.append(text_item)
                        break
                # If boxes significantly overlap, consider it a duplicate
                elif _boxes_overlap((x1, y1, w1, h1), (x2, y2, w2, h2)):
                    is_duplicate = True
                    # Keep the detection with higher confidence or longer text
                    if text_item["confidence"] > filtered["confidence"] or len(text1) > len(text2):
                        filtered_text.remove(filtered)
                        filtered_text.append(text_item)
                    break
            
            if not is_duplicate:
                filtered_text.append(text_item)
                
        print(f"Found {len(filtered_text)} unique text elements in gaming UI")
        return filtered_text
    except Exception as e:
        import traceback
        print(f"Error extracting gaming UI text: {str(e)}")
        traceback.print_exc()
        return []

def _boxes_overlap(box1, box2):
    """Check if two boxes overlap"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate the coordinates of the corners
    left1, right1 = x1, x1 + w1
    top1, bottom1 = y1, y1 + h1
    left2, right2 = x2, x2 + w2
    top2, bottom2 = y2, y2 + h2
    
    # Check if boxes overlap
    return not (right1 < left2 or left1 > right2 or bottom1 < top2 or top1 > bottom2)

def _merge_boxes(box1, box2):
    """Merge two overlapping boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate the coordinates of the corners
    left1, right1 = x1, x1 + w1
    top1, bottom1 = y1, y1 + h1
    left2, right2 = x2, x2 + w2
    top2, bottom2 = y2, y2 + h2
    
    # Calculate the coordinates of the merged box
    left = min(left1, left2)
    top = min(top1, top2)
    right = max(right1, right2)
    bottom = max(bottom1, bottom2)
    
    # Convert back to x, y, w, h format
    x = left
    y = top
    w = right - left
    h = bottom - top
    
    return (x, y, w, h)

def distance_between_boxes(box1, box2):
    """Calculate distance between centers of two boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    center1 = (x1 + w1//2, y1 + h1//2)
    center2 = (x2 + w2//2, y2 + h2//2)
    
    return ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5

def analyze_window(window_title: str = None, window_id: str = None) -> Dict[str, Any]:
    """
    Capture a window and analyze it to extract UI elements like text and buttons.
    
    Args:
        window_title: Title of the window to analyze
        window_id: ID of the window to analyze (platform-specific, optional)
    
    Returns:
        Dict containing analysis results with:
        - caption: Description of the window content
        - text_elements: List of text elements with positions
        - buttons: List of button-like elements with positions
        - window_info: Information about the captured window
    """
    system = platform.system()
    window_image = None
    window_info = None
    
    print(f"Analyzing window with ID {window_id} and title {window_title}")
    
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
    
    try:
        if not window_image or len(window_image) == 0:
            print("Error: Empty window image data")
            return {"error": "Empty window image data"}

        # Initialize result components with default values
        caption = ""
        text_elements = []
        buttons = []
        checkboxes = []
        ui_regions = []
            
        # Get window image description - in a try-except block for each step
        try:
            image_analysis = caption_image(window_image)
            caption = image_analysis.get("caption", "")
            print(f"Generated caption: {caption[:50]}...")
        except Exception as e:
            print(f"Error generating caption: {e}")
            import traceback
            traceback.print_exc()
            caption = "Unable to generate caption"
        
        # Extract text elements from the window image
        try:
            text_elements = extract_text_from_window(window_image)
            print(f"Found {len(text_elements)} text elements")
        except Exception as e:
            print(f"Error extracting text: {e}")
            import traceback
            traceback.print_exc()
        
        # Find button-like elements
        try:
            buttons = cv_find_all_buttons(window_image)
            print(f"Found {len(buttons)} buttons")
        except Exception as e:
            print(f"Error finding buttons: {e}")
            import traceback
            traceback.print_exc()
        
        # Find checkbox elements
        try:
            checkboxes = cv_find_checkboxes(window_image)
            print(f"Found {len(checkboxes)} checkboxes")
        except Exception as e:
            print(f"Error finding checkboxes: {e}")
            import traceback
            traceback.print_exc()
        
        # Extract image regions that might contain important content
        try:
            ui_regions = cv_detect_and_analyze_regions(window_image)
            # Remove binary image_data from regions to avoid serialization issues
            for region in ui_regions:
                if "image_data" in region:
                    # Remove binary image data that can't be JSON serialized
                    del region["image_data"]
            print(f"Found {len(ui_regions)} UI regions")
        except Exception as e:
            print(f"Error detecting UI regions: {e}")
            import traceback
            traceback.print_exc()
        
        # Create and sanitize window_info to ensure it's serializable
        safe_window_info = {}
        try:
            for key, value in target_window.items():
                # Only include serializable types
                if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                    safe_window_info[key] = value
        except Exception as e:
            print(f"Error sanitizing window info: {e}")
            safe_window_info = {"id": window_id, "title": window_title}
        
        # Create result dictionary with safe values
        result = {
            "caption": caption,
            "text_elements": text_elements,
            "buttons": buttons,
            "checkboxes": checkboxes,
            "ui_regions": ui_regions,
            "window_info": safe_window_info
        }
        
        # Final check - remove any binary data that might cause serialization issues
        def remove_binary(obj):
            if isinstance(obj, dict):
                for key in list(obj.keys()):
                    if isinstance(obj[key], bytes):
                        del obj[key]
                    elif isinstance(obj[key], (dict, list)):
                        remove_binary(obj[key])
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        remove_binary(item)
        
        remove_binary(result)
        print("Successfully created analysis result")
        return result
    except Exception as e:
        import traceback
        print(f"Error analyzing window content: {e}")
        traceback.print_exc()
        return {"error": f"Error analyzing window content: {str(e)}"}

def get_screenshot_at_mouse(width: int = 300, height: int = 300) -> bytes:
    """
    Capture a screenshot of a rectangular area extending down and to the right from the current mouse position.
    
    Args:
        width: Width of the area to capture in pixels
        height: Height of the area to capture in pixels
        
    Returns:
        The screenshot image as bytes
    """
    try:
        # Get current mouse position
        mouse_x, mouse_y = pyautogui.position()
        print(f"Mouse position: {mouse_x}, {mouse_y}")
        
        # Define the region to capture (extending down and right from mouse position)
        with mss.mss() as sct:
            # Get primary monitor to ensure we stay within its bounds
            monitors_info = get_available_monitors()
            primary_monitor_id = monitors_info.get("primary", 1)
            primary_monitor = sct.monitors[primary_monitor_id]
            
            # Calculate capture region, ensuring it stays within monitor bounds
            region = {
                "left": mouse_x,
                "top": mouse_y,
                "width": min(width, primary_monitor["width"] - mouse_x),
                "height": min(height, primary_monitor["height"] - mouse_y)
            }
            
            # Capture the screenshot
            screenshot = sct.grab(region)
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            
            # Convert to bytes
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return buf.getvalue()
    except Exception as e:
        import traceback
        print(f"Error capturing screenshot at mouse: {e}")
        traceback.print_exc()
        return None

def extract_dropdown_options(image_data: bytes) -> List[Dict[str, Any]]:
    """
    Extract text options from a dropdown menu screenshot.
    
    This function is optimized for extracting menu items from dropdown/context menus
    with their coordinates, allowing for interaction with individual menu options.
    
    Args:
        image_data: Raw image data in bytes (from get_screenshot_at_mouse)
        
    Returns:
        List of dictionaries containing menu options with text and absolute coordinates
    """
    try:
        print("Extracting dropdown menu lines...")
        
        if not image_data:
            print("Error: No image data provided")
            return []
            
        # Convert bytes to OpenCV image
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            print("Error: Failed to decode image data")
            return []
        
        # Get current mouse position for absolute coordinates calculation
        mouse_x, mouse_y = pyautogui.position()
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Use pytesseract to detect text with positioning information
        ocr_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        
        menu_options = []
        for i, text in enumerate(ocr_data["text"]):
            text = text.strip()
            if not text:  # Skip empty results
                continue
                
            conf = int(ocr_data["conf"][i])
            if conf < 40:  # Skip very low confidence results
                continue
                
            # Get relative coordinates from OCR
            x_rel = ocr_data["left"][i]
            y_rel = ocr_data["top"][i]
            w = ocr_data["width"][i]
            h = ocr_data["height"][i]
            
            # Calculate absolute screen coordinates
            x_abs = mouse_x + x_rel
            y_abs = mouse_y + y_rel
            
            menu_option = {
                "text": text,
                "absolute_position": {
                    "x": x_abs,
                    "y": y_abs,
                    "width": w,
                    "height": h,
                    "center_x": x_abs + w // 2,
                    "center_y": y_abs + h // 2
                },
                "relative_position": {
                    "x": x_rel,
                    "y": y_rel,
                    "width": w,
                    "height": h
                },
                "confidence": conf
            }
            menu_options.append(menu_option)
            print(f"Found menu line: '{text}' at abs({x_abs}, {y_abs})")
        
        # Sort menu options by vertical position (top to bottom)
        menu_options.sort(key=lambda opt: opt["absolute_position"]["y"])
        
        return menu_options
    except Exception as e:
        import traceback
        print(f"Error extracting dropdown options: {str(e)}")
        traceback.print_exc()
        return []
 