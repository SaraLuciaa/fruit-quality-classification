import os
import random
import cv2
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================================================
#                   CONFIGURATION SETTINGS
# ==========================================================
# Choose between:
# 1. "rembg": Fast, CPU-friendly background removal.
# 2. "sam": Meta's Segment Anything. Removes hands perfectly using center-point prompts.
#           Requires 'sam_vit_b_01ec64.pth' and 'torch' / 'segment_anything' packages.
SEGMENTATION_ENGINE = "rembg"  

SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"
TARGET_SIZE = 224

print(os.cpu_count())
# Set maximum parallel processes (default: half of your CPU threads to prevent thrashing)
MAX_WORKERS = max(1, (os.cpu_count() or 2) // 2)

# ==========================================================
#                 GLOBAL WORKER INITIALIZER
# ==========================================================
_predictor = None

def init_worker(engine, sam_checkpoint, sam_model_type):
    """
    Initializes the segmentation model once per worker process to optimize memory and speed.
    """
    global _predictor
    if engine == "sam":
        try:
            import torch
            from segment_anything import sam_model_registry, SamPredictor
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[Worker] Loading SAM checkpoint '{sam_checkpoint}' on {device}...")
            
            sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)
            _predictor = SamPredictor(sam)
            print("[Worker] SAM model loaded successfully.")
        except Exception as e:
            print(f"[Worker Error] Failed to load SAM: {e}")
            print("Verify that 'torch' and 'segment_anything' are installed, and the checkpoint file exists.")

# ==========================================================
#                IMAGE PROCESSING PIPELINE
# ==========================================================
def aspect_ratio_preserved_resize(image, target_size=224):
    """
    Resizes an image preserving its aspect ratio and pads the remaining space 
    with black pixels (letterboxing) to match target_size x target_size.
    """
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return None
        
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize keeping the aspect ratio
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create the black canvas
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # Calculate centering offsets
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def generate_augmentations(image):
    """
    Generates geometric augmentations for training samples.
    """
    augmented_list = []
    
    # Horizontal Flip
    h_flip = cv2.flip(image, 1)
    augmented_list.append(("hflip", h_flip))
    
    # Vertical Flip
    v_flip = cv2.flip(image, 0)
    augmented_list.append(("vflip", v_flip))
    
    # 90 Degree Rotation
    rot_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    augmented_list.append(("rot90", rot_90))
    
    # 180 Degree Rotation
    rot_180 = cv2.rotate(image, cv2.ROTATE_180)
    augmented_list.append(("rot180", rot_180))
    
    return augmented_list

def process_and_segment_fruit_rembg(image_path, target_size=224):
    """
    Removes background using rembg, filters noise via geometry thresholds,
    and returns segmented fruit images.
    """
    from rembg import remove
    try:
        img_pil = Image.open(image_path)
        img_no_bg = remove(img_pil)
        img_np = np.array(img_no_bg)
    except Exception:
        return []

    if img_np.shape[2] == 4:
        r, g, b, a = cv2.split(img_np)
        mask = a
        img_bgr = cv2.merge([b, g, r])
        clean_image = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    else:
        clean_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        mask = cv2.cvtColor(clean_image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    valid_segments = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Geometry constraints to discard border lines and shadows
            if w < 40 or h < 40:
                continue
                
            aspect_ratio = float(w) / h
            if aspect_ratio > 3.5 or aspect_ratio < 0.28:
                continue
                
            extent = float(area) / (w * h)
            if extent < 0.25:
                continue
                
            cropped_fruit = clean_image[y:y+h, x:x+w]
            
            # Non-black pixels ratio constraint
            gray_crop = cv2.cvtColor(cropped_fruit, cv2.COLOR_BGR2GRAY)
            active_pixels = np.sum(gray_crop > 15)
            useful_ratio = active_pixels / gray_crop.size
            if useful_ratio < 0.20:
                continue
                
            valid_segments.append({
                'area': area,
                'img': cropped_fruit
            })
            
    processed_outputs = []
    if len(valid_segments) > 0:
        # Process primary object
        primary = valid_segments[0]['img']
        resized_primary = aspect_ratio_preserved_resize(primary, target_size)
        if resized_primary is not None:
            processed_outputs.append(("crop_0", resized_primary))
        
        # Process secondary objects if they meet relative area ratios
        primary_area = valid_segments[0]['area']
        counter = 1
        for idx, candidate in enumerate(valid_segments[1:]):
            if candidate['area'] >= 0.30 * primary_area:
                resized_secondary = aspect_ratio_preserved_resize(candidate['img'], target_size)
                if resized_secondary is not None:
                    processed_outputs.append((f"crop_{counter}", resized_secondary))
                    counter += 1
    else:
        # Fallback to the largest raw contour if strict filters fail
        if len(contours) > 0:
            x, y, w, h = cv2.boundingRect(contours[0])
            fallback_crop = clean_image[y:y+h, x:x+w]
            resized_fallback = aspect_ratio_preserved_resize(fallback_crop, target_size)
            if resized_fallback is not None:
                processed_outputs.append(("crop_fallback", resized_fallback))
                
    return processed_outputs

def process_and_segment_fruit_sam(image_path, predictor, target_size=224):
    """
    Isolates the fruit using Segment Anything (SAM) with a center point hint.
    By targeting the geometric center, it cleanly ignores hands holding the fruit.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return []
            
        h_orig, w_orig, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Feed image to SAM
        predictor.set_image(img_rgb)
        
        # Center point prompt (assuming the fruit is in the center)
        center_point = np.array([[w_orig // 2, h_orig // 2]])
        point_label = np.array([1]) # Include this point
        
        masks, scores, _ = predictor.predict(
            point_coords=center_point,
            point_labels=point_label,
            multimask_output=False
        )
        
        mask = masks[0].astype(np.uint8) * 255
        
        # Check if the segmented area is not empty
        if np.sum(mask > 0) < 1000:
            return []
            
        # Isolate foreground
        clean_image = cv2.bitwise_and(img, img, mask=mask)
        
        # Crop to bounding box
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
        else:
            return []
            
        cropped_fruit = clean_image[y:y+h, x:x+w]
        
        # Resize with aspect ratio preservation
        resized = aspect_ratio_preserved_resize(cropped_fruit, target_size)
        if resized is not None:
            return [("crop_0", resized)]
    except Exception as e:
        print(f"Error in SAM segmenting {image_path}: {e}")
        
    return []

# ==========================================================
#                 PARALLEL TASK WORKER
# ==========================================================
def process_single_task(args):
    """
    Worker function executed by each parallel process.
    """
    path, target_folder, split_name, base_name, engine, target_size, augs_to_generate = args
    
    # Check if this file was already processed to skip it (Cache validation)
    expected_primary_file = os.path.join(target_folder, f"{base_name}_crop_0.jpg")
    expected_fallback_file = os.path.join(target_folder, f"{base_name}_crop_fallback.jpg")
    if os.path.exists(expected_primary_file) or os.path.exists(expected_fallback_file):
        return "skipped"
        
    # Segment using selected engine
    if engine == "sam":
        global _predictor
        if _predictor is None:
            return "error_no_predictor"
        segments = process_and_segment_fruit_sam(path, _predictor, target_size)
    else:
        segments = process_and_segment_fruit_rembg(path, target_size)
        
    if not segments:
        return "no_segments"
        
    # Save results
    for suffix, img_data in segments:
        filename = f"{base_name}_{suffix}.jpg"
        out_path = os.path.join(target_folder, filename)
        cv2.imwrite(out_path, img_data)
        
        # If train split, also generate geometric augmentations to balance the class
        if split_name == "train" and augs_to_generate > 0:
            augmentations = generate_augmentations(img_data)
            for aug_suffix, aug_img in augmentations[:augs_to_generate]:
                aug_filename = f"{base_name}_{suffix}_{aug_suffix}.jpg"
                aug_out_path = os.path.join(target_folder, aug_filename)
                cv2.imwrite(aug_out_path, aug_img)
                
    return "success"


# ==========================================================
#               MAIN PIPELINE MANAGER
# ==========================================================
def build_split_dataset(input_dir="dataset", output_dir="dataset_processed", split_ratios=(0.70, 0.15, 0.15)):
    """
    Traverses dataset structure, aggregates files dynamically, handles stratified splitting,
    runs the background removal, letterboxing, and training augmentation pipeline.
    Also removes any orphan processed images in output_dir that are no longer in input_dir.
    """
    random.seed(42) # Set seed for reproducibility
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    ignored_folders = ['.venv', '.git', 'notebooks', 'experiments', 'tests', output_dir]
    
    # 1. Fallback check for SAM checkpoint
    engine_to_use = SEGMENTATION_ENGINE
    if SEGMENTATION_ENGINE == "sam" and not os.path.exists(SAM_CHECKPOINT):
        print(f"[-] Warning: SAM checkpoint '{SAM_CHECKPOINT}' not found.")
        print("    Please place the file in the project folder to use SAM.")
        print("    Falling back to 'rembg' engine...")
        engine_to_use = "rembg"
        
    # 2. Clean up orphan files in output_dir that are no longer present in input_dir
    existing_sources = set()
    print("Scanning source dataset for cleanup reference...")
    for root, dirs, files in os.walk(input_dir):
        parts = os.path.relpath(root, input_dir).split(os.sep)
        if any(ignored in parts for ignored in ignored_folders) or parts[0] == '.':
            continue
        rel_class_path = os.path.relpath(root, input_dir)
        for file in files:
            if file.lower().endswith(valid_extensions):
                base = os.path.splitext(file)[0]
                existing_sources.add((rel_class_path, base))
                
    if os.path.exists(output_dir):
        print("Cleaning up orphan processed images in destination...")
        deleted_count = 0
        for root, dirs, files in os.walk(output_dir):
            rel_to_output = os.path.relpath(root, output_dir)
            parts = rel_to_output.split(os.sep)
            if len(parts) < 2:
                continue
            
            # parts[0] is train/val/test, the rest is the relative class path
            rel_class_path = os.path.join(*parts[1:])
            
            for file in files:
                if "_crop_" in file:
                    base_name = file.split("_crop_")[0]
                    if (rel_class_path, base_name) not in existing_sources:
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                            deleted_count += 1
                        except Exception as e:
                            print(f"Error deleting orphan file {file_path}: {e}")
                            
        if deleted_count > 0:
            print(f"[+] Cleaned up {deleted_count} orphan processed files from '{output_dir}'.")
        else:
            print("[+] No orphan files found to clean up.")
            
    # 3. Gather dataset files (for all fruits, no banana filter)
    category_map = {}
    print("Gathering dataset files...")
    for root, dirs, files in os.walk(input_dir):
        parts = os.path.relpath(root, input_dir).split(os.sep)
        if any(ignored in parts for ignored in ignored_folders) or parts[0] == '.':
            continue
            
        # Filter files
        image_files = [f for f in files if f.lower().endswith(valid_extensions)]
        if not image_files:
            continue
            
        # Keep track of relative class paths (e.g., 'Bad Quality_Fruits/Apple_Bad')
        relative_class_path = os.path.relpath(root, input_dir)
        if relative_class_path not in category_map:
            category_map[relative_class_path] = []
            
        for file in image_files:
            category_map[relative_class_path].append(os.path.join(root, file))
            
    # 4. Calculate class sizes and determine majority class size in the train split
    train_ratio, val_ratio, test_ratio = split_ratios
    train_sizes = {}
    for class_path, file_list in category_map.items():
        train_sizes[class_path] = int(len(file_list) * train_ratio)
    
    max_train_size = max(train_sizes.values()) if train_sizes else 0
    print(f"[*] Target train size (majority class raw size): {max_train_size}")
    
    # 5. Prepare task list across splits to process them in parallel with adaptive balancing
    tasks = []
    for class_path, file_list in category_map.items():
        random.shuffle(file_list)
        total_files = len(file_list)
        train_end = int(total_files * train_ratio)
        val_end = train_end + int(total_files * val_ratio)
        
        splits = {
            "train": file_list[:train_end],
            "val": file_list[train_end:val_end],
            "test": file_list[val_end:]
        }
        
        for split_name, paths in splits.items():
            target_folder = os.path.join(output_dir, split_name, class_path)
            os.makedirs(target_folder, exist_ok=True)
            
            # Calculate total augmentations needed to balance this class in the train split
            if split_name == "train" and len(paths) < max_train_size:
                num_augs_needed = max_train_size - len(paths)
            else:
                num_augs_needed = 0
                
            for path_idx, path in enumerate(paths):
                base_name = os.path.splitext(os.path.basename(path))[0]
                
                # Calculate augmentations for this specific image task
                if split_name == "train" and num_augs_needed > 0:
                    base_augs = num_augs_needed // len(paths)
                    extra_aug = 1 if (path_idx < (num_augs_needed % len(paths))) else 0
                    augs_to_generate = min(4, base_augs + extra_aug)
                else:
                    augs_to_generate = 0
                    
                tasks.append((path, target_folder, split_name, base_name, engine_to_use, TARGET_SIZE, augs_to_generate))

    # 5. Process tasks in parallel
    print(f"[*] Starting parallel execution. Tasks count: {len(tasks)}")
    print(f"[*] Engine: {engine_to_use.upper()} | Workers: {MAX_WORKERS}")
    
    success_count = 0
    skipped_count = 0
    error_count = 0
    
    # Initialize the worker processes with the respective segmentation models
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker, initargs=(engine_to_use, SAM_CHECKPOINT, SAM_MODEL_TYPE)) as executor:
        futures = {executor.submit(process_single_task, task): task for task in tasks}
        
        for idx, future in enumerate(as_completed(futures)):
            result = future.result()
            if result == "success":
                success_count += 1
            elif result == "skipped":
                skipped_count += 1
            else:
                error_count += 1
                
            # Log progress
            if (idx + 1) % 50 == 0 or (idx + 1) == len(tasks):
                print(f"    -> Progress: {idx + 1}/{len(tasks)} processed | Success: {success_count} | Skipped (Cached): {skipped_count} | Failures: {error_count}")
                
    print(f"\n✨ Data preparation completed successfully!")
    print(f"   - Total images handled: {len(tasks)}")
    print(f"   - Newly processed: {success_count}")
    print(f"   - Skipped (already exists): {skipped_count}")
    print(f"   - Failed: {error_count}")
    print(f"   - Output location: '{output_dir}'")

def preprocess_for_cnn(image, target_size=224):
    """
    Resizes an image and normalizes it for CNN.
    """
    # If image is RGB, resize and normalize
    resized = cv2.resize(image, (target_size, target_size))
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def segment_fruit(image):
    """
    Isolates the fruit from its background using morphological filters 
    and Otsu binarization on the Saturation channel of the HSV color space.
    """
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Otsu binarization on Saturation channel
    _, thresh = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Segmented image
    segmented = cv2.bitwise_and(image, image, mask=mask)
    return mask, segmented


if __name__ == "__main__":
    build_split_dataset(input_dir="dataset", output_dir="dataset_processed")
