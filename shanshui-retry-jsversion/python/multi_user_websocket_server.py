import asyncio
import concurrent
import websockets
import json
import threading
import time
import os
import cv2
import numpy as np
from PIL import Image
from moviepy.editor import ImageSequenceClip
import torch
import random
import uuid
from image_processing import resize, merge_images, draw_tangent_and_fill
from frame_interpolation import generate_interpolated_frames
from pipeline import prompt_embeds, pooled_prompt_embeds, pipe_reduced, aug_embs, dtype

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
WIDTH = 1047
HEIGHT = 1544
CIRCLE_RADIUS = 65
SPEED_REDUCE = 0.2
CROP_SIZE = 512
DILATE_KERNEL_SIZE = 20
CIRCLE_NUM = 20
GAUSSIAN_KERNEL_SIZE = 41
BLUR_RADIUS = (GAUSSIAN_KERNEL_SIZE + DILATE_KERNEL_SIZE) // 2

prompts_list = [
    "Abandoned Chernobyl amusement park with a rusted Ferris wheel against a radioactive sky.",
    "Exxon Valdez oil spill covering the waters of Prince William Sound, with distressed wildlife.",
    "Dense haze from Southeast Asian forest fires, with obscured sun and masked city residents.",
    "European city under a scorching sun during the 2003 heat wave, streets deserted and hot.",
    "Ruins of the Fukushima nuclear plant post-tsunami, under a cloudy sky with radioactive symbols.",
    "California forest engulfed in flames at sunset, with firefighters battling the intense wildfire.",
    "Stark contrast of lush Amazon rainforest and adjacent deforested barren land with stumps.",
    "Polar bear on a melting ice fragment in the Arctic, surrounded by water and distant icebergs.",
    "Australian bushfires scene with fleeing kangaroos and a landscape engulfed in red flames.",
    "Bleached coral in the Great Barrier Reef, with vibrant living coral and swimming small fish.",
    "Sea turtle navigating through ocean cluttered with plastic debris, near a shadowy city skyline.",
    "Brazilian Amazon in flames, with rising smoke depicting rainforest destruction.",
    "Australian bushfires from above, showing fire consuming forests and causing wildlife distress.",
    "California's scorched earth and barren landscapes with wildfires and smoke clouds.",
    "East African farmlands overrun by swarms of locusts, devastating crops and causing despair.",
]

def img2img(img, mask, i):
    image = torch.tensor(np.array(img)[None, ...].transpose((0, 3, 1, 2)), device=device, dtype=torch.bfloat16) / 255.0
    mask = torch.tensor(np.array(mask.resize((512, 512)), dtype=np.bool_)[None, None], device=device, dtype=dtype)
    with torch.inference_mode():
        img2img_result = pipe_reduced(
            prompt_embeds=prompt_embeds[i],
            aug_emb=aug_embs[i],
            image=image,
            mask=mask,
        )
    return Image.fromarray((img2img_result[0] * 255).to('cpu', dtype=torch.float32).numpy().transpose(1, 2, 0).astype(np.uint8))

class UserSession:
    """Individual user session with isolated state"""
    def __init__(self, user_id, websocket, server=None):
        self.user_id = user_id
        self.websocket = websocket
        self.server = server  # Reference to the server for thread-safe operations
        self.xy_queue = []
        self.latest_mask = None
        self.latest_mask_info = None
        self.input_image = None
        self.images_to_show = []
        self.prompt_idx = 0
        self.is_active = True
        self.last_activity = time.time()
        
        # Video storage related attributes
        self.reverse_show_images = False
        self.archiving = False
        self.video_writer = None
        self._progress_total = 0
        self._progress_count = 0
        self.reverse_index = 0
        self.video_saved = False  # Flag to track if video has been saved for this session
        
        # Image interaction bounds (for limiting interaction to actual image area)
        self.image_bounds = None  # (offset_x, offset_y, width, height)
        
        # Initialize user's input image
        self.reset_system()
    
    def _aspect_fit_to_canvas(self, image, canvas_width, canvas_height, fill_color=(128, 128, 128)):
        """Resize image to fit within a fixed canvas while preserving aspect ratio (letterbox).
        Returns (canvas_image, image_bounds) where image_bounds is (offset_x, offset_y, width, height)
        """
        try:
            src_width, src_height = image.size
            if src_width <= 0 or src_height <= 0:
                raise ValueError("Invalid source image size")
            scale = min(canvas_width / src_width, canvas_height / src_height)
            new_width = max(1, int(round(src_width * scale)))
            new_height = max(1, int(round(src_height * scale)))
            resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            canvas = Image.new('RGB', (canvas_width, canvas_height), fill_color)
            offset_x = (canvas_width - new_width) // 2
            offset_y = (canvas_height - new_height) // 2
            canvas.paste(resized, (offset_x, offset_y))
            
            # Return canvas and image bounds for interaction limiting
            image_bounds = (offset_x, offset_y, new_width, new_height)
            return canvas, image_bounds
        except Exception:
            # Fallback to simple resize to ensure pipeline continuity
            canvas = image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            image_bounds = (0, 0, canvas_width, canvas_height)  # Full canvas is interactive
            return canvas, image_bounds

    def reset_system(self):
        """Reset user's system state"""
        try:
            ground_img = Image.open("test/image/ground.jpg")
            # Fit any input ground image into the working canvas while preserving aspect ratio
            self.input_image, self.image_bounds = self._aspect_fit_to_canvas(ground_img, WIDTH, HEIGHT)
            print(f"User {self.user_id} - Image bounds: {self.image_bounds}")
        except Exception as e:
            print(f"Failed to load image for user {self.user_id}: {e}")
            self.input_image = Image.new('RGB', (WIDTH, HEIGHT), (128, 128, 128))
            self.image_bounds = (0, 0, WIDTH, HEIGHT)  # Full canvas is interactive
        
        self.latest_mask = None
        self.latest_mask_info = None
        self.xy_queue.clear()
        self.images_to_show.clear()
        
        # Reset video storage state
        self.reverse_show_images = True
        self.archiving = False
        self.video_saved = False  # Reset video saved flag
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self._progress_total = 0
        self._progress_count = 0
        self.reverse_index = 0
    
    def is_point_in_image_bounds(self, x, y):
        """Check if a point (x, y) is within the actual image area (not gray padding)"""
        if self.image_bounds is None:
            return True  # If no bounds set, allow all points
        
        offset_x, offset_y, width, height = self.image_bounds
        return (offset_x <= x < offset_x + width and 
                offset_y <= y < offset_y + height)
    
    def save_images_to_new_video(self, fps=10):
        """Save images to new video file using MoviePy"""
        if not self.images_to_show:
            return
        
        # Get next video number in a thread-safe way
        if self.server:
            next_video_idx = self.server.get_next_video_number()
        else:
            # Fallback to local method if server reference is not available
            video_files = [f for f in os.listdir("history") if f.endswith('.mp4')]
            existing_numbers = []
            for video_file in video_files:
                try:
                    number = int(video_file.split('.')[0])
                    existing_numbers.append(number)
                except (ValueError, IndexError):
                    continue
            next_video_idx = max(existing_numbers) + 1 if existing_numbers else 0
        
        next_video_path = os.path.join("history", f"{next_video_idx}.mp4")
        
        try:
            print(f"[INFO] User {self.user_id} - Converting {len(self.images_to_show)} frames to RGB format")
            
            # Convert images from BGR to RGB format (MoviePy needs RGB)
            images_rgb = []
            for i, img in enumerate(self.images_to_show):
                if len(img.shape) == 3 and img.shape[2] == 3:
                    # Already BGR, convert to RGB
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif len(img.shape) == 2:
                    # Grayscale, convert to RGB
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                else:
                    # Other format, try to convert
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images_rgb.append(rgb_img)
                
                # Show progress
                if (i + 1) % 10 == 0 or i == len(self.images_to_show) - 1:
                    progress = (i + 1) / len(self.images_to_show) * 100
                    print(f"\r[User {self.user_id} Video Progress] Converting frames: {progress:.1f}% ({i+1}/{len(self.images_to_show)})", end='', flush=True)
            
            print(f"\n[INFO] User {self.user_id} - Creating video with MoviePy...")
            
            # Create video using MoviePy
            clip = ImageSequenceClip(images_rgb, fps=fps)
            clip.write_videofile(next_video_path, codec='libx264', verbose=False, logger=None)
            print(f"[INFO] User {self.user_id} - Video successfully saved: {next_video_path}")
            
        except Exception as e:
            print(f"\n[ERROR] User {self.user_id} - Failed to save video with MoviePy: {e}")
            # Try alternative approach
            try:
                print(f"[INFO] User {self.user_id} - Attempting to save as individual frames...")
                # Save as individual frames if video creation fails
                frame_dir = os.path.join("history", f"frames_{next_video_idx}")
                os.makedirs(frame_dir, exist_ok=True)
                for i, img in enumerate(self.images_to_show):
                    frame_path = os.path.join(frame_dir, f"frame_{i:04d}.png")
                    cv2.imwrite(frame_path, img)
                print(f"[INFO] User {self.user_id} - Frames saved to: {frame_dir}")
            except Exception as e2:
                print(f"[ERROR] User {self.user_id} - Failed to save frames: {e2}")
    
    def start_video_archiving(self):
        """Start video archiving process - Use MoviePy directly for better reliability"""
        if not self.images_to_show or self.archiving or self.video_saved:
            return
        
        print(f"[INFO] User {self.user_id} - Starting video archiving with MoviePy")
        
        # Use MoviePy directly for better reliability
        self.video_writer = None  # We won't use OpenCV VideoWriter
        self.archiving = True
        self.video_saved = True  # Mark as saved to prevent duplicate saves
        self._progress_total = len(self.images_to_show)
        self._progress_count = 0
        self.reverse_index = 0
        
        # Start MoviePy video creation in a separate thread to avoid blocking
        import threading
        def create_video():
            try:
                self.save_images_to_new_video()
                self.finish_video_archiving()
            except Exception as e:
                print(f"[ERROR] User {self.user_id} - MoviePy video creation failed: {e}")
                self.finish_video_archiving()
        
        video_thread = threading.Thread(target=create_video, daemon=True)
        video_thread.start()
    
    def archive_frame(self):
        """Archive current frame to video - Not used with MoviePy approach"""
        # Since we're using MoviePy directly, this method is not needed
        # But we keep it for compatibility and to handle any remaining OpenCV-based archiving
        if not self.archiving or not self.images_to_show:
            return False
        
        # If we're using MoviePy (video_writer is None), just return True to indicate completion
        if self.video_writer is None:
            return True
        
        # Legacy OpenCV VideoWriter code (kept for fallback)
        if self.reverse_index < len(self.images_to_show):
            current_img = self.images_to_show[-self.reverse_index - 1]  # Reverse order
            
            # Validate frame before writing
            if current_img is not None and current_img.size > 0:
                try:
                    # Ensure frame is in correct format (BGR, uint8)
                    if current_img.dtype != np.uint8:
                        current_img = current_img.astype(np.uint8)
                    
                    # Ensure frame has 3 channels (BGR)
                    if len(current_img.shape) == 2:
                        current_img = cv2.cvtColor(current_img, cv2.COLOR_GRAY2BGR)
                    elif len(current_img.shape) == 3 and current_img.shape[2] == 4:
                        current_img = cv2.cvtColor(current_img, cv2.COLOR_BGRA2BGR)
                    elif len(current_img.shape) == 3 and current_img.shape[2] == 1:
                        current_img = cv2.cvtColor(current_img, cv2.COLOR_GRAY2BGR)
                    
                    # Verify VideoWriter is still valid
                    if not self.video_writer.isOpened():
                        print(f"\n[ERROR] User {self.user_id} - VideoWriter is no longer opened")
                        self.finish_video_archiving()
                        return True
                    
                    # Write current frame to video
                    success = self.video_writer.write(current_img)
                    if not success:
                        print(f"\n[WARNING] User {self.user_id} - Failed to write frame {self.reverse_index}")
                        # Continue with next frame instead of failing completely
                    else:
                        self._progress_count += 1
                        
                        # Print progress bar only every 5 frames to reduce console spam
                        if self._progress_count % 5 == 0 or self._progress_count == self._progress_total:
                            bar_len = 30
                            filled_len = int(round(bar_len * self._progress_count / float(self._progress_total)))
                            percents = round(100.0 * self._progress_count / float(self._progress_total), 1)
                            bar = '█' * filled_len + '-' * (bar_len - filled_len)
                            print(f'\r[User {self.user_id} Archiving Progress] |{bar}| {percents}% ({self._progress_count}/{self._progress_total})', end='', flush=True)
                
                except Exception as e:
                    print(f"\n[ERROR] User {self.user_id} - Error writing frame {self.reverse_index}: {e}")
                    # Continue with next frame
            
            self.reverse_index += 1
            
            # Check if archiving is complete
            if self.reverse_index >= len(self.images_to_show):
                self.finish_video_archiving()
                return True
        
        return False
    
    def finish_video_archiving(self):
        """Finish video archiving process"""
        try:
            if self.video_writer is not None:
                # Clear the progress line and print completion message
                print(f"\r{' ' * 80}\r[INFO] User {self.user_id} - Video Successfully Saved ({self._progress_count} frames)")
                self.video_writer.release()
                self.video_writer = None
        except Exception as e:
            print(f"\n[ERROR] User {self.user_id} - Error finishing video archiving: {e}")
        
        self.archiving = False
        self.reverse_show_images = False
        self.images_to_show.clear()
        self.reverse_index = 0
        self._progress_total = 0
        self._progress_count = 0
        # Note: Don't reset video_saved here, as we want to prevent duplicate saves
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = time.time()
    
    def is_idle(self, timeout_seconds=300):  # 5 minutes timeout
        """Check if user session is idle"""
        return time.time() - self.last_activity > timeout_seconds

class MultiUserWebSocketServer:
    def __init__(self):
        self.user_sessions = {}  # user_id -> UserSession
        self.running = True
        self.history_dir = "history"
        
        # Add pending images queue for each user
        self.pending_images = {}  # user_id -> list of (session, image) tuples
        
        # Video numbering lock for thread safety
        self.video_numbering_lock = threading.Lock()
        
        # Ensure history directory exists
        os.makedirs(self.history_dir, exist_ok=True)
        
        # Start processing threads
        self.start_processing_threads()
    
    def generate_user_id(self):
        """Generate unique user ID"""
        return str(uuid.uuid4())
    
    def get_next_video_number(self):
        """Get next available video number in a thread-safe way"""
        with self.video_numbering_lock:
            video_files = [f for f in os.listdir(self.history_dir) if f.endswith('.mp4')]
            
            # Extract existing video numbers and find the next available number
            existing_numbers = []
            for video_file in video_files:
                try:
                    # Extract number from filename (e.g., "123.mp4" -> 123)
                    number = int(video_file.split('.')[0])
                    existing_numbers.append(number)
                except (ValueError, IndexError):
                    # Skip files that don't match the expected naming pattern
                    continue
            
            # Find the next available number
            if existing_numbers:
                next_video_idx = max(existing_numbers) + 1
            else:
                next_video_idx = 0
            
            return next_video_idx
    
    def start_processing_threads(self):
        """Start processing threads for all users"""
        def processing_loop():
            while self.running:
                time.sleep(0.1)  # Check every 100ms
                
                # Process each user's data
                for user_id, session in list(self.user_sessions.items()):
                    if not session.is_active:
                        continue
                    
                    try:
                        if session.xy_queue:
                            self._generate_and_update_mask(session)
                        if session.latest_mask is not None:
                            self._process_latest_mask(session)
                    except Exception as e:
                        print(f"Error processing user {user_id}: {e}")
                
                # Clean up idle sessions
                self._cleanup_idle_sessions()
        
        def video_archiving_loop():
            while self.running:
                time.sleep(0.03)  # ~30 FPS for video archiving
                
                # Process video archiving for each user
                for user_id, session in list(self.user_sessions.items()):
                    if not session.is_active:
                        continue
                    
                    try:
                        if session.archiving:
                            session.archive_frame()
                    except Exception as e:
                        # Suppress error output during video archiving to avoid interruption
                        pass
        
        # Start main processing thread
        processing_thread = threading.Thread(target=processing_loop, daemon=True)
        processing_thread.start()
        
        # Start video archiving thread
        archiving_thread = threading.Thread(target=video_archiving_loop, daemon=True)
        archiving_thread.start()

    # def start_processing_threads(self):
    #     """multi-thread processing"""
    #     def process_user(session):
    #         try:
    #             if session.xy_queue:
    #                 self._generate_and_update_mask(session)
    #             if session.latest_mask is not None:
    #                 self._process_latest_mask(session)
    #         except Exception as e:
    #             print(f"Error processing user {session.user_id}: {e}")

    #     def processing_loop():
    #         while self.running:
    #             time.sleep(0.1)
    #             active_sessions = [s for s in self.user_sessions.values() if s.is_active]
                
    #             # 使用线程池并行处理
    #             with concurrent.futures.ThreadPoolExecutor() as executor:
    #                 executor.map(process_user, active_sessions)
                
    #             self._cleanup_idle_sessions()

    #     thread = threading.Thread(target=processing_loop, daemon=True)
    #     thread.start()
    
    def _cleanup_idle_sessions(self):
        """Clean up idle user sessions"""
        current_time = time.time()
        idle_users = []
        
        for user_id, session in self.user_sessions.items():
            if session.is_idle():
                idle_users.append(user_id)
        
        for user_id in idle_users:
            print(f"Cleaning up idle session for user {user_id}")
            # Save video before cleaning up session
            session = self.user_sessions[user_id]
            if session.images_to_show and not session.archiving and not session.video_saved:
                session.start_video_archiving()
                # Wait a bit for archiving to start
                time.sleep(0.1)
                # Continue archiving until complete
                while session.archiving:
                    session.archive_frame()
                    time.sleep(0.03)
            del self.user_sessions[user_id]
            if user_id in self.pending_images:
                del self.pending_images[user_id]
    
    def _generate_and_update_mask(self, session):
        """Generate and update mask for specific user"""
        if not session.xy_queue:
            return

        xy_array = np.array(session.xy_queue)

        center_x = int(np.mean([np.min(xy_array[:, 0]), np.max(xy_array[:, 0])]))
        center_y = int(np.mean([np.min(xy_array[:, 1]), np.max(xy_array[:, 1])]))

        start_x = max(0, min(center_x - 256, WIDTH - 512))
        start_y = max(0, min(center_y - 256, HEIGHT - 512))
        end_x = start_x + 512
        end_y = start_y + 512

        cropped_mask = np.zeros((512, 512), dtype=np.uint8)

        valid_points = xy_array[
            (xy_array[:, 0] >= start_x + CIRCLE_RADIUS + BLUR_RADIUS) &
            (xy_array[:, 0] < end_x - CIRCLE_RADIUS - BLUR_RADIUS) &
            (xy_array[:, 1] >= start_y + CIRCLE_RADIUS + BLUR_RADIUS) &
            (xy_array[:, 1] < end_y - CIRCLE_RADIUS - BLUR_RADIUS)
        ]
        
        if len(valid_points) == 0:
            session.xy_queue.clear()
            return
            
        relative_points = valid_points[:, :2] - [start_x, start_y]

        i = 0
        while i < len(relative_points) - 1:
            x1, y1 = relative_points[i].astype(int)
            r1 = int(max(CIRCLE_RADIUS - valid_points[i, 2] * 0.02, 15))

            for j in range(i + 1, len(relative_points)):
                x2, y2 = relative_points[j].astype(int)
                r2 = int(max(CIRCLE_RADIUS - valid_points[j, 2] * 0.02, 15))

                dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                if dist + r2 <= r1/2:
                    continue
                else:
                    draw_tangent_and_fill(cropped_mask, x1, y1, r1, x2, y2, r2)
                    i = j - 1 
                    break
            i += 1

        # Morphological processing
        kernel = np.ones((DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE), np.uint8)
        continuous_mask = cv2.dilate(cropped_mask, kernel, iterations=1)

        # Gaussian blur processing
        blurred_mask = cv2.GaussianBlur(continuous_mask, (GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE), 0)

        session.xy_queue.clear()
        session.latest_mask = blurred_mask
        session.latest_mask_info = (start_x, start_y)

    def _process_latest_mask(self, session):
        """Process the latest mask for specific user"""
        if session.input_image is None:
            print(f"input_image is None for user {session.user_id}, skipping processing.")
            session.reset_system()
            return
            
        mask_img = session.latest_mask
        x, y = session.latest_mask_info
        session.latest_mask = None
        session.latest_mask_info = None

        prompt_idx = session.prompt_idx
        square_ground = session.input_image.crop((x, y, x + 512, y + 512))

        try:
            # Generate image
            img2img_result = img2img(square_ground, Image.fromarray(mask_img), prompt_idx)
            
            # Generate interpolated frames
            images = generate_interpolated_frames(square_ground, img2img_result, exp=4)
            
            # Add initial frame to images_to_show (BGR format for OpenCV)
            session.images_to_show.append(cv2.cvtColor(np.array(session.input_image), cv2.COLOR_RGB2BGR))
            
            # Process each frame
            for interpolated_image in images:
                final_image = merge_images(session.input_image, interpolated_image, Image.fromarray(mask_img), x, y)
                
                # Add to images_to_show (BGR format for OpenCV)
                session.images_to_show.append(cv2.cvtColor(np.array(final_image), cv2.COLOR_RGB2BGR))
                
                # Add to pending images queue instead of direct async call
                if session.user_id not in self.pending_images:
                    self.pending_images[session.user_id] = []
                self.pending_images[session.user_id].append((session, final_image))
            
            session.input_image = final_image
            
            # Start video archiving if reverse_show_images is True and we have content
            if session.reverse_show_images and not session.archiving and len(session.images_to_show) > 1:
                session.start_video_archiving()
            
        except Exception as e:
            print(f"Image processing error for user {session.user_id}: {e}")

    async def send_pending_images(self):
        """Send pending images to all users"""
        while self.running:
            try:
                # Process pending images for each user
                for user_id in list(self.pending_images.keys()):
                    if user_id not in self.user_sessions:
                        # User disconnected, remove pending images
                        del self.pending_images[user_id]
                        continue
                    
                    session = self.user_sessions[user_id]
                    if not session.is_active:
                        continue
                    
                    # Send all pending images for this user
                    while self.pending_images[user_id]:
                        session, image = self.pending_images[user_id].pop(0)
                        await self.send_image_to_user(session, image)
                
                await asyncio.sleep(0.1)  # Check every 100ms
                
            except Exception as e:
                print(f"Error in send_pending_images: {e}")
                await asyncio.sleep(0.1)

    async def send_image_to_user(self, session, image):
        """Send image to specific user"""
        try:
            import base64
            import io
            
            # Resize image to fit frontend
            display_image = image.resize((800, 600), Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffer = io.BytesIO()
            display_image.save(buffer, format='JPEG', quality=85)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Create message
            message = {
                'type': 'image',
                'data': f'data:image/jpeg;base64,{img_base64}',
                'userId': session.user_id
            }
            
            await session.websocket.send(json.dumps(message))
        except Exception as e:
            print(f"Failed to send image to user {session.user_id}: {e}")
            session.is_active = False

    async def handle_client(self, websocket, path):
        """Handle client connection"""
        user_id = self.generate_user_id()
        session = UserSession(user_id, websocket, self)  # Pass server reference
        self.user_sessions[user_id] = session
        self.pending_images[user_id] = []  # Initialize pending images queue
        
        print(f"New client connected: {websocket.remote_address}, User ID: {user_id}")
        print(f"Total active users: {len(self.user_sessions)}")
        
        # Send initial image and user ID
        if session.input_image:
            await self.send_image_to_user(session, session.input_image)
        
        # Send user ID and image bounds to client
        await websocket.send(json.dumps({
            'type': 'user_registered',
            'userId': user_id,
            'imageBounds': session.image_bounds
        }))
        
        try:
            async for message in websocket:
                session.update_activity()
                await self.process_message(session, message)
        except websockets.exceptions.ConnectionClosed:
            print(f"Client disconnected: {websocket.remote_address}, User ID: {user_id}")
        except Exception as e:
            print(f"Error handling client message for user {user_id}: {e}")
        finally:
            # Save video before cleaning up session
            if user_id in self.user_sessions:
                session = self.user_sessions[user_id]
                if session.images_to_show and not session.archiving and not session.video_saved:
                    session.start_video_archiving()
                    # Wait a bit for archiving to start
                    time.sleep(0.1)
                    # Continue archiving until complete
                    while session.archiving:
                        session.archive_frame()
                        time.sleep(0.03)
                del self.user_sessions[user_id]
            if user_id in self.pending_images:
                del self.pending_images[user_id]
            print(f"User {user_id} session ended. Total active users: {len(self.user_sessions)}")

    async def process_message(self, session, message):
        """Process client messages for specific user"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'mouse_data':
                # Process mouse data
                x, y, speed = data['x'], data['y'], data['speed']
                # Only process if point is within image bounds
                if session.is_point_in_image_bounds(x, y):
                    session.xy_queue.append((x, y, speed))
                else:
                    # print(f"User {session.user_id} mouse data ignored - outside image bounds: ({x:.1f}, {y:.1f})")
                    pass
                
            elif msg_type == 'gaze_data':
                # Process gaze tracking data
                x, y, speed = data['x'], data['y'], data['speed']
                confidence = data.get('confidence', 1.0)
                
                # Only process gaze data with sufficient confidence AND within image bounds
                if confidence >= 0.7 and session.is_point_in_image_bounds(x, y):  # Confidence threshold
                    session.xy_queue.append((x, y, speed))
                    print(f"User {session.user_id} gaze data: ({x:.1f}, {y:.1f}), speed: {speed:.1f}, confidence: {confidence:.2f}")
                elif confidence < 0.7:
                    print(f"User {session.user_id} gaze data rejected due to low confidence: {confidence:.2f}")
                else:
                    # print(f"User {session.user_id} gaze data ignored - outside image bounds: ({x:.1f}, {y:.1f})")
                    pass
                
            elif msg_type == 'prompt_change':
                # Process prompt change
                session.prompt_idx = data['prompt_idx']
                print(f"User {session.user_id} prompt changed to: {prompts_list[session.prompt_idx]}")
                
            elif msg_type == 'reset':
                # Process reset request
                # Start video archiving before reset if there are images to save
                if session.images_to_show and not session.archiving and not session.video_saved:
                    session.start_video_archiving()
                
                session.reset_system()
                print(f"User {session.user_id} system reset")
                
                # Send reset confirmation
                response = {
                    'type': 'reset_confirmed',
                    'userId': session.user_id
                }
                await session.websocket.send(json.dumps(response))
                
        except json.JSONDecodeError:
            # Handle non-JSON format messages (backward compatibility)
            try:
                parts = message.split(',')
                if len(parts) == 3:
                    # Mouse data format: x,y,speed
                    x, y, speed = map(float, parts)
                    # Only process if point is within image bounds
                    if session.is_point_in_image_bounds(x, y):
                        session.xy_queue.append((x, y, speed))
                    else:
                        # print(f"User {session.user_id} mouse data ignored - outside image bounds: ({x:.1f}, {y:.1f})")
                        pass
                elif len(parts) == 1:
                    # Prompt index format: index
                    session.prompt_idx = int(parts[0])
                    print(f"User {session.user_id} prompt changed to: {prompts_list[session.prompt_idx]}")
            except Exception as e:
                print(f"Error processing message format for user {session.user_id}: {e}")

    async def start_server(self, host='localhost', port=12346):
        """Start WebSocket server"""
        print(f"Starting Multi-User WebSocket server on {host}:{port}")
        
        async with websockets.serve(self.handle_client, host, port):
            print(f"Multi-User WebSocket server is running...")
            print(f"Server supports multiple concurrent users")
            
            # Start server stats broadcasting and pending images sending
            asyncio.create_task(self.broadcast_server_stats())
            asyncio.create_task(self.send_pending_images())
            
            await asyncio.Future()  # Keep server running

    async def broadcast_server_stats(self):
        """Broadcast server statistics to all clients"""
        while self.running:
            try:
                stats = {
                    'type': 'server_stats',
                    'stats': {
                        'totalUsers': len(self.user_sessions),
                        'activeUsers': len([s for s in self.user_sessions.values() if s.is_active])
                    }
                }
                
                # Send stats to all connected clients
                for session in list(self.user_sessions.values()):
                    try:
                        await session.websocket.send(json.dumps(stats))
                    except Exception as e:
                        print(f"Failed to send stats to user {session.user_id}: {e}")
                        session.is_active = False
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                print(f"Error broadcasting server stats: {e}")
                await asyncio.sleep(5)

def main():
    server = MultiUserWebSocketServer()
    
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        print("Server is shutting down...")
        server.running = False
    except Exception as e:
        print(f"Server error: {e}")

if __name__ == "__main__":
    main() 