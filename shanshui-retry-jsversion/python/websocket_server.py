import asyncio
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
from image_processing import resize, merge_images, draw_tangent_and_fill
from frame_interpolation import generate_interpolated_frames
from pipeline import prompt_embeds, pooled_prompt_embeds, pipe_reduced, aug_embs, dtype

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 配置
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

class WebSocketShanshuiServer:
    def __init__(self):
        self.xy_queue = []
        self.latest_mask = None
        self.latest_mask_info = None
        self.input_image = None
        self.images_to_show = []
        self.history_dir = "history"
        self.prompt_idx = 0
        self.running = True
        self.connected_clients = set()
        self.pending_images = []  # 待发送的图像队列
        
        # 确保历史目录存在
        os.makedirs(self.history_dir, exist_ok=True)
        
        # 初始化输入图像
        self.reset_system()
        
        # 启动处理线程
        self.start_processing_thread()

    def reset_system(self):
        """Reset system state"""
        try:
            ground_img = Image.open("test/image/ground.jpg")
            scale = 0.5
            self.input_image = ground_img.resize((int(ground_img.width * scale), int(ground_img.height * scale)))
        except Exception as e:
            print(f"Failed to load image when resetting system: {e}")
            # Create a default gray image
            self.input_image = Image.new('RGB', (WIDTH, HEIGHT), (128, 128, 128))
        
        self.latest_mask = None
        self.latest_mask_info = None
        self.xy_queue.clear()
        self.images_to_show.clear()

    def start_processing_thread(self):
        """Start processing thread"""
        def processing_loop():
            while self.running:
                time.sleep(0.1)  # Check every 100ms
                if self.xy_queue:
                    self._generate_and_update_mask()
                if self.latest_mask is not None:
                    self._process_latest_mask()
        
        thread = threading.Thread(target=processing_loop, daemon=True)
        thread.start()

    def _generate_and_update_mask(self):
        """Generate and update mask"""
        if not self.xy_queue:
            return

        xy_array = np.array(self.xy_queue)

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
            self.xy_queue.clear()
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

        # 形态学处理
        kernel = np.ones((DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE), np.uint8)
        continuous_mask = cv2.dilate(cropped_mask, kernel, iterations=1)

        # 高斯模糊处理
        blurred_mask = cv2.GaussianBlur(continuous_mask, (GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE), 0)

        self.xy_queue.clear()
        self.latest_mask = blurred_mask
        self.latest_mask_info = (start_x, start_y)

    def _process_latest_mask(self):
        """Process the latest mask"""
        if self.input_image is None:
            print("input_image is None, skipping processing.")
            self.reset_system()
            return
            
        mask_img = self.latest_mask
        x, y = self.latest_mask_info
        self.latest_mask = None
        self.latest_mask_info = None

        prompt_idx = self.prompt_idx
        square_ground = self.input_image.crop((x, y, x + 512, y + 512))

        try:
            # Generate image
            img2img_result = img2img(square_ground, Image.fromarray(mask_img), prompt_idx)
            
            # Generate interpolated frames
            images = generate_interpolated_frames(square_ground, img2img_result, exp=4)
            
            # Process each frame
            for interpolated_image in images:
                final_image = merge_images(self.input_image, interpolated_image, Image.fromarray(mask_img), x, y)
                self.images_to_show.append(final_image)
                
                # Send image to all connected clients
                self.broadcast_image(final_image)
            
            self.input_image = self.images_to_show[-1] if self.images_to_show else self.input_image
            
        except Exception as e:
            print(f"Image processing error: {e}")

    def broadcast_image(self, image):
        """Broadcast image to all connected clients"""
        try:
            # Convert PIL image to base64 encoding
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
                'data': f'data:image/jpeg;base64,{img_base64}'
            }
            
            # Add message to pending queue
            self.pending_images.append(message)
                    
        except Exception as e:
            print(f"Error preparing image data: {e}")

    async def handle_client(self, websocket, path):
        """Handle client connection"""
        print(f"New client connected: {websocket.remote_address}")
        self.connected_clients.add(websocket)
        
        # Send initial image
        if self.input_image:
            await self.send_image_to_client(websocket, self.input_image)
        
        try:
            async for message in websocket:
                await self.process_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            print(f"Client disconnected: {websocket.remote_address}")
        except Exception as e:
            print(f"Error handling client message: {e}")
        finally:
            self.connected_clients.discard(websocket)

    async def send_image_to_client(self, websocket, image):
        """Send image to a single client"""
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
                'data': f'data:image/jpeg;base64,{img_base64}'
            }
            
            await websocket.send(json.dumps(message))
        except Exception as e:
            print(f"Failed to send image to client: {e}")

    async def send_pending_images(self):
        """Send pending image queue"""
        while self.pending_images and self.connected_clients:
            message = self.pending_images.pop(0)
            disconnected_clients = set()
            
            for websocket in self.connected_clients:
                try:
                    await websocket.send(json.dumps(message))
                except Exception as e:
                    print(f"Failed to send image to client: {e}")
                    disconnected_clients.add(websocket)
            
            # Remove disconnected clients
            self.connected_clients -= disconnected_clients

    async def process_message(self, websocket, message):
        """Process client messages"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'mouse_data':
                # Process mouse data
                x, y, speed = data['x'], data['y'], data['speed']
                self.xy_queue.append((x, y, speed))
                
            elif msg_type == 'prompt_change':
                # Process prompt change
                self.prompt_idx = data['prompt_idx']
                print(f"Prompt changed to: {prompts_list[self.prompt_idx]}")
                
            elif msg_type == 'reset':
                # Process reset request
                self.reset_system()
                print("System reset")
                
                # Send reset confirmation
                response = {'type': 'reset_confirmed'}
                await websocket.send(json.dumps(response))
                
        except json.JSONDecodeError:
            # Handle non-JSON format messages (backward compatibility)
            try:
                parts = message.split(',')
                if len(parts) == 3:
                    # Mouse data format: x,y,speed
                    x, y, speed = map(float, parts)
                    self.xy_queue.append((x, y, speed))
                elif len(parts) == 1:
                    # Prompt index format: index
                    self.prompt_idx = int(parts[0])
                    print(f"Prompt changed to: {prompts_list[self.prompt_idx]}")
            except Exception as e:
                print(f"Error processing message format: {e}")

    async def start_server(self, host='localhost', port=12346):
        """Start WebSocket server"""
        print(f"Starting WebSocket server on {host}:{port}")
        
        async with websockets.serve(self.handle_client, host, port):
            print(f"WebSocket server is running...")
            
            # Periodically send pending images
            async def send_pending_images_task():
                while self.running:
                    await self.send_pending_images()
                    await asyncio.sleep(0.1)  # Check every 100ms
            
            # Start sending task
            asyncio.create_task(send_pending_images_task())
            
            await asyncio.Future()  # Keep server running

def main():
    server = WebSocketShanshuiServer()
    
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        print("Server is shutting down...")
        server.running = False
    except Exception as e:
        print(f"Server error: {e}")

if __name__ == "__main__":
    main() 