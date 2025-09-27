# negativeshanshui_gaze
Negative Shanshui Experience through Webcam and Gaze Tracking

An AI-powered real-time landscape painting generation system that supports mouse and eye-tracking interaction, capable of generating and modifying landscape paintings based on user interactions.

## Features

### Landscape Painting Generation
- AI image generation based on Stable Diffusion
- Real-time image interpolation and frame generation
- Adaptive display for both landscape and portrait images

### Multiple Interaction Methods
- **Mouse Tracking**: Interact with landscape paintings by moving the mouse
- **Eye Tracking**: Support for GazeRecorder eye-tracking devices (to be done)
- **Fullscreen Immersive Experience**: Automatic fullscreen display with landscape images filling the screen
- **Real-time Response**: Low-latency interaction feedback

### User Experience Optimization
- One-click connection and startup, automatic fullscreen after theme selection
- Coordinate mapping ensuring interaction precision
- Responsive interface design adapting to different screen sizes
- Keyboard shortcut support (ESC to exit, R to reset, D to disconnect)

## Quick Start

### Requirements
- Python 3.8+
- CUDA-supported GPU (recommended)
- Modern browser (Chrome, Firefox, Edge)

### Installation Steps

1. **Start Backend Server**
```bash
cd python
python multi_user_websocket_server.py
```

2. **Open Frontend Interface**
```bash
cd frontend
# Use any HTTP server to open index.html
# Or double-click index.html file directly
```

3. **Start Using**
   - Select an environment theme
   - Click "Connect & Start" button
   - System automatically enters fullscreen mode
   - Move mouse or use eye-tracking device to interact with landscape paintings

## Usage Instructions

### Basic Operation Flow
1. **Select Theme**: Choose environment theme from dropdown menu
2. **Connect & Start**: Click "Connect & Start" to automatically enter fullscreen
3. **Interactive Creation**: Move mouse or gaze to interact with landscape paintings
4. **Exit & End**: Press ESC to exit fullscreen, click "Disconnect" to end

### Keyboard Shortcuts
- `ESC`: Exit fullscreen mode
- `R`: Reset landscape painting
- `D`: Disconnect

### Eye Tracking Setup
- Requires GazeRecorder device support
- Configure API key in settings
- Adjustable confidence threshold and smoothing parameters

## Technical Architecture

### Backend Tech Stack
- **WebSocket Server**: Multi-user real-time communication
- **AI Model**: Stable Diffusion + custom pipeline
- **Image Processing**: OpenCV + PIL
- **Frame Interpolation**: Custom interpolation algorithm

### Frontend Tech Stack
- **Vanilla JavaScript**: No framework dependencies
- **Canvas 2D**: Image rendering and interaction
- **WebSocket**: Real-time data transmission
- **Responsive CSS**: Multi-device adaptation

### Core Algorithms
- **Coordinate Mapping**: Precise conversion from canvas coordinates to server coordinates
- **Image Interpolation**: Smooth frame transition effects
- **Boundary Detection**: Intelligent interaction area recognition

## Project Structure

```
shanshui-retry/
├── python/                    # Backend server
│   ├── multi_user_websocket_server.py  # Main server
│   ├── pipeline.py           # AI generation pipeline
│   ├── image_processing.py   # Image processing
│   └── frame_interpolation.py # Frame interpolation
├── frontend/                 # Frontend interface
│   ├── index.html           # Main interface
│   ├── multi_user_app.js    # Core logic
│   ├── styles.css           # Style files
│   └── gaze_tracking.js     # Eye tracking
└── README.md                # Project documentation
```

## Troubleshooting

### WebSocket Connection Error (1005)
**Error Cause**: Client abnormally disconnected
- Browser tab closed
- Network connection interrupted
- Browser crash or refresh

**Solution**: 
- This is a normal connection disconnection, no special handling required
- Server will automatically clean up disconnected sessions
- Reconnect to continue using

### Common Issues
1. **Cannot connect to server**: Ensure backend server is running
2. **Eye tracking not working**: Check device connection and API configuration
3. **Image display abnormal**: Refresh page and reconnect

## Development Guide

### Adding New Environment Themes
Add new prompts to `prompts_list` in `python/multi_user_websocket_server.py`.

### Customizing Interaction Parameters
Modify configuration constants in `python/multi_user_websocket_server.py`:
- `CIRCLE_RADIUS`: Interaction circle radius
- `SPEED_REDUCE`: Speed decay coefficient
- `GAUSSIAN_KERNEL_SIZE`: Blur kernel size

## License

This project is for learning and research purposes only.

## Contributing

Issues and Pull Requests are welcome to improve the project.

---

**Enjoy creating landscape paintings!** 🎨✨
