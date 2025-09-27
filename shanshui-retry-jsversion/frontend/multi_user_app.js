class MultiUserShanshuiApp {
    constructor() {
        console.log('MultiUserShanshuiApp constructor called');
        
        this.websocket = null;
        this.isConnected = false;
        this.isTracking = true;
        this.currentPrompt = 0;
        this.mousePosition = { x: 0, y: 0 };
        this.lastMousePosition = { x: 0, y: 0 };
        this.mouseSpeed = 0;
        this.lastMouseTime = Date.now();
        
        // Get canvas element
        this.canvas = document.getElementById('main-canvas');
        if (!this.canvas) {
            console.error('Canvas element not found!');
            return;
        }
        this.ctx = this.canvas.getContext('2d');
        
        this.historyImages = [];
        this.isFullscreen = false;
        
        // Base logical canvas size expected by server for coordinates
        this.baseWidth = 1047;
        this.baseHeight = 1544;

        // Current render rectangle on the canvas for the image (for coord mapping)
        // Structure: { x, y, width, height }
        this.renderTransform = null;

        // Multi-user specific properties
        this.userId = null;
        this.sessionId = null;
        this.connectionStatus = 'disconnected';
        this.serverStats = {
            totalUsers: 0,
            activeUsers: 0
        };
        
        // Image bounds for interaction limiting
        this.imageBounds = null; // {offsetX, offsetY, width, height}
        
        // Gaze tracking integration
        this.gazeTracker = null;
        this.useGazeTracking = false;
        
        console.log('MultiUserShanshuiApp constructor completed');
        this.init();
    }

    init() {
        console.log('Initializing MultiUserShanshuiApp...');
        try {
            this.setupEventListeners();
            console.log('Event listeners setup completed');
            
            this.setupCanvas();
            console.log('Canvas setup completed');
            
            this.initializeGazeTracking();
            console.log('Gaze tracking initialization completed');
            
            // Initialize UI state
            this.updateTrackingMode();
            this.updateMouseInfo();
            this.updateGazeStatus('Initializing...');
            
            this.updateUI();
            console.log('UI update completed');
            
            this.updateUserInfo();
            console.log('User info update completed');
            
            console.log('MultiUserShanshuiApp initialization completed successfully');
        } catch (error) {
            console.error('Error during initialization:', error);
        }
    }

    isPointInImageBounds(x, y) {
        // Prefer precise renderTransform if available; fallback to imageBounds
        const bounds = this.renderTransform
            ? { offsetX: this.renderTransform.x, offsetY: this.renderTransform.y, width: this.renderTransform.width, height: this.renderTransform.height }
            : this.imageBounds;

        if (!bounds) return true;

        const { offsetX, offsetY, width, height } = bounds;
        return (offsetX <= x && x < offsetX + width && offsetY <= y && y < offsetY + height);
    }

    // Map a canvas-space point to the server's base logical coordinate system (1047x1544)
    mapToBaseCoordinates(x, y) {
        // If we know exactly where the image is drawn on the canvas, map within that rect
        const rt = this.renderTransform;
        if (rt) {
            const nx = (x - rt.x) / rt.width;   // 0..1 within drawn image
            const ny = (y - rt.y) / rt.height; // 0..1 within drawn image
            // Clamp to [0,1] just in case
            const cx = Math.max(0, Math.min(1, nx));
            const cy = Math.max(0, Math.min(1, ny));
            return {
                x: cx * this.baseWidth,
                y: cy * this.baseHeight
            };
        }

        // Fallback: assume full canvas maps linearly to base size
        return {
            x: x / this.canvas.width * this.baseWidth,
            y: y / this.canvas.height * this.baseHeight
        };
    }

    setupEventListeners() {
        try {
            // Connection buttons
            const connectBtn = document.getElementById('connect-btn');
            if (connectBtn) {
                connectBtn.addEventListener('click', () => {
                    this.connect();
                });
            } else {
                console.warn('Connect button not found');
            }

            const disconnectBtn = document.getElementById('disconnect-btn');
            if (disconnectBtn) {
                disconnectBtn.addEventListener('click', () => {
                    this.disconnect();
                });
            } else {
                console.warn('Disconnect button not found');
            }

            // Prompt selection
            const promptSelector = document.getElementById('prompt-selector');
            if (promptSelector) {
                promptSelector.addEventListener('change', (e) => {
                    this.currentPrompt = parseInt(e.target.value);
                    this.sendPromptToServer();
                    this.updateCurrentPrompt();
                });
            } else {
                console.warn('Prompt selector not found');
            }

            // Mouse tracking toggle
            const trackingToggle = document.getElementById('tracking-toggle');
            if (trackingToggle) {
                trackingToggle.addEventListener('change', (e) => {
                    this.isTracking = e.target.checked;
                    console.log('Mouse tracking toggled:', this.isTracking);
                    
                    // If mouse tracking is disabled and gaze tracking is enabled, keep tracking
                    if (!this.isTracking && this.useGazeTracking) {
                        console.log('Mouse tracking disabled but gaze tracking is active');
                    }
                });
            } else {
                console.warn('Tracking toggle not found');
            }

                    // Gaze tracking toggle
        const gazeTrackingToggle = document.getElementById('gaze-tracking-toggle');
        if (gazeTrackingToggle) {
            gazeTrackingToggle.addEventListener('change', async (e) => {
                await this.toggleGazeTracking(e.target.checked);
            });
        } else {
                console.warn('Gaze tracking toggle not found');
            }

            // Reset button
            const resetBtn = document.getElementById('reset-btn');
            if (resetBtn) {
                resetBtn.addEventListener('click', () => {
                    this.resetSystem();
                });
            } else {
                console.warn('Reset button not found');
            }

            // Fullscreen button
            const fullscreenBtn = document.getElementById('fullscreen-btn');
            if (fullscreenBtn) {
                fullscreenBtn.addEventListener('click', () => {
                    this.toggleFullscreen();
                });
            } else {
                console.warn('Fullscreen button not found');
            }

            // Settings button
            const settingsBtn = document.getElementById('settings-btn');
            if (settingsBtn) {
                settingsBtn.addEventListener('click', () => {
                    this.openSettings();
                });
            } else {
                console.warn('Settings button not found');
            }

            // Mouse move events (always available as fallback)
            document.addEventListener('mousemove', (e) => {
                this.handleMouseMove(e);
            });

            // Keyboard events
            document.addEventListener('keydown', (e) => {
                if (e.key === 'r' || e.key === 'R') {
                    this.resetSystem();
                } else if (e.key === 'Escape' && this.isFullscreen) {
                    this.exitFullscreen();
                } else if (e.key === 'd' || e.key === 'D') {
                    if (this.isFullscreen) {
                        this.disconnect();
                    }
                }
            });

            // Window resize
            window.addEventListener('resize', () => {
                this.handleResize();
            });
            
            console.log('All event listeners setup completed');
        } catch (error) {
            console.error('Error setting up event listeners:', error);
        }
    }

    setupCanvas() {
        // Set canvas dimensions
        this.canvas.width = 1047;
        this.canvas.height = 1544;
        
        // Draw initial background
        this.ctx.fillStyle = '#f0f0f0';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Add prompt text
        this.ctx.fillStyle = '#666';
        this.ctx.font = '24px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('Waiting for server connection...', this.canvas.width / 2, this.canvas.height / 2);
    }

    connect() {
        if (this.isConnected) return;

        this.updateConnectionStatus('connecting');
        
        try {
            // Create WebSocket connection
            this.websocket = new WebSocket('ws://localhost:12346');
            
            this.websocket.onopen = () => {
                console.log('WebSocket connection established');
                this.isConnected = true;
                this.updateConnectionStatus('connected');
                
                // Update UI when connected
                this.updateTrackingMode();
                this.updateMouseInfo();
                
                // Wait for user registration from server
            };

            this.websocket.onmessage = (event) => {
                this.handleServerMessage(event.data);
            };

            this.websocket.onclose = () => {
                console.log('WebSocket connection closed');
                this.isConnected = false;
                this.userId = null;
                this.updateConnectionStatus('disconnected');
                this.updateUserInfo();
                // Exit fullscreen if connection is lost
                if (this.isFullscreen) {
                    this.exitFullscreen();
                }
            };

            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('disconnected');
            };

        } catch (error) {
            console.error('Connection failed:', error);
            this.updateConnectionStatus('disconnected');
        }
    }

    disconnect() {
        // Exit fullscreen first if in fullscreen mode
        if (this.isFullscreen) {
            this.exitFullscreen();
        }
        
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        this.isConnected = false;
        this.userId = null;
        this.updateConnectionStatus('disconnected');
        this.updateUserInfo();
        
        // Reset system state
        this.resetSystem();
    }

    initializeGazeTracking() {
        try {
            console.log('Initializing gaze tracking...');
            
            // Check if gaze tracking script is already loaded
            if (typeof GazeTracker !== 'undefined') {
                console.log('GazeTracker already available');
                this.createGazeTracker();
                return;
            }
            
            // Wait a bit for scripts to load
            setTimeout(() => {
                if (typeof GazeTracker !== 'undefined') {
                    console.log('GazeTracker loaded after delay');
                    this.createGazeTracker();
                } else {
                    console.warn('GazeTracker not available, using mouse tracking only');
                    this.updateGazeStatus('Script not found');
                }
            }, 1000);
            
        } catch (error) {
            console.error('Failed to initialize gaze tracking:', error);
            this.updateGazeStatus('Failed to initialize');
        }
    }
    
    createGazeTracker() {
        try {
            console.log('Creating gaze tracker...');
            this.gazeTracker = new GazeTracker();
            
            // Listen for gaze data events
            window.addEventListener('gazeData', (event) => {
                console.log('Gaze data event received:', event.detail);
                this.handleGazeData(event.detail);
            });
            
            console.log('Gaze tracking initialized successfully');
            this.updateGazeStatus('Ready');
            
            // Update UI to reflect that gaze tracking is available
            this.updateTrackingMode();
        } catch (error) {
            console.error('Failed to create gaze tracker:', error);
            this.updateGazeStatus('Failed to create');
        }
    }
    
    updateGazeStatus(status) {
        const gazeStatusElement = document.getElementById('gaze-status');
        if (gazeStatusElement) {
            gazeStatusElement.textContent = status;
            
            // Update CSS class based on status
            gazeStatusElement.className = '';
            if (status.includes('Ready')) {
                gazeStatusElement.classList.add('ready');
            } else if (status.includes('Recording')) {
                gazeStatusElement.classList.add('recording');
            } else if (status.includes('Error') || status.includes('Failed') || status.includes('Not')) {
                gazeStatusElement.classList.add('error');
            }
            
            console.log('Gaze status updated:', status);
        }
    }

    async toggleGazeTracking(enabled) {
        console.log('Toggling gaze tracking:', enabled);
        
        this.useGazeTracking = enabled;
        
        if (enabled) {
            if (this.gazeTracker) {
                try {
                    await this.gazeTracker.toggleGazeTracking(true);
                    console.log('Switched to gaze tracking');
                } catch (error) {
                    console.error('Failed to start gaze tracking:', error);
                    this.useGazeTracking = false;
                    // Update checkbox to reflect actual state
                    const gazeToggle = document.getElementById('gaze-tracking-toggle');
                    if (gazeToggle) {
                        gazeToggle.checked = false;
                    }
                    this.updateGazeStatus('Failed to start: ' + error.message);
                }
            } else {
                console.warn('Gaze tracker not available, using mouse tracking');
                this.useGazeTracking = false;
                // Update checkbox to reflect actual state
                const gazeToggle = document.getElementById('gaze-tracking-toggle');
                if (gazeToggle) {
                    gazeToggle.checked = false;
                }
            }
        } else {
            if (this.gazeTracker) {
                this.gazeTracker.toggleGazeTracking(false);
            }
            console.log('Using mouse tracking');
        }
        
        // Update UI
        this.updateTrackingMode();
        this.updateMouseInfo();
        
        // Log current state
        console.log('Current tracking state:', {
            useGazeTracking: this.useGazeTracking,
            isTracking: this.isTracking,
            isConnected: this.isConnected,
            userId: this.userId
        });
    }

    handleGazeData(gazeData) {
        if (!this.isTracking || !this.isConnected) {
            console.log('Gaze data ignored:', {
                isTracking: this.isTracking,
                isConnected: this.isConnected
            });
            return;
        }

        console.log('Processing gaze data in main app:', gazeData);

        // Check if gaze point is within image bounds
        if (!this.isPointInImageBounds(gazeData.x, gazeData.y)) {
            console.log('Gaze data ignored - outside image bounds:', { x: gazeData.x, y: gazeData.y });
            return;
        }

        // Update position from gaze data (already in canvas coordinates)
        this.mousePosition = { x: gazeData.x, y: gazeData.y };
        this.mouseSpeed = gazeData.speed || 0;

        // Update gaze status to show it's active
        this.updateGazeStatus('Recording');

        // Send data to server immediately (this will also update UI)
        this.sendMouseData();
        
        console.log('Gaze data sent to server:', {
            x: this.mousePosition.x,
            y: this.mousePosition.y,
            speed: this.mouseSpeed,
            useGazeTracking: this.useGazeTracking
        });
    }

    handleMouseMove(event) {
        // Only process mouse events if not using gaze tracking
        if (this.useGazeTracking) {
            return;
        }

        if (!this.isTracking || !this.isConnected) {
            console.log('Mouse move ignored:', {
                isTracking: this.isTracking,
                isConnected: this.isConnected,
                useGazeTracking: this.useGazeTracking
            });
            return;
        }

        const rect = this.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        // Check if point is within image bounds (rendered image rect)
        if (!this.isPointInImageBounds(x, y)) {
            console.log('Mouse move ignored - outside image bounds:', { x, y });
            return;
        }

        // Calculate mouse speed
        const currentTime = Date.now();
        const timeDiff = currentTime - this.lastMouseTime;
        const distance = Math.sqrt(
            Math.pow(x - this.lastMousePosition.x, 2) + 
            Math.pow(y - this.lastMousePosition.y, 2)
        );
        
        this.mouseSpeed = timeDiff > 0 ? distance / (timeDiff / 1000) : 0;

        // Update position
        this.lastMousePosition = this.mousePosition;
        this.mousePosition = { x, y };
        this.lastMouseTime = currentTime;

        // Send data to server (this will also update UI)
        this.sendMouseData();
        
        console.log('Mouse data sent:', { x, y, speed: this.mouseSpeed });
    }

    sendMouseData() {
        // Always update UI first
        this.updateMouseInfo();
        
        // Only send data to server if user is registered
        if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN || !this.userId) {
            console.log('Mouse data updated in UI but not sent to server:', {
                websocket: !!this.websocket,
                readyState: this.websocket?.readyState,
                userId: this.userId
            });
            return;
        }

        // Map current mousePosition to server's base coordinates using current render transform
        const mapped = this.mapToBaseCoordinates(this.mousePosition.x, this.mousePosition.y);
        const clampedX = Math.max(0, Math.min(this.baseWidth - 1, mapped.x));
        const clampedY = Math.max(0, Math.min(this.baseHeight - 1, mapped.y));

        const data = {
            type: this.useGazeTracking ? 'gaze_data' : 'mouse_data',
            x: clampedX,
            y: clampedY,
            speed: this.mouseSpeed,
            confidence: 1.0
        };
        
        // Add confidence for gaze data
        if (this.useGazeTracking && this.gazeTracker && this.gazeTracker.getGazeData()) {
            data.confidence = this.gazeTracker.getGazeData().confidence || 1.0;
        }
        
        console.log('Sending data to server:', {
            canvasPoint: { x: this.mousePosition.x, y: this.mousePosition.y },
            mappedBase: { x: mapped.x, y: mapped.y },
            clampedBase: { x: clampedX, y: clampedY },
            baseSize: { width: this.baseWidth, height: this.baseHeight },
            renderTransform: this.renderTransform,
            type: data.type
        });
        this.websocket.send(JSON.stringify(data));
    }

    sendPromptToServer() {
        if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN || !this.userId) return;

        const data = {
            type: 'prompt_change',
            prompt_idx: this.currentPrompt
        };
        this.websocket.send(JSON.stringify(data));
    }

    handleServerMessage(data) {
        try {
            const message = JSON.parse(data);
            
            switch (message.type) {
                case 'user_registered':
                    this.userId = message.userId;
                    // Convert server bounds format (array) to frontend format (object)
                    if (message.imageBounds && Array.isArray(message.imageBounds) && message.imageBounds.length === 4) {
                        this.imageBounds = {
                            offsetX: message.imageBounds[0],
                            offsetY: message.imageBounds[1],
                            width: message.imageBounds[2],
                            height: message.imageBounds[3]
                        };
                    } else {
                        this.imageBounds = null;
                    }
                    console.log('User registered with ID:', this.userId);
                    console.log('Image bounds:', this.imageBounds);
                    this.updateUserInfo();
                    this.updateTrackingMode();
                    this.updateMouseInfo();
                    this.sendPromptToServer(); // Send initial prompt
                    
                    // Auto-enter fullscreen mode after successful registration
                    setTimeout(() => {
                        this.enterFullscreen();
                    }, 1000); // Small delay to ensure image is loaded
                    break;
                    
                case 'image':
                    this.handleImageData(message.data);
                    break;
                    
                case 'reset_confirmed':
                    console.log('Server confirmed reset');
                    break;
                    
                case 'server_stats':
                    this.serverStats = message.stats;
                    this.updateServerStats();
                    break;
                    
                default:
                    console.log('Unknown message type:', message.type);
            }
        } catch (error) {
            console.error('Error processing server message:', error);
        }
    }

    handleImageData(data) {
        try {
            const img = new Image();
            img.onload = () => {
                this.displayImage(img);
                this.addToHistory(img);
            };
            img.src = data;
        } catch (error) {
            console.error('Image data processing error:', error);
        }
    }

    displayImage(img) {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    
        if (this.isFullscreen) {
            // Fullscreen mode: "contain" the image within the canvas (window)
            const canvasWidth = this.canvas.width;
            const canvasHeight = this.canvas.height;
            const imageAspect = img.width / img.height;
            const canvasAspect = canvasWidth / canvasHeight;
    
            let displayWidth, displayHeight, x, y;
    
            if (imageAspect > canvasAspect) {
                // Image is wider than the canvas, so fit to width
                displayWidth = canvasWidth;
                displayHeight = displayWidth / imageAspect;
                x = 0;
                y = (canvasHeight - displayHeight) / 2;
            } else {
                // Image is taller than or has the same aspect as the canvas, so fit to height
                displayHeight = canvasHeight;
                displayWidth = displayHeight * imageAspect;
                y = 0;
                x = (canvasWidth - displayWidth) / 2;
            }
    
            this.ctx.drawImage(img, x, y, displayWidth, displayHeight);
    
            // Update render transform for accurate coordinate mapping
            this.renderTransform = { x, y, width: displayWidth, height: displayHeight };
            this.imageBounds = { offsetX: x, offsetY: y, width: displayWidth, height: displayHeight };
    
        } else {
            // Non-fullscreen mode: stretch image to fit the fixed-size canvas
            this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
            
            // The image fills the entire canvas in this mode
            this.renderTransform = { x: 0, y: 0, width: this.canvas.width, height: this.canvas.height };
            this.imageBounds = { offsetX: 0, offsetY: 0, width: this.canvas.width, height: this.canvas.height };
    
            this.drawInteractionAreaIndicator();
        }
    }
    
    drawInteractionAreaIndicator() {
        if (!this.imageBounds) {
            return;
        }
        
        const { offsetX, offsetY, width, height } = this.imageBounds;
        
        // Draw a subtle border around the interactive area
        this.ctx.save();
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([5, 5]);
        this.ctx.strokeRect(offsetX, offsetY, width, height);
        this.ctx.restore();
        
        // Add a subtle overlay to non-interactive areas
        this.ctx.save();
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
        
        // Top area
        if (offsetY > 0) {
            this.ctx.fillRect(0, 0, this.canvas.width, offsetY);
        }
        // Bottom area
        if (offsetY + height < this.canvas.height) {
            this.ctx.fillRect(0, offsetY + height, this.canvas.width, this.canvas.height - offsetY - height);
        }
        // Left area
        if (offsetX > 0) {
            this.ctx.fillRect(0, offsetY, offsetX, height);
        }
        // Right area
        if (offsetX + width < this.canvas.width) {
            this.ctx.fillRect(offsetX + width, offsetY, this.canvas.width - offsetX - width, height);
        }
        
        this.ctx.restore();
    }

    addToHistory(img) {
        const historyItem = {
            image: img,
            timestamp: new Date().toLocaleTimeString()
        };

        this.historyImages.unshift(historyItem);
        
        // Limit history count
        if (this.historyImages.length > 10) {
            this.historyImages.pop();
        }

        this.updateHistoryDisplay();
    }

    updateHistoryDisplay() {
        const container = document.getElementById('history-container');
        container.innerHTML = '';

        this.historyImages.forEach((item, index) => {
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            historyItem.innerHTML = `
                <img src="${item.image.src}" alt="History image ${index + 1}">
                <div class="timestamp">${item.timestamp}</div>
            `;
            
            historyItem.addEventListener('click', () => {
                this.displayImage(item.image);
            });

            container.appendChild(historyItem);
        });
    }

    resetSystem() {
        console.log('Resetting system');
        
        // Send reset signal to server
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN && this.userId) {
            const data = {
                type: 'reset'
            };
            this.websocket.send(JSON.stringify(data));
        }

        // Reset local state
        this.historyImages = [];
        this.updateHistoryDisplay();
        this.setupCanvas();
    }

    toggleFullscreen() {
        if (!this.isFullscreen) {
            this.enterFullscreen();
        } else {
            this.exitFullscreen();
        }
    }

    enterFullscreen() {
        const canvas = this.canvas;
        
        if (canvas.requestFullscreen) {
            canvas.requestFullscreen();
        } else if (canvas.webkitRequestFullscreen) {
            canvas.webkitRequestFullscreen();
        } else if (canvas.msRequestFullscreen) {
            canvas.msRequestFullscreen();
        }

        this.isFullscreen = true;
        
        // Resize canvas to full screen
        this.resizeCanvasForFullscreen();
        
        this.addFullscreenControls();
        document.getElementById('fullscreen-btn').textContent = 'Exit Fullscreen';
    }

    exitFullscreen() {
        if (document.exitFullscreen) {
            document.exitFullscreen();
        } else if (document.webkitExitFullscreen) {
            document.webkitExitFullscreen();
        } else if (document.msExitFullscreen) {
            document.msExitFullscreen();
        }

        this.isFullscreen = false;
        
        // Resize canvas back to normal
        this.resizeCanvasForNormal();
        
        this.removeFullscreenControls();
        document.getElementById('fullscreen-btn').textContent = 'Fullscreen Display';
    }

    updateConnectionStatus(status) {
        this.connectionStatus = status;
        const statusElement = document.getElementById('connection-status');
        const connectBtn = document.getElementById('connect-btn');
        const disconnectBtn = document.getElementById('disconnect-btn');

        statusElement.className = `status-indicator ${status}`;

        switch (status) {
            case 'connected':
                statusElement.querySelector('.status-text').textContent = 'Connected';
                connectBtn.disabled = true;
                disconnectBtn.disabled = false;
                break;
            case 'connecting':
                statusElement.querySelector('.status-text').textContent = 'Connecting...';
                connectBtn.disabled = true;
                disconnectBtn.disabled = true;
                break;
            case 'disconnected':
                statusElement.querySelector('.status-text').textContent = 'Disconnected';
                connectBtn.disabled = false;
                disconnectBtn.disabled = true;
                break;
        }
    }

    updateUserInfo() {
        const userInfoElement = document.getElementById('user-info');
        if (userInfoElement) {
            if (this.userId) {
                userInfoElement.innerHTML = `
                    <div class="user-id">User ID: ${this.userId.substring(0, 8)}...</div>
                    <div class="connection-status">Status: ${this.connectionStatus}</div>
                `;
            } else {
                userInfoElement.innerHTML = `
                    <div class="user-id">Not connected</div>
                    <div class="connection-status">Status: ${this.connectionStatus}</div>
                `;
            }
        }
    }

    updateServerStats() {
        const statsElement = document.getElementById('server-stats');
        if (statsElement) {
            statsElement.innerHTML = `
                <div class="total-users">Total Users: ${this.serverStats.totalUsers}</div>
                <div class="active-users">Active Users: ${this.serverStats.activeUsers}</div>
            `;
        }
    }

    updateMouseInfo() {
        const positionElement = document.getElementById('mouse-position');
        const speedElement = document.getElementById('mouse-speed');
        
        if (positionElement) {
            positionElement.textContent = `(${Math.round(this.mousePosition.x)}, ${Math.round(this.mousePosition.y)})`;
        }
        
        if (speedElement) {
            speedElement.textContent = `${Math.round(this.mouseSpeed)}`;
        }
        
        // Also update tracking mode when mouse info is updated
        this.updateTrackingMode();
        
        // Update fullscreen info if in fullscreen mode
        this.updateFullscreenInfo();
    }

    updateTrackingMode() {
        const trackingModeElement = document.getElementById('tracking-mode');
        if (trackingModeElement) {
            if (this.useGazeTracking) {
                trackingModeElement.textContent = 'Gaze Tracking';
                trackingModeElement.className = 'tracking-mode-gaze';
            } else {
                trackingModeElement.textContent = 'Mouse Tracking';
                trackingModeElement.className = 'tracking-mode-mouse';
            }
        }
        
        // Update checkbox states to reflect current mode
        const mouseToggle = document.getElementById('tracking-toggle');
        const gazeToggle = document.getElementById('gaze-tracking-toggle');
        
        if (mouseToggle) {
            mouseToggle.checked = this.isTracking;
        }
        
        if (gazeToggle) {
            gazeToggle.checked = this.useGazeTracking;
        }
    }

    openSettings() {
        const modal = document.getElementById('settings-modal');
        const apiKeyInput = document.getElementById('api-key-input');
        const confidenceThreshold = document.getElementById('confidence-threshold');
        const confidenceValue = document.getElementById('confidence-value');
        const smoothingEnabled = document.getElementById('smoothing-enabled');
        const mouseFallbackEnabled = document.getElementById('mouse-fallback-enabled');

        // Load current settings
        apiKeyInput.value = getGazeRecorderAPIKey() || '';
        confidenceThreshold.value = CONFIG.GAZE_RECORDER.CONFIDENCE_THRESHOLD;
        confidenceValue.textContent = CONFIG.GAZE_RECORDER.CONFIDENCE_THRESHOLD;
        smoothingEnabled.checked = CONFIG.TRACKING.SMOOTHING_ENABLED;
        mouseFallbackEnabled.checked = CONFIG.TRACKING.MOUSE_FALLBACK_ENABLED;

        // Show modal
        modal.style.display = 'block';

        // Event listeners
        const closeBtn = modal.querySelector('.close');
        const saveApiKeyBtn = document.getElementById('save-api-key');

        closeBtn.onclick = () => {
            modal.style.display = 'none';
        };

        window.onclick = (event) => {
            if (event.target === modal) {
                modal.style.display = 'none';
            }
        };

        confidenceThreshold.oninput = () => {
            confidenceValue.textContent = confidenceThreshold.value;
        };

        saveApiKeyBtn.onclick = () => {
            const apiKey = apiKeyInput.value.trim();
            if (apiKey) {
                setGazeRecorderAPIKey(apiKey);
                if (this.gazeTracker) {
                    this.gazeTracker.config.apiKey = apiKey;
                }
                alert('API key saved successfully!');
            } else {
                alert('Please enter a valid API key.');
            }
        };

        // Save settings on change
        confidenceThreshold.onchange = () => {
            CONFIG.GAZE_RECORDER.CONFIDENCE_THRESHOLD = parseFloat(confidenceThreshold.value);
            if (this.gazeTracker) {
                this.gazeTracker.config.confidenceThreshold = CONFIG.GAZE_RECORDER.CONFIDENCE_THRESHOLD;
            }
            saveConfig();
        };

        smoothingEnabled.onchange = () => {
            CONFIG.TRACKING.SMOOTHING_ENABLED = smoothingEnabled.checked;
            saveConfig();
        };

        mouseFallbackEnabled.onchange = () => {
            CONFIG.TRACKING.MOUSE_FALLBACK_ENABLED = mouseFallbackEnabled.checked;
            saveConfig();
        };
    }

    updateCurrentPrompt() {
        const prompts = [
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
            "East African farmlands overrun by swarms of locusts, devastating crops and causing despair."
        ];

        const currentPromptText = prompts[this.currentPrompt] || '';
        document.getElementById('current-prompt').textContent = currentPromptText;
    }

    updateUI() {
        this.updateCurrentPrompt();
        this.updateMouseInfo();
        this.updateUserInfo();
        this.updateTrackingMode();
    }

    handleResize() {
        // Handle window resize
        if (this.isFullscreen) {
            // Resize canvas for fullscreen
            this.resizeCanvasForFullscreen();
        }
    }

    showLoading(show = true) {
        const loadingIndicator = document.getElementById('loading-indicator');
        const overlay = document.querySelector('.canvas-overlay');
        
        if (show) {
            loadingIndicator.style.display = 'flex';
            overlay.classList.add('show');
        } else {
            loadingIndicator.style.display = 'none';
            overlay.classList.remove('show');
        }
    }

    addFullscreenControls() {
        // Create fullscreen control overlay
        const overlay = document.createElement('div');
        overlay.id = 'fullscreen-overlay';
        overlay.className = 'fullscreen-overlay';
        
        // Add control buttons
        overlay.innerHTML = `
            <div class="fullscreen-controls">
                <div class="control-info">
                    <div class="interaction-hint">Move your mouse or gaze to interact with the landscape</div>
                    <div class="tracking-status">
                        <span id="fullscreen-tracking-mode">Mouse Tracking</span>
                        <span id="fullscreen-position">(0, 0)</span>
                    </div>
                    <div class="keyboard-hints">
                        <span>ESC: Exit Fullscreen</span>
                        <span>R: Reset</span>
                        <span>D: Disconnect</span>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(overlay);
        
        // Update fullscreen tracking info
        this.updateFullscreenInfo();
    }

    removeFullscreenControls() {
        const overlay = document.getElementById('fullscreen-overlay');
        if (overlay) {
            overlay.remove();
        }
    }

    updateFullscreenInfo() {
        if (!this.isFullscreen) return;
        
        const trackingModeElement = document.getElementById('fullscreen-tracking-mode');
        const positionElement = document.getElementById('fullscreen-position');
        
        if (trackingModeElement) {
            trackingModeElement.textContent = this.useGazeTracking ? 'Gaze Tracking' : 'Mouse Tracking';
        }
        
        if (positionElement) {
            positionElement.textContent = `(${Math.round(this.mousePosition.x)}, ${Math.round(this.mousePosition.y)})`;
        }
    }

    resizeCanvasForFullscreen() {
        if (!this.isFullscreen) return;
        
        // Set canvas to full screen dimensions
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
        
        // Add fullscreen class to canvas container
        this.canvas.classList.add('fullscreen');
        
        // Redraw current image if available
        if (this.historyImages.length > 0) {
            const currentImage = this.historyImages[0].image;
            this.displayImage(currentImage);
        }
    }

    resizeCanvasForNormal() {
        if (this.isFullscreen) return;
        
        // Reset canvas to original dimensions
        this.canvas.width = 1047;
        this.canvas.height = 1544;
        
        // Remove fullscreen class
        this.canvas.classList.remove('fullscreen');
        
        // Redraw current image if available
        if (this.historyImages.length > 0) {
            const currentImage = this.historyImages[0].image;
            this.displayImage(currentImage);
        }
    }
}

// Start application
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing MultiUserShanshuiApp...');
    try {
        window.multiUserShanshuiApp = new MultiUserShanshuiApp();
        console.log('MultiUserShanshuiApp initialized successfully');
    } catch (error) {
        console.error('Failed to initialize MultiUserShanshuiApp:', error);
    }
});

// Export class for use elsewhere
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MultiUserShanshuiApp;
}