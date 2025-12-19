class GazeTracker {
    constructor() {
        this.isTracking = false;
        this.gazeData = null;
        this.lastGazePosition = { x: 0, y: 0 };
        this.gazeSpeed = 0;
        this.lastGazeTime = Date.now();
        this.isCalibrated = false;
        this.gazeRecorderAPI = null;
        
        this.config = {
            apiKey: null,
            sampleRate: CONFIG.GAZE_RECORDER.SAMPLE_RATE,
            smoothingFactor: CONFIG.GAZE_RECORDER.SMOOTHING_FACTOR,
            confidenceThreshold: CONFIG.GAZE_RECORDER.CONFIDENCE_THRESHOLD,
            maxGazeSpeed: CONFIG.GAZE_RECORDER.MAX_GAZE_SPEED
        };
        
        this.init();
    }
    
    async init() {
        try {
            await this.initializeGazeRecorder();
            this.setupEventListeners();
            console.log('Gaze tracker initialized');
        } catch (error) {
            console.error('Gaze tracker init failed:', error);
            this.fallbackToMouseTracking();
        }
    }
    
    async initializeGazeRecorder() {
        try {
            // Check if GazeRecorderAPI is available
            if (!window.GazeRecorderAPI) {
                console.warn('GazeRecorderAPI not available, using fallback');
                this.updateGazeStatus('API not available');
                this.fallbackToMouseTracking();
                return;
            }
            
            // Initialize GazeRecorderAPI
            this.gazeRecorderAPI = window.GazeRecorderAPI;
            
            // Request camera permission first
            await this.requestCameraPermission();
            
            // Set up gaze data polling
            this.setupGazeDataPolling();
            
            console.log('GazeRecorderAPI initialized successfully');
            this.updateGazeStatus('Ready');
            
        } catch (error) {
            console.error('Failed to initialize GazeRecorderAPI:', error);
            this.updateGazeStatus('Failed to initialize');
            this.fallbackToMouseTracking();
        }
    }
    
    async requestCameraPermission() {
        try {
            console.log('Requesting camera permission...');
            this.updateGazeStatus('Requesting camera permission...');
            
            // Check if we already have camera permission
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(device => device.kind === 'videoinput');
            
            if (videoDevices.length === 0) {
                throw new Error('No camera found');
            }
            
            console.log('Found video devices:', videoDevices.map(d => d.label || 'Unknown camera'));
            
            // Request camera access with specific constraints
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 1280, min: 640 },
                    height: { ideal: 720, min: 480 },
                    facingMode: 'user',
                    frameRate: { ideal: 30 }
                },
                audio: false
            });
            
            console.log('Camera permission granted, stream active:', stream.active);
            this.updateGazeStatus('Camera ready');
            
            // Keep the stream active for a moment to ensure GazeRecorderAPI can access it
            setTimeout(() => {
                stream.getTracks().forEach(track => {
                    console.log('Stopping camera track:', track.label);
                    track.stop();
                });
            }, 1000);
            
        } catch (error) {
            console.error('Camera permission error:', error);
            let errorMessage = 'Camera permission denied';
            
            if (error.name === 'NotAllowedError') {
                errorMessage = 'Please allow camera access in your browser';
            } else if (error.name === 'NotFoundError') {
                errorMessage = 'No camera found on your device';
            } else if (error.name === 'NotReadableError') {
                errorMessage = 'Camera is already in use by another application';
            }
            
            this.updateGazeStatus(errorMessage);
            throw new Error(errorMessage);
        }
    }
    
    
    loadFallback() {
        try {
            const fallbackScript = document.createElement('script');
            fallbackScript.src = 'gazerecorder_fallback.js';
            fallbackScript.async = true;
            
            fallbackScript.onload = () => {
                console.log('GazeRecorder fallback loaded successfully');
            };
            
            fallbackScript.onerror = (error) => {
                console.error('Failed to load fallback:', error);
            };
            
            document.head.appendChild(fallbackScript);
        } catch (error) {
            console.error('Error loading fallback:', error);
        }
    }
    
    setupGazeDataPolling() {
        // Poll for gaze data every 100ms
        this.gazeDataInterval = setInterval(() => {
            if (this.isTracking && window.GazeRecorderAPI) {
                try {
                    // Get current gaze data from GazeRecorderAPI
                    const sessionReplayData = window.GazeRecorderAPI.GetRecData();
                    
                    if (sessionReplayData && sessionReplayData.length > 0) {
                        // Process the latest gaze data point
                        const latestData = sessionReplayData[sessionReplayData.length - 1];
                        this.processGazeData(latestData);
                    } else if (sessionReplayData) {
                        console.log('Gaze data received but empty array:', sessionReplayData);
                    }
                } catch (error) {
                    console.warn('Error getting gaze data:', error);
                    // Track error count
                    this.errorCount = (this.errorCount || 0) + 1;
                    if (this.errorCount > 20) {
                        console.error('Too many gaze data errors, stopping polling');
                        this.updateGazeStatus('Data error');
                        clearInterval(this.gazeDataInterval);
                    }
                }
            }
        }, 100);
    }
    
    processGazeData(gazeData) {
        if (!this.isTracking) return;
        
        // GazeRecorderAPI data format: {x, y, timestamp}
        const { x, y, timestamp } = gazeData;
        
        // Get canvas element and its position
        const canvas = document.getElementById('main-canvas');
        if (!canvas) {
            console.warn('Canvas not found for gaze coordinate mapping');
            return;
        }
        
        // Use canvas dimensions directly instead of bounding rect
        const canvasWidth = canvas.width;  // 1047
        const canvasHeight = canvas.height; // 1544
        
        // Convert normalized coordinates (0-1) to canvas coordinates
        const canvasX = Math.max(0, Math.min(canvasWidth, x * canvasWidth));
        const canvasY = Math.max(0, Math.min(canvasHeight, y * canvasHeight));
        
        const currentTime = Date.now();
        const timeDiff = currentTime - this.lastGazeTime;
        const distance = Math.sqrt(
            Math.pow(canvasX - this.lastGazePosition.x, 2) + 
            Math.pow(canvasY - this.lastGazePosition.y, 2)
        );
        
        this.gazeSpeed = timeDiff > 0 ? distance / (timeDiff / 1000) : 0;
        
        // Apply smoothing
        const smoothingFactor = this.config.smoothingFactor;
        this.lastGazePosition.x = canvasX * smoothingFactor + this.lastGazePosition.x * (1 - smoothingFactor);
        this.lastGazePosition.y = canvasY * smoothingFactor + this.lastGazePosition.y * (1 - smoothingFactor);
        this.lastGazeTime = currentTime;
        
        this.gazeData = {
            x: this.lastGazePosition.x,
            y: this.lastGazePosition.y,
            speed: this.gazeSpeed,
            confidence: 1.0, // GazeRecorderAPI doesn't provide confidence, assume high confidence
            timestamp: timestamp || currentTime
        };
        
        console.log('Gaze data processed:', {
            original: { x, y },
            canvas: { width: canvasWidth, height: canvasHeight },
            mapped: { x: canvasX, y: canvasY },
            final: this.gazeData
        });
        this.updateGazeInfo();
        this.emitGazeEvent();
    }
    
    updateGazeInfo() {
        const gazePositionElement = document.getElementById('gaze-position');
        const gazeSpeedElement = document.getElementById('gaze-speed');
        
        if (gazePositionElement) {
            gazePositionElement.textContent = 
                `(${Math.round(this.lastGazePosition.x)}, ${Math.round(this.lastGazePosition.y)})`;
        }
        
        if (gazeSpeedElement) {
            gazeSpeedElement.textContent = `${Math.round(this.gazeSpeed)}`;
        }
    }
    
    updateGazeStatus(status) {
        const gazeStatusElement = document.getElementById('gaze-status');
        if (gazeStatusElement) {
            gazeStatusElement.textContent = status;
            
            // Add CSS classes for visual feedback
            gazeStatusElement.className = '';
            if (status.includes('Ready') || status.includes('ready')) {
                gazeStatusElement.classList.add('ready');
            } else if (status.includes('Recording') || status.includes('recording')) {
                gazeStatusElement.classList.add('recording');
            } else if (status.includes('error') || status.includes('Error') || status.includes('denied') || status.includes('Failed')) {
                gazeStatusElement.classList.add('error');
            }
        }
    }
    
    emitGazeEvent() {
        const gazeEvent = new CustomEvent('gazeData', {
            detail: this.gazeData
        });
        window.dispatchEvent(gazeEvent);
    }
    
    async toggleGazeTracking(enabled) {
        this.isTracking = enabled;
        
        if (enabled) {
            await this.startGazeTracking();
        } else {
            this.stopGazeTracking();
        }
    }
    
    async startGazeTracking() {
        if (!window.GazeRecorderAPI) {
            console.warn('GazeRecorderAPI not available');
            this.updateGazeStatus('API not available');
            return;
        }
        
        try {
            console.log('Starting gaze tracking...');
            this.updateGazeStatus('Starting...');
            
            // Ensure camera permission is granted
            await this.requestCameraPermission();
            
            // Wait a moment for camera to be ready
            await new Promise(resolve => setTimeout(resolve, 500));
            
            // Start recording on current webpage using the official API
            console.log('Calling GazeRecorderAPI.Rec()...');
            window.GazeRecorderAPI.Rec();
            
            // Mark as calibrated since GazeRecorderAPI handles calibration automatically
            this.isCalibrated = true;
            
            this.updateGazeStatus('Recording');
            console.log('Gaze tracking started successfully');
            
            // Check if recording is actually active
            setTimeout(() => {
                try {
                    const sessionReplayData = window.GazeRecorderAPI.GetRecData();
                    console.log('Initial gaze data check:', sessionReplayData);
                } catch (error) {
                    console.warn('Could not get initial gaze data:', error);
                }
            }, 1000);
            
        } catch (error) {
            console.error('Failed to start gaze tracking:', error);
            this.updateGazeStatus('Failed to start: ' + error.message);
        }
    }
    
    stopGazeTracking() {
        if (window.GazeRecorderAPI) {
            window.GazeRecorderAPI.StopRec();
            console.log('Gaze tracking stopped');
            this.updateGazeStatus('Ready');
        }
    }
    
    fallbackToMouseTracking() {
        console.warn('Using mouse tracking fallback');
        
        document.addEventListener('mousemove', (e) => {
            if (!this.isTracking) return;
            
            const rect = document.getElementById('main-canvas')?.getBoundingClientRect();
            if (!rect) return;
            
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            this.gazeData = {
                x: x,
                y: y,
                speed: 0,
                confidence: 1.0,
                timestamp: Date.now()
            };
            
            this.emitGazeEvent();
        });
    }
    
    getGazeData() {
        return this.gazeData;
    }
    
    destroy() {
        if (window.GazeRecorderAPI) {
            window.GazeRecorderAPI.StopRec();
        }
        
        if (this.gazeDataInterval) {
            clearInterval(this.gazeDataInterval);
            this.gazeDataInterval = null;
        }
        
        this.isTracking = false;
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = GazeTracker;
} 