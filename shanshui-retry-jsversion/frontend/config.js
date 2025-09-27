// Configuration file for the application
const CONFIG = {
            // Gaze Recorder API Configuration
        GAZE_RECORDER: {
            SAMPLE_RATE: 60, // Hz
            SMOOTHING_FACTOR: 0.3,
            CONFIDENCE_THRESHOLD: 0.7,
            MAX_GAZE_SPEED: 1000, // px/s
            POLLING_INTERVAL: 100, // ms
            CALIBRATION_POINTS: [
                { x: 0.1, y: 0.1 },
                { x: 0.9, y: 0.1 },
                { x: 0.5, y: 0.5 },
                { x: 0.1, y: 0.9 },
                { x: 0.9, y: 0.9 }
            ]
        },
    
    // WebSocket Configuration
    WEBSOCKET: {
        HOST: 'localhost',
        PORT: 12346,
        RECONNECT_INTERVAL: 5000, // ms
        MAX_RECONNECT_ATTEMPTS: 10
    },
    
    // Image Processing Configuration
    IMAGE_PROCESSING: {
        CANVAS_WIDTH: 1047,
        CANVAS_HEIGHT: 1544,
        DISPLAY_WIDTH: 800,
        DISPLAY_HEIGHT: 600,
        JPEG_QUALITY: 85
    },
    
    // Tracking Configuration
    TRACKING: {
        MOUSE_FALLBACK_ENABLED: true,
        GAZE_TRACKING_ENABLED: true,
        CONFIDENCE_THRESHOLD: 0.7,
        SMOOTHING_ENABLED: true
    },
    
    // UI Configuration
    UI: {
        HISTORY_MAX_ITEMS: 10,
        UPDATE_INTERVAL: 100, // ms
        STATS_UPDATE_INTERVAL: 5000 // ms
    }
};

// Load configuration from localStorage
function loadConfig() {
    try {
        const savedConfig = localStorage.getItem('shanshuiConfig');
        if (savedConfig) {
            const parsed = JSON.parse(savedConfig);
            Object.assign(CONFIG, parsed);
        }
    } catch (error) {
        console.warn('Failed to load saved configuration:', error);
    }
}

// Save configuration to localStorage
function saveConfig() {
    try {
        localStorage.setItem('shanshuiConfig', JSON.stringify(CONFIG));
    } catch (error) {
        console.warn('Failed to save configuration:', error);
    }
}

// Get API key with fallback
function getGazeRecorderAPIKey() {
    if (CONFIG.GAZE_RECORDER.API_KEY) {
        return CONFIG.GAZE_RECORDER.API_KEY;
    }
    
    // Try to get from localStorage
    const savedKey = localStorage.getItem('gazeRecorderAPIKey');
    if (savedKey) {
        CONFIG.GAZE_RECORDER.API_KEY = savedKey;
        return savedKey;
    }
    
    return null;
}

// Set API key
function setGazeRecorderAPIKey(apiKey) {
    CONFIG.GAZE_RECORDER.API_KEY = apiKey;
    localStorage.setItem('gazeRecorderAPIKey', apiKey);
    saveConfig();
}

// Initialize configuration
loadConfig();

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { CONFIG, loadConfig, saveConfig, getGazeRecorderAPIKey, setGazeRecorderAPIKey };
} 