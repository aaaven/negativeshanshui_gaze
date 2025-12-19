// GazeRecorder Fallback Implementation
// This is a fallback implementation when the external GazeRecorderAPI.js is not available

(function() {
    'use strict';
    
    // Check if GazeRecorderAPI is already loaded
    if (window.GazeRecorderAPI) {
        console.log('GazeRecorderAPI already loaded, skipping fallback');
        return;
    }
    
    // Fallback implementation
    window.GazeRecorderAPI = {
        isRecording: false,
        data: [],
        startTime: null,
        
        // Start recording
        Rec: function() {
            console.log('GazeRecorder Fallback: Starting recording');
            this.isRecording = true;
            this.startTime = Date.now();
            this.data = [];
            
            // Simulate gaze data collection
            this.startSimulation();
        },
        
        // Stop recording
        StopRec: function() {
            console.log('GazeRecorder Fallback: Stopping recording');
            this.isRecording = false;
            this.stopSimulation();
        },
        
        // Get recording data (SessionReplayData)
        GetRecData: function() {
            return this.data;
        },
        
        // Simulate gaze data
        startSimulation: function() {
            this.simulationInterval = setInterval(() => {
                if (this.isRecording) {
                    // Generate simulated gaze data
                    const gazePoint = {
                        x: Math.random(), // Random X position (0-1)
                        y: Math.random(), // Random Y position (0-1)
                        timestamp: Date.now()
                    };
                    
                    this.data.push(gazePoint);
                    
                    // Limit data points to prevent memory issues
                    if (this.data.length > 1000) {
                        this.data = this.data.slice(-500);
                    }
                }
            }, 100); // 10Hz simulation
        },
        
        stopSimulation: function() {
            if (this.simulationInterval) {
                clearInterval(this.simulationInterval);
                this.simulationInterval = null;
            }
        }
    };
    
    console.log('GazeRecorder Fallback loaded');
    
})();


