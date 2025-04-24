document.addEventListener('DOMContentLoaded', () => {
    const videoFeed = document.getElementById('videoFeed');
    const startCameraBtn = document.getElementById('startCameraBtn');
    const stopCameraBtn = document.getElementById('stopCameraBtn');
    const recordBtn = document.getElementById('recordBtn');
    const status = document.getElementById('status');

    let isRecording = false;
    const API_BASE_URL = 'http://localhost:5001';

    // Check camera status on page load
    checkCameraStatus();

    // Function to check camera status
    async function checkCameraStatus() {
        try {
            const response = await fetch(`${API_BASE_URL}/camera_status`);
            const data = await response.json();

            if (data.status === 'running') {
                // Camera is already running, update UI
                videoFeed.src = `${API_BASE_URL}/video_feed`;
                startCameraBtn.style.display = 'none';
                stopCameraBtn.style.display = 'inline-block';
                recordBtn.style.display = 'inline-block';
                status.textContent = 'Camera is running with violence detection';
                status.className = 'status-message success';
            }
        } catch (error) {
            console.error('Error checking camera status:', error);
        }
    }

    startCameraBtn.addEventListener('click', async () => {
        try {
            // Show loading state
            startCameraBtn.disabled = true;
            status.textContent = 'Starting camera and loading models...';
            status.className = 'status-message loading';

            const response = await fetch(`${API_BASE_URL}/start_camera`, {
                method: 'POST'
            });

            const data = await response.json();

            if (response.ok) {
                // Add a timestamp to prevent caching
                videoFeed.src = `${API_BASE_URL}/video_feed?t=${new Date().getTime()}`;
                startCameraBtn.style.display = 'none';
                stopCameraBtn.style.display = 'inline-block';
                recordBtn.style.display = 'inline-block';
                status.textContent = data.message || 'Camera started with violence detection';
                status.className = 'status-message success';
            } else {
                throw new Error(data.message || 'Failed to start camera');
            }
        } catch (error) {
            console.error('Error:', error);
            status.textContent = error.message || 'Failed to start camera';
            status.className = 'status-message error';
            startCameraBtn.disabled = false;
        }
    });

    stopCameraBtn.addEventListener('click', async () => {
        try {
            stopCameraBtn.disabled = true;
            status.textContent = 'Stopping camera...';

            const response = await fetch(`${API_BASE_URL}/stop_camera`, {
                method: 'POST'
            });

            const data = await response.json();

            if (response.ok) {
                videoFeed.src = '';
                startCameraBtn.style.display = 'inline-block';
                startCameraBtn.disabled = false;
                stopCameraBtn.style.display = 'none';
                recordBtn.style.display = 'none';
                recordBtn.textContent = 'Start Recording';
                recordBtn.classList.remove('recording');
                isRecording = false;
                status.textContent = data.message || 'Camera stopped';
                status.className = 'status-message';
            } else {
                throw new Error(data.message || 'Failed to stop camera');
            }
        } catch (error) {
            console.error('Error:', error);
            status.textContent = error.message || 'Failed to stop camera';
            status.className = 'status-message error';
            stopCameraBtn.disabled = false;
        }
    });

    recordBtn.addEventListener('click', async () => {
        try {
            recordBtn.disabled = true;

            if (!isRecording) {
                status.textContent = 'Starting recording...';

                const response = await fetch(`${API_BASE_URL}/start_recording`, {
                    method: 'POST'
                });

                const data = await response.json();

                if (response.ok) {
                    isRecording = true;
                    recordBtn.textContent = 'Stop Recording';
                    recordBtn.classList.add('recording');
                    status.textContent = data.message || 'Recording violence detection...';
                    status.className = 'status-message recording';
                } else {
                    throw new Error(data.message || 'Failed to start recording');
                }
            } else {
                status.textContent = 'Stopping recording...';

                const response = await fetch(`${API_BASE_URL}/stop_recording`, {
                    method: 'POST'
                });

                const data = await response.json();

                if (response.ok) {
                    isRecording = false;
                    recordBtn.textContent = 'Start Recording';
                    recordBtn.classList.remove('recording');
                    status.textContent = data.message || 'Recording saved';
                    status.className = 'status-message success';
                } else {
                    throw new Error(data.message || 'Failed to stop recording');
                }
            }
        } catch (error) {
            console.error('Error:', error);
            status.textContent = error.message || 'Failed to toggle recording';
            status.className = 'status-message error';
        } finally {
            recordBtn.disabled = false;
        }
    });

    // Add event listener for video feed errors
    videoFeed.addEventListener('error', (e) => {
        console.error('Video feed error:', e);
        status.textContent = 'Error loading video feed. Please try again.';
        status.className = 'status-message error';
        startCameraBtn.style.display = 'inline-block';
        startCameraBtn.disabled = false;
        stopCameraBtn.style.display = 'none';
        recordBtn.style.display = 'none';

        // Try to stop the camera on the backend
        fetch(`${API_BASE_URL}/stop_camera`, {
            method: 'POST'
        }).catch(err => console.error('Error stopping camera after video feed error:', err));
    });

    // Add a retry button
    const retryBtn = document.createElement('button');
    retryBtn.textContent = 'Retry Connection';
    retryBtn.className = 'control-btn';
    retryBtn.style.display = 'none';
    document.querySelector('.controls').appendChild(retryBtn);

    retryBtn.addEventListener('click', async () => {
        retryBtn.style.display = 'none';
        startCameraBtn.click();
    });

    // Function to check server health
    async function checkServerHealth() {
        try {
            const response = await fetch(`${API_BASE_URL}/health`);
            if (response.ok) {
                console.log('Server is healthy');
                return true;
            } else {
                console.error('Server health check failed');
                status.textContent = 'Server is not responding properly. Please check the backend server.';
                status.className = 'status-message error';
                retryBtn.style.display = 'inline-block';
                return false;
            }
        } catch (error) {
            console.error('Server health check error:', error);
            status.textContent = 'Cannot connect to server. Please check if the backend is running.';
            status.className = 'status-message error';
            retryBtn.style.display = 'inline-block';
            return false;
        }
    }

    // Check server health on page load
    checkServerHealth();
});

