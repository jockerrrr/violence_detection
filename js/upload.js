document.addEventListener('DOMContentLoaded', () => {
    // Check if user is logged in
    const currentUser = JSON.parse(localStorage.getItem('currentUser'));
    if (!currentUser) {
        window.location.href = 'login.html';
        return;
    }

    // Display welcome message
    const welcomeMessage = document.getElementById('welcomeMessage');
    if (welcomeMessage) {
        welcomeMessage.textContent = `Welcome, ${currentUser.fullName}! Upload a video for analysis.`;
    }
    const fileInput = document.getElementById('fileInput');
    const dropZone = document.getElementById('dropZone');
    const progressBar = document.getElementById('uploadProgress');
    const progressText = document.getElementById('progressText');

    // Check if server is available
    async function checkServer() {
        try {
            const response = await fetch('http://localhost:5001/health', {
                method: 'GET',
                timeout: 5000
            });
            return response.ok;
        } catch (error) {
            console.error('Server check failed:', error);
            return false;
        }
    }

    async function handleFileUpload(e) {
        console.log('File upload initiated');
        const file = e.target.files[0];
        if (!file) return;
        console.log('File selected:', file.name, file.type, file.size);

        // Check if it's a video file
        if (!file.type.startsWith('video/')) {
            showError('Please upload a video file');
            return;
        }

        // Check server availability first
        console.log('Checking server availability...');
        const isServerAvailable = await checkServer();
        console.log('Server available:', isServerAvailable);
        if (!isServerAvailable) {
            showError('Server is not available. Please check if the backend server is running.');
            return;
        }

        // Show progress bar and loading state
        progressBar.style.display = 'block';
        const progressFill = document.querySelector('.progress-fill');
        const filenameSpan = document.querySelector('.filename');
        filenameSpan.textContent = `Uploading: ${file.name}`;
        progressText.textContent = 'Preparing upload...';
        progressFill.style.width = '0%';
        dropZone.style.opacity = '0.5';

        const formData = new FormData();
        formData.append('video', file);

        // Create an XMLHttpRequest to track upload progress
        const xhr = new XMLHttpRequest();
        xhr.upload.addEventListener('progress', (event) => {
            if (event.lengthComputable) {
                const percentComplete = Math.round((event.loaded / event.total) * 100);
                progressFill.style.width = `${percentComplete}%`;
                progressText.textContent = `${percentComplete}%`;
                console.log(`Upload progress: ${percentComplete}%`);
            }
        });

        // We'll use fetch for the actual upload, but track progress with XHR

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minute timeout

            console.log('Sending upload request to server...');
            const response = await fetch('http://localhost:5001/upload', {
                method: 'POST',
                body: formData,
                signal: controller.signal,
                headers: {
                    'Accept': 'application/json'
                },
                // Don't set Content-Type header - let the browser set it with the boundary
                // This is important for multipart/form-data uploads
                mode: 'cors',
                credentials: 'omit'
            });
            console.log('Upload response status:', response.status);

            clearTimeout(timeoutId);

            if (!response.ok) {
                let errorMessage = 'Upload failed';
                try {
                    const errorData = await response.json();
                    errorMessage = errorData.error || `HTTP error! status: ${response.status}`;
                } catch (e) {
                    errorMessage = `Server error: ${response.status}`;
                }
                throw new Error(errorMessage);
            }

            const data = await response.json();
            console.log('Response data:', data);

            // Hide progress bar and restore drop zone
            progressBar.style.display = 'none';
            dropZone.style.opacity = '1';

            // Show the video container
            const videoContainer = document.querySelector('.video-container');
            videoContainer.style.display = 'block';

            // Create video players for both original and processed videos
            // Use relative paths to access the videos directly from the backend
            const originalVideoUrl = `http://localhost:5001${data.original_video}`;
            const processedVideoUrl = `http://localhost:5001${data.processed_video}`;

            console.log('Original video URL:', originalVideoUrl);
            console.log('Processed video URL:', processedVideoUrl);
            console.log('All response data:', data);

            // Create video display without canvas overlay (bounding boxes are drawn directly on the video)
            videoContainer.innerHTML = `
                <h3>Analysis Results</h3>
                <div class="video-comparison">
                    <div class="video-wrapper">
                        <h3>Original Video</h3>
                        <video id="originalVideo" controls playsinline width="100%" src="${originalVideoUrl}"></video>
                    </div>
                    <div class="video-wrapper">
                        <h3>Processed Video with Detection</h3>
                        <video id="processedVideo" controls playsinline width="100%" src="${processedVideoUrl}"></video>
                    </div>
                </div>
                <div class="detection-info">
                    <h3>Detection Information</h3>
                    <div id="currentDetections"></div>
                </div>
                <div class="debug-info">
                    <h3>Debug Information</h3>
                    <p class="url-info">Original Video URL: <a href="${originalVideoUrl}" target="_blank">View</a></p>
                    <p class="url-info">Processed Video URL: <a href="${processedVideoUrl}" target="_blank">View</a></p>
                    <p><button id="downloadProcessedVideo" class="btn">Download Processed Video</button></p>
                </div>
            `;

            // Add download button functionality
            document.getElementById('downloadProcessedVideo').addEventListener('click', function () {
                window.open(processedVideoUrl, '_blank');
            });

            // Add detection information display
            const processedVideo = document.getElementById('processedVideo');
            const currentDetectionsDiv = document.getElementById('currentDetections');

            // Display all detections in a list
            function displayAllDetections() {
                if (!data.detections || data.detections.length === 0) {
                    currentDetectionsDiv.innerHTML = '<p>No detections found in this video</p>';
                    return;
                }

                // Add overall video classification at the top
                let isVideoViolent = data.is_violent;
                let violenceFrameCount = data.violence_frame_count || 0;
                let nonViolenceFrameCount = data.non_violence_frame_count || 0;
                let totalFrames = violenceFrameCount + nonViolenceFrameCount;

                let overallHtml = `
                    <div class="overall-classification ${isVideoViolent ? 'violence' : 'non-violence'}">
                        <h4>Overall Video Classification</h4>
                        <div class="classification-result">Violence: ${isVideoViolent}</div>
                        <div class="frame-stats">
                            <div>Violent Frames: ${violenceFrameCount} (${totalFrames > 0 ? Math.round((violenceFrameCount / totalFrames) * 100) : 0}%)</div>
                            <div>Non-Violent Frames: ${nonViolenceFrameCount} (${totalFrames > 0 ? Math.round((nonViolenceFrameCount / totalFrames) * 100) : 0}%)</div>
                        </div>
                    </div>
                `;

                // Add individual detections
                let html = overallHtml + '<div class="detection-list">';
                data.detections.forEach(detection => {
                    const colorClass = detection.is_violence ? 'violence' : 'non-violence';
                    html += `
                        <div class="detection-item ${colorClass}" data-time="${detection.timestamp}">
                            <span class="detection-class">Violence: ${detection.is_violence}</span>
                            <span class="detection-time">Time: ${detection.timestamp}</span>
                        </div>
                    `;
                });
                html += '</div>';
                currentDetectionsDiv.innerHTML = html;
            }

            // Display all detections immediately
            displayAllDetections();

            // Helper function to convert timestamp to seconds
            function timeToSeconds(timestamp) {
                const [hours, minutes, seconds] = timestamp.split(':').map(Number);
                return hours * 3600 + minutes * 60 + seconds;
            }

            // Update current detection highlight when video is playing
            processedVideo.addEventListener('timeupdate', function () {
                const currentTime = processedVideo.currentTime;

                // Highlight current detections in the list
                const detectionItems = document.querySelectorAll('.detection-item');
                detectionItems.forEach(item => {
                    const itemTime = timeToSeconds(item.getAttribute('data-time'));
                    if (Math.abs(itemTime - currentTime) < 0.5) {
                        item.classList.add('current');
                    } else {
                        item.classList.remove('current');
                    }
                });
            });

            // Add event listener to seek to detection time when clicking on a detection item
            currentDetectionsDiv.addEventListener('click', function (e) {
                const item = e.target.closest('.detection-item');
                if (item) {
                    const timestamp = item.getAttribute('data-time');
                    if (timestamp) {
                        processedVideo.currentTime = timeToSeconds(timestamp);
                        processedVideo.play();
                    }
                }
            });

        } catch (error) {
            console.error('Error:', error);
            // Check if it's an abort error (timeout)
            if (error.name === 'AbortError') {
                showError('Upload timed out. The video might be too large or the server is taking too long to respond.');
            } else {
                showError(error.message || 'Failed to upload video. Please try again.');
            }

            // Log additional information for debugging
            console.log('File details:', {
                name: file.name,
                type: file.type,
                size: `${(file.size / (1024 * 1024)).toFixed(2)} MB`
            });
        }
    }

    function showError(message) {
        progressBar.style.display = 'none';
        dropZone.style.opacity = '1';

        const videoContainer = document.querySelector('.video-container');
        videoContainer.innerHTML = `
            <div class="error-message">
                <p>⚠️ ${message}</p>
                <button onclick="location.reload()" class="btn">Try Again</button>
                <p class="error-help">If the problem persists:</p>
                <ul>
                    <li>Check if the backend server is running</li>
                    <li>Ensure the video file is not too large</li>
                    <li>Check your internet connection</li>
                    <li>Try a different browser</li>
                </ul>
            </div>
        `;
    }

    // Handle file selection
    fileInput.addEventListener('change', handleFileUpload);

    // Handle drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileUpload({ target: fileInput });
        }
    });

    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    // Helper function to convert timestamp to seconds is defined above
});





