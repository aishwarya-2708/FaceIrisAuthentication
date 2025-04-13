let videoElement = document.getElementById('video');
let videoContainer = document.getElementById('video-container');
let startButton = document.getElementById('start-face-recognition');
let statusElement = document.getElementById('status');

// Start Face Authentication
startButton.addEventListener('click', () => {
    statusElement.textContent = 'Starting face authentication...';
    fetch('/start_face_recognition')
        .then(response => response.json())
        .then(data => {
            statusElement.textContent = data.status;
            startFaceCapture();
        });
});

// Start video feed and capture face
function startFaceCapture() {
    videoContainer.style.display = 'block';
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            videoElement.srcObject = stream;
            videoElement.play();
        })
        .catch(err => console.log('Error: ' + err));

    // Capture face when spacebar is pressed
    document.addEventListener('keydown', (event) => {
        if (event.key === ' ') { // Space key pressed
            fetch('/capture_face', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    statusElement.textContent = data.status + ' Name: ' + data.name;
                    startIrisCapture();
                });
        }
    });
}

// Start Iris Authentication
function startIrisCapture() {
    statusElement.textContent = 'Starting iris authentication...';
    fetch('/start_iris_recognition')
        .then(response => response.json())
        .then(data => {
            statusElement.textContent = data.status;
            startIrisCaptureProcess();
        });
}

// Capture Iris when spacebar is pressed
function startIrisCaptureProcess() {
    document.addEventListener('keydown', (event) => {
        if (event.key === ' ') { // Space key pressed
            fetch('/capture_iris', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    statusElement.textContent = data.status;
                });
        }
    });
}
