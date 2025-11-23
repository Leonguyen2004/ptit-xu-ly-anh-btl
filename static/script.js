document.addEventListener('DOMContentLoaded', () => {
    // State
    let currentImage = null;
    let currentMode = 'detect'; // 'detect' or 'process'

    // Elements
    const fileInput = document.getElementById('file-input');
    const fileNameDisplay = document.getElementById('file-name');
    const originalImage = document.getElementById('original-image');
    const resultImage = document.getElementById('result-image');
    const imageWrapperOriginal = document.querySelector('.image-wrapper.original');
    const imageWrapperResult = document.querySelector('.image-wrapper.result');
    const placeholderText = document.querySelector('.placeholder-text');
    const loader = document.querySelector('.loader');
    
    // Tabs
    const navBtns = document.querySelectorAll('.nav-btn');
    const detectControls = document.getElementById('detect-controls');
    const processControls = document.getElementById('process-controls');
    const resultsPanel = document.getElementById('results-panel');

    // Controls
    const runDetectionBtn = document.getElementById('run-detection');
    const runProcessBtn = document.getElementById('run-process');
    const detectionModeSelect = document.getElementById('detection-mode');
    const procOperationSelect = document.getElementById('proc-operation');
    const dynamicParamsContainer = document.getElementById('dynamic-params');

    // Event Listeners
    fileInput.addEventListener('change', handleFileUpload);
    
    navBtns.forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });

    runDetectionBtn.addEventListener('click', runDetection);
    runProcessBtn.addEventListener('click', runProcessing);
    procOperationSelect.addEventListener('change', updateDynamicParams);

    // Initial setup
    updateDynamicParams();

    // Functions
    function handleFileUpload(e) {
        const file = e.target.files[0];
        if (!file) return;

        currentImage = file;
        fileNameDisplay.textContent = file.name;

        const reader = new FileReader();
        reader.onload = (e) => {
            originalImage.src = e.target.result;
            originalImage.classList.remove('hidden');
            placeholderText.classList.add('hidden');
            imageWrapperOriginal.classList.add('has-image');
            
            // Reset result
            resultImage.classList.add('hidden');
            resultImage.src = '';
            resultsPanel.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }

    function switchTab(tab) {
        currentMode = tab;
        
        // Update nav
        navBtns.forEach(btn => {
            if (btn.dataset.tab === tab) btn.classList.add('active');
            else btn.classList.remove('active');
        });

        // Update controls
        if (tab === 'detect') {
            detectControls.classList.remove('hidden');
            processControls.classList.add('hidden');
        } else {
            detectControls.classList.add('hidden');
            processControls.classList.remove('hidden');
        }
    }

    async function runDetection() {
        if (!currentImage) {
            alert('Please upload an image first');
            return;
        }

        showLoading(true);
        resultsPanel.classList.add('hidden');

        const formData = new FormData();
        formData.append('image', currentImage);
        formData.append('mode', detectionModeSelect.value);

        try {
            const response = await fetch('/api/pipeline', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (response.ok) {
                displayDetectionResults(data);
            } else {
                alert('Error: ' + (data.error || 'Unknown error'));
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during detection');
        } finally {
            showLoading(false);
        }
    }

    async function runProcessing() {
        if (!currentImage) {
            alert('Please upload an image first');
            return;
        }

        showLoading(true);
        
        const formData = new FormData();
        formData.append('image', currentImage);
        formData.append('operation', procOperationSelect.value);
        
        // Collect params
        const params = {};
        const inputs = dynamicParamsContainer.querySelectorAll('input');
        inputs.forEach(input => {
            params[input.name] = input.value;
        });
        formData.append('params', JSON.stringify(params));

        try {
            const response = await fetch('/api/process_image', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (response.ok) {
                resultImage.src = 'data:image/jpeg;base64,' + data.image;
                resultImage.classList.remove('hidden');
                imageWrapperResult.classList.add('has-image');
            } else {
                alert('Error: ' + (data.error || 'Unknown error'));
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during processing');
        } finally {
            showLoading(false);
        }
    }

    function displayDetectionResults(data) {
        // Show processed image with box
        resultImage.src = 'data:image/jpeg;base64,' + data.plate_crop; // Or full image if available
        // Actually, let's show the full image with boxes if we had it, but the pipeline returns crop.
        // Let's use the crop for now or if we want full visualization we might need to adjust API.
        // The pipeline API returns 'plate_crop'. Let's stick to that for "Result" image in detection mode.
        // Wait, the user might want to see the full image with the box.
        // The API /api/detect returns full image with box. /api/pipeline returns structured data.
        // Let's use the plate crop for the result image area for now, as it focuses on the plate.
        
        resultImage.src = 'data:image/jpeg;base64,' + data.plate_crop;
        resultImage.classList.remove('hidden');
        imageWrapperResult.classList.add('has-image');

        // Show text and conf
        document.getElementById('plate-text').textContent = data.plate_text;
        document.getElementById('plate-conf').textContent = (data.plate_conf * 100).toFixed(1) + '%';

        // Show characters
        const grid = document.getElementById('chars-grid');
        grid.innerHTML = '';
        
        data.characters.forEach(char => {
            const card = document.createElement('div');
            card.className = 'char-card';
            card.innerHTML = `
                <img src="data:image/jpeg;base64,${char.crop}">
                <div class="char-val">${char.label}</div>
                <div class="char-conf">${(char.conf * 100).toFixed(0)}%</div>
            `;
            grid.appendChild(card);
        });

        resultsPanel.classList.remove('hidden');
    }

    function showLoading(isLoading) {
        if (isLoading) {
            loader.classList.remove('hidden');
            resultImage.classList.add('hidden');
        } else {
            loader.classList.add('hidden');
        }
    }

    function updateDynamicParams() {
        const op = procOperationSelect.value;
        dynamicParamsContainer.innerHTML = '';
        
        let params = [];
        
        if (op === 'brightness_contrast') {
            params = [
                { name: 'brightness', label: 'Brightness', type: 'number', value: 0, min: -127, max: 127 },
                { name: 'contrast', label: 'Contrast', type: 'number', value: 0, min: -127, max: 127 }
            ];
        } else if (op === 'gaussian_blur' || op === 'median_blur') {
            params = [
                { name: 'kernel_size', label: 'Kernel Size (Odd)', type: 'number', value: 5, min: 1, step: 2 }
            ];
        } else if (op === 'canny') {
            params = [
                { name: 'threshold1', label: 'Threshold 1', type: 'number', value: 100 },
                { name: 'threshold2', label: 'Threshold 2', type: 'number', value: 200 }
            ];
        } else if (op === 'threshold_adaptive') {
            params = [
                { name: 'block_size', label: 'Block Size', type: 'number', value: 11, min: 3, step: 2 },
                { name: 'C', label: 'C', type: 'number', value: 2 }
            ];
        }

        params.forEach(p => {
            const div = document.createElement('div');
            div.className = 'control-group';
            div.innerHTML = `
                <label>${p.label}</label>
                <input type="${p.type}" name="${p.name}" value="${p.value}" min="${p.min}" max="${p.max}" step="${p.step}">
            `;
            dynamicParamsContainer.appendChild(div);
        });
    }
});
