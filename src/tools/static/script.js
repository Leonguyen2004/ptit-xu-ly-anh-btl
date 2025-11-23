function openTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(tabName).classList.add('active');
    document.querySelector(`button[onclick="openTab('${tabName}')"]`).classList.add('active');

    if (tabName === 'history') {
        loadHistory();
    }
}

// Sliders
const sliders = ['conf', 'bright', 'contrast', 'rot'];
sliders.forEach(id => {
    const slider = document.getElementById(`${id}-slider`);
    const val = document.getElementById(`${id}-val`);
    if (slider && val) {
        slider.addEventListener('input', (e) => {
            val.textContent = e.target.value;
        });
    }
});

// Single Process
document.getElementById('single-upload').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('image', file);
    formData.append('yolo_conf', document.getElementById('conf-slider').value);
    formData.append('brightness', document.getElementById('bright-slider').value);
    formData.append('contrast', document.getElementById('contrast-slider').value);
    formData.append('rotation', document.getElementById('rot-slider').value);
    
    document.getElementById('single-loading').classList.remove('hidden');
    document.getElementById('single-result').classList.add('hidden');

    try {
        const response = await fetch('/api/process', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.error) {
            alert('Error: ' + data.error);
        } else {
            document.getElementById('result-img').src = 'data:image/jpeg;base64,' + data.annotated_image;
            document.getElementById('plate-text').textContent = data.text;
            
            // Debug grid
            const debugGrid = document.getElementById('debug-grid');
            debugGrid.innerHTML = '';
            data.debug_crops.forEach(item => {
                const div = document.createElement('div');
                div.className = 'debug-item';
                div.innerHTML = `
                    <img src="data:image/jpeg;base64,${item.image}">
                    <div><strong>${item.char}</strong></div>
                    <small>${(item.conf * 100).toFixed(1)}%</small>
                `;
                debugGrid.appendChild(div);
            });

            document.getElementById('single-result').classList.remove('hidden');
        }
    } catch (err) {
        alert('Network Error');
        console.error(err);
    } finally {
        document.getElementById('single-loading').classList.add('hidden');
    }
});

// History
async function loadHistory() {
    const tbody = document.getElementById('history-table-body');
    tbody.innerHTML = '<tr><td colspan="4">Loading...</td></tr>';

    try {
        const response = await fetch('/api/history');
        const data = await response.json();
        
        tbody.innerHTML = '';
        data.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${row.id}</td>
                <td>${new Date(row.timestamp).toLocaleString()}</td>
                <td style="font-weight:bold; color:#2563eb">${row.plate_text}</td>
                <td>${row.filename}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (err) {
        tbody.innerHTML = '<tr><td colspan="4">Error loading history</td></tr>';
    }
}

// Batch Process
document.getElementById('batch-upload').addEventListener('change', async (e) => {
    const files = e.target.files;
    if (files.length === 0) return;

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('images', files[i]);
    }

    document.getElementById('batch-loading').classList.remove('hidden');
    document.getElementById('batch-result').classList.add('hidden');
    const tbody = document.getElementById('batch-table-body');
    tbody.innerHTML = '';

    try {
        const response = await fetch('/api/batch', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        data.results.forEach(res => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${res.filename}</td>
                <td style="color: ${res.status === 'success' ? 'green' : 'red'}">${res.status}</td>
                <td>${res.text || res.error}</td>
            `;
            tbody.appendChild(tr);
        });

        document.getElementById('batch-result').classList.remove('hidden');
    } catch (err) {
        alert('Network Error');
    } finally {
        document.getElementById('batch-loading').classList.add('hidden');
    }
});

// Analysis
document.getElementById('analysis-upload').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('image', file);

    document.getElementById('analysis-loading').classList.remove('hidden');
    document.getElementById('analysis-result').classList.add('hidden');

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.error) {
            alert('Error: ' + data.error);
        } else {
            const ids = ['original', 'gray', 'blur', 'edges', 'equalized', 'threshold', 'morphology'];
            ids.forEach(id => {
                const img = document.getElementById(`an-${id}`);
                if (img && data[id]) {
                    img.src = 'data:image/jpeg;base64,' + data[id];
                }
            });

            document.getElementById('analysis-result').classList.remove('hidden');
        }
    } catch (err) {
        alert('Network Error');
    } finally {
        document.getElementById('analysis-loading').classList.add('hidden');
    }
});
