const textInput = document.getElementById('text-input');
const analyzeBtn = document.getElementById('analyze-btn');
const resultSection = document.getElementById('result-section');
const predictedEmotion = document.getElementById('predicted-emotion');
const emotionEmoji = document.getElementById('emotion-emoji');
const confidenceVal = document.getElementById('confidence-val');
const probBars = document.getElementById('prob-bars');

const emojiMap = {
    'joy': '😊',
    'sadness': '😢',
    'anger': '😡',
    'fear': '😨',
    'love': '❤️',
    'surprise': '😲'
};

const colorMap = {
    'joy': '#eab308',
    'sadness': '#3b82f6',
    'anger': '#ef4444',
    'fear': '#8b5cf6',
    'love': '#ec4899',
    'surprise': '#22c55e'
};

async function analyzeEmotion() {
    const text = textInput.value.trim();
    if (!text) return;

    analyzeBtn.classList.add('loading');
    analyzeBtn.disabled = true;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });

        if (!response.ok) throw new Error('Prediction failed');

        const data = await response.json();
        displayResults(data);
    } catch (err) {
        console.error(err);
        alert('An error occurred while analyzing the text.');
    } finally {
        analyzeBtn.classList.remove('loading');
        analyzeBtn.disabled = false;
    }
}

function displayResults(data) {
    resultSection.classList.remove('hidden');
    
    // Top Result
    predictedEmotion.textContent = data.emotion;
    confidenceVal.textContent = `${(data.confidence * 100).toFixed(1)}%`;
    emotionEmoji.textContent = emojiMap[data.emotion] || '✨';
    
    // Detailed bars
    probBars.innerHTML = '';
    const sortedProbs = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
    
    sortedProbs.forEach(([emotion, prob]) => {
        const item = document.createElement('div');
        item.className = 'prob-item';
        
        const percentage = (prob * 100).toFixed(1);
        const color = colorMap[emotion] || '#8b5cf6';
        
        item.innerHTML = `
            <div class="prob-labels">
                <span>${emotion}</span>
                <span>${percentage}%</span>
            </div>
            <div class="bar-bg">
                <div class="bar-fill" style="width: 0%; background: ${color}"></div>
            </div>
        `;
        
        probBars.appendChild(item);
        
        // Trigger animation
        setTimeout(() => {
            item.querySelector('.bar-fill').style.width = `${percentage}%`;
        }, 100);
    });

    // Scroll to results
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

analyzeBtn.addEventListener('click', analyzeEmotion);

// Optional: Ctrl + Enter to submit
textInput.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') analyzeEmotion();
});
