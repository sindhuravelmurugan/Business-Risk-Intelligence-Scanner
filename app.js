import scaler from './scaler_stats.js';

let model;

async function loadModel() {
    model = await tf.loadLayersModel('./tfjs_model/model.json');
    console.log('Model loaded');
}

loadModel();

document.getElementById('riskForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    if (!model) {
        alert("⏳ Model is still loading. Please wait...");
        return;
    }

    // 1. Get form inputs
    const businessData = {
        revenue: parseFloat(document.getElementById('revenue').value),
        debtToEquity: parseFloat(document.getElementById('debtToEquity').value),
        cashFlow: parseFloat(document.getElementById('cashFlow').value),
        creditScore: parseFloat(document.getElementById('creditScore').value),
        yearsActive: parseFloat(document.getElementById('yearsActive').value),
        employees: parseFloat(document.getElementById('employees').value)
    };

    // 2. Normalize inputs using scaler
    const inputArray = [
        (businessData.revenue - scaler.mean[0]) / scaler.std[0],
        (businessData.debtToEquity - scaler.mean[1]) / scaler.std[1],
        (businessData.cashFlow - scaler.mean[2]) / scaler.std[2],
        (businessData.creditScore - scaler.mean[3]) / scaler.std[3],
        (businessData.yearsActive - scaler.mean[4]) / scaler.std[4],
        (businessData.employees - scaler.mean[5]) / scaler.std[5]
    ];

    // 3. Predict risk
    const inputTensor = tf.tensor2d([inputArray]);
    const prediction = await model.predict(inputTensor).data();
    const riskScore = Math.round(prediction[0] * 100);
    inputTensor.dispose();

    // 4. Display result
    document.getElementById('riskOutput').innerText = `Risk Score: ${riskScore} (${getRiskLevel(riskScore)})`;

    // 5. Optional: Add suggestions
    const suggestions = generateSuggestions(riskScore);
    suggestions.forEach(s => {
        const p = document.createElement('p');
        p.textContent = "💡 " + s;
        document.body.appendChild(p);
    });
});

function getRiskLevel(score) {
    if (score <= 30) return 'Low Risk';
    if (score <= 60) return 'Medium Risk';
    return 'High Risk';
}

function generateSuggestions(score) {
    const suggestions = [];
    if (score > 70) {
        suggestions.push("Consider reducing debt and improving cash flow.");
        suggestions.push("Review your credit profile and risk exposure.");
    } else if (score > 40) {
        suggestions.push("Maintain consistent revenue and reduce liabilities.");
    } else {
        suggestions.push("Keep up the good performance! Focus on sustainable growth.");
    }
    return suggestions;
}
