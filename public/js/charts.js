/**
 * Agentic DermaGuard - Chart.js Visualization Helpers
 * Shared charting functions for screening, benchmark, and batch dashboards
 */

const CLASS_COLORS = {
  mel: '#ef4444',    // Melanoma (Red)
  nv: '#10b981',     // Nevus (Emerald)
  bcc: '#f97316',    // Basal cell (Orange)
  akiec: '#eab308',  // Actinic keratosis (Yellow)
  bkl: '#8b5cf6',    // Benign keratosis (Purple)
  df: '#06b6d4',     // Dermatofibroma (Cyan)
  vasc: '#ec4899'    // Vascular (Pink)
};

const CLASS_NAMES = {
  mel: 'Melanoma',
  nv: 'Nevus',
  bcc: 'Basal Cell Ca.',
  akiec: 'Actinic Keratosis',
  bkl: 'Benign Keratosis',
  df: 'Dermatofibroma',
  vasc: 'Vascular Lesion'
};

let activeCharts = {};

function destroyChart(canvasId) {
  if (activeCharts[canvasId]) {
    activeCharts[canvasId].destroy();
    delete activeCharts[canvasId];
  }
}

/**
 * Creates 7-class differential probability horizontal bar chart.
 */
function createProbabilityBarChart(canvasId, probabilities) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return null;
  destroyChart(canvasId);

  const codes = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc'];
  const labels = codes.map(c => `${CLASS_NAMES[c]} [${c}]`);
  const values = codes.map(c => ((probabilities[c] || 0) * 100).toFixed(1));
  const bgColors = codes.map(c => CLASS_COLORS[c] || '#3b82f6');

  const ctx = canvas.getContext('2d');
  activeCharts[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        data: values,
        backgroundColor: bgColors,
        borderRadius: 6,
        borderWidth: 0
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `Probability: ${ctx.raw}%`
          }
        }
      },
      scales: {
        x: {
          min: 0,
          max: 100,
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#94a3b8', callback: v => `${v}%`, font: { family: 'JetBrains Mono', size: 10 } }
        },
        y: {
          grid: { display: false },
          ticks: { color: '#cbd5e1', font: { family: 'Plus Jakarta Sans', size: 11, weight: '600' } }
        }
      }
    }
  });

  return activeCharts[canvasId];
}

/**
 * Creates 4-axis Image Quality Radar Chart.
 */
function createQualityRadarChart(canvasId, scores) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return null;
  destroyChart(canvasId);

  const labels = ['Sharpness', 'Lighting', 'Glare Void Free', 'Centering'];
  const values = [
    scores.blur_score || 0,
    scores.lighting_score || 0,
    scores.glare_score || 0,
    scores.occlusion_score || 0
  ];

  const ctx = canvas.getContext('2d');
  activeCharts[canvasId] = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Quality Score',
        data: values,
        backgroundColor: 'rgba(6, 182, 212, 0.2)',
        borderColor: '#06b6d4',
        borderWidth: 2,
        pointBackgroundColor: '#22d3ee',
        pointBorderColor: '#ffffff',
        pointRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        r: {
          min: 0,
          max: 100,
          ticks: { stepSize: 25, color: '#64748b', backdropColor: 'transparent', font: { size: 9 } },
          grid: { color: 'rgba(255, 255, 255, 0.08)' },
          pointLabels: { color: '#cbd5e1', font: { size: 11, weight: '600' } }
        }
      }
    }
  });

  return activeCharts[canvasId];
}

/**
 * Renders interactive 7x7 confusion matrix as an HTML table or canvas.
 */
function renderConfusionMatrixTable(containerId, matrix, classLabels) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const labels = classLabels || ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc'];
  let maxVal = 1;
  matrix.forEach(row => row.forEach(val => { if (val > maxVal) maxVal = val; }));

  let html = `
    <div class="overflow-x-auto custom-scroll">
      <table class="w-full text-center border-collapse font-mono text-xs">
        <thead>
          <tr>
            <th class="p-2 text-left font-bold text-slate-400 uppercase text-[10px]">True \\ Pred</th>
  `;

  labels.forEach(l => {
    html += `<th class="p-2 font-bold text-slate-300 uppercase text-[11px]">${l}</th>`;
  });
  html += `</tr></thead><tbody>`;

  matrix.forEach((row, rIdx) => {
    html += `<tr><td class="p-2 text-left font-bold text-slate-300 uppercase text-[11px]">${labels[rIdx]}</td>`;
    row.forEach((val, cIdx) => {
      const isDiagonal = rIdx === cIdx;
      const intensity = Math.min(1.0, val / maxVal);
      let cellBg = isDiagonal 
        ? `rgba(16, 185, 129, ${Math.max(0.15, intensity * 0.85)})`
        : (val > 0 ? `rgba(239, 68, 68, ${Math.max(0.12, intensity * 0.75)})` : 'transparent');
      
      let textColor = val === 0 ? 'text-slate-600' : (isDiagonal ? 'text-emerald-300 font-bold' : 'text-rose-300 font-medium');

      html += `
        <td style="background-color: ${cellBg};" class="p-2 border border-slate-800/60 ${textColor} transition hover:ring-1 hover:ring-cyan-400">
          ${val}
        </td>
      `;
    });
    html += `</tr>`;
  });

  html += `</tbody></table></div>`;
  container.innerHTML = html;
}

window.createProbabilityBarChart = createProbabilityBarChart;
window.createQualityRadarChart = createQualityRadarChart;
window.renderConfusionMatrixTable = renderConfusionMatrixTable;
window.destroyChart = destroyChart;
