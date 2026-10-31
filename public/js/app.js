/**
 * Agentic DermaGuard - Universal Client & App Core
 * Multi-Agent Skin Lesion Screening & Triage Workstation
 */

// --- API Client ---
async function apiRequest(endpoint, method = 'GET', data = null) {
  const options = {
    method,
    headers: {
      'Content-Type': 'application/json'
    }
  };
  if (data && (method === 'POST' || method === 'PUT')) {
    options.body = JSON.stringify(data);
  }

  const response = await fetch(endpoint, options);
  if (!response.ok) {
    let errMsg = `HTTP ${response.status}: ${response.statusText}`;
    try {
      const errJson = await response.json();
      if (errJson.error) errMsg = errJson.error;
      else if (errJson.message) errMsg = errJson.message;
    } catch (e) {}
    throw new Error(errMsg);
  }
  return await response.json();
}

// --- Toast Notifications ---
function showToast(message, type = 'info', duration = 3500) {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    document.body.appendChild(container);
  }

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;

  const iconClass = type === 'success' 
    ? 'fa-solid fa-circle-check text-emerald-400' 
    : (type === 'error' ? 'fa-solid fa-triangle-exclamation text-rose-400' : 'fa-solid fa-circle-info text-cyan-400');

  toast.innerHTML = `
    <i class="${iconClass}"></i>
    <span class="flex-1">${message}</span>
    <button onclick="this.parentElement.remove()" class="text-slate-400 hover:text-white transition">
      <i class="fa-solid fa-xmark text-xs"></i>
    </button>
  `;

  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(8px) scale(0.95)';
    setTimeout(() => toast.remove(), 250);
  }, duration);
}

// --- Theme Management ---
function initTheme() {
  const savedTheme = localStorage.getItem('dermaguard_theme') || 'dark';
  document.documentElement.setAttribute('data-theme', savedTheme);
  updateThemeIcon(savedTheme);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('dermaguard_theme', next);
  updateThemeIcon(next);
  showToast(`Switched to ${next} theme`, 'info', 2000);
}

function updateThemeIcon(theme) {
  const icon = document.getElementById('theme-toggle-icon');
  if (icon) {
    icon.className = theme === 'dark' ? 'fa-solid fa-sun text-amber-400' : 'fa-solid fa-moon text-slate-600';
  }
}

// --- Active Navigation Highlighting ---
function initNavigation() {
  const currentPath = window.location.pathname.split('/').pop() || 'index.html';
  const navLinks = document.querySelectorAll('.nav-link');
  navLinks.forEach(link => {
    const href = link.getAttribute('href');
    if (href === currentPath || (currentPath === '' && href === 'index.html')) {
      link.classList.add('border-cyan-400', 'text-cyan-400');
      link.classList.remove('border-transparent', 'text-slate-400');
    } else {
      link.classList.remove('border-cyan-400', 'text-cyan-400');
      link.classList.add('border-transparent', 'text-slate-400');
    }
  });
}

// --- Common Formatter Helpers ---
function formatPercentage(num) {
  if (num === null || num === undefined || isNaN(num)) return '0.0%';
  return `${(num * 100).toFixed(1)}%`;
}

function formatScore(num) {
  if (num === null || num === undefined || isNaN(num)) return '0.0';
  return Number(num).toFixed(2);
}

function getDecisionBadgeClass(decision) {
  if (decision === 'AUTO_TRIAGE') return 'badge-auto';
  if (decision === 'RE_IMAGE') return 'badge-reimage';
  return 'badge-refer';
}

function renderRequiredStrip(stripStr, containerId = 'required-example-strip') {
  const container = document.getElementById(containerId);
  if (!container) return;

  if (!stripStr) {
    stripStr = 'A1 PASS "Image is usable" | A2 nv • 0.73 "Model predicts nevus" | A3+A4 SUSPICIOUS "Other evidence raises concern" | A5 SAFETY FLAG "Conflict detected" | A6 REFER "Final disposition"';
  }

  // Parse strip components
  const parts = stripStr.split('|').map(s => s.trim());
  let html = `<div class="example-strip-container">`;

  parts.forEach((part, idx) => {
    let pillClass = 'strip-pill-a1';
    let icon = 'fa-check';

    if (part.startsWith('A1')) {
      pillClass = 'strip-pill-a1';
      icon = 'fa-camera';
    } else if (part.startsWith('A2')) {
      pillClass = 'strip-pill-a2';
      icon = 'fa-dna';
    } else if (part.startsWith('A3') || part.includes('A3+A4')) {
      pillClass = 'strip-pill-a3a4';
      icon = 'fa-circle-notch';
    } else if (part.startsWith('A5')) {
      pillClass = 'strip-pill-a5';
      icon = 'fa-shield-halved';
    } else if (part.startsWith('A6')) {
      pillClass = 'strip-pill-a6';
      icon = 'fa-clipboard-check';
    }

    // Extract quote if present
    const quoteMatch = part.match(/"([^"]+)"/);
    const quote = quoteMatch ? quoteMatch[1] : '';
    const mainToken = quoteMatch ? part.replace(quoteMatch[0], '').trim() : part;

    html += `
      <div class="strip-pill ${pillClass}">
        <i class="fa-solid ${icon} text-[10px]"></i>
        <span>${mainToken}</span>
        ${quote ? `<span class="strip-quote">"${quote}"</span>` : ''}
      </div>
    `;

    if (idx < parts.length - 1) {
      html += `<span class="strip-separator">|</span>`;
    }
  });

  html += `</div>`;
  container.innerHTML = html;
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initNavigation();
});

window.apiRequest = apiRequest;
window.showToast = showToast;
window.toggleTheme = toggleTheme;
window.formatPercentage = formatPercentage;
window.formatScore = formatScore;
window.getDecisionBadgeClass = getDecisionBadgeClass;
window.renderRequiredStrip = renderRequiredStrip;
