/**
 * GALTTCMC CI/CD Web UI - JavaScript Utilities
 */

// ==================== Estado / insignias ====================

var RUNNING_STATUSES = ['compiling', 'analyzing', 'deploying'];

function isRunning(status) {
    return RUNNING_STATUSES.indexOf(status) !== -1;
}

// Clase de insignia por estado de deployment (color siempre semántico)
function statusClass(status) {
    if (status === 'success') return 'b-pass';
    if (status === 'failed') return 'b-fail';
    if (isRunning(status)) return 'b-info';
    if (status === 'pending') return 'b-inc';
    return 'b-dim';
}

// Clase de insignia para el Quality Gate de SonarQube
function gateClass(gate) {
    if (gate === 'PASSED' || gate === 'OK') return 'b-pass';
    if (gate === 'FAILED' || gate === 'ERROR') return 'b-fail';
    if (gate === 'WARN') return 'b-inc';
    return 'b-dim';
}

// Segmentos de progreso (5 fases) a partir del estado global del deployment
var PIPELINE_PHASES = [
    { key: 'checkout', label: 'Git Checkout' },
    { key: 'compile',  label: 'Compilación' },
    { key: 'analyze',  label: 'SonarQube' },
    { key: 'deploy',   label: 'vCenter Deploy' },
    { key: 'notify',   label: 'SSH Install' }
];

function phaseSegs(status) {
    var numDone = { pending: 0, compiling: 1, analyzing: 2, deploying: 3, success: 5, failed: 0 }[status] || 0;
    return PIPELINE_PHASES.map(function(p, i) {
        var cls = '';
        if (status === 'success' || i < numDone) {
            cls = 'done';
        } else if (isRunning(status) && i === numDone) {
            cls = 'run';
        }
        return { key: p.key, label: p.label, cls: cls };
    });
}

// ==================== Tema de Chart.js ====================

function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// Color hex (#rrggbb) con transparencia -> #rrggbbaa
function alpha(hex, a) {
    var h = Math.round(a * 255).toString(16);
    return hex + (h.length === 1 ? '0' + h : h);
}

function applyChartTheme() {
    if (typeof Chart === 'undefined') return;
    Chart.defaults.color = cssVar('--dim');
    Chart.defaults.borderColor = cssVar('--line');
    Chart.defaults.font.family = cssVar('--sans');
    Chart.defaults.font.size = 12;
    Chart.defaults.plugins.legend.labels.boxWidth = 10;
    Chart.defaults.plugins.legend.labels.boxHeight = 10;
    Chart.defaults.plugins.tooltip.backgroundColor = cssVar('--panel2');
    Chart.defaults.plugins.tooltip.borderColor = cssVar('--line');
    Chart.defaults.plugins.tooltip.borderWidth = 1;
    Chart.defaults.plugins.tooltip.titleColor = cssVar('--text');
    Chart.defaults.plugins.tooltip.bodyColor = cssVar('--text');
    Chart.defaults.plugins.tooltip.padding = 10;
    Chart.defaults.plugins.tooltip.titleFont = { family: cssVar('--mono'), size: 12 };
}

// ==================== Toasts ====================

var TOAST_ICONS = {
    success: '<svg class="ico" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>',
    error:   '<svg class="ico" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>',
    warning: '<svg class="ico" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/></svg>',
    info:    '<svg class="ico" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>'
};

function showToast(message, type) {
    type = TOAST_ICONS[type] ? type : 'info';
    var container = document.getElementById('toast-container');
    if (!container) return;

    var toast = document.createElement('div');
    toast.className = 'toast ' + type;
    toast.setAttribute('role', 'status');

    // Icono (SVG estático de confianza, no controlado por el usuario)
    var iconWrapper = document.createElement('div');
    iconWrapper.innerHTML = TOAST_ICONS[type];
    toast.appendChild(iconWrapper.firstElementChild);

    // Mensaje: textContent para evitar XSS
    var msgSpan = document.createElement('span');
    msgSpan.className = 'msg';
    msgSpan.textContent = message;
    toast.appendChild(msgSpan);

    var closeBtn = document.createElement('button');
    closeBtn.className = 'x';
    closeBtn.setAttribute('aria-label', 'Cerrar');
    closeBtn.textContent = '×';
    closeBtn.addEventListener('click', function() { toast.remove(); });
    toast.appendChild(closeBtn);

    container.appendChild(toast);

    setTimeout(function() {
        toast.classList.add('out');
        setTimeout(function() { toast.remove(); }, 300);
    }, 5000);
}

// ==================== Formatters ====================

function formatBytes(bytes, decimals) {
    if (decimals === undefined) decimals = 2;
    if (bytes === 0) return '0 Bytes';
    var k = 1024;
    var dm = decimals < 0 ? 0 : decimals;
    var sizes = ['Bytes', 'KB', 'MB', 'GB'];
    var i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

function formatDuration(seconds) {
    if (seconds === null || seconds === undefined || seconds < 0) return 'N/A';
    var hours = Math.floor(seconds / 3600);
    var minutes = Math.floor((seconds % 3600) / 60);
    var secs = Math.floor(seconds % 60);
    if (hours > 0) return hours + 'h ' + minutes + 'm ' + secs + 's';
    if (minutes > 0) return minutes + 'm ' + secs + 's';
    return secs + 's';
}

// ==================== Clipboard ====================

function copyToClipboard(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function() {
            showToast('Copiado al portapapeles', 'success');
        }).catch(function(err) {
            console.error('Failed to copy:', err);
            showToast('No se pudo copiar al portapapeles', 'error');
        });
    } else {
        var textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        document.body.appendChild(textArea);
        textArea.select();
        try {
            document.execCommand('copy');
            showToast('Copiado al portapapeles', 'success');
        } catch (err) {
            console.error('Failed to copy:', err);
            showToast('No se pudo copiar al portapapeles', 'error');
        }
        document.body.removeChild(textArea);
    }
}

// ==================== Auto-refresh ====================

var autoRefreshInterval = null;

function startAutoRefresh(callback, intervalSeconds) {
    intervalSeconds = intervalSeconds || 30;
    if (autoRefreshInterval) clearInterval(autoRefreshInterval);
    autoRefreshInterval = setInterval(function() { callback(); }, intervalSeconds * 1000);
    showToast('Auto-refresh activado (cada ' + intervalSeconds + 's)', 'info');
}

function stopAutoRefresh() {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
        showToast('Auto-refresh desactivado', 'info');
    }
}

// ==================== Shell (layout global) ====================

function appShell() {
    return {
        navOpen: false,
        autoRefresh: false,
        service: { status: 'unknown', running: false },

        init: function() {
            var self = this;
            this.checkService();
            setInterval(function() { self.checkService(); }, 60000);
        },

        checkService: function() {
            var self = this;
            fetch('/api/pipeline/status')
                .then(function(r) { return r.json(); })
                .then(function(d) { self.service = d; })
                .catch(function() { self.service = { status: 'unknown', running: false }; });
        },

        serviceDot: function() {
            if (this.service.running) return 'ok';
            if (this.service.status === 'failed') return 'err';
            if (this.service.status === 'inactive') return 'warn';
            return '';
        },

        toggleAutoRefresh: function() {
            this.autoRefresh = !this.autoRefresh;
            if (this.autoRefresh) {
                startAutoRefresh(window.pageRefreshCallback || function() { location.reload(); }, 30);
            } else {
                stopAutoRefresh();
            }
        }
    };
}

// Export
window.isRunning = isRunning;
window.statusClass = statusClass;
window.gateClass = gateClass;
window.phaseSegs = phaseSegs;
window.cssVar = cssVar;
window.alpha = alpha;
window.applyChartTheme = applyChartTheme;
window.showToast = showToast;
window.formatBytes = formatBytes;
window.formatDuration = formatDuration;
window.copyToClipboard = copyToClipboard;
window.startAutoRefresh = startAutoRefresh;
window.stopAutoRefresh = stopAutoRefresh;
window.appShell = appShell;

window.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled promise rejection:', event.reason);
    showToast('Se produjo un error. Revisa la consola para más detalles.', 'error');
});
