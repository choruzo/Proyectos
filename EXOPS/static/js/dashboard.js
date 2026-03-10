/**
 * dashboard.js — KPIs en tiempo real + gráfico histórico de métricas
 */

const spinner      = document.getElementById('spinner');
const kpiContainer = document.getElementById('kpi-container');
const toastEl      = document.getElementById('action-toast');
const toastMsg     = document.getElementById('toast-msg');
const bsToast      = new bootstrap.Toast(toastEl, { delay: 5000 });

// ── Estado ────────────────────────────────────────────────────────
let historyChart  = null;
let lastSnapshots = [];
let currentHours  = 24;

// ── Helpers ───────────────────────────────────────────────────────

function fmtGb(gb) {
  if (gb >= 1024) return `${(gb / 1024).toFixed(1)} TB`;
  return `${gb.toFixed(1)} GB`;
}

function barColor(pct) {
  if (pct >= 90) return 'danger';
  if (pct >= 70) return 'warning';
  return 'success';
}

function showToast(msg, type = 'danger') {
  toastEl.className = `toast align-items-center border-0 text-bg-${type}`;
  toastMsg.textContent = msg;
  bsToast.show();
}

function isDarkMode() {
  return document.documentElement.getAttribute('data-bs-theme') === 'dark';
}

function chartColors() {
  return {
    grid: isDarkMode() ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)',
    text: isDarkMode() ? '#adb5bd' : '#495057',
  };
}

// ── Render KPIs ───────────────────────────────────────────────────

function renderKPIs(vms, hosts, datastores) {
  // — Tarjeta 1: VMs —
  const vmsOn    = vms.filter(v => v.power_state === 'poweredOn').length;
  const vmsTotal = vms.length;
  document.getElementById('kpi-vms-on').textContent  = vmsOn;
  document.getElementById('kpi-vms-sub').textContent = `encendidas de ${vmsTotal} VMs totales`;

  // — Tarjeta 2: Hosts —
  const hostsOk    = hosts.filter(h => h.connection_state === 'connected' && !h.in_maintenance).length;
  const hostsTotal = hosts.length;
  const hostsMaint = hosts.filter(h => h.in_maintenance).length;
  document.getElementById('kpi-hosts-ok').textContent  = hostsOk;
  document.getElementById('kpi-hosts-sub').textContent = `conectados de ${hostsTotal} hosts totales`;

  const maintBadge = document.getElementById('kpi-hosts-maint');
  if (hostsMaint > 0) {
    maintBadge.innerHTML = `<span class="badge bg-warning text-dark"><i class="bi bi-cone-striped me-1"></i>${hostsMaint} en mantenimiento</span>`;
    maintBadge.classList.remove('d-none');
  } else {
    maintBadge.classList.add('d-none');
  }

  // — Tarjeta 3: Datastore más lleno —
  const fullestDs = datastores[0];
  if (fullestDs) {
    const pct   = Math.round(fullestDs.used_pct);
    const color = barColor(pct);

    document.getElementById('card-ds').className      = `card h-100 border-${color}`;
    document.getElementById('kpi-ds-icon').className  = `bi bi-device-hdd fs-1 text-${color}`;
    const dsName = document.getElementById('kpi-ds-name');
    dsName.textContent = fullestDs.name;
    dsName.title       = fullestDs.name;

    const bar = document.getElementById('kpi-ds-bar');
    bar.style.width  = `${pct}%`;
    bar.className    = `progress-bar bg-${color}`;
    bar.setAttribute('aria-valuenow', pct);

    document.getElementById('kpi-ds-pct').textContent = `${pct}% usado`;
  }

  // — Tarjeta 4: Espacio libre total —
  const totalFree = datastores.reduce((s, d) => s + d.free_gb, 0);
  const totalCap  = datastores.reduce((s, d) => s + d.capacity_gb, 0);
  document.getElementById('kpi-free').textContent     = fmtGb(totalFree);
  document.getElementById('kpi-free-sub').textContent = `libre de ${fmtGb(totalCap)} totales`;
}

// ── Gráfico histórico ─────────────────────────────────────────────

function renderChart(snapshots) {
  const chartSection = document.getElementById('chart-section');
  const noDataEl     = document.getElementById('chart-no-data');

  if (!snapshots || snapshots.length < 2) {
    chartSection.classList.add('d-none');
    noDataEl.classList.remove('d-none');
    return;
  }

  noDataEl.classList.add('d-none');
  chartSection.classList.remove('d-none');

  const colors      = chartColors();
  const showPoints  = snapshots.length <= 60;

  const labels = snapshots.map(s => {
    const d = new Date(s.timestamp);
    return currentHours <= 24
      ? d.toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit' })
      : d.toLocaleDateString('es-ES', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' });
  });

  const datasets = [
    {
      label: 'CPU %',
      data: snapshots.map(s => s.cpu_usage_pct),
      borderColor: '#0d6efd',
      backgroundColor: 'rgba(13,110,253,0.07)',
      fill: true, tension: 0.4,
      pointRadius: showPoints ? 3 : 0, borderWidth: 2,
    },
    {
      label: 'Memoria %',
      data: snapshots.map(s => s.mem_usage_pct),
      borderColor: '#198754',
      backgroundColor: 'rgba(25,135,84,0.07)',
      fill: true, tension: 0.4,
      pointRadius: showPoints ? 3 : 0, borderWidth: 2,
    },
    {
      label: 'Datastores %',
      data: snapshots.map(s => s.datastores_used_pct),
      borderColor: '#ffc107',
      backgroundColor: 'rgba(255,193,7,0.07)',
      fill: true, tension: 0.4,
      pointRadius: showPoints ? 3 : 0, borderWidth: 2,
    },
  ];

  if (historyChart) {
    historyChart.data.labels   = labels;
    historyChart.data.datasets = datasets;
    historyChart.options.scales.x.ticks.color          = colors.text;
    historyChart.options.scales.x.grid.color           = colors.grid;
    historyChart.options.scales.y.ticks.color          = colors.text;
    historyChart.options.scales.y.grid.color           = colors.grid;
    historyChart.options.plugins.legend.labels.color   = colors.text;
    historyChart.update('none');
    return;
  }

  const ctx = document.getElementById('metrics-chart');
  if (!ctx) return;

  historyChart = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          position: 'top',
          labels: { color: colors.text, usePointStyle: true, pointStyleWidth: 10 },
        },
        tooltip: {
          callbacks: { label: item => ` ${item.dataset.label}: ${item.raw}%` },
        },
      },
      scales: {
        x: {
          ticks: { color: colors.text, maxTicksLimit: 10 },
          grid:  { color: colors.grid },
        },
        y: {
          min: 0, max: 100,
          ticks: { color: colors.text, callback: v => v + '%' },
          grid:  { color: colors.grid },
        },
      },
    },
  });
}

async function loadHistory(hours) {
  currentHours = hours;
  document.querySelectorAll('[data-hours]').forEach(btn => {
    btn.classList.toggle('active', parseInt(btn.dataset.hours) === hours);
  });
  try {
    const data = await api.get(`/api/v1/metrics/history?hours=${hours}`);
    if (!data) return;
    lastSnapshots = data.snapshots;
    renderChart(lastSnapshots);
  } catch (_) { /* silencioso si no hay caché aún */ }
}

// ── Carga principal ───────────────────────────────────────────────

async function loadDashboard(silent = false) {
  if (!silent) {
    spinner.classList.remove('d-none');
    kpiContainer.classList.add('d-none');
  }

  try {
    const data = await api.get('/api/v1/metrics/kpis');
    if (!data) return;

    renderKPIs(data.vms, data.hosts, data.datastores);

    spinner.classList.add('d-none');
    kpiContainer.classList.remove('d-none');
  } catch (err) {
    spinner.classList.add('d-none');
    kpiContainer.classList.remove('d-none');
    if (!silent) showToast(`Error al cargar datos: ${err.message}`);
  }
}

// ── Adaptar gráfico al cambio de tema ─────────────────────────────

new MutationObserver(() => {
  if (historyChart && lastSnapshots.length >= 2) renderChart(lastSnapshots);
}).observe(document.documentElement, { attributes: true, attributeFilter: ['data-bs-theme'] });

// ── Eventos y arranque ────────────────────────────────────────────

document.getElementById('btn-refresh').addEventListener('click', () => {
  loadDashboard();
  loadHistory(currentHours);
});

document.querySelectorAll('[data-hours]').forEach(btn => {
  btn.addEventListener('click', () => loadHistory(parseInt(btn.dataset.hours)));
});

loadDashboard();
loadHistory(currentHours);
setInterval(() => { loadDashboard(true); loadHistory(currentHours); }, 30000);
