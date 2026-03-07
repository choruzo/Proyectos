/**
 * hosts.js — Listado y modo mantenimiento de hosts ESXi
 */

let allHosts = [];

const spinner = document.getElementById('spinner');
const tableContainer = document.getElementById('table-container');
const tbody = document.getElementById('host-tbody');
const noResults = document.getElementById('no-results');

const toastEl = document.getElementById('action-toast');
const toastMsg = document.getElementById('toast-msg');
const bsToast = new bootstrap.Toast(toastEl, { delay: 4000 });

const confirmModal = new bootstrap.Modal(document.getElementById('confirm-modal'));
let pendingAction = null;

// ── Helpers ───────────────────────────────────────────────────────
function esc(str) {
  return (str || '').replace(/'/g, "\\'");
}

function fmtMhz(mhz) {
  return mhz >= 1000 ? `${(mhz / 1000).toFixed(1)} GHz` : `${mhz} MHz`;
}

function fmtMb(mb) {
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb} MB`;
}

// ── Badge de estado ───────────────────────────────────────────────
function stateBadge(host) {
  if (host.in_maintenance) {
    return '<span class="badge bg-warning text-dark"><i class="bi bi-cone-striped me-1"></i>Mantenimiento</span>';
  }
  if (host.connection_state === 'connected') {
    return '<span class="badge bg-success"><i class="bi bi-check-circle me-1"></i>Conectado</span>';
  }
  if (host.connection_state === 'disconnected') {
    return '<span class="badge bg-danger"><i class="bi bi-x-circle me-1"></i>Desconectado</span>';
  }
  return `<span class="badge bg-secondary">${host.connection_state}</span>`;
}

// ── Barra de progreso ─────────────────────────────────────────────
function progressBar(pct, label, colorClass) {
  const safeColor = colorClass || (pct >= 90 ? 'bg-danger' : pct >= 70 ? 'bg-warning' : 'bg-success');
  return `
    <div class="d-flex align-items-center gap-2">
      <div class="progress flex-grow-1" style="height:8px">
        <div class="progress-bar ${safeColor}" role="progressbar" style="width:${pct}%" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100"></div>
      </div>
      <small class="text-nowrap">${label}</small>
    </div>`;
}

function cpuBar(host) {
  if (!host.cpu_total_mhz) return '<span class="text-muted">—</span>';
  const color = host.cpu_usage_pct >= 90 ? 'bg-danger' : host.cpu_usage_pct >= 70 ? 'bg-warning' : 'bg-success';
  return progressBar(host.cpu_usage_pct, `${host.cpu_usage_pct}% (${fmtMhz(host.cpu_usage_mhz)})`, color);
}

function ramBar(host) {
  if (!host.mem_total_mb) return '<span class="text-muted">—</span>';
  const color = host.mem_usage_pct >= 90 ? 'bg-danger' : host.mem_usage_pct >= 70 ? 'bg-warning' : 'bg-success';
  return progressBar(host.mem_usage_pct, `${host.mem_usage_pct}% (${fmtMb(host.mem_usage_mb)})`, color);
}

// ── Botones de acción ─────────────────────────────────────────────
function actionButtons(host) {
  if (host.connection_state !== 'connected') return '';
  if (host.in_maintenance) {
    return `<button class="btn btn-sm btn-outline-success" onclick="doMaintenanceAction('${host.id}','${esc(host.name)}','exit')">
      <i class="bi bi-cone-striped me-1"></i>Salir mantenimiento
    </button>`;
  }
  return `<button class="btn btn-sm btn-outline-warning" onclick="confirmAction('${host.id}','${esc(host.name)}','enter')">
    <i class="bi bi-cone-striped me-1"></i>Entrar mantenimiento
  </button>`;
}

// ── Render tabla ──────────────────────────────────────────────────
function renderTable(hosts) {
  tbody.innerHTML = '';
  if (hosts.length === 0) {
    noResults.classList.remove('d-none');
    return;
  }
  noResults.classList.add('d-none');
  hosts.forEach(host => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="fw-semibold">${host.name}</td>
      <td>${stateBadge(host)}</td>
      <td>${cpuBar(host)}</td>
      <td>${ramBar(host)}</td>
      <td>${host.vms_on} / ${host.vms_total}</td>
      <td>${host.esxi_version || '<span class="text-muted">—</span>'}</td>
      <td>${actionButtons(host)}</td>`;
    tbody.appendChild(tr);
  });
}

// ── Filtro local ──────────────────────────────────────────────────
function applyFilter() {
  const text = document.getElementById('search-input').value.toLowerCase();
  const state = document.getElementById('filter-state').value;

  const filtered = allHosts.filter(host => {
    const matchText = !text || host.name.toLowerCase().includes(text);
    let matchState = true;
    if (state === 'maintenance') matchState = host.in_maintenance;
    else if (state === 'connected') matchState = host.connection_state === 'connected' && !host.in_maintenance;
    else if (state === 'disconnected') matchState = host.connection_state === 'disconnected';
    return matchText && matchState;
  });
  renderTable(filtered);
}

// ── Carga de hosts ────────────────────────────────────────────────
async function loadHosts(silent = false) {
  if (!silent) {
    spinner.classList.remove('d-none');
    tableContainer.classList.add('d-none');
  }
  try {
    const data = await api.get('/api/v1/hosts/');
    if (!data) return;
    allHosts = data;
    spinner.classList.add('d-none');
    tableContainer.classList.remove('d-none');
    applyFilter();
  } catch (err) {
    spinner.classList.add('d-none');
    tableContainer.classList.remove('d-none');
    showToast(`Error al cargar hosts: ${err.message}`, 'danger');
  }
}

// ── Toast ─────────────────────────────────────────────────────────
function showToast(msg, type = 'success') {
  toastEl.className = `toast align-items-center border-0 text-bg-${type}`;
  toastMsg.textContent = msg;
  bsToast.show();
}

// ── Confirmación (solo para "enter") ─────────────────────────────
function confirmAction(hostId, hostName, action) {
  document.getElementById('confirm-title').textContent = 'Confirmar modo mantenimiento';
  document.getElementById('confirm-body').textContent =
    `¿Seguro que quieres poner en modo mantenimiento el host "${hostName}"? Las VMs encendidas en ese host podrían verse afectadas.`;
  pendingAction = { hostId, hostName, action };
  confirmModal.show();
}

document.getElementById('confirm-btn').addEventListener('click', async () => {
  confirmModal.hide();
  if (pendingAction) {
    const { hostId, hostName, action } = pendingAction;
    pendingAction = null;
    await doMaintenanceAction(hostId, hostName, action);
  }
});

// ── Ejecutar acción de mantenimiento ─────────────────────────────
async function doMaintenanceAction(hostId, hostName, action) {
  const labels = { enter: 'Entrando en mantenimiento', exit: 'Saliendo de mantenimiento' };
  showToast(`${labels[action]}: "${hostName}"…`, 'info');

  const buttons = tbody.querySelectorAll('button');
  buttons.forEach(b => b.disabled = true);

  try {
    await api.post(`/api/v1/hosts/${hostId}/maintenance`, { action });
    const msg = action === 'enter'
      ? `Host "${hostName}" en modo mantenimiento.`
      : `Host "${hostName}" fuera del modo mantenimiento.`;
    showToast(msg, 'success');
  } catch (err) {
    showToast(`Error: ${err.message}`, 'danger');
  } finally {
    await loadHosts(true);
  }
}

// ── Eventos ───────────────────────────────────────────────────────
document.getElementById('search-input').addEventListener('input', applyFilter);
document.getElementById('filter-state').addEventListener('change', applyFilter);
document.getElementById('btn-refresh').addEventListener('click', () => loadHosts());

// ── Init ──────────────────────────────────────────────────────────
loadHosts();
setInterval(() => loadHosts(true), 30000);
