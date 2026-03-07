/**
 * vms.js — Listado y operaciones de poder sobre VMs
 */

let allVMs = [];
let refreshInterval = null;

const spinner = document.getElementById('spinner');
const tableContainer = document.getElementById('table-container');
const tbody = document.getElementById('vm-tbody');
const noResults = document.getElementById('no-results');

const toastEl = document.getElementById('action-toast');
const toastMsg = document.getElementById('toast-msg');
const bsToast = new bootstrap.Toast(toastEl, { delay: 4000 });

const confirmModal = new bootstrap.Modal(document.getElementById('confirm-modal'));
let pendingAction = null;

// ── Estado → badge ──────────────────────────────────────────────
function stateBadge(state) {
  const map = {
    poweredOn:  '<span class="badge bg-success"><i class="bi bi-play-fill me-1"></i>Encendida</span>',
    poweredOff: '<span class="badge bg-secondary"><i class="bi bi-stop-fill me-1"></i>Apagada</span>',
    suspended:  '<span class="badge bg-warning text-dark"><i class="bi bi-pause-fill me-1"></i>Suspendida</span>',
  };
  return map[state] || `<span class="badge bg-secondary">${state}</span>`;
}

// ── Botones según estado ─────────────────────────────────────────
function actionButtons(vm) {
  const id = vm.id;
  const name = vm.name;
  if (vm.power_state === 'poweredOn') {
    return `
      <button class="btn btn-sm btn-outline-danger me-1" onclick="confirmAction('${id}','${esc(name)}','off')">
        <i class="bi bi-stop-fill me-1"></i>Apagar
      </button>
      <button class="btn btn-sm btn-outline-warning" onclick="confirmAction('${id}','${esc(name)}','reboot')">
        <i class="bi bi-arrow-clockwise me-1"></i>Reiniciar
      </button>`;
  }
  if (vm.power_state === 'poweredOff') {
    return `
      <button class="btn btn-sm btn-outline-success" onclick="doPowerAction('${id}','${esc(name)}','on')">
        <i class="bi bi-play-fill me-1"></i>Encender
      </button>`;
  }
  if (vm.power_state === 'suspended') {
    return `
      <button class="btn btn-sm btn-outline-success" onclick="doPowerAction('${id}','${esc(name)}','on')">
        <i class="bi bi-play-fill me-1"></i>Reanudar
      </button>`;
  }
  return '';
}

function esc(str) {
  return (str || '').replace(/'/g, "\\'");
}

// ── Render tabla ─────────────────────────────────────────────────
function renderTable(vms) {
  tbody.innerHTML = '';
  if (vms.length === 0) {
    noResults.classList.remove('d-none');
    return;
  }
  noResults.classList.add('d-none');
  vms.forEach(vm => {
    const ram = vm.memory_mb >= 1024
      ? `${(vm.memory_mb / 1024).toFixed(1)} GB`
      : `${vm.memory_mb} MB`;
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="fw-semibold">${vm.name}</td>
      <td>${stateBadge(vm.power_state)}</td>
      <td>${vm.num_cpu} vCPU</td>
      <td>${ram}</td>
      <td>${vm.ip_address || '<span class="text-muted">—</span>'}</td>
      <td class="text-truncate" style="max-width:160px" title="${vm.host_name || ''}">${vm.host_name || '<span class="text-muted">—</span>'}</td>
      <td class="text-truncate" style="max-width:180px" title="${vm.guest_os || ''}">${vm.guest_os || '<span class="text-muted">—</span>'}</td>
      <td>${actionButtons(vm)}</td>`;
    tbody.appendChild(tr);
  });
}

// ── Filtro local ─────────────────────────────────────────────────
function applyFilter() {
  const text = document.getElementById('search-input').value.toLowerCase();
  const state = document.getElementById('filter-state').value;
  const filtered = allVMs.filter(vm => {
    const matchText = !text || vm.name.toLowerCase().includes(text) ||
      (vm.ip_address || '').includes(text) ||
      (vm.host_name || '').toLowerCase().includes(text);
    const matchState = !state || vm.power_state === state;
    return matchText && matchState;
  });
  renderTable(filtered);
}

// ── Carga de VMs ─────────────────────────────────────────────────
async function loadVMs(silent = false) {
  if (!silent) {
    spinner.classList.remove('d-none');
    tableContainer.classList.add('d-none');
  }
  try {
    const data = await api.get('/api/v1/vms/');
    if (!data) return;
    allVMs = data;
    spinner.classList.add('d-none');
    tableContainer.classList.remove('d-none');
    applyFilter();
  } catch (err) {
    spinner.classList.add('d-none');
    tableContainer.classList.remove('d-none');
    showToast(`Error al cargar VMs: ${err.message}`, 'danger');
  }
}

// ── Toast ─────────────────────────────────────────────────────────
function showToast(msg, type = 'success') {
  toastEl.className = `toast align-items-center border-0 text-bg-${type}`;
  toastMsg.textContent = msg;
  bsToast.show();
}

// ── Acción con confirmación ───────────────────────────────────────
function confirmAction(vmId, vmName, action) {
  const labels = { off: 'apagar', reboot: 'reiniciar' };
  document.getElementById('confirm-title').textContent =
    action === 'off' ? 'Confirmar apagado' : 'Confirmar reinicio';
  document.getElementById('confirm-body').textContent =
    `¿Seguro que quieres ${labels[action]} la VM "${vmName}"?`;
  pendingAction = { vmId, vmName, action };
  confirmModal.show();
}

document.getElementById('confirm-btn').addEventListener('click', async () => {
  confirmModal.hide();
  if (pendingAction) {
    const { vmId, vmName, action } = pendingAction;
    pendingAction = null;
    await doPowerAction(vmId, vmName, action);
  }
});

// ── Ejecutar acción de poder ──────────────────────────────────────
async function doPowerAction(vmId, vmName, action) {
  const labels = { on: 'encendiendo', off: 'apagando', reboot: 'reiniciando' };

  // Deshabilitar botones de esa VM
  const buttons = tbody.querySelectorAll('button');
  buttons.forEach(b => b.disabled = true);

  showToast(`${labels[action].charAt(0).toUpperCase() + labels[action].slice(1)} "${vmName}"…`, 'info');
  try {
    await api.post(`/api/v1/vms/${vmId}/power`, { action });
    showToast(`VM "${vmName}" ${labels[action].replace('ando', 'ada').replace('iendo', 'ida')} correctamente.`, 'success');
  } catch (err) {
    showToast(`Error: ${err.message}`, 'danger');
  } finally {
    await loadVMs(true);
  }
}

// ── Eventos ───────────────────────────────────────────────────────
document.getElementById('search-input').addEventListener('input', applyFilter);
document.getElementById('filter-state').addEventListener('change', applyFilter);
document.getElementById('btn-refresh').addEventListener('click', () => loadVMs());

// ── Init ──────────────────────────────────────────────────────────
loadVMs();
refreshInterval = setInterval(() => loadVMs(true), 30000);
