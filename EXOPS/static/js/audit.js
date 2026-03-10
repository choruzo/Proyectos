/**
 * audit.js — Registro de auditoría con filtros y paginación
 */

const LIMIT = 50;
let currentPage = 1;
let lastResultCount = 0;

const spinner        = document.getElementById('spinner');
const tableContainer = document.getElementById('table-container');
const tbody          = document.getElementById('audit-tbody');
const noResults      = document.getElementById('no-results');
const pageLabel      = document.getElementById('page-label');
const btnPrev        = document.getElementById('btn-prev');
const btnNext        = document.getElementById('btn-next');

const toastEl  = document.getElementById('action-toast');
const toastMsg = document.getElementById('toast-msg');
const bsToast  = new bootstrap.Toast(toastEl, { delay: 5000 });

const detailModal   = new bootstrap.Modal(document.getElementById('detail-modal'));
const detailFields  = document.getElementById('detail-fields');

// ── Mapeo acción → [color badge, etiqueta] ────────────────────────
const ACTION_BADGES = {
  login:                  ['secondary', 'Login'],
  logout:                 ['secondary', 'Logout'],
  vm_power_on:            ['primary',   'VM — Encender'],
  vm_power_off:           ['primary',   'VM — Apagar'],
  vm_power_reboot:        ['primary',   'VM — Reiniciar'],
  host_maintenance_enter: ['info',      'Host — Mantenimiento ▲'],
  host_maintenance_exit:  ['info',      'Host — Mantenimiento ▼'],
  snapshot_create:        ['success',   'Snapshot — Crear'],
  snapshot_restore:       ['warning',   'Snapshot — Restaurar'],
  snapshot_delete:        ['danger',    'Snapshot — Eliminar'],
  user_create:            ['primary',   'Usuario — Crear'],
  user_update:            ['primary',   'Usuario — Actualizar'],
  user_deactivate:        ['warning',   'Usuario — Desactivar'],
  user_delete:            ['danger',    'Usuario — Eliminar'],
};

function actionBadge(action, result) {
  const [color, label] = ACTION_BADGES[action] || ['secondary', action];
  // login fallido → danger
  const badgeColor = (action === 'login' && result === 'error') ? 'danger' : color;
  return `<span class="badge text-bg-${badgeColor}">${label}</span>`;
}

function resultBadge(result) {
  return result === 'success'
    ? '<span class="badge bg-success">OK</span>'
    : '<span class="badge bg-danger">Error</span>';
}

function formatDate(ts) {
  if (!ts) return '—';
  return new Date(ts).toLocaleString('es-ES', {
    day: '2-digit', month: '2-digit', year: 'numeric',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

function dash(val) {
  return val || '<span class="text-muted">—</span>';
}

// ── Render tabla ──────────────────────────────────────────────────
function renderTable(logs) {
  tbody.innerHTML = '';
  if (logs.length === 0) {
    noResults.classList.remove('d-none');
    return;
  }
  noResults.classList.add('d-none');

  logs.forEach(log => {
    const resource = (log.resource_type && log.resource_id)
      ? `${log.resource_type} · ${log.resource_id}`
      : null;

    const tr = document.createElement('tr');
    tr.style.cursor = 'pointer';
    tr.innerHTML = `
      <td class="text-nowrap">${formatDate(log.timestamp)}</td>
      <td>${dash(log.username)}</td>
      <td>${actionBadge(log.action, log.result)}</td>
      <td>${dash(resource)}</td>
      <td>${resultBadge(log.result)}</td>
      <td>${dash(log.ip_address)}</td>`;
    tr.addEventListener('click', () => openDetail(log));
    tbody.appendChild(tr);
  });
}

// ── Modal detalle ─────────────────────────────────────────────────
function openDetail(log) {
  const fields = [
    ['Fecha completa',   formatDate(log.timestamp)],
    ['Usuario',          log.username],
    ['Acción',           (ACTION_BADGES[log.action] || ['', log.action])[1]],
    ['Resultado',        log.result === 'success' ? 'OK' : 'Error'],
    ['IP',               log.ip_address],
    ['vCenter Host',     log.vcenter_host],
    ['Tipo de recurso',  log.resource_type],
    ['ID de recurso',    log.resource_id],
    ['Nombre de recurso',log.resource_name],
    ['Detalles',         log.details],
    ['Mensaje de error', log.error_message],
  ];

  detailFields.innerHTML = fields.map(([label, val]) => `
    <dt class="col-sm-4 text-muted">${label}</dt>
    <dd class="col-sm-8">${val || '—'}</dd>`).join('');

  detailModal.show();
}

// ── Paginación ────────────────────────────────────────────────────
function updatePagination() {
  pageLabel.textContent = `Página ${currentPage}`;
  btnPrev.disabled = currentPage === 1;
  btnNext.disabled = lastResultCount < LIMIT;
}

// ── Carga principal ───────────────────────────────────────────────
async function loadLogs(page) {
  spinner.classList.remove('d-none');
  tableContainer.classList.add('d-none');

  const params = new URLSearchParams();
  params.set('skip', (page - 1) * LIMIT);
  params.set('limit', LIMIT);

  const filterUser   = document.getElementById('filter-user');
  const filterAction = document.getElementById('filter-action');
  const filterFrom   = document.getElementById('filter-from');
  const filterTo     = document.getElementById('filter-to');

  if (filterUser && filterUser.value.trim())   params.set('user',      filterUser.value.trim());
  if (filterAction && filterAction.value)       params.set('action',    filterAction.value);
  if (filterFrom && filterFrom.value)           params.set('from_date', new Date(filterFrom.value).toISOString());
  if (filterTo && filterTo.value)               params.set('to_date',   new Date(filterTo.value + 'T23:59:59').toISOString());

  try {
    const data = await api.get(`/api/v1/audit/?${params.toString()}`);
    if (!data) return;
    currentPage = page;
    lastResultCount = data.length;
    renderTable(data);
    updatePagination();
  } catch (err) {
    showToast(`Error al cargar registros: ${err.message}`, 'danger');
  } finally {
    spinner.classList.add('d-none');
    tableContainer.classList.remove('d-none');
  }
}

// ── Toast ─────────────────────────────────────────────────────────
function showToast(msg, type = 'info') {
  toastEl.className = `toast align-items-center border-0 text-bg-${type}`;
  toastMsg.textContent = msg;
  bsToast.show();
}

// ── Eventos ───────────────────────────────────────────────────────
document.getElementById('btn-search').addEventListener('click', () => {
  currentPage = 1;
  loadLogs(1);
});

document.getElementById('btn-clear').addEventListener('click', () => {
  const filterUser   = document.getElementById('filter-user');
  const filterAction = document.getElementById('filter-action');
  const filterFrom   = document.getElementById('filter-from');
  const filterTo     = document.getElementById('filter-to');
  if (filterUser)   filterUser.value   = '';
  if (filterAction) filterAction.value = '';
  if (filterFrom)   filterFrom.value   = '';
  if (filterTo)     filterTo.value     = '';
  currentPage = 1;
  loadLogs(1);
});

btnPrev.addEventListener('click', () => {
  if (currentPage > 1) loadLogs(currentPage - 1);
});

btnNext.addEventListener('click', () => {
  if (lastResultCount >= LIMIT) loadLogs(currentPage + 1);
});

// ── Init ──────────────────────────────────────────────────────────
loadLogs(1);
