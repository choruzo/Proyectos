/**
 * datastores.js — Listado de datastores (solo lectura)
 */

let allDatastores = [];

const spinner = document.getElementById('spinner');
const tableContainer = document.getElementById('table-container');
const tbody = document.getElementById('ds-tbody');
const noResults = document.getElementById('no-results');

const toastEl = document.getElementById('action-toast');
const toastMsg = document.getElementById('toast-msg');
const bsToast = new bootstrap.Toast(toastEl, { delay: 4000 });

// ── Helpers ───────────────────────────────────────────────────────
function typeBadge(type) {
  const t = (type || '').toUpperCase();
  if (t === 'VMFS')          return `<span class="badge bg-primary">VMFS</span>`;
  if (t === 'NFS')           return `<span class="badge bg-info text-dark">NFS</span>`;
  if (t === 'NFS41')         return `<span class="badge bg-info text-dark">NFS 4.1</span>`;
  if (t === 'VSAN')          return `<span class="badge" style="background:#6f42c1">vSAN</span>`;
  return `<span class="badge bg-secondary">${type || '—'}</span>`;
}

function progressBar(pct) {
  const color = pct >= 90 ? 'bg-danger' : pct >= 70 ? 'bg-warning' : 'bg-success';
  return `
    <div class="d-flex align-items-center gap-2">
      <div class="progress flex-grow-1" style="height:8px">
        <div class="progress-bar ${color}" role="progressbar" style="width:${pct}%"
             aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100"></div>
      </div>
      <small class="text-nowrap">${pct}%</small>
    </div>`;
}

// ── Render tabla ──────────────────────────────────────────────────
function renderTable(datastores) {
  tbody.innerHTML = '';
  if (datastores.length === 0) {
    noResults.classList.remove('d-none');
    return;
  }
  noResults.classList.add('d-none');
  datastores.forEach(ds => {
    const inaccessible = ds.accessible === false
      ? ' <span class="badge bg-danger ms-1">No accesible</span>'
      : '';
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="fw-semibold">${ds.name}${inaccessible}</td>
      <td>${typeBadge(ds.type)}</td>
      <td class="text-nowrap">${ds.capacity_gb} GB</td>
      <td class="text-nowrap">${ds.free_gb} GB</td>
      <td>${progressBar(ds.used_pct)}</td>
      <td>${ds.vm_count}</td>`;
    tbody.appendChild(tr);
  });
}

// ── Filtro local ──────────────────────────────────────────────────
function applyFilter() {
  const text = document.getElementById('search-input').value.toLowerCase();
  const type = document.getElementById('filter-type').value.toUpperCase();

  const filtered = allDatastores.filter(ds => {
    const matchText = !text || ds.name.toLowerCase().includes(text);
    const matchType = !type || (ds.type || '').toUpperCase() === type;
    return matchText && matchType;
  });
  renderTable(filtered);
}

// ── Carga de datastores ───────────────────────────────────────────
async function loadDatastores(silent = false) {
  if (!silent) {
    spinner.classList.remove('d-none');
    tableContainer.classList.add('d-none');
  }
  try {
    const data = await api.get('/api/v1/datastores/');
    if (!data) return;
    allDatastores = data;
    spinner.classList.add('d-none');
    tableContainer.classList.remove('d-none');
    applyFilter();
  } catch (err) {
    spinner.classList.add('d-none');
    tableContainer.classList.remove('d-none');
    showToast(`Error al cargar datastores: ${err.message}`, 'danger');
  }
}

// ── Toast ─────────────────────────────────────────────────────────
function showToast(msg, type = 'success') {
  toastEl.className = `toast align-items-center border-0 text-bg-${type}`;
  toastMsg.textContent = msg;
  bsToast.show();
}

// ── Eventos ───────────────────────────────────────────────────────
document.getElementById('search-input').addEventListener('input', applyFilter);
document.getElementById('filter-type').addEventListener('change', applyFilter);
document.getElementById('btn-refresh').addEventListener('click', () => loadDatastores());

// ── Init ──────────────────────────────────────────────────────────
loadDatastores();
setInterval(() => loadDatastores(true), 60000);
