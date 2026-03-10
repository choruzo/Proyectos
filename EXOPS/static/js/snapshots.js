/**
 * snapshots.js — Árbol de snapshots por VM con acciones crear/restaurar/eliminar
 */

let currentVmId = null;
let currentVmName = null;
let allSnapshots = [];
let currentSnapshotId = null;
let pendingAction = null;

const spinner = document.getElementById('spinner');
const tableContainer = document.getElementById('table-container');
const placeholder = document.getElementById('placeholder');
const tbody = document.getElementById('snap-tbody');
const noResults = document.getElementById('no-results');
const noFilterResults = document.getElementById('no-filter-results');

const toastEl = document.getElementById('action-toast');
const toastMsg = document.getElementById('toast-msg');
const bsToast = new bootstrap.Toast(toastEl, { delay: 5000 });

const createModal = new bootstrap.Modal(document.getElementById('create-modal'));
const restoreModal = new bootstrap.Modal(document.getElementById('restore-modal'));
const deleteModal = new bootstrap.Modal(document.getElementById('delete-modal'));

// ── Sanitización HTML ─────────────────────────────────────────────
function esc(str) {
  return (str || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function escAttr(str) {
  return (str || '').replace(/'/g, "\\'");
}

// ── Toast ─────────────────────────────────────────────────────────
function showToast(msg, type = 'success') {
  toastEl.className = `toast align-items-center border-0 text-bg-${type}`;
  toastMsg.textContent = msg;
  bsToast.show();
}

// ── Formatear fecha ───────────────────────────────────────────────
function formatDate(iso) {
  if (!iso) return '<span class="text-muted">—</span>';
  const d = new Date(iso);
  return d.toLocaleString('es-ES', { dateStyle: 'short', timeStyle: 'short' });
}

// ── Badge estado VM en snapshot ───────────────────────────────────
function stateBadge(state) {
  const map = {
    poweredOn:  '<span class="badge bg-success">Encendida</span>',
    poweredOff: '<span class="badge bg-secondary">Apagada</span>',
    suspended:  '<span class="badge bg-warning text-dark">Suspendida</span>',
  };
  return map[state] || `<span class="badge bg-secondary">${esc(state)}</span>`;
}

// ── Render árbol (recursivo → filas planas con indentación) ───────
function flattenTree(snapshots, rows = []) {
  snapshots.forEach(snap => {
    rows.push(snap);
    if (snap.children && snap.children.length > 0) {
      flattenTree(snap.children, rows);
    }
  });
  return rows;
}

function renderTree(snapshots) {
  tbody.innerHTML = '';
  noResults.classList.add('d-none');
  noFilterResults.classList.add('d-none');

  const searchText = document.getElementById('search-input').value.toLowerCase();
  const flat = flattenTree(snapshots);

  if (flat.length === 0) {
    noResults.classList.remove('d-none');
    return;
  }

  const filtered = searchText
    ? flat.filter(s => s.name.toLowerCase().includes(searchText) || (s.description || '').toLowerCase().includes(searchText))
    : flat;

  if (filtered.length === 0) {
    noFilterResults.classList.remove('d-none');
    return;
  }

  filtered.forEach(snap => {
    const isCurrent = snap.id === currentSnapshotId;
    const indent = snap.depth * 20;
    const prefix = snap.depth > 0 ? '└─ ' : '';
    const currentBadge = isCurrent ? ' <span class="badge bg-success ms-1">actual</span>' : '';

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="text-muted small">${esc(currentVmName)}</td>
      <td>
        <span style="padding-left:${indent}px">${esc(prefix)}<span class="fw-semibold">${esc(snap.name)}</span>${currentBadge}</span>
      </td>
      <td class="text-muted small text-truncate" style="max-width:200px" title="${esc(snap.description)}">${esc(snap.description) || '<span class="text-muted">—</span>'}</td>
      <td class="small">${formatDate(snap.created_at)}</td>
      <td>${stateBadge(snap.state)}</td>
      <td>
        <button class="btn btn-sm btn-outline-warning me-1" onclick="confirmRestore('${escAttr(snap.id)}','${escAttr(snap.name)}')" title="Restaurar a este snapshot">
          <i class="bi bi-arrow-counterclockwise me-1"></i>Restaurar
        </button>
        <button class="btn btn-sm btn-outline-danger" onclick="confirmDelete('${escAttr(snap.id)}','${escAttr(snap.name)}')" title="Eliminar snapshot">
          <i class="bi bi-trash"></i>
        </button>
      </td>`;
    tbody.appendChild(tr);
  });
}

// ── Cargar VMs en el selector ─────────────────────────────────────
async function loadVMs() {
  const select = document.getElementById('vm-select');
  try {
    const vms = await api.get('/api/v1/vms/');
    if (!vms) return;
    vms.sort((a, b) => a.name.localeCompare(b.name));
    vms.forEach(vm => {
      const opt = document.createElement('option');
      opt.value = vm.id;
      opt.textContent = vm.name;
      select.appendChild(opt);
    });
  } catch (err) {
    showToast(`Error al cargar VMs: ${err.message}`, 'danger');
  }
}

// ── Cargar snapshots de la VM seleccionada ────────────────────────
async function loadSnapshots(vmId, silent = false) {
  if (!vmId) return;
  if (!silent) {
    spinner.classList.remove('d-none');
    tableContainer.classList.add('d-none');
    placeholder.classList.add('d-none');
  }
  try {
    const data = await api.get(`/api/v1/snapshots/${vmId}`);
    if (!data) return;
    currentVmId = vmId;
    currentVmName = data.vm_name;
    allSnapshots = data.snapshots;
    currentSnapshotId = data.current_snapshot_id;

    spinner.classList.add('d-none');
    tableContainer.classList.remove('d-none');
    renderTree(allSnapshots);
  } catch (err) {
    spinner.classList.add('d-none');
    placeholder.classList.add('d-none');
    tableContainer.classList.remove('d-none');
    showToast(`Error al cargar snapshots: ${err.message}`, 'danger');
  }
}

// ── Confirmar restaurar ───────────────────────────────────────────
function confirmRestore(snapId, snapName) {
  document.getElementById('restore-body').innerHTML =
    `<p>¿Seguro que quieres restaurar la VM <strong>${esc(currentVmName)}</strong> al snapshot <strong>${esc(snapName)}</strong>?</p>
     <p class="text-warning fw-semibold"><i class="bi bi-exclamation-triangle me-1"></i>Esta acción revertirá el estado actual de la VM. Los cambios posteriores al snapshot se perderán.</p>`;
  pendingAction = { type: 'restore', snapId, snapName };
  restoreModal.show();
}

// ── Confirmar eliminar ────────────────────────────────────────────
function confirmDelete(snapId, snapName) {
  document.getElementById('delete-body').innerHTML =
    `<p>¿Seguro que quieres eliminar el snapshot <strong>${esc(snapName)}</strong> de la VM <strong>${esc(currentVmName)}</strong>?</p>
     <p class="text-muted small">No se eliminarán los snapshots hijo.</p>`;
  pendingAction = { type: 'delete', snapId, snapName };
  deleteModal.show();
}

// ── Ejecutar restaurar ────────────────────────────────────────────
document.getElementById('restore-confirm-btn').addEventListener('click', async () => {
  restoreModal.hide();
  if (!pendingAction || pendingAction.type !== 'restore') return;
  const { snapId, snapName } = pendingAction;
  pendingAction = null;
  showToast(`Restaurando a "${snapName}"…`, 'info');
  try {
    await api.post(`/api/v1/snapshots/${currentVmId}/${snapId}/restore`);
    showToast(`VM restaurada al snapshot "${snapName}" correctamente.`, 'success');
  } catch (err) {
    showToast(`Error al restaurar: ${err.message}`, 'danger');
  } finally {
    await loadSnapshots(currentVmId, true);
  }
});

// ── Ejecutar eliminar ─────────────────────────────────────────────
document.getElementById('delete-confirm-btn').addEventListener('click', async () => {
  deleteModal.hide();
  if (!pendingAction || pendingAction.type !== 'delete') return;
  const { snapId, snapName } = pendingAction;
  pendingAction = null;
  showToast(`Eliminando snapshot "${snapName}"…`, 'info');
  try {
    await api.delete(`/api/v1/snapshots/${currentVmId}/${snapId}`);
    showToast(`Snapshot "${snapName}" eliminado correctamente.`, 'success');
  } catch (err) {
    showToast(`Error al eliminar: ${err.message}`, 'danger');
  } finally {
    await loadSnapshots(currentVmId, true);
  }
});

// ── Crear snapshot ────────────────────────────────────────────────
document.getElementById('btn-new-snapshot').addEventListener('click', () => {
  document.getElementById('snap-name').value = '';
  document.getElementById('snap-desc').value = '';
  createModal.show();
});

document.getElementById('create-confirm-btn').addEventListener('click', async () => {
  const name = document.getElementById('snap-name').value.trim();
  if (!name) {
    document.getElementById('snap-name').classList.add('is-invalid');
    return;
  }
  document.getElementById('snap-name').classList.remove('is-invalid');
  const description = document.getElementById('snap-desc').value.trim();
  createModal.hide();
  showToast(`Creando snapshot "${name}"…`, 'info');
  try {
    await api.post(`/api/v1/snapshots/${currentVmId}`, { name, description });
    showToast(`Snapshot "${name}" creado correctamente.`, 'success');
  } catch (err) {
    showToast(`Error al crear snapshot: ${err.message}`, 'danger');
  } finally {
    await loadSnapshots(currentVmId, true);
  }
});

// ── Eventos ───────────────────────────────────────────────────────
document.getElementById('vm-select').addEventListener('change', (e) => {
  const vmId = e.target.value;
  if (!vmId) {
    currentVmId = null;
    currentVmName = null;
    tableContainer.classList.add('d-none');
    placeholder.classList.remove('d-none');
    document.getElementById('search-input').disabled = true;
    document.getElementById('btn-refresh').disabled = true;
    document.getElementById('btn-new-snapshot').disabled = true;
    return;
  }
  document.getElementById('search-input').disabled = false;
  document.getElementById('btn-refresh').disabled = false;
  document.getElementById('btn-new-snapshot').disabled = false;
  loadSnapshots(vmId);
});

document.getElementById('search-input').addEventListener('input', () => {
  if (currentVmId) renderTree(allSnapshots);
});

document.getElementById('btn-refresh').addEventListener('click', () => {
  if (currentVmId) loadSnapshots(currentVmId);
});

// ── Init ──────────────────────────────────────────────────────────
loadVMs();
