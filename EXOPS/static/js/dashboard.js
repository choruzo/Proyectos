/**
 * dashboard.js — KPIs en tiempo real para el dashboard principal
 */

const spinner      = document.getElementById('spinner');
const kpiContainer = document.getElementById('kpi-container');
const toastEl      = document.getElementById('action-toast');
const toastMsg     = document.getElementById('toast-msg');
const bsToast      = new bootstrap.Toast(toastEl, { delay: 5000 });

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

// ── Carga principal ───────────────────────────────────────────────

async function loadDashboard(silent = false) {
  if (!silent) {
    spinner.classList.remove('d-none');
    kpiContainer.classList.add('d-none');
  }

  try {
    const [vms, hosts, datastores] = await Promise.all([
      api.get('/api/v1/vms/'),
      api.get('/api/v1/hosts/'),
      api.get('/api/v1/datastores/'),
    ]);

    if (!vms || !hosts || !datastores) return;

    renderKPIs(vms, hosts, datastores);

    spinner.classList.add('d-none');
    kpiContainer.classList.remove('d-none');
  } catch (err) {
    spinner.classList.add('d-none');
    kpiContainer.classList.remove('d-none');
    showToast(`Error al cargar datos: ${err.message}`);
  }
}

// ── Eventos y arranque ────────────────────────────────────────────

document.getElementById('btn-refresh').addEventListener('click', () => loadDashboard());

loadDashboard();
setInterval(() => loadDashboard(true), 30000);
