#  Web - Frontend Components

## Visión General

Componentes Alpine.js, templates Jinja2 y estructura del frontend de la Web UI.

**Relacionado con**:
- [[Arquitectura Web UI#Frontend]]
- [[Web - API Endpoints]] - APIs consumidas
- [[Web - Visualización de Datos]] - Charts y gráficos

---

## Stack Frontend

- **Alpine.js 3.x**: Reactividad ligera
- **Tailwind CSS**: Utility-first CSS
- **Chart.js 4.x**: Gráficos
- **Jinja2**: Templating server-side

---

## Template Base

**Archivo**: `templates/base.html`

**Estructura**:
```html
<!DOCTYPE html>
<html lang="es" x-data="appData()" x-init="init()" :class="{ 'dark': darkMode }">
<head>
    <meta charset="UTF-8">
    <title>{% block title %}CI/CD Pipeline{% endblock %}</title>
    
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    
    <!-- Alpine.js -->
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
    
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.x.x"></script>
    
    <!-- Custom CSS -->
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body class="bg-gray-100 dark:bg-gray-900 text-gray-900 dark:text-gray-100">
    
    <!-- Navbar -->
    {% include 'partials/navbar.html' %}
    
    <!-- Sidebar + Content -->
    <div class="flex">
        {% include 'partials/sidebar.html' %}
        
        <main class="flex-1 p-6">
            {% block content %}{% endblock %}
        </main>
    </div>
    
    <!-- Toast Notifications -->
    <div x-show="toast.visible" x-transition class="toast">
        <span x-text="toast.message"></span>
    </div>
    
    <script src="{{ url_for('static', filename='js/app.js') }}"></script>
</body>
</html>
```

---

## Componente: Dashboard

**Archivo**: `templates/dashboard.html`

**Alpine.js Data**:
```javascript
function dashboardData() {
    return {
        stats: {},
        recentDeployments: [],
        chart: null,
        autoRefresh: true,
        refreshInterval: 30000, // 30 seconds
        
        init() {
            this.loadStats();
            this.loadRecentDeployments();
            this.loadChartData();
            
            if (this.autoRefresh) {
                setInterval(() => this.refresh(), this.refreshInterval);
            }
        },
        
        async loadStats() {
            try {
                const response = await fetch('/api/dashboard/stats');
                this.stats = await response.json();
            } catch (error) {
                console.error('Failed to load stats:', error);
                showToast('Error loading stats', 'error');
            }
        },
        
        async loadRecentDeployments() {
            const response = await fetch('/api/dashboard/recent-deployments');
            this.recentDeployments = await response.json();
        },
        
        async loadChartData() {
            const response = await fetch('/api/dashboard/chart-data');
            const data = await response.json();
            this.renderChart(data);
        },
        
        renderChart(data) {
            const ctx = document.getElementById('deploymentChart').getContext('2d');
            
            if (this.chart) {
                this.chart.destroy();
            }
            
            this.chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.labels,
                    datasets: data.datasets.map(ds => ({
                        label: ds.label,
                        data: ds.data,
                        borderColor: ds.label === 'Successful' ? 'rgb(16, 185, 129)' : 'rgb(239, 68, 68)',
                        backgroundColor: ds.label === 'Successful' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                        tension: 0.4
                    }))
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { position: 'top' }
                    }
                }
            });
        },
        
        refresh() {
            this.loadStats();
            this.loadRecentDeployments();
        },
        
        getBadgeClass(status) {
            const classes = {
                'success': 'bg-green-500',
                'failed': 'bg-red-500',
                'pending': 'bg-yellow-500',
                'compiling': 'bg-blue-500',
                'analyzing': 'bg-blue-500',
                'deploying': 'bg-purple-500'
            };
            return classes[status] || 'bg-gray-500';
        }
    }
}
```

**Template HTML**:
```html
{% extends "base.html" %}
{% block content %}

<div x-data="dashboardData()" x-init="init()">
    
    <!-- Stats Cards -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <h3 class="text-gray-500 text-sm font-medium">Total Deployments</h3>
            <p class="text-3xl font-bold mt-2" x-text="stats.total_deployments || '-'"></p>
        </div>
        
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <h3 class="text-gray-500 text-sm font-medium">Success Rate</h3>
            <p class="text-3xl font-bold mt-2" x-text="(stats.success_rate || 0).toFixed(1) + '%'"></p>
        </div>
        
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <h3 class="text-gray-500 text-sm font-medium">Last 24h</h3>
            <p class="text-3xl font-bold mt-2" x-text="stats.last_24h || '0'"></p>
        </div>
        
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <h3 class="text-gray-500 text-sm font-medium">Avg Duration</h3>
            <p class="text-3xl font-bold mt-2" x-text="(stats.avg_duration_minutes || 0).toFixed(0) + ' min'"></p>
        </div>
    </div>
    
    <!-- Recent Deployments -->
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6 mb-8">
        <h2 class="text-xl font-bold mb-4">Recent Deployments</h2>
        <table class="w-full">
            <thead>
                <tr class="border-b dark:border-gray-700">
                    <th class="text-left p-2">Tag</th>
                    <th class="text-left p-2">Status</th>
                    <th class="text-left p-2">Started At</th>
                    <th class="text-left p-2">Duration</th>
                </tr>
            </thead>
            <tbody>
                <template x-for="deployment in recentDeployments" :key="deployment.id">
                    <tr class="border-b dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700">
                        <td class="p-2" x-text="deployment.tag_name"></td>
                        <td class="p-2">
                            <span class="px-2 py-1 rounded-full text-xs text-white" 
                                  :class="getBadgeClass(deployment.status)"
                                  x-text="deployment.status"></span>
                        </td>
                        <td class="p-2" x-text="formatDate(deployment.started_at)"></td>
                        <td class="p-2" x-text="formatDuration(deployment.duration_seconds)"></td>
                    </tr>
                </template>
            </tbody>
        </table>
    </div>
    
    <!-- Chart -->
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h2 class="text-xl font-bold mb-4">Deployment Trend (Last 7 Days)</h2>
        <canvas id="deploymentChart"></canvas>
    </div>
    
</div>

{% endblock %}
```

---

## Dark Mode

**Toggle button**:
```html
<button @click="darkMode = !darkMode; saveDarkMode()" 
        class="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700">
    <span x-show="!darkMode">🌙</span>
    <span x-show="darkMode">☀️</span>
</button>
```

**Persistence**:
```javascript
function appData() {
    return {
        darkMode: localStorage.getItem('darkMode') === 'true',
        
        saveDarkMode() {
            localStorage.setItem('darkMode', this.darkMode);
        }
    }
}
```

---

## Utilidades JavaScript

**Archivo**: `static/js/app.js`

```javascript
// Date formatting
function formatDate(dateString) {
    if (!dateString) return '-';
    const date = new Date(dateString);
    return date.toLocaleString('es-ES');
}

// Duration formatting
function formatDuration(seconds) {
    if (!seconds) return '-';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) {
        return `${hours}h ${minutes}m`;
    } else if (minutes > 0) {
        return `${minutes}m ${secs}s`;
    } else {
        return `${secs}s`;
    }
}

// Toast notifications
function showToast(message, type = 'info') {
    Alpine.store('toast', {
        visible: true,
        message: message,
        type: type
    });
    
    setTimeout(() => {
        Alpine.store('toast', { visible: false });
    }, 3000);
}

// Copy to clipboard
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showToast('Copied to clipboard', 'success');
    });
}
```

---

## Custom CSS

**Archivo**: `static/css/style.css`

```css
/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
}

.dark ::-webkit-scrollbar-track {
    background: #1f2937;
}

::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #555;
}

/* Badges */
.badge {
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Toast */
.toast {
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    background: #1f2937;
    color: white;
    padding: 1rem 1.5rem;
    border-radius: 0.5rem;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    z-index: 9999;
}
```

---

## Enlaces Relacionados

- [[Arquitectura Web UI]]
- [[Web - API Endpoints]]
- [[Web - Visualización de Datos]]
