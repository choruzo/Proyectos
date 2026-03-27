#  Web - Visualización de Datos

## Visión General

Gráficos, charts y visualizaciones de métricas del pipeline usando Chart.js.

**Relacionado con**:
- [[Web - Frontend Components]] - Implementación Alpine.js
- [[Web - API Endpoints]] - Datos consumidos
- [[Modelo de Datos]] - Fuente de métricas

---

## Chart.js - Configuración

### Gráfico de Deployments (Line Chart)

```javascript
const deploymentChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'],
        datasets: [
            {
                label: 'Successful',
                data: [3, 4, 2, 5, 3, 4, 3],
                borderColor: 'rgb(16, 185, 129)',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                tension: 0.4
            },
            {
                label: 'Failed',
                data: [1, 0, 1, 0, 2, 1, 0],
                borderColor: 'rgb(239, 68, 68)',
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                tension: 0.4
            }
        ]
    },
    options: {
        responsive: true,
        plugins: {
            legend: { position: 'top' },
            title: { display: true, text: 'Deployments (Last 7 Days)' }
        },
        scales: {
            y: { beginAtZero: true }
        }
    }
});
```

---

### Gráfico de Coverage (Line Chart)

```javascript
const coverageChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: tagNames,
        datasets: [{
            label: 'Coverage %',
            data: coverageValues,
            borderColor: 'rgb(59, 130, 246)',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            tension: 0.4
        }]
    },
    options: {
        scales: {
            y: {
                beginAtZero: true,
                max: 100,
                ticks: {
                    callback: function(value) {
                        return value + '%';
                    }
                }
            }
        }
    }
});
```

---

### Gráfico de Distribución (Doughnut Chart)

```javascript
const statusChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
        labels: ['Success', 'Failed', 'In Progress'],
        datasets: [{
            data: [129, 27, 3],
            backgroundColor: [
                'rgb(16, 185, 129)',
                'rgb(239, 68, 68)',
                'rgb(59, 130, 246)'
            ]
        }]
    },
    options: {
        plugins: {
            legend: { position: 'right' }
        }
    }
});
```

---

## Badges de Estado

**HTML + Tailwind**:
```html
<span class="px-2 py-1 rounded-full text-xs font-semibold text-white"
      :class="{
          'bg-green-500': status === 'success',
          'bg-red-500': status === 'failed',
          'bg-yellow-500': status === 'pending',
          'bg-blue-500': status === 'compiling' || status === 'analyzing',
          'bg-purple-500': status === 'deploying'
      }">
    <span x-text="status"></span>
</span>
```

---

## Progress Bar

**Para deployments en progreso**:
```html
<div class="w-full bg-gray-200 rounded-full h-2">
    <div class="bg-blue-500 h-2 rounded-full transition-all duration-500"
         :style="'width: ' + progress + '%'">
    </div>
</div>
```

**Cálculo de progreso**:
```javascript
function calculateProgress(deployment) {
    const phases = {
        'pending': 0,
        'compiling': 25,
        'analyzing': 50,
        'deploying': 75,
        'success': 100,
        'failed': 100
    };
    return phases[deployment.status] || 0;
}
```

---

## Auto-refresh

**Alpine.js implementation**:
```javascript
{
    autoRefresh: true,
    refreshInterval: 30000, // 30 seconds
    
    init() {
        if (this.autoRefresh) {
            setInterval(() => {
                this.loadData();
            }, this.refreshInterval);
        }
    },
    
    toggleAutoRefresh() {
        this.autoRefresh = !this.autoRefresh;
        if (this.autoRefresh) {
            this.startAutoRefresh();
        }
    }
}
```

---

## Enlaces Relacionados

- [[Web - Frontend Components]]
- [[Web - API Endpoints]]
- [[Arquitectura Web UI]]
