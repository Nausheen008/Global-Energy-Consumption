// Global variables and data
let globalData = {
    countries: ["United States", "China", "India", "Germany", "United Kingdom", "France", "Brazil", "Canada", "Russia", "Japan"],
    years: [2000, 2005, 2010, 2015, 2020, 2022],
    developmentStatus: ["Developed", "Developing", "Least Developed"],
    sampleData: {
        co2_per_capita: [15.2, 7.4, 1.9, 9.1, 5.6, 4.8, 2.3, 15.8, 11.2, 8.7],
        gdp_per_capita: [45000, 12000, 2100, 48000, 42000, 41000, 8500, 46000, 11000, 40000],
        renewables_share: [8, 26, 21, 42, 38, 19, 45, 18, 1, 18],
        fossil_share: [92, 74, 79, 58, 62, 81, 55, 82, 99, 82],
        development: ["Developed", "Developing", "Developing", "Developed", "Developed", "Developed", "Developing", "Developed", "Developing", "Developed"]
    }
};

let currentFilters = {
    yearStart: 2000,
    yearEnd: 2022,
    development: 'all',
    countries: []
};

let charts = {};
let currentPage = 1;
const itemsPerPage = 10;

// Initialize the dashboard
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
    setupEventListeners();
    updateMetrics();
    initializeCharts();
    updateDataTable();
});

function initializeDashboard() {
    // Initialize year range displays
    document.getElementById('yearStartDisplay').textContent = '2000';
    document.getElementById('yearEndDisplay').textContent = '2022';
}

function setupEventListeners() {
    // Tab navigation
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.addEventListener('click', function() {
            const tabId = this.dataset.tab;
            switchTab(tabId);
        });
    });

    // Year range sliders
    const yearStartSlider = document.getElementById('yearStart');
    const yearEndSlider = document.getElementById('yearEnd');
    
    yearStartSlider.addEventListener('input', function() {
        document.getElementById('yearStartDisplay').textContent = this.value;
        if (parseInt(this.value) > parseInt(yearEndSlider.value)) {
            yearEndSlider.value = this.value;
            document.getElementById('yearEndDisplay').textContent = this.value;
        }
    });

    yearEndSlider.addEventListener('input', function() {
        document.getElementById('yearEndDisplay').textContent = this.value;
        if (parseInt(this.value) < parseInt(yearStartSlider.value)) {
            yearStartSlider.value = this.value;
            document.getElementById('yearStartDisplay').textContent = this.value;
        }
    });

    // Filter controls
    document.getElementById('applyFilters').addEventListener('click', applyFilters);
    document.getElementById('resetFilters').addEventListener('click', resetFilters);

    // Data explorer controls
    document.getElementById('exportData').addEventListener('click', exportData);
    document.getElementById('viewStats').addEventListener('click', showStatsModal);
    document.getElementById('closeModal').addEventListener('click', hideStatsModal);
    
    // Pagination
    document.getElementById('prevPage').addEventListener('click', () => changePage(-1));
    document.getElementById('nextPage').addEventListener('click', () => changePage(1));

    // Modal click outside to close
    document.getElementById('statsModal').addEventListener('click', function(e) {
        if (e.target === this) {
            hideStatsModal();
        }
    });
}

function switchTab(tabId) {
    // Update tab buttons
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(`${tabId}-tab`).classList.add('active');

    // Resize charts if needed
    setTimeout(() => {
        Object.values(charts).forEach(chart => {
            if (chart && chart.resize) {
                chart.resize();
            }
        });
    }, 100);
}

function updateMetrics() {
    const filteredData = getFilteredData();
    
    document.getElementById('totalCountries').textContent = filteredData.length;
    
    if (filteredData.length > 0) {
        const avgCO2 = (filteredData.reduce((sum, item) => sum + item.co2_per_capita, 0) / filteredData.length).toFixed(1);
        const avgRenewable = Math.round(filteredData.reduce((sum, item) => sum + item.renewables_share, 0) / filteredData.length);
        const avgFossil = Math.round(filteredData.reduce((sum, item) => sum + item.fossil_share, 0) / filteredData.length);
        
        document.getElementById('avgCO2').textContent = avgCO2;
        document.getElementById('renewablePercent').textContent = avgRenewable + '%';
        document.getElementById('fossilPercent').textContent = avgFossil + '%';
    }
}

function getFilteredData() {
    const data = [];
    for (let i = 0; i < globalData.countries.length; i++) {
        if (currentFilters.countries.length === 0 || currentFilters.countries.includes(globalData.countries[i])) {
            if (currentFilters.development === 'all' || globalData.sampleData.development[i] === currentFilters.development) {
                data.push({
                    country: globalData.countries[i],
                    co2_per_capita: globalData.sampleData.co2_per_capita[i],
                    gdp_per_capita: globalData.sampleData.gdp_per_capita[i],
                    renewables_share: globalData.sampleData.renewables_share[i],
                    fossil_share: globalData.sampleData.fossil_share[i],
                    development: globalData.sampleData.development[i],
                    year: 2022 // Using latest year as default
                });
            }
        }
    }
    return data;
}

function initializeCharts() {
    createGlobalCO2Chart();
    createEnergyMixChart();
    createTrendsChart();
    createTopEmittersChart();
    createTopPerCapitaChart();
    createGDPCO2Chart();
    createClusterChart();
}

function createGlobalCO2Chart() {
    const ctx = document.getElementById('globalCO2Chart').getContext('2d');
    const years = [2000, 2005, 2010, 2015, 2020, 2022];
    const co2Data = [24.2, 28.1, 33.5, 35.7, 34.8, 36.7]; // Global CO2 in billions of tons

    charts.globalCO2 = new Chart(ctx, {
        type: 'line',
        data: {
            labels: years,
            datasets: [{
                label: 'Global CO2 Emissions (Gt)',
                data: co2Data,
                borderColor: '#1FB8CD',
                backgroundColor: 'rgba(31, 184, 205, 0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 5,
                pointHoverRadius: 7
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: 'CO2 Emissions (Gt)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Year'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

function createEnergyMixChart() {
    const ctx = document.getElementById('energyMixChart').getContext('2d');
    const years = [2000, 2005, 2010, 2015, 2020, 2022];
    
    charts.energyMix = new Chart(ctx, {
        type: 'line',
        data: {
            labels: years,
            datasets: [{
                label: 'Renewables',
                data: [7, 8, 10, 15, 22, 29],
                borderColor: '#B4413C',
                backgroundColor: 'rgba(180, 65, 60, 0.1)',
                fill: false,
                tension: 0.4
            }, {
                label: 'Fossil Fuels',
                data: [78, 79, 77, 73, 68, 63],
                borderColor: '#5D878F',
                backgroundColor: 'rgba(93, 135, 143, 0.1)',
                fill: false,
                tension: 0.4
            }, {
                label: 'Nuclear',
                data: [15, 13, 13, 12, 10, 8],
                borderColor: '#FFC185',
                backgroundColor: 'rgba(255, 193, 133, 0.1)',
                fill: false,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Energy Share (%)'
                    }
                }
            }
        }
    });
}

function createTrendsChart() {
    const ctx = document.getElementById('trendsChart').getContext('2d');
    const years = [2000, 2005, 2010, 2015, 2020, 2022];
    
    charts.trends = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: years,
            datasets: [{
                label: 'Solar',
                data: [0.1, 0.5, 2.1, 7.2, 15.4, 28.3],
                backgroundColor: '#1FB8CD'
            }, {
                label: 'Wind',
                data: [1.2, 3.4, 8.7, 18.9, 35.2, 42.1],
                backgroundColor: '#FFC185'
            }, {
                label: 'Hydro',
                data: [45.2, 47.1, 48.9, 52.3, 58.1, 61.7],
                backgroundColor: '#B4413C'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    stacked: true,
                    title: {
                        display: true,
                        text: 'Year'
                    }
                },
                y: {
                    stacked: true,
                    title: {
                        display: true,
                        text: 'Energy Generation (TWh)'
                    }
                }
            }
        }
    });
}

function createTopEmittersChart() {
    const ctx = document.getElementById('topEmittersChart').getContext('2d');
    const countries = ['China', 'United States', 'India', 'Russia', 'Japan'];
    const emissions = [11680, 4713, 2654, 1661, 1107]; // Million tons CO2
    
    charts.topEmitters = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: countries,
            datasets: [{
                data: emissions,
                backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C', '#5D878F', '#DB4545']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'CO2 Emissions (Mt)'
                    }
                }
            }
        }
    });
}

function createTopPerCapitaChart() {
    const ctx = document.getElementById('topPerCapitaChart').getContext('2d');
    const countries = ['Canada', 'United States', 'Russia', 'Germany', 'Japan'];
    const perCapita = [15.8, 15.2, 11.2, 9.1, 8.7]; // Tons per capita
    
    charts.topPerCapita = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: countries,
            datasets: [{
                data: perCapita,
                backgroundColor: ['#ECEBD5', '#1FB8CD', '#5D878F', '#FFC185', '#DB4545']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'CO2 per Capita (tons)'
                    }
                }
            }
        }
    });
}

function createGDPCO2Chart() {
    const ctx = document.getElementById('gdpCO2Chart').getContext('2d');
    const filteredData = getFilteredData();
    
    const developedData = filteredData.filter(d => d.development === 'Developed');
    const developingData = filteredData.filter(d => d.development === 'Developing');
    
    charts.gdpCO2 = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Developed',
                data: developedData.map(d => ({x: d.gdp_per_capita, y: d.co2_per_capita})),
                backgroundColor: '#1FB8CD',
                borderColor: '#1FB8CD'
            }, {
                label: 'Developing',
                data: developingData.map(d => ({x: d.gdp_per_capita, y: d.co2_per_capita})),
                backgroundColor: '#B4413C',
                borderColor: '#B4413C'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'GDP per Capita (USD)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'CO2 per Capita (tons)'
                    }
                }
            }
        }
    });
}

function createClusterChart() {
    const ctx = document.getElementById('clusterChart').getContext('2d');
    
    charts.cluster = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['High Renewable', 'Fossil Dependent'],
            datasets: [{
                data: [4, 6],
                backgroundColor: ['#B4413C', '#5D878F'],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

function applyFilters() {
    currentFilters.yearStart = parseInt(document.getElementById('yearStart').value);
    currentFilters.yearEnd = parseInt(document.getElementById('yearEnd').value);
    currentFilters.development = document.getElementById('developmentFilter').value;
    
    const countrySelect = document.getElementById('countryFilter');
    currentFilters.countries = Array.from(countrySelect.selectedOptions).map(option => option.value);
    
    updateMetrics();
    updateCharts();
    updateDataTable();
    currentPage = 1;
}

function resetFilters() {
    document.getElementById('yearStart').value = 2000;
    document.getElementById('yearEnd').value = 2022;
    document.getElementById('yearStartDisplay').textContent = '2000';
    document.getElementById('yearEndDisplay').textContent = '2022';
    document.getElementById('developmentFilter').value = 'all';
    document.getElementById('countryFilter').selectedIndex = -1;
    
    currentFilters = {
        yearStart: 2000,
        yearEnd: 2022,
        development: 'all',
        countries: []
    };
    
    updateMetrics();
    updateCharts();
    updateDataTable();
    currentPage = 1;
}

function updateCharts() {
    if (charts.gdpCO2) {
        const filteredData = getFilteredData();
        const developedData = filteredData.filter(d => d.development === 'Developed');
        const developingData = filteredData.filter(d => d.development === 'Developing');
        
        charts.gdpCO2.data.datasets[0].data = developedData.map(d => ({x: d.gdp_per_capita, y: d.co2_per_capita}));
        charts.gdpCO2.data.datasets[1].data = developingData.map(d => ({x: d.gdp_per_capita, y: d.co2_per_capita}));
        charts.gdpCO2.update();
    }
}

function updateDataTable() {
    const filteredData = getFilteredData();
    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    const pageData = filteredData.slice(startIndex, endIndex);
    
    const tbody = document.getElementById('dataTableBody');
    tbody.innerHTML = '';
    
    pageData.forEach(item => {
        const row = tbody.insertRow();
        row.innerHTML = `
            <td>${item.country}</td>
            <td>${item.year}</td>
            <td>${item.co2_per_capita.toFixed(1)}</td>
            <td>$${item.gdp_per_capita.toLocaleString()}</td>
            <td>${item.renewables_share}%</td>
            <td>${item.development}</td>
        `;
    });
    
    const totalPages = Math.ceil(filteredData.length / itemsPerPage);
    document.getElementById('currentPage').textContent = currentPage;
    document.getElementById('totalPages').textContent = totalPages;
    
    document.getElementById('prevPage').disabled = currentPage === 1;
    document.getElementById('nextPage').disabled = currentPage === totalPages;
}

function changePage(direction) {
    const filteredData = getFilteredData();
    const totalPages = Math.ceil(filteredData.length / itemsPerPage);
    
    currentPage += direction;
    if (currentPage < 1) currentPage = 1;
    if (currentPage > totalPages) currentPage = totalPages;
    
    updateDataTable();
}

function exportData() {
    const filteredData = getFilteredData();
    const csv = convertToCSV(filteredData);
    downloadCSV(csv, 'global_energy_data.csv');
}

function convertToCSV(data) {
    const headers = ['Country', 'Year', 'CO2 per Capita', 'GDP per Capita', 'Renewables Share', 'Development Status'];
    const csvContent = [
        headers.join(','),
        ...data.map(row => [
            row.country,
            row.year,
            row.co2_per_capita,
            row.gdp_per_capita,
            row.renewables_share,
            row.development
        ].join(','))
    ].join('\n');
    
    return csvContent;
}

function downloadCSV(csv, filename) {
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.setAttribute('hidden', '');
    a.setAttribute('href', url);
    a.setAttribute('download', filename);
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

function showStatsModal() {
    const filteredData = getFilteredData();
    const statsContent = document.getElementById('statsContent');
    
    const stats = calculateStatistics(filteredData);
    
    statsContent.innerHTML = `
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-value">${stats.count}</div>
                <div class="stat-label">Total Records</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${stats.avgCO2}</div>
                <div class="stat-label">Avg CO2 per Capita</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">$${stats.avgGDP}</div>
                <div class="stat-label">Avg GDP per Capita</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${stats.maxCO2}</div>
                <div class="stat-label">Max CO2 per Capita</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${stats.minCO2}</div>
                <div class="stat-label">Min CO2 per Capita</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${stats.avgRenewable}%</div>
                <div class="stat-label">Avg Renewable Share</div>
            </div>
        </div>
    `;
    
    document.getElementById('statsModal').classList.remove('hidden');
}

function hideStatsModal() {
    document.getElementById('statsModal').classList.add('hidden');
}

function calculateStatistics(data) {
    if (data.length === 0) {
        return {
            count: 0,
            avgCO2: '0.0',
            avgGDP: '0',
            maxCO2: '0.0',
            minCO2: '0.0',
            avgRenewable: '0'
        };
    }
    
    const co2Values = data.map(d => d.co2_per_capita);
    const gdpValues = data.map(d => d.gdp_per_capita);
    const renewableValues = data.map(d => d.renewables_share);
    
    return {
        count: data.length,
        avgCO2: (co2Values.reduce((a, b) => a + b, 0) / co2Values.length).toFixed(1),
        avgGDP: Math.round(gdpValues.reduce((a, b) => a + b, 0) / gdpValues.length).toLocaleString(),
        maxCO2: Math.max(...co2Values).toFixed(1),
        minCO2: Math.min(...co2Values).toFixed(1),
        avgRenewable: Math.round(renewableValues.reduce((a, b) => a + b, 0) / renewableValues.length)
    };
}