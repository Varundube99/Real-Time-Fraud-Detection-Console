let sseSource = null;
let currentTab = "dashboard";
let globalData = null;
let audioEnabled = true;
let lastSeenAlertId = null;

document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    fetchInitialData();
    connectSSE();
});

async function fetchInitialData() {
    try {
        const [statsRes, txRes, alertsRes] = await Promise.all([
            fetch('/api/stats'),
            fetch('/api/transactions?limit=20'),
            fetch('/api/alerts?limit=10')
        ]);
        const stats = await statsRes.json();
        const txList = await txRes.json();
        const alertList = await alertsRes.json();

        globalData = {
            kpis: stats.kpis,
            is_streaming: stats.is_streaming,
            is_paused: stats.is_paused,
            recent_transactions: txList.slice(0, 10),
            all_transactions: txList,
            recent_alerts: alertList.slice(0, 5),
            all_alerts: alertList
        };
        updateDashboard(globalData);
    } catch (err) {
        console.warn("Initial data fetch waiting for SSE:", err);
    }
}

function initNavigation() {
    const navButtons = document.querySelectorAll(".sidebar-nav .nav-icon-btn");
    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const page = btn.getAttribute("data-page");
            if (page) switchPage(page);
        });
    });
}

function switchPage(pageName) {
    currentTab = pageName;
    document.querySelectorAll(".sidebar-nav .nav-icon-btn").forEach(b => {
        if (b.getAttribute("data-page") === pageName) b.classList.add("active");
        else b.classList.remove("active");
    });
    document.querySelectorAll(".page-view").forEach(pv => pv.classList.remove("active"));
    const targetView = document.getElementById("page-" + pageName);
    if (targetView) targetView.classList.add("active");

    if (globalData) {
        updateDashboard(globalData);
    }
}

function connectSSE() {
    if (sseSource) sseSource.close();
    sseSource = new EventSource("/api/stream/sse");
    sseSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        globalData = data;
        updateDashboard(data);
        checkNewAlerts(data);
    };
    sseSource.onerror = (err) => { console.error("SSE connection error:", err); };
}

function updateDashboard(data) {
    const kpis = data.kpis || {};
    const total = kpis.processed || 0;
    const approved = kpis.approved || 0;
    const fraud = kpis.fraud || 0;
    const tpm = kpis.tpm || 0;
    const isStreaming = data.is_streaming;
    const isPaused = data.is_paused;

    const genuinePct = total > 0 ? ((approved / total) * 100).toFixed(1) : "0.0";

    // Dashboard KPI Updates
    const elTotal = document.getElementById("kpi-total");
    if (elTotal) elTotal.textContent = total.toLocaleString();
    const elGenuine = document.getElementById("kpi-genuine");
    if (elGenuine) elGenuine.textContent = approved.toLocaleString();
    const elGenuinePct = document.getElementById("kpi-genuine-pct");
    if (elGenuinePct) elGenuinePct.textContent = genuinePct + "%";
    const elFraud = document.getElementById("kpi-fraud");
    if (elFraud) elFraud.textContent = fraud.toLocaleString();
    const elTpm = document.getElementById("kpi-tpm");
    if (elTpm) elTpm.textContent = (isStreaming && !isPaused) ? (tpm || 0) : "0";

    // Status Indicator Dot & Badges
    const pulseDot = document.getElementById("header-pulse-dot");
    const pulseText = document.getElementById("header-status-text");
    const liveCtrlStatus = document.getElementById("live-controller-status");

    if (isStreaming && !isPaused) {
        if (pulseDot) pulseDot.className = "pulse-dot active";
        if (pulseText) pulseText.textContent = "Monitoring Active";
        if (liveCtrlStatus) {
            liveCtrlStatus.textContent = "Monitoring Active";
            liveCtrlStatus.className = "kpi-pill-badge badge-green";
        }
    } else if (isPaused) {
        if (pulseDot) pulseDot.className = "pulse-dot";
        if (pulseText) pulseText.textContent = "Monitoring Paused";
        if (liveCtrlStatus) {
            liveCtrlStatus.textContent = "Monitoring Paused";
            liveCtrlStatus.className = "kpi-pill-badge";
        }
    } else {
        if (pulseDot) pulseDot.className = "pulse-dot";
        if (pulseText) pulseText.textContent = "Monitoring Standby";
        if (liveCtrlStatus) {
            liveCtrlStatus.textContent = "Monitoring Standby";
            liveCtrlStatus.className = "kpi-pill-badge badge-red";
        }
    }

    // Charts & Tables
    renderDonutChart(total, approved, fraud);
    
    const recentTx = data.recent_transactions || [];
    renderTimelineFromTx(data.all_transactions || recentTx);
    renderDynamicTimeline(data.all_transactions || recentTx);

    // Dashboard View: ONLY 5 newest alerts
    renderRecentAlerts(data.recent_alerts || []);
    // Dashboard View: ONLY 10 newest transactions
    renderTransactionsTable("recent-transactions-tbody", recentTx.slice(0, 10));

    // Dedicated Alerts Page: Filterable alerts
    filterAlertsTable();

    // Dedicated Transactions Page: Filterable transactions
    filterTransactionsTable();
}

function renderDynamicTimeline(txList) {
    const container = document.getElementById("timeline-nodes-container");
    if (!container) return;

    if (!txList || txList.length === 0) {
        container.innerHTML = '<div style="color:#777; font-size:0.85rem; padding:20px; text-align:center; width:100%;">Awaiting real-time stream telemetry events...</div>';
        return;
    }

    // Pick 5 newest transactions for timeline nodes (chronological left-to-right)
    const newestFive = txList.slice(0, 5).reverse();
    let html = "";

    newestFive.forEach((tx) => {
        const status = String(tx.Status || "").toLowerCase();
        const txId = tx["Transaction ID"] || "";
        const amt = tx["Amount (₹)"] || "";
        const prob = tx.Probability || "0%";

        let bubbleClass = "speech-bubble-callout";
        let bubbleText = `Verified Genuine<br><strong style="color:#FFF;">${amt}</strong>`;
        let dotStyle = "border-color: #00F2FE; box-shadow: 0 0 12px #00F2FE;";
        let dotInnerStyle = "background-color: #00F2FE;";

        if (status.includes("blocked") || status.includes("fraud")) {
            bubbleClass = "speech-bubble-callout alert-bubble";
            bubbleText = `🚨 Blocked Fraud ${amt}<br><small style="opacity:0.8;">${txId}</small>`;
            dotStyle = "border-color: #FF3155; box-shadow: 0 0 14px #FF3155;";
            dotInnerStyle = "background-color: #FF3155;";
        } else if (status.includes("review")) {
            bubbleClass = "speech-bubble-callout ai-bubble";
            bubbleText = `⚠️ Under AI Review (${prob})<br><small style="opacity:0.8;">${txId}</small>`;
            dotStyle = "border-color: #FFB800; box-shadow: 0 0 14px #FFB800;";
            dotInnerStyle = "background-color: #FFB800;";
        }

        html += `
            <div class="timeline-node-item" onclick="openInvestigationModal('${txId}')" style="cursor:pointer;" title="Click to inspect ${txId}">
                <div class="${bubbleClass}">${bubbleText}</div>
                <div class="node-dot-outer" style="${dotStyle}"><div class="node-dot-inner" style="${dotInnerStyle}"></div></div>
            </div>`;
    });

    container.innerHTML = html;
}

function checkNewAlerts(data) {
    const alerts = data.recent_alerts || [];
    if (alerts.length > 0) {
        const topAlert = alerts[0];
        const txId = topAlert["Transaction ID"] || "";
        if (txId && txId !== lastSeenAlertId) {
            lastSeenAlertId = txId;
            if (audioEnabled) playAlertAudioSound();
        }
    }
}

function playAlertAudioSound() {
    try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();

        osc.type = "sine";
        osc.frequency.setValueAtTime(880, ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.3);

        gain.gain.setValueAtTime(0.15, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.3);

        osc.connect(gain);
        gain.connect(ctx.destination);

        osc.start();
        osc.stop(ctx.currentTime + 0.3);
    } catch (e) {
        console.log("Audio play blocked by browser policy until user interaction");
    }
}

function toggleAudioChime() {
    audioEnabled = !audioEnabled;
    const label = document.getElementById("audio-status-label");
    if (label) label.textContent = audioEnabled ? "Sound: On" : "Sound: Off";
}

function renderDonutChart(total, genuine, fraud) {
    const chartElem = document.getElementById('chart-donut');
    if (!chartElem) return;

    if (!total || total === 0) {
        const data = [{
            values: [1],
            labels: ['Engine Reset / 0'],
            type: 'pie',
            hole: 0.65,
            marker: { colors: ['#1E293B'] },
            textinfo: 'none',
            hoverinfo: 'label',
            showlegend: true
        }];

        const layout = {
            width: 260,
            height: 200,
            margin: { l: 5, r: 5, t: 5, b: 35 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            legend: {
                orientation: 'h',
                x: 0.5,
                xanchor: 'center',
                y: -0.15,
                font: { color: '#64748B', size: 10 }
            }
        };

        Plotly.react('chart-donut', data, layout, { displayModeBar: false, responsive: false });
        return;
    }

    const safeVal = Math.max(0, genuine || 0);
    const fraudVal = Math.max(0, fraud || 0);

    const data = [{
        values: [safeVal, fraudVal],
        labels: ['Safe / Verified', 'Critical Fraud'],
        type: 'pie',
        hole: 0.62,
        marker: { colors: ['#00E5FF', '#FF3366'] },
        textinfo: 'percent',
        textposition: 'inside',
        insidetextfont: { color: '#000000', size: 11, family: 'Plus Jakarta Sans' },
        showlegend: true
    }];

    const layout = {
        width: 260,
        height: 200,
        margin: { l: 5, r: 5, t: 5, b: 35 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        legend: {
            orientation: 'h',
            x: 0.5,
            xanchor: 'center',
            y: -0.15,
            font: { color: '#AAAAAA', size: 10 }
        }
    };

    Plotly.react('chart-donut', data, layout, { displayModeBar: false, responsive: false });
}

function renderTimelineFromTx(txList) {
    const xVals = [];
    const yVals1 = [];
    const yVals2 = [];

    if (txList && txList.length > 0) {
        txList.slice(0, 15).reverse().forEach((tx, idx) => {
            xVals.push(idx + 1);
            const probStr = String(tx.Probability || "0").replace("%", "");
            const probNum = parseFloat(probStr) || 0.0;
            yVals1.push(probNum);
            yVals2.push(Math.max(5, probNum * 0.4));
        });
    }
    renderTimelineChart(xVals, yVals1, yVals2);
}

function renderTimelineChart(xVals, yVals1, yVals2) {
    const chartElem = document.getElementById('chart-timeline');
    if (!chartElem) return;

    if (xVals.length === 0) {
        xVals = [1, 2, 3, 4, 5];
        yVals1 = [0, 0, 0, 0, 0];
        yVals2 = [0, 0, 0, 0, 0];
    }

    const data = [
        {
            x: xVals, y: yVals1, type: 'scatter', mode: 'lines+markers',
            name: 'Fraud Risk Prob',
            line: { color: '#D9FD00', width: 2.5 },
            marker: { size: 6, color: '#D9FD00' }
        },
        {
            x: xVals, y: yVals2, type: 'scatter', mode: 'lines+markers',
            name: 'Baseline Threat',
            line: { color: '#FF3155', width: 1.5, dash: 'dot' },
            marker: { size: 4, color: '#FF3155' }
        }
    ];

    const layout = {
        height: 200,
        margin: { l: 30, r: 10, t: 10, b: 30 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        showlegend: false,
        xaxis: { gridcolor: '#1A1A1A', tickfont: { color: '#888888', size: 9 } },
        yaxis: { gridcolor: '#1A1A1A', tickfont: { color: '#888888', size: 9 } }
    };

    Plotly.react('chart-timeline', data, layout, { displayModeBar: false, responsive: true });
}

function renderRecentAlerts(alerts) {
    const container = document.getElementById("recent-alerts-list");
    if (!container) return;

    if (!alerts || alerts.length === 0) {
        container.innerHTML = '<div class="empty-state">No Active Fraud Alerts</div>';
        return;
    }

    const newestTwo = alerts.slice(0, 3);
    let html = "";
    newestTwo.forEach(al => {
        const prob = al.Probability || "0.0%";
        const amt = al["Amount (₹)"] || "₹0.00";
        const txId = al["Transaction ID"] || "TXN-000";
        const time = al.Time || "Just now";
        html += `
            <div class="alert-card-item" onclick="openInvestigationModal('${txId}')">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span class="badge-pill badge-high">🚨 CRITICAL FRAUD</span>
                    <span style="font-size: 0.68rem; color: #888888;">${time}</span>
                </div>
                <div style="font-size: 0.78rem; color: #AAAAAA;">ID: <strong style="color: #FFFFFF;">${txId}</strong></div>
                <div style="font-size: 0.78rem; color: #AAAAAA;">Probability: <strong style="color: #FF3155;">${prob}</strong></div>
                <div style="font-size: 0.78rem; color: #AAAAAA; display: flex; justify-content: space-between; margin-top: 4px;">
                    <span>Amount: <strong style="color: #FFFFFF;">${amt}</strong></span>
                    <span class="badge-pill badge-high">BLOCKED</span>
                </div>
            </div>`;
    });
    container.innerHTML = html;
}

function renderTransactionsTable(elementId, txList) {
    const tbody = document.getElementById(elementId);
    if (!tbody) return;

    if (!txList || txList.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="loading-td">No transactions recorded yet.</td></tr>';
        return;
    }

    let html = "";
    txList.forEach(tx => {
        const txId = tx["Transaction ID"] || "";
        const status = String(tx.Status || "").toLowerCase();
        let statusClass = "status-cell-approved";
        if (status.includes("blocked") || status.includes("fraud")) statusClass = "status-cell-blocked";
        else if (status.includes("review")) statusClass = "status-cell-review";

        html += `
            <tr onclick="openInvestigationModal('${txId}')">
                <td><strong>${txId}</strong></td>
                <td>${tx.Time || ""}</td>
                <td>${tx["Amount (₹)"] || ""}</td>
                <td>${tx.Probability || ""}</td>
                <td><span class="${statusClass}">${tx.Status || ""}</span></td>
            </tr>`;
    });
    tbody.innerHTML = html;
}

function filterAlertsTable() {
    const tbody = document.getElementById("all-alerts-tbody");
    if (!tbody || !globalData) return;

    const searchInput = document.getElementById("alerts-search-input");
    const term = searchInput ? searchInput.value.toLowerCase().trim() : "";
    const allAlerts = globalData.all_alerts || globalData.recent_alerts || [];

    const filtered = allAlerts.filter(al => {
        const txId = String(al["Transaction ID"] || "").toLowerCase();
        const loc = String(al.Location || "").toLowerCase();
        const status = String(al.Status || "").toLowerCase();
        return !term || txId.includes(term) || loc.includes(term) || status.includes(term);
    });

    if (filtered.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="loading-td">No matching fraud alerts.</td></tr>';
        return;
    }

    let html = "";
    filtered.forEach(al => {
        const txId = al["Transaction ID"] || "";
        const status = String(al.Status || "").toLowerCase();
        let statusClass = "status-cell-blocked";
        if (status.includes("review")) statusClass = "status-cell-review";

        html += `
            <tr onclick="openInvestigationModal('${txId}')">
                <td><strong>${txId}</strong></td>
                <td>${al.Time || ""}</td>
                <td>${al["Amount (₹)"] || ""}</td>
                <td><strong style="color: #FF3155;">${al.Probability || ""}</strong></td>
                <td>${al["Risk Level"] || "High"}</td>
                <td><span class="${statusClass}">${al.Status || "Blocked"}</span></td>
                <td>${al.Location || "Mumbai, IN"}</td>
            </tr>`;
    });
    tbody.innerHTML = html;
}

function filterTransactionsTable() {
    const tbody = document.getElementById("all-transactions-tbody");
    if (!tbody || !globalData) return;

    const searchInput = document.getElementById("tx-search-input");
    const statusSelect = document.getElementById("tx-status-filter");
    const term = searchInput ? searchInput.value.toLowerCase().trim() : "";
    const statusFilter = statusSelect ? statusSelect.value : "ALL";

    const allTx = globalData.all_transactions || globalData.recent_transactions || [];

    const filtered = allTx.filter(tx => {
        const txId = String(tx["Transaction ID"] || "").toLowerCase();
        const loc = String(tx.Location || "").toLowerCase();
        const status = String(tx.Status || "");
        const statusLower = status.toLowerCase();

        const matchesTerm = !term || txId.includes(term) || loc.includes(term) || statusLower.includes(term);

        let matchesStatus = true;
        if (statusFilter === "Blocked") matchesStatus = statusLower.includes("blocked") || statusLower.includes("fraud");
        else if (statusFilter === "Under Review") matchesStatus = statusLower.includes("review");
        else if (statusFilter === "Approved") matchesStatus = statusLower.includes("approved");

        return matchesTerm && matchesStatus;
    });

    if (filtered.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="loading-td">No matching transactions found.</td></tr>';
        return;
    }

    let html = "";
    filtered.forEach(tx => {
        const txId = tx["Transaction ID"] || "";
        const status = String(tx.Status || "").toLowerCase();
        let statusClass = "status-cell-approved";
        if (status.includes("blocked") || status.includes("fraud")) statusClass = "status-cell-blocked";
        else if (status.includes("review")) statusClass = "status-cell-review";

        html += `
            <tr onclick="openInvestigationModal('${txId}')">
                <td><strong>${txId}</strong></td>
                <td>${tx.Time || ""}</td>
                <td>${tx["Amount (₹)"] || ""}</td>
                <td>${tx.Probability || ""}</td>
                <td>${tx["Risk Level"] || "Low"}</td>
                <td><span class="${statusClass}">${tx.Status || "Approved"}</span></td>
                <td>${tx.Location || "Mumbai, IN"}</td>
            </tr>`;
    });
    tbody.innerHTML = html;
}

function openInvestigationModal(txId) {
    const modal = document.getElementById("investigation-modal");
    if (!modal) return;

    document.getElementById("modal-tx-id").textContent = txId;
    document.getElementById("modal-timestamp").textContent = "Loading transaction data...";
    document.getElementById("modal-amount").textContent = "₹--";
    document.getElementById("modal-prob").textContent = "--%";
    document.getElementById("modal-risk-level").textContent = "--";
    document.getElementById("modal-features-grid").innerHTML = '<div class="loading-td">Fetching anonymized feature vector...</div>';

    modal.classList.add("active");

    fetch('/api/transaction/' + encodeURIComponent(txId))
        .then(res => {
            if (!res.ok) throw new Error("Not found");
            return res.json();
        })
        .then(data => {
            document.getElementById("modal-tx-id").textContent = data.transaction_id;
            document.getElementById("modal-timestamp").textContent = "Timestamp: " + data.timestamp;
            document.getElementById("modal-amount").textContent = "₹" + Number(data.amount).toLocaleString(undefined, {minimumFractionDigits: 2});
            document.getElementById("modal-prob").textContent = data.probability + "%";
            document.getElementById("modal-risk-level").textContent = data.risk_level;

            const badge = document.getElementById("modal-decision-badge");
            badge.textContent = data.decision.toUpperCase();
            if (data.decision === "Blocked") badge.className = "kpi-pill-badge badge-red";
            else if (data.decision === "Under Review") badge.className = "kpi-pill-badge";
            else badge.className = "kpi-pill-badge badge-green";

            document.getElementById("modal-merchant").textContent = "Merchant: " + (data.merchant || "Amazon");
            document.getElementById("modal-location").textContent = "Location: " + (data.location || "Mumbai, IN");
            document.getElementById("modal-device").textContent = "Device: " + (data.device || "Mobile");
            document.getElementById("modal-latency").textContent = "Inference Time: " + (data.latency_ms || 1.8) + " ms";

            // Features Grid
            const fGrid = document.getElementById("modal-features-grid");
            let fHtml = "";
            const feats = data.features || {};
            for (let i = 1; i <= 28; i++) {
                const key = "V" + i;
                const val = feats[key] !== undefined ? Number(feats[key]).toFixed(4) : "0.0000";
                fHtml += `<div class="feature-tag"><span>${key}:</span> <span>${val}</span></div>`;
            }
            fGrid.innerHTML = fHtml;
        })
        .catch(err => {
            document.getElementById("modal-timestamp").textContent = "Failed to load detailed record.";
        });
}

function closeModal() {
    const modal = document.getElementById("investigation-modal");
    if (modal) modal.classList.remove("active");
}

function closeModalOnBackdrop(e) {
    if (e.target.id === "investigation-modal") {
        closeModal();
    }
}

function updateSpeedVal(val) {
    const el = document.getElementById("speed-val");
    if (el) el.textContent = parseFloat(val).toFixed(1) + "x";
}

function sendSpeed(val) {
    fetch('/api/control/speed', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speed: parseFloat(val) })
    }).catch(err => console.error("Speed update error:", err));
}

function sendFraudRate(val) {
    const rateFloat = parseFloat(val) / 100.0;
    fetch('/api/control/fraud_rate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fraud_rate: rateFloat })
    }).catch(err => console.error("Fraud rate update error:", err));
}

function sendControl(action) {
    fetch('/api/control/' + action, { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            console.log('Control action:', data);
            if (action === 'reset') {
                globalData = {
                    kpis: { processed: 0, approved: 0, review: 0, fraud: 0, tpm: 0, fraud_rate: 0 },
                    is_streaming: false,
                    is_paused: false,
                    alert_count: 0,
                    recent_transactions: [],
                    all_transactions: [],
                    recent_alerts: [],
                    all_alerts: []
                };
                updateDashboard(globalData);
            }
        })
        .catch(err => console.error('Control error:', err));
}

function downloadAlertsCSV() {
    window.location.href = '/api/export/alerts';
}

function downloadTransactionsCSV() {
    window.location.href = '/api/export/transactions';
}
