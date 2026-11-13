// AeroSentinel Dashboard Client — Real-Time WebSocket Controller

document.addEventListener("DOMContentLoaded", () => {
    // Clock
    setInterval(() => {
        const now = new Date();
        document.getElementById("live-time").textContent = now.toTimeString().split(" ")[0];
    }, 1000);

    // WebSocket Setup
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${proto}//${window.location.host}/ws`;
    let socket = null;
    let reconnectTimer = null;

    function connectWS() {
        const statusPill = document.getElementById("connection-status");
        socket = new WebSocket(wsUrl);

        socket.onopen = () => {
            statusPill.textContent = "CONNECTED";
            statusPill.className = "status-pill status-connected";
            if (reconnectTimer) clearInterval(reconnectTimer);
        };

        socket.onclose = () => {
            statusPill.textContent = "DISCONNECTED";
            statusPill.className = "status-pill status-disconnected";
            setTimeout(connectWS, 2000);
        };

        socket.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                handleMessage(msg);
            } catch (e) {
                console.error("WS Parse Error:", e);
            }
        };
    }

    function handleMessage(msg) {
        if (!msg || !msg.type) return;

        switch (msg.type) {
            case "system_status":
                updateSystemStatus(msg.data);
                break;
            case "rgb_frame":
                document.getElementById("rgb-stream").src = `data:image/jpeg;base64,${msg.data}`;
                break;
            case "depth_frame":
                document.getElementById("depth-stream").src = `data:image/jpeg;base64,${msg.data}`;
                break;
            case "telemetry":
                updateTelemetry(msg.data);
                break;
            case "obstacle":
                updateObstacle(msg.data);
                break;
            case "imu":
                updateIMU(msg.data);
                break;
            case "detections":
                updateDetections(msg.data);
                break;
        }
    }

    function setDot(id, status) {
        const el = document.getElementById(id);
        if (!el) return;
        el.className = "dot " + (status === "connected" || status === "running" || status === "active" || status === "online" ? "dot-green" : "dot-red");
    }

    function updateSystemStatus(s) {
        if (!s) return;
        setDot("ind-rgb", s.rgb_camera);
        setDot("ind-depth", s.d435i);
        setDot("ind-ai", s.ai_engine);
        setDot("ind-mavlink", s.mavlink);
        setDot("ind-mission", s.mission);
        setDot("ind-net", s.network);

        const mTag = document.getElementById("mission-state-tag");
        if (s.mission === "active") {
            mTag.textContent = "ACTIVE";
            mTag.className = "badge badge-active";
        } else {
            mTag.textContent = "INACTIVE";
            mTag.className = "badge badge-inactive";
        }
    }

    function updateTelemetry(t) {
        if (!t) return;
        if (t.battery_pct !== undefined && t.battery_pct !== null) {
            document.getElementById("tel-bat").textContent = `${t.battery_pct}%`;
        }
        if (t.alt_m !== undefined && t.alt_m !== null) {
            document.getElementById("tel-alt").textContent = `${t.alt_m.toFixed(1)} m`;
        }
        if (t.flight_mode) {
            document.getElementById("tel-mode").textContent = t.flight_mode;
        }
        if (t.armed !== undefined) {
            document.getElementById("tel-armed").textContent = t.armed ? "ARMED" : "DISARMED";
        }
        if (t.lat !== null && t.lat !== undefined && t.lon !== null && t.lon !== undefined) {
            document.getElementById("tel-gps").textContent = `${t.lat.toFixed(5)}, ${t.lon.toFixed(5)}`;
        } else {
            document.getElementById("tel-gps").textContent = "No Fix";
        }
        if (t.yaw_deg !== undefined && t.yaw_deg !== null) {
            document.getElementById("tel-yaw").textContent = `${t.yaw_deg.toFixed(0)}°`;
        }
    }

    function updateIMU(imu) {
        if (!imu) return;
        const badge = document.getElementById("imu-valid-badge");
        if (badge) {
            badge.textContent = imu.is_valid ? "VALID" : "NO DATA";
            badge.className = "badge " + (imu.is_valid ? "badge-active" : "badge-inactive");
        }
        const fmt = (v, u) => v !== null && v !== undefined ? `${v.toFixed(3)} ${u}` : `-- ${u}`;
        const fmtDeg = (v) => v !== null && v !== undefined ? `${v.toFixed(1)}°` : "--°";
        const el = (id) => document.getElementById(id);
        if (el("imu-ax")) el("imu-ax").textContent = fmt(imu.accel_x, "m/s²");
        if (el("imu-ay")) el("imu-ay").textContent = fmt(imu.accel_y, "m/s²");
        if (el("imu-az")) el("imu-az").textContent = fmt(imu.accel_z, "m/s²");
        if (el("imu-gx")) el("imu-gx").textContent = fmt(imu.gyro_x, "rad/s");
        if (el("imu-gy")) el("imu-gy").textContent = fmt(imu.gyro_y, "rad/s");
        if (el("imu-gz")) el("imu-gz").textContent = fmt(imu.gyro_z, "rad/s");
        if (el("imu-roll"))  el("imu-roll").textContent  = fmtDeg(imu.roll_estimate);
        if (el("imu-pitch")) el("imu-pitch").textContent = fmtDeg(imu.pitch_estimate);
    }

    function updateObstacle(obs) {
        if (!obs) return;
        const banner = document.getElementById("obstacle-alert");
        if (obs.overall_status === "CRITICAL" || obs.warning_active) {
            banner.classList.remove("alert-hidden");
            const d = obs.closest_obstacle_m ? obs.closest_obstacle_m.toFixed(2) : "0.0";
            document.getElementById("alert-dist").textContent = d;
        } else {
            banner.classList.add("alert-hidden");
        }
    }

    function updateDetections(dets) {
        if (!dets || !Array.isArray(dets)) return;
        const tbody = document.getElementById("detections-tbody");
        document.getElementById("det-count").textContent = `${dets.length} Detections`;

        if (dets.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="text-muted">No detections recorded in current session.</td></tr>';
            return;
        }

        tbody.innerHTML = dets.map(d => `
            <tr>
                <td>${d.timestamp || "--"}</td>
                <td><strong>${d.class_name || d.class || "Potential Survivor"}</strong></td>
                <td>${((d.confidence || 0) * 100).toFixed(1)}%</td>
                <td>${d.distance_m ? d.distance_m.toFixed(2) + "m" : "N/A"}</td>
                <td><span class="badge badge-inactive">${d.verification_status || "PENDING"}</span></td>
                <td>${d.lat ? `${d.lat.toFixed(4)}, ${d.lon.toFixed(4)}` : "None"}</td>
            </tr>
        `).join("");
    }

    // Button Listeners
    document.getElementById("btn-start").onclick = async () => {
        const name = document.getElementById("mission-name-input").value;
        await fetch("/api/mission/start", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({name: name, operator: "Local Operator"})
        });
    };

    document.getElementById("btn-pause").onclick = async () => {
        await fetch("/api/mission/pause", {method: "POST"});
    };

    document.getElementById("btn-stop").onclick = async () => {
        await fetch("/api/mission/stop", {method: "POST"});
    };

    // Initial WebSocket connect
    connectWS();
});

