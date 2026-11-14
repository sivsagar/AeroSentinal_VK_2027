// AeroSentinel Ground Control Station — Real-Time WebSocket Controller (v1.2)

document.addEventListener("DOMContentLoaded", () => {
    // -----------------------------------------------------------------------
    // Clocks and Uptime
    // -----------------------------------------------------------------------
    const liveTimeEl = document.getElementById("live-time");
    const uptimeEl = document.getElementById("sys-uptime");
    let clientStartTime = Date.now();

    setInterval(() => {
        const now = new Date();
        if (liveTimeEl) {
            liveTimeEl.textContent = now.toISOString().substring(11, 19);
        }
        if (uptimeEl) {
            const elapsed = Math.floor((Date.now() - clientStartTime) / 1000);
            const hrs = String(Math.floor(elapsed / 3600)).padStart(2, "0");
            const mins = String(Math.floor((elapsed % 3600) / 60)).padStart(2, "0");
            const secs = String(elapsed % 60).padStart(2, "0");
            uptimeEl.textContent = `${hrs}:${mins}:${secs}`;
        }
    }, 1000);

    // -----------------------------------------------------------------------
    // WebSocket Connection Manager
    // -----------------------------------------------------------------------
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${proto}//${window.location.host}/ws`;
    let socket = null;
    let reconnectTimeout = null;

    const statusPill = document.getElementById("connection-status");
    const connDot = statusPill ? statusPill.querySelector(".conn-dot") : null;
    const connText = statusPill ? statusPill.querySelector(".conn-text") : null;

    function connectWS() {
        if (socket && (socket.readyState === WebSocket.CONNECTING || socket.readyState === WebSocket.OPEN)) {
            return;
        }

        socket = new WebSocket(wsUrl);

        socket.onopen = () => {
            if (statusPill) {
                statusPill.className = "status-pill status-connected";
            }
            if (connText) connText.textContent = "LINK ACTIVE";
            if (reconnectTimeout) {
                clearTimeout(reconnectTimeout);
                reconnectTimeout = null;
            }
        };

        socket.onclose = () => {
            if (statusPill) {
                statusPill.className = "status-pill status-disconnected";
            }
            if (connText) connText.textContent = "RECONNECTING";
            reconnectTimeout = setTimeout(connectWS, 2000);
        };

        socket.onerror = (err) => {
            console.debug("WS Error:", err);
            try { socket.close(); } catch (_) {}
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

    // -----------------------------------------------------------------------
    // Dispatch Incoming WebSocket Messages
    // -----------------------------------------------------------------------
    function handleMessage(msg) {
        if (!msg || !msg.type) return;

        switch (msg.type) {
            case "system_status":
                updateSystemStatus(msg.data);
                break;
            case "health":
                updateHealth(msg.data);
                break;
            case "rgb_frame":
                updateFrame("rs-rgb-stream", msg.data);
                break;
            case "depth_frame":
                updateFrame("rs-depth-stream", msg.data);
                break;
            case "picam_frame":
                updateFrame("picam-stream", msg.data);
                break;
            case "obstacle":
                updateObstacle(msg.data);
                break;
            case "tracks":
                updateTracks(msg.data);
                break;
            case "events":
                updateEvents(msg.data);
                break;
            case "telemetry":
                updateTelemetry(msg.data);
                break;
        }
    }

    function updateFrame(imgId, base64Data) {
        const el = document.getElementById(imgId);
        if (el && base64Data) {
            el.src = `data:image/jpeg;base64,${base64Data}`;
        }
    }

    // -----------------------------------------------------------------------
    // Subsystem Health Ribbon Indicators
    // -----------------------------------------------------------------------
    function setDot(id, active) {
        const el = document.getElementById(id);
        if (!el) return;
        el.className = "dot " + (active ? "dot-green" : "dot-red");
    }

    function updateSystemStatus(s) {
        if (!s) return;
        setDot("ind-rs-rgb", s.realsense_rgb === "connected" || s.rgb_camera === "connected");
        setDot("ind-rs-depth", s.realsense_depth === "connected" || s.d435i === "connected");
        setDot("ind-picam", s.picam_camera === "connected");
        setDot("ind-ai", s.ai_engine === "running");
        setDot("ind-tracker", s.person_tracker === "running");
        setDot("ind-obstacle", s.obstacle_detector === "running" || s.depth_processing === "running");
        setDot("ind-mavlink", s.mavlink === "connected");
        setDot("ind-net", s.network === "online");

        // FPS badge
        const fpsBadge = document.getElementById("rs-rgb-fps");
        if (fpsBadge && s.camera_fps !== undefined) {
            fpsBadge.textContent = `${s.camera_fps.toFixed(1)} FPS`;
        }
    }

    // -----------------------------------------------------------------------
    // System Health & Hardware Performance Cards (Section 2.E)
    // -----------------------------------------------------------------------
    function updateHealth(h) {
        if (!h) return;

        // Stale warning indicator
        const staleWarn = document.getElementById("stale-metrics-warning");
        if (staleWarn) {
            if (h.is_stale) {
                staleWarn.classList.remove("hidden");
            } else {
                staleWarn.classList.add("hidden");
            }
        }

        // CPU Utilization
        const cpuVal = document.getElementById("hc-cpu-val");
        const cpuBar = document.getElementById("hc-cpu-bar");
        if (cpuVal && h.cpu_percent !== null && h.cpu_percent !== undefined) {
            cpuVal.textContent = `${h.cpu_percent.toFixed(1)}%`;
            if (cpuBar) cpuBar.style.width = `${Math.min(100, h.cpu_percent)}%`;
        } else if (cpuVal) {
            cpuVal.textContent = "--%";
        }

        // CPU Temperature
        const tempVal = document.getElementById("hc-temp-val");
        const tempBar = document.getElementById("hc-temp-bar");
        if (tempVal && h.cpu_temp_c !== null && h.cpu_temp_c !== undefined) {
            tempVal.textContent = `${h.cpu_temp_c.toFixed(1)}°C`;
            if (tempBar) {
                const tempPct = Math.min(100, Math.max(0, (h.cpu_temp_c - 30) / 50 * 100));
                tempBar.style.width = `${tempPct}%`;
                if (h.cpu_temp_c > 75) {
                    tempBar.style.background = "#ff1744";
                } else if (h.cpu_temp_c > 60) {
                    tempBar.style.background = "#ff9100";
                } else {
                    tempBar.style.background = "#00e676";
                }
            }
        } else if (tempVal) {
            tempVal.textContent = "N/A";
            if (tempBar) tempBar.style.width = "0%";
        }

        // RAM Usage
        const ramVal = document.getElementById("hc-ram-val");
        const ramBar = document.getElementById("hc-ram-bar");
        if (ramVal && h.ram_used_mb && h.ram_total_mb) {
            ramVal.textContent = `${h.ram_used_mb.toFixed(0)} / ${h.ram_total_mb.toFixed(0)} MB`;
            if (ramBar && h.ram_percent !== undefined) {
                ramBar.style.width = `${Math.min(100, h.ram_percent)}%`;
            }
        }

        // FPS and Drops
        const fpsVal = document.getElementById("hc-fps-val");
        const inferFps = document.getElementById("hc-infer-fps");
        const dropsVal = document.getElementById("hc-drops");
        if (fpsVal && h.camera_fps !== undefined) fpsVal.textContent = `${h.camera_fps.toFixed(1)} FPS`;
        if (inferFps && h.inference_fps !== undefined) inferFps.textContent = h.inference_fps.toFixed(1);
        if (dropsVal && h.dropped_frames !== undefined) dropsVal.textContent = h.dropped_frames;
    }

    // -----------------------------------------------------------------------
    // Obstacle Proximity Monitoring (Section 2.C)
    // -----------------------------------------------------------------------
    function getChipClass(status) {
        switch ((status || "").toUpperCase()) {
            case "CRITICAL": return "chip-critical";
            case "WARNING": return "chip-warning";
            case "CAUTION": return "chip-caution";
            case "CLEAR": case "SAFE": return "chip-clear";
            default: return "chip-unknown";
        }
    }

    function updateObstacle(obs) {
        if (!obs) return;

        // 1. Proximity Banner Alert
        const banner = document.getElementById("proximity-alert-banner");
        const alertTitle = document.getElementById("alert-title");
        const alertDesc = document.getElementById("alert-desc");
        const alertDistVal = document.getElementById("alert-dist-val");

        const isCritical = obs.critical_active || obs.overall_status === "CRITICAL";
        const isWarning = obs.warning_active || obs.overall_status === "WARNING";
        const isCaution = obs.caution_active || obs.overall_status === "CAUTION";

        if (banner) {
            if (isCritical) {
                banner.className = "alert-banner banner-critical";
                if (alertTitle) alertTitle.textContent = "CRITICAL PROXIMITY HAZARD";
                if (alertDesc) alertDesc.textContent = "Immediate close-range surface/obstacle in camera flight corridor!";
                if (alertDistVal) alertDistVal.textContent = obs.closest_obstacle_m ? `${obs.closest_obstacle_m.toFixed(2)} m` : "-- m";
            } else if (isWarning) {
                banner.className = "alert-banner banner-warning";
                if (alertTitle) alertTitle.textContent = "PROXIMITY WARNING";
                if (alertDesc) alertDesc.textContent = "Reliable close-range obstacle detected in camera corridor.";
                if (alertDistVal) alertDistVal.textContent = obs.closest_obstacle_m ? `${obs.closest_obstacle_m.toFixed(2)} m` : "-- m";
            } else if (isCaution) {
                banner.className = "alert-banner banner-caution";
                if (alertTitle) alertTitle.textContent = "PROXIMITY CAUTION";
                if (alertDesc) alertDesc.textContent = "Approaching proximity boundary zone.";
                if (alertDistVal) alertDistVal.textContent = obs.closest_obstacle_m ? `${obs.closest_obstacle_m.toFixed(2)} m` : "-- m";
            } else {
                banner.className = "alert-banner banner-hidden";
            }
        }

        // Overall status badge
        const ovrBadge = document.getElementById("overall-proximity-badge");
        if (ovrBadge) {
            ovrBadge.textContent = obs.overall_status || "CLEAR";
            ovrBadge.className = `metric-pill ${getChipClass(obs.overall_status)}`;
        }

        // HUD overlay badge on video stream
        const hudStatus = document.getElementById("hud-status");
        if (hudStatus) {
            hudStatus.textContent = `PROXIMITY: ${obs.overall_status || "CLEAR"}`;
            hudStatus.className = `hud-chip ${getChipClass(obs.overall_status)}`;
        }

        // 2. Sector Cards (Left, Centre, Right)
        const regions = obs.regions || {};
        const leftSec = regions["left"] || obs.left;
        const centerSec = regions["center"] || obs.center;
        const rightSec = regions["right"] || obs.right;

        function populateSector(secName, secData) {
            if (!secData) return;
            const statusEl = document.getElementById(`sec-${secName}-status`);
            const medEl = document.getElementById(`sec-${secName}-med`);
            const p10El = document.getElementById(`sec-${secName}-p10`);
            const covEl = document.getElementById(`sec-${secName}-cov`);
            const occEl = document.getElementById(`sec-${secName}-occ`);
            const humanEl = document.getElementById(`sec-${secName}-human`);

            if (statusEl) {
                statusEl.textContent = secData.status || "UNKNOWN";
                statusEl.className = `sec-status-chip ${getChipClass(secData.status)}`;
            }
            if (medEl) {
                medEl.textContent = (secData.median_distance_m !== undefined && secData.median_distance_m !== null)
                    ? `${secData.median_distance_m.toFixed(2)} m`
                    : (secData.min_distance_m ? `${secData.min_distance_m.toFixed(2)} m` : "-- m");
            }
            if (p10El) {
                p10El.textContent = (secData.percentile_10_distance_m !== undefined && secData.percentile_10_distance_m !== null)
                    ? `${secData.percentile_10_distance_m.toFixed(2)} m`
                    : "-- m";
            }
            if (covEl) {
                covEl.textContent = (secData.valid_depth_percent !== undefined)
                    ? `${secData.valid_depth_percent.toFixed(1)}%`
                    : "--%";
            }
            if (occEl) {
                occEl.textContent = (secData.near_range_occupancy_percent !== undefined)
                    ? `${secData.near_range_occupancy_percent.toFixed(1)}%`
                    : "--%";
            }
            if (humanEl) {
                if (secData.human_associated) {
                    const ids = (secData.human_track_ids || []).join(", ");
                    humanEl.innerHTML = `<span class="tag-human">👤 HUMAN PRESENT (#${ids || "1"})</span>`;
                } else {
                    humanEl.textContent = "No human detected in sector";
                }
            }
        }

        populateSector("left", leftSec);
        populateSector("center", centerSec);
        populateSector("right", rightSec);

        // Closest proximity candidate summary
        const candText = document.getElementById("closest-cand-text");
        if (candText) {
            if (obs.closest_reliable_candidate) {
                const c = obs.closest_reliable_candidate;
                candText.textContent = `${c.distance_m.toFixed(2)} m (${(c.region || "centre").toUpperCase()})`;
            } else if (obs.closest_obstacle_m) {
                candText.textContent = `${obs.closest_obstacle_m.toFixed(2)} m`;
            } else {
                candText.textContent = "CLEAR (None < 3.5m)";
            }
        }

        const totalCov = document.getElementById("total-coverage-text");
        if (totalCov && obs.total_valid_coverage_percent !== undefined) {
            totalCov.textContent = `${obs.total_valid_coverage_percent.toFixed(1)}%`;
            const depthCovBadge = document.getElementById("depth-coverage-badge");
            if (depthCovBadge) depthCovBadge.textContent = `Cov: ${obs.total_valid_coverage_percent.toFixed(1)}%`;
        }
    }

    // -----------------------------------------------------------------------
    // Multi-Person Tracking Table (Section 2.D)
    // -----------------------------------------------------------------------
    function updateTracks(tracks) {
        const tbody = document.getElementById("tracking-tbody");
        const countBadge = document.getElementById("active-track-counter");
        const rsBadge = document.getElementById("rs-track-count-badge");

        const trackList = Array.isArray(tracks) ? tracks : [];
        if (countBadge) countBadge.textContent = `${trackList.length} Active`;
        if (rsBadge) rsBadge.textContent = `${trackList.length} Tracks`;

        if (!tbody) return;

        if (trackList.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="empty-table-cell">No active person tracks in RealSense FOV.</td></tr>';
            return;
        }

        tbody.innerHTML = trackList.map(t => {
            const tid = t.track_id !== undefined ? t.track_id : t.detection_id || 1;
            const conf = t.confidence !== undefined ? (t.confidence * 100).toFixed(1) : "--";
            const dist = t.distance_m !== null && t.distance_m !== undefined
                ? `${t.distance_m.toFixed(2)} m`
                : (t.estimated_distance_m ? `${t.estimated_distance_m.toFixed(2)} m` : "-- m");
            const isValid = t.distance_valid === true || t.depth_measurement_validity === true;
            const validTag = isValid
                ? '<span class="status-chip chip-clear">VALID</span>'
                : '<span class="status-chip chip-warning">INVALID</span>';
            const state = (t.state || "CONFIRMED").toUpperCase();
            const age = t.age_frames !== undefined ? `${t.age_frames} f` : "--";
            const lastSeen = t.last_seen_ts ? new Date(t.last_seen_ts * 1000).toISOString().substring(11, 19) : "--:--:--";

            return `
                <tr>
                    <td><strong class="track-id-badge">TRK-${String(tid).padStart(2, "0")}</strong></td>
                    <td>${conf}%</td>
                    <td><strong>${dist}</strong></td>
                    <td>${validTag}</td>
                    <td><span class="track-state-pill ${state === "CONFIRMED" ? "state-confirmed" : "state-tentative"}">${state}</span></td>
                    <td>${age}</td>
                    <td>${lastSeen}</td>
                </tr>
            `;
        }).join("");
    }

    // -----------------------------------------------------------------------
    // Chronological Event History Timeline (Section 2.F)
    // -----------------------------------------------------------------------
    function getEventIcon(eventType) {
        switch (eventType) {
            case "PERSON_DETECTED": return "👤";
            case "TRACK_CREATED": return "🎯";
            case "PERSON_REACQUIRED": return "🔄";
            case "TRACK_LOST": return "❓";
            case "TRACK_EXPIRED": return "⏱️";
            case "PROXIMITY_CAUTION": return "🟡";
            case "PROXIMITY_WARNING": return "🟠";
            case "PROXIMITY_CRITICAL": return "🔴";
            case "CAMERA_UNAVAILABLE": return "📹";
            case "SENSOR_RECONNECT": return "🔌";
            case "SYSTEM_ERROR": return "❌";
            default: return "ℹ️";
        }
    }

    function updateEvents(events) {
        const timeline = document.getElementById("events-timeline");
        const countBadge = document.getElementById("event-count-badge");
        const eventList = Array.isArray(events) ? events : [];

        if (countBadge) countBadge.textContent = `${eventList.length} Events`;
        if (!timeline) return;

        if (eventList.length === 0) {
            timeline.innerHTML = '<div class="event-row-empty">No system or perception events recorded.</div>';
            return;
        }

        timeline.innerHTML = eventList.map(e => {
            const countTag = (e.count && e.count > 1) ? `<span class="event-count-pill">× ${e.count}</span>` : "";
            const sevClass = `sev-${(e.severity || "INFO").toLowerCase()}`;
            const icon = getEventIcon(e.event_type);

            return `
                <div class="event-row ${sevClass}">
                    <span class="event-time">${e.timestamp_iso || "--:--:--"}</span>
                    <span class="event-icon">${icon}</span>
                    <span class="event-type-badge">${e.event_type}</span>
                    <span class="event-msg">${e.message}</span>
                    ${countTag}
                </div>
            `;
        }).join("");
    }

    // -----------------------------------------------------------------------
    // Vehicle Flight Telemetry (Read-Only)
    // -----------------------------------------------------------------------
    function updateTelemetry(t) {
        if (!t) return;
        const setVal = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val;
        };

        if (t.battery_pct !== undefined && t.battery_pct !== null) setVal("t-bat", `${t.battery_pct}%`);
        if (t.alt_m !== undefined && t.alt_m !== null) setVal("t-alt", `${t.alt_m.toFixed(1)} m`);
        if (t.speed_ms !== undefined && t.speed_ms !== null) setVal("t-spd", `${t.speed_ms.toFixed(1)} m/s`);
        if (t.flight_mode) setVal("t-mode", t.flight_mode);
        if (t.armed !== undefined) setVal("t-arm", t.armed ? "ARMED" : "DISARMED");
        if (t.yaw_deg !== undefined && t.yaw_deg !== null) setVal("t-yaw", `${t.yaw_deg.toFixed(0)}°`);

        const mavBadge = document.getElementById("mavlink-conn-badge");
        if (mavBadge) {
            if (t.is_connected) {
                mavBadge.textContent = "LINK ACTIVE";
                mavBadge.className = "metric-pill pill-green";
            } else {
                mavBadge.textContent = "NO LINK";
                mavBadge.className = "metric-pill pill-grey";
            }
        }
    }

    // Clear events button
    const clearBtn = document.getElementById("btn-clear-events");
    if (clearBtn) {
        clearBtn.onclick = () => {
            const timeline = document.getElementById("events-timeline");
            if (timeline) timeline.innerHTML = '<div class="event-row-empty">Event timeline view cleared.</div>';
        };
    }

    // Connect WebSocket on load
    connectWS();
});
