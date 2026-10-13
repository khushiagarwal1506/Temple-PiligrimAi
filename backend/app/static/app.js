/**
 * PilgrimAI — Intelligent Pilgrimage Assistant
 * Frontend Controller: Multi-Turn Session Memory, Active Agent Display,
 * Structured Cards, Google Maps JavaScript, Places & Routes Integration,
 * Native Browser Voice UI (STT/TTS), and Multilingual Support.
 */

document.addEventListener("DOMContentLoaded", () => {
    // -------------------------------------------------------------
    // DOM Elements
    // -------------------------------------------------------------
    const chatForm = document.getElementById("chatForm");
    const userQuery = document.getElementById("userQuery");
    const messageArea = document.getElementById("messageArea");
    const submitBtn = document.getElementById("submitBtn");
    const newChatBtn = document.getElementById("newChatBtn");
    const micBtn = document.getElementById("micBtn");
    const toggleMapBtn = document.getElementById("toggleMapBtn");
    const tabMapBtn = document.getElementById("tabMapBtn");
    const tabPlacesBtn = document.getElementById("tabPlacesBtn");
    const mapPanel = document.getElementById("mapPanel");
    const mapFallback = document.getElementById("mapFallback");
    const placesList = document.getElementById("placesList");
    const placesTargetTitle = document.getElementById("placesTargetTitle");
    const templeList = document.getElementById("templeList");
    const langChips = document.querySelectorAll(".lang-chip");

    // -------------------------------------------------------------
    // State Management
    // -------------------------------------------------------------
    let currentSessionId = getOrCreateSessionId();
    let currentLang = "en";
    let currentSelectedTemple = null;
    let currentLat = null;
    let currentLng = null;
    let activeNavigationState = {
        origin: "",
        destination: "",
        travelMode: "driving",
        distance_km: null,
        duration_minutes: null,
        polyline: null,
        origin_coordinates: null,
        destination_coordinates: null
    };
    let placesRequestStarted = false;
    let placesLastStatus = "NOT_STARTED";
    let placesLastResultCount = 0;
    let placesLastError = null;
    let googleMap = null;
    let mapMarkers = [];
    let directionsRenderer = null;
    let directionsService = null;
    let activeRoutePolyline = null;
    let activeRouteMarkers = [];
    let pendingRouteToRestore = null;
    const VERIFIED_TEMPLE_REGISTRY = [
        { canonical_name: "Ahobilam", display_name: "Sri Ahobila Narasimha Swamy Temple", latitude: 15.127814, longitude: 78.718048, district: "Nandyal", state: "Andhra Pradesh" },
        { canonical_name: "Badrachalam", display_name: "Sri Sita Ramachandraswamy Temple, Bhadrachalam", latitude: 17.669822, longitude: 80.893641, district: "Bhadradri Kothagudem", state: "Telangana" },
        { canonical_name: "Balkampet", display_name: "Balkampet Yellamma Temple, Hyderabad", latitude: 17.447528, longitude: 78.451639, district: "Hyderabad", state: "Telangana" },
        { canonical_name: "Basara", display_name: "Sri Gnana Saraswathi Devasthanam, Basara", latitude: 18.870451, longitude: 77.954153, district: "Nirmal", state: "Telangana" },
        { canonical_name: "Birla Mandir", display_name: "Birla Mandir, Hyderabad", latitude: 17.40429, longitude: 78.469146, district: "Hyderabad", state: "Telangana" },
        { canonical_name: "Chilkur", display_name: "Chilkur Balaji Temple", latitude: 17.378281, longitude: 78.297491, district: "Rangareddy", state: "Telangana" },
        { canonical_name: "Dharmapuri", display_name: "Sri Lakshmi Narasimha Swamy Temple, Dharmapuri", latitude: 18.95152, longitude: 79.09171, district: "Jagtial", state: "Telangana" },
        { canonical_name: "Edupayal", display_name: "Sri Edupayala Vana Durga Bhavani Devalayam", latitude: 17.97121, longitude: 78.17552, district: "Medak", state: "Telangana" },
        { canonical_name: "Jogulamba(New)", display_name: "Jogulamba Devi Temple, Alampur", latitude: 15.876984, longitude: 78.134864, district: "Jogulamba Gadwal", state: "Telangana" },
        { canonical_name: "Kaleshvaram", display_name: "Sri Kaleshwara Mukteshwara Swamy Temple, Kaleshwaram", latitude: 18.811821, longitude: 79.904026, district: "Jayashankar Bhupalpally", state: "Telangana" },
        { canonical_name: "Karmanghat", display_name: "Karmanghat Hanuman Temple, Hyderabad", latitude: 17.341166, longitude: 78.522387, district: "Hyderabad", state: "Telangana" },
        { canonical_name: "Medaram", display_name: "Sammakka Saralamma Gaddelu, Medaram", latitude: 18.2259, longitude: 79.9463, district: "Mulugu", state: "Telangana" },
        { canonical_name: "Ramappa", display_name: "Ramappa Temple, Palampet", latitude: 18.256242, longitude: 79.940562, district: "Mulugu", state: "Telangana" },
        { canonical_name: "Srisaila", display_name: "Sri Bhramaramba Mallikarjuna Swamy Temple, Srisailam", latitude: 16.0742, longitude: 78.868199, district: "Nandyal", state: "Andhra Pradesh" },
        { canonical_name: "Thousand Pillar", display_name: "Thousand Pillar Temple, Hanamkonda", latitude: 18.003737, longitude: 79.574771, district: "Hanamkonda", state: "Telangana" },
        { canonical_name: "Ttd", display_name: "Sri Venkateswara Swamy Temple, Tirumala", latitude: 13.680736, longitude: 79.350898, district: "Tirupati", state: "Andhra Pradesh" },
        { canonical_name: "Ujjaini", display_name: "Sri Ujjaini Mahakali Temple, Secunderabad", latitude: 17.441235, longitude: 78.498123, district: "Hyderabad", state: "Telangana" },
        { canonical_name: "Veera Satyanarayana", display_name: "Sri Veera Venkata Satyanarayana Swamy Temple, Annavaram", latitude: 17.280512, longitude: 82.404523, district: "Kakinada", state: "Andhra Pradesh" },
        { canonical_name: "Vemulavada", display_name: "Sri Raja Rajeshwara Swamy Temple, Vemulawada", latitude: 18.467234, longitude: 78.868728, district: "Rajanna Sircilla", state: "Telangana" },
        { canonical_name: "Yadadri", display_name: "Sri Lakshmi Narasimha Swamy Temple, Yadadri", latitude: 17.587714, longitude: 78.948325, district: "Yadadri Bhuvanagiri", state: "Telangana" }
    ];
    let templeCoordinates = [...VERIFIED_TEMPLE_REGISTRY];
    let isListening = false;
    let speechRecognition = null;

    // Initialize Icons
    if (window.lucide) {
        lucide.createIcons();
    }

    // -------------------------------------------------------------
    // Route Duration Formatter (Hours & Minutes)
    // -------------------------------------------------------------
    function formatDurationHoursMinutes(minutes) {
        if (minutes === null || minutes === undefined || minutes === "" || isNaN(Number(minutes))) {
            return "";
        }
        const m = Math.round(Number(minutes));
        if (m <= 0) {
            return "0 mins";
        }
        if (m < 60) {
            return m === 1 ? "1 min" : `${m} mins`;
        }
        const hrs = Math.floor(m / 60);
        const rem = m % 60;
        const hrStr = hrs === 1 ? "1 hr" : `${hrs} hrs`;
        if (rem === 0) {
            return hrStr;
        }
        const minStr = rem === 1 ? "1 min" : `${rem} mins`;
        return `${hrStr} ${minStr}`;
    }
    window.formatDurationHoursMinutes = formatDurationHoursMinutes;

    // -------------------------------------------------------------
    // Session Lifecycle Management
    // -------------------------------------------------------------
    function getOrCreateSessionId() {
        let sid = localStorage.getItem("pilgrim_session_id");
        if (!sid) {
            sid = "session_" + (crypto.randomUUID ? crypto.randomUUID() : Date.now() + "_" + Math.random().toString(36).substring(2, 9));
            localStorage.setItem("pilgrim_session_id", sid);
        }
        return sid;
    }

    function createFreshSession() {
        const freshId = "session_" + (crypto.randomUUID ? crypto.randomUUID() : Date.now() + "_" + Math.random().toString(36).substring(2, 9));
        localStorage.setItem("pilgrim_session_id", freshId);
        currentSessionId = freshId;

        // Cancel any active speech synthesis and reset controls
        if (window.speechSynthesis) {
            window.speechSynthesis.cancel();
        }
        resetActiveSpeechUi();
        clearActiveRoute();
        currentSelectedTemple = null;
        currentLat = null;
        currentLng = null;
        if (activeInAppPlaceMarker) {
            try { activeInAppPlaceMarker.setMap(null); } catch (e) {}
            activeInAppPlaceMarker = null;
        }
        if (placesTargetTitle) {
            placesTargetTitle.innerText = "Nearby Discoveries";
        }
        if (placesList) {
            placesList.innerHTML = `<div style="font-size: 12px; color: var(--text-muted); text-align: center; padding: 20px;">Select a temple from the list or ask a navigation/location question to discover live places.</div>`;
        }
        
        // Remove active state from history items
        if (chatHistoryList) {
            chatHistoryList.querySelectorAll(".chat-history-item").forEach(el => el.classList.remove("active"));
        }

        // Reset message area to initial welcome box
        messageArea.innerHTML = `
            <div class="welcome-box" id="welcomeBox">
                <h2>Namaste! New Session Started</h2>
                <p>Conversation memory reset. Ask your question below or click a suggestion:</p>
                <div class="prompt-grid">
                    <div class="prompt-card" onclick="sendPrompt('What are the timings of Yadadri temple?')">
                        🕒 <strong>Yadadri Timings</strong><br>Darshan schedule & poojas
                    </div>
                    <div class="prompt-card" onclick="sendPrompt('How far is Hyderabad from Yadadri?')">
                        🧭 <strong>Hyderabad to Yadadri</strong><br>Driving distance & duration
                    </div>
                    <div class="prompt-card" onclick="sendPrompt('What is the weather at Yadadri?')">
                        🌤️ <strong>Yadadri Weather</strong><br>Live temperature & conditions
                    </div>
                    <div class="prompt-card" onclick="sendPrompt('What will a visit to Yadadri cost?')">
                        💰 <strong>Yadadri Budget</strong><br>Verified sevas & prasadam costs
                    </div>
                </div>
            </div>
        `;
        if (directionsRenderer) {
            directionsRenderer.setDirections({ routes: [] });
        }
        if (window.lucide) lucide.createIcons();
    }

    if (newChatBtn) {
        newChatBtn.addEventListener("click", () => {
            createFreshSession();
        });
    }

    // -------------------------------------------------------------
    // Chat History Management (MongoDB-backed User Isolation)
    // -------------------------------------------------------------
    const chatHistoryList = document.getElementById("chatHistoryList");

    async function loadChatHistory(autoRestore = false) {
        if (!chatHistoryList) return;
        const token = localStorage.getItem("pilgrim_jwt");
        if (!token) {
            chatHistoryList.innerHTML = '<div class="history-empty-msg">Sign in to view history</div>';
            return;
        }

        try {
            const res = await fetch("/api/chat/history", {
                headers: { "Authorization": `Bearer ${token}` }
            });
            if (!res.ok) {
                if (res.status === 401) {
                    chatHistoryList.innerHTML = '<div class="history-empty-msg">Sign in to view history</div>';
                }
                return;
            }

            const sessions = await res.json();
            if (!sessions || sessions.length === 0) {
                chatHistoryList.innerHTML = '<div class="history-empty-msg">No conversations yet</div>';
                return;
            }

            // Group sessions by Today, Yesterday, Earlier
            const now = new Date();
            const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
            const yesterday = new Date(today);
            yesterday.setDate(yesterday.getDate() - 1);

            const groups = {
                today: [],
                yesterday: [],
                earlier: []
            };

            sessions.forEach(s => {
                const sDate = s.updated_at ? new Date(s.updated_at) : new Date(s.created_at);
                const sDay = new Date(sDate.getFullYear(), sDate.getMonth(), sDate.getDate());

                if (sDay.getTime() >= today.getTime()) {
                    groups.today.push(s);
                } else if (sDay.getTime() >= yesterday.getTime()) {
                    groups.yesterday.push(s);
                } else {
                    groups.earlier.push(s);
                }
            });

            let html = "";
            const renderGroup = (title, items) => {
                if (!items || items.length === 0) return "";
                let gHtml = `<div class="chat-history-group-title">${title}</div>`;
                items.forEach(item => {
                    const isActive = item.session_id === currentSessionId ? "active" : "";
                    const safeTitle = escapeHtml(item.title || "Untitled Conversation");
                    gHtml += `
                        <div class="chat-history-item ${isActive}" data-session-id="${escapeHtml(item.session_id)}" title="${safeTitle}">
                            <div class="chat-history-item-left">
                                <span class="chat-history-icon">💬</span>
                                <span class="chat-history-title">${safeTitle}</span>
                            </div>
                            <button type="button" class="btn-delete-session" title="Delete conversation" data-session-id="${escapeHtml(item.session_id)}">
                                <i data-lucide="trash-2" style="width: 12px; height: 12px;"></i>
                            </button>
                        </div>
                    `;
                });
                return gHtml;
            };

            html += renderGroup("Today", groups.today);
            html += renderGroup("Yesterday", groups.yesterday);
            html += renderGroup("Earlier", groups.earlier);

            chatHistoryList.innerHTML = html;
            if (window.lucide) lucide.createIcons();

            // Attach event listeners for clicking conversation and deleting conversation
            chatHistoryList.querySelectorAll(".chat-history-item").forEach(el => {
                el.addEventListener("click", (e) => {
                    if (e.target.closest(".btn-delete-session")) return;
                    const sessId = el.getAttribute("data-session-id");
                    if (sessId) {
                        loadChatConversation(sessId);
                    }
                });
            });

            chatHistoryList.querySelectorAll(".btn-delete-session").forEach(btn => {
                btn.addEventListener("click", (e) => {
                    e.stopPropagation();
                    const sessId = btn.getAttribute("data-session-id");
                    if (sessId) {
                        deleteChatConversation(sessId);
                    }
                });
            });

            // If autoRestore is requested (e.g. on page reload or sign-in), automatically restore active session
            if (autoRestore) {
                const activeId = localStorage.getItem("pilgrim_session_id");
                let targetSessionId = null;
                if (activeId && sessions.some(s => s.session_id === activeId)) {
                    targetSessionId = activeId;
                } else if (sessions.length > 0) {
                    targetSessionId = sessions[0].session_id;
                }
                if (targetSessionId) {
                    await loadChatConversation(targetSessionId);
                }
            }

        } catch (err) {
            console.warn("Failed to load chat history:", err);
        }
    }

    let activeLoadingSessionId = null;

    async function loadChatConversation(sessionId) {
        if (!sessionId) return;
        const token = localStorage.getItem("pilgrim_jwt");
        if (!token) return;

        // Cancel active voice and routes
        if (window.speechSynthesis) window.speechSynthesis.cancel();
        resetActiveSpeechUi();
        clearActiveRoute();

        // Update active session id and loading token to guard against async session race conditions
        currentSessionId = sessionId;
        activeLoadingSessionId = sessionId;
        localStorage.setItem("pilgrim_session_id", sessionId);

        // Update active class in sidebar history
        if (chatHistoryList) {
            chatHistoryList.querySelectorAll(".chat-history-item").forEach(el => {
                if (el.getAttribute("data-session-id") === sessionId) {
                    el.classList.add("active");
                } else {
                    el.classList.remove("active");
                }
            });
        }

        // Show loading state in message area
        messageArea.innerHTML = `
            <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-muted); font-size: 13px; gap: 8px;">
                <span class="dot-online" style="animation: pulse 1s infinite;"></span>
                Loading conversation...
            </div>
        `;

        try {
            const res = await fetch(`/api/chat/history/${encodeURIComponent(sessionId)}`, {
                headers: { "Authorization": `Bearer ${token}` }
            });

            // If user switched to another session while this fetch was in flight, abort rendering
            if (activeLoadingSessionId !== sessionId) return;

            if (!res.ok) {
                appendErrorMessage("Unable to load conversation messages.");
                return;
            }

            const messages = await res.json();

            // Check again after JSON parsing
            if (activeLoadingSessionId !== sessionId) return;

            messageArea.innerHTML = "";

            if (!messages || messages.length === 0) {
                messageArea.innerHTML = `
                    <div class="welcome-box">
                        <h2>Empty Conversation</h2>
                        <p>No messages in this conversation. Ask a question to begin.</p>
                    </div>
                `;
                return;
            }

            let lastNavMeta = null;
            const renderedMessageIds = new Set();
            let lastUserContent = null;
            let lastRole = null;

            messages.forEach(msg => {
                const msgId = msg.message_id || msg.client_message_id;
                if (msgId && renderedMessageIds.has(msgId)) {
                    return; // Skip duplicate message record by stable message ID
                }
                if (msgId) {
                    renderedMessageIds.add(msgId);
                }

                if (msg.role === "user") {
                    const cleanContent = (msg.content || "").trim();
                    // Guard against consecutive duplicate unanswered user messages (e.g. from accidental duplicate submissions)
                    if (lastRole === "user" && lastUserContent === cleanContent) {
                        return; // Suppress duplicate unresponded user prompt on render
                    }
                    lastUserContent = cleanContent;
                    lastRole = "user";
                    appendUserMessage(msg.content, msgId);
                } else {
                    lastRole = "assistant";
                    lastUserContent = null;
                    const botData = {
                        answer: msg.content,
                        agent_name: msg.agent_name || "temple_information",
                        metadata: msg.metadata || {},
                        sources: msg.sources || [],
                        confidence_score: msg.confidence_score,
                        message_id: msgId
                    };
                    appendBotResponse(botData);
                    if (botData.agent_name === "live_navigation" || (botData.metadata && (botData.metadata.polyline || botData.metadata.distance_km !== undefined))) {
                        lastNavMeta = botData.metadata;
                    }
                }
            });

            // Restore route on map if valid saved geometry exists without making new routing provider calls
            if (lastNavMeta && lastNavMeta.origin && lastNavMeta.destination) {
                latestNavigationMeta = lastNavMeta;
                if (lastNavMeta.polyline) {
                    const btn = document.querySelector(".nav-btn-view-map");
                    if (googleMap) {
                        focusRouteOnMap(
                            lastNavMeta.origin,
                            lastNavMeta.destination,
                            btn,
                            lastNavMeta.origin_canonical,
                            lastNavMeta.destination_canonical,
                            lastNavMeta.polyline,
                            lastNavMeta.origin_coordinates,
                            lastNavMeta.destination_coordinates
                        );
                    } else {
                        pendingRouteToRestore = {
                            origin: lastNavMeta.origin,
                            destination: lastNavMeta.destination,
                            btn: btn,
                            originCanonical: lastNavMeta.origin_canonical,
                            destCanonical: lastNavMeta.destination_canonical,
                            polyline: lastNavMeta.polyline,
                            origCoords: lastNavMeta.origin_coordinates,
                            destCoords: lastNavMeta.destination_coordinates
                        };
                    }
                }
            }

            scrollToBottom();
        } catch (err) {
            console.error("Error loading chat conversation:", err);
            appendErrorMessage("Failed to load conversation from server.");
        }
    }

    async function deleteChatConversation(sessionId) {
        if (!sessionId) return;
        const token = localStorage.getItem("pilgrim_jwt");
        if (!token) return;

        try {
            const res = await fetch(`/api/chat/history/${encodeURIComponent(sessionId)}`, {
                method: "DELETE",
                headers: { "Authorization": `Bearer ${token}` }
            });
            if (res.ok) {
                if (currentSessionId === sessionId) {
                    createFreshSession();
                }
                loadChatHistory();
            }
        } catch (err) {
            console.error("Failed to delete session:", err);
        }
    }

    // -------------------------------------------------------------
    // Multilingual Selection Chips
    // -------------------------------------------------------------
    langChips.forEach(chip => {
        chip.addEventListener("click", () => {
            langChips.forEach(c => c.classList.remove("active"));
            chip.classList.add("active");
            currentLang = chip.getAttribute("data-lang");
        });
    });

    // -------------------------------------------------------------
    // Map Panel Toggle
    // -------------------------------------------------------------
    if (toggleMapBtn && mapPanel) {
        toggleMapBtn.addEventListener("click", () => {
            mapPanel.classList.toggle("collapsed");
            if (!mapPanel.classList.contains("collapsed") && googleMap) {
                google.maps.event.trigger(googleMap, "resize");
            }
        });
    }

    // -------------------------------------------------------------
    // Map / Places Tabs
    // -------------------------------------------------------------
    if (tabPlacesBtn) {
        tabPlacesBtn.addEventListener("click", () => {
            if (mapPanel && mapPanel.classList.contains("collapsed")) {
                mapPanel.classList.remove("collapsed");
            }
            if (tabMapBtn) tabMapBtn.classList.remove("active");
            tabPlacesBtn.classList.add("active");

            // Show loading state on button
            const originalText = tabPlacesBtn.innerText;
            tabPlacesBtn.innerText = "Finding Places...";
            tabPlacesBtn.disabled = true;

            const placesPanelElem = document.getElementById("placesPanel");
            if (placesPanelElem) {
                placesPanelElem.scrollIntoView({ behavior: "smooth", block: "nearest" });
            }

            // Trigger discovery for current temple if one is selected
            if (currentSelectedTemple && currentLat && currentLng) {
                fetchNearbyPlaces(currentSelectedTemple, currentLat, currentLng, () => {
                    tabPlacesBtn.innerText = originalText;
                    tabPlacesBtn.disabled = false;
                });
            } else {
                tabPlacesBtn.innerText = originalText;
                tabPlacesBtn.disabled = false;
                if (placesTargetTitle) placesTargetTitle.innerText = "Nearby Discoveries";
                if (placesList) {
                    placesList.innerHTML = `<div style="font-size: 12px; color: var(--text-muted); text-align: center; padding: 20px;">Please select a temple from the covered temples list or ask a temple question to discover nearby places.</div>`;
                }
            }
        });
    }

    if (tabMapBtn) {
        tabMapBtn.addEventListener("click", () => {
            if (tabPlacesBtn) tabPlacesBtn.classList.remove("active");
            tabMapBtn.classList.add("active");
            if (googleMap) {
                google.maps.event.trigger(googleMap, "resize");
            }
        });
    }

    // Initialize Map Panel Route Planner controls
    initMapRoutePlanner();

    // -------------------------------------------------------------
    // Chat Submission & Orchestration
    // -------------------------------------------------------------
    let isSubmittingChat = false;

    window.sendPrompt = function(text) {
        if (isSubmittingChat) return;
        userQuery.value = text;
        chatForm.dispatchEvent(new Event("submit"));
    };

    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (isSubmittingChat) return;

        let question = userQuery.value.trim();
        if (!question) return;

        isSubmittingChat = true;
        submitBtn.disabled = true;
        userQuery.disabled = true;

        // Apply language directive if non-English chip active
        if (currentLang === "te" && !question.toLowerCase().includes("telugu")) {
            question = `Answer in Telugu: ${question}`;
        } else if (currentLang === "hi" && !question.toLowerCase().includes("hindi")) {
            question = `Answer in Hindi: ${question}`;
        }

        // Hide initial welcome box if present
        const welcomeBox = document.getElementById("welcomeBox");
        if (welcomeBox) welcomeBox.style.display = "none";

        // Generate stable client message identifier for idempotency & deduplication
        const clientMessageId = "msg_" + Date.now() + "_" + Math.random().toString(36).substring(2, 9);

        // Append User Message Bubble
        appendUserMessage(question, clientMessageId);
        userQuery.value = "";

        // Append Loading Indicator
        const loadingId = appendLoadingBubble();

        try {
            const chatHeaders = { "Content-Type": "application/json" };
            const storedJwt = localStorage.getItem("pilgrim_jwt");
            if (storedJwt) {
                chatHeaders["Authorization"] = `Bearer ${storedJwt}`;
            }

            const res = await fetch("/chat", {
                method: "POST",
                headers: chatHeaders,
                body: JSON.stringify({
                    question: question,
                    session_id: currentSessionId,
                    client_message_id: clientMessageId
                })
            });

            removeLoadingBubble(loadingId);

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                appendErrorMessage(errData.detail || "Server communication error. Please try again.");
                return;
            }

            const data = await res.json();
            
            // Preserve session_id from response
            if (data.session_id) {
                currentSessionId = data.session_id;
                localStorage.setItem("pilgrim_session_id", currentSessionId);
            }

            // Render structured bot response
            appendBotResponse(data);

            // Trigger Map / Places side-effects if applicable
            handleMapSideEffects(data);

            // Refresh chat history sidebar without re-rendering active conversation
            loadChatHistory(false);

        } catch (err) {
            removeLoadingBubble(loadingId);
            appendErrorMessage("Connection error. Ensure local PilgrimAI backend is running on http://localhost:8000.");
        } finally {
            isSubmittingChat = false;
            submitBtn.disabled = false;
            userQuery.disabled = false;
            userQuery.focus();
            scrollToBottom();
        }
    });

    // -------------------------------------------------------------
    // Message Rendering Helpers
    // -------------------------------------------------------------
    function appendUserMessage(text, messageId = null) {
        const row = document.createElement("div");
        row.className = "msg-row user";
        if (messageId) {
            row.setAttribute("data-message-id", messageId);
        }
        row.innerHTML = `
            <div class="msg-avatar">YOU</div>
            <div class="msg-bubble-wrapper">
                <div class="msg-bubble">${escapeHtml(text)}</div>
            </div>
        `;
        messageArea.appendChild(row);
        scrollToBottom();
    }

    function appendErrorMessage(errText) {
        const row = document.createElement("div");
        row.className = "msg-row bot";
        row.innerHTML = `
            <div class="msg-avatar" style="background: #ef4444; color: #fff;">⚠️</div>
            <div class="msg-bubble-wrapper">
                <div class="msg-bubble" style="border-color: #ef4444; color: #fca5a5;">
                    ${escapeHtml(errText)}
                </div>
            </div>
        `;
        messageArea.appendChild(row);
        scrollToBottom();
    }

    function appendLoadingBubble() {
        const id = "load_" + Date.now();
        const row = document.createElement("div");
        row.className = "msg-row bot";
        row.id = id;
        row.innerHTML = `
            <div class="msg-avatar">AI</div>
            <div class="msg-bubble-wrapper">
                <div class="msg-bubble" style="color: var(--text-muted); font-style: italic;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span class="dot-online" style="animation: pulse 1s infinite;"></span>
                        Consulting certified multi-agent engine...
                    </div>
                </div>
            </div>
        `;
        messageArea.appendChild(row);
        scrollToBottom();
        return id;
    }

    function removeLoadingBubble(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function appendBotResponse(data) {
        const row = document.createElement("div");
        row.className = "msg-row bot";
        if (data && (data.message_id || data.client_message_id)) {
            row.setAttribute("data-message-id", data.message_id || data.client_message_id);
        }

        // 1. Agent Name & Badge Mapping
        const badgeConfig = getAgentBadgeConfig(data.agent_name);
        const badgeHtml = `
            <div class="agent-badge-bar">
                <span class="agent-badge ${badgeConfig.className}">
                    ${badgeConfig.icon} ${badgeConfig.label}
                </span>
                ${data.latency_ms ? `<span class="latency-tag">${Math.round(data.latency_ms)}ms</span>` : ""}
                <div class="speech-controls" data-speech-state="idle">
                    ${renderSpeechControlsHtml("idle")}
                </div>
            </div>
        `;

        // 2. Structured Metadata Cards
        let structuredCardsHtml = "";
        const meta = data.metadata || {};

        // Weather Card
        if (data.agent_name === "live_weather" || meta.current_weather) {
            structuredCardsHtml += renderWeatherCard(meta);
        }
        let hasBudgetCard = false;
        // Budget Card: render when budget specialist responds or structured budget parameters present
        if (data.agent_name === "budget" || meta.is_budget_request === true || (meta.parameters && meta.parameters.budget_limit !== null && meta.parameters.budget_limit !== undefined) || (meta.line_items && meta.line_items.length > 0)) {
            structuredCardsHtml += renderBudgetCard(meta);
            hasBudgetCard = true;
        }
        // Navigation Card
        else if (data.agent_name === "live_navigation" || meta.distance_km !== undefined || meta.missing_origin) {
            structuredCardsHtml += renderNavigationCard(meta);
        }
        // Itinerary Card
        else if (data.agent_name === "itinerary_planning" || meta.itinerary) {
            structuredCardsHtml += renderItineraryCard(meta);
        }

        // 3. Format Answer Markdown
        let answerText = data.answer || "";
        if (hasBudgetCard) {
            // Structured HTML budget card already renders the budget metrics grid and line-item table.
            // Strip duplicate Markdown budget section (title, summary, warning, and table)
            // while preserving the separate accommodation options list, day-by-day itinerary, and connectivity.
            answerText = answerText.replace(/##.*?Pilgrimage Budget & Expense Estimate[\s\S]*?(?=\n\s*(?:---\s*\n\s*)?#{3,4}\s*Day|\n###.*?Accommodation Options|\n###.*?Travel & Connectivity|$)/i, "");
            answerText = answerText.replace(/###.*?Expense Breakdown[\s\S]*?(?=\n\s*(?:---\s*\n\s*)?#{3,4}\s*Day|\n###.*?Accommodation Options|\n###.*?Travel & Connectivity|$)/i, "");
            answerText = answerText.replace(/###\s*🏨?\s*Accommodation Summary[\s\S]*?(?=\n###|\n<details|$)/i, "");
            answerText = answerText.replace(/(?:\n\s*---\s*){2,}/g, "\n\n---\n\n").trim();
        }
        const formattedAnswer = formatMarkdownText(answerText);

        // 4. Sources & Confidence Bar (Expandable, unobtrusive evidence drawer)
        let sourcesHtml = "";
        if (data.sources && data.sources.length > 0) {
            const chips = data.sources.map(s => `<span class="src-chip">${escapeHtml(s)}</span>`).join("");
            let confHtml = "";
            if (data.confidence_score !== undefined && data.confidence_score >= 0.0) {
                confHtml = `<span class="conf-chip">Confidence: ${data.confidence_score.toFixed(2)}</span>`;
            }
            sourcesHtml = `
                <details class="sources-details">
                    <summary class="sources-summary">
                        <span>📄 View Evidence & Sources (${data.sources.length})</span>
                        ${confHtml}
                    </summary>
                    <div class="sources-chips">
                        ${chips}
                    </div>
                </details>
            `;
        }

        const attrRawAnswer = (data.answer || "").replace(/<\/?(?:details|summary)[^>]*>/gi, "");
        row.innerHTML = `
            <div class="msg-avatar">AI</div>
            <div class="msg-bubble-wrapper">
                ${badgeHtml}
                <div class="msg-bubble" data-raw-answer="${escapeHtml(attrRawAnswer)}">
                    ${structuredCardsHtml}
                    <div class="answer-content">${formattedAnswer}</div>
                    ${sourcesHtml}
                </div>
            </div>
        `;

        messageArea.appendChild(row);
        if (window.lucide) lucide.createIcons();
        scrollToBottom();
    }

    // -------------------------------------------------------------
    // Agent Badges Config
    // -------------------------------------------------------------
    function getAgentBadgeConfig(agentName) {
        switch (agentName) {
            case "temple_information":
                return { label: "Temple Information", icon: "🛕", className: "badge-temple_info" };
            case "comparison":
                return { label: "Temple Comparison", icon: "⚖️", className: "badge-comparison" };
            case "cross_temple_aggregation":
                return { label: "Cross-Temple Search", icon: "🔍", className: "badge-temple_info" };
            case "itinerary_planning":
                return { label: "Itinerary Planner", icon: "📅", className: "badge-itinerary" };
            case "live_weather":
                return { label: "Live Weather", icon: "🌤️", className: "badge-weather" };
            case "budget":
                return { label: "Budget Specialist", icon: "💰", className: "badge-budget" };
            case "live_navigation":
                return { label: "Live Navigation", icon: "🧭", className: "badge-navigation" };
            case "voice_multilingual":
                return { label: "Multilingual Guide", icon: "🌐", className: "badge-multilingual" };
            case "emergency":
                return { label: "Emergency Services", icon: "🚨", className: "badge-emergency" };
            case "recommendation":
                return { label: "Temple Recommendation", icon: "⭐", className: "badge-temple_info" };
            case "multi_intent":
                return { label: "Multi-Intent Pipeline", icon: "⚡", className: "badge-comparison" };
            default:
                return { label: agentName ? agentName.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase()) : "Pilgrim Specialist", icon: "✨", className: "badge-default" };
        }
    }

    // -------------------------------------------------------------
    // Structured Card Renderers
    // -------------------------------------------------------------
    function renderWeatherCard(meta) {
        const cur = meta.current_weather || {};
        const temp = cur.temperature !== undefined ? `${cur.temperature}°C` : "N/A";
        const cond = cur.condition || "Live Observation";
        const loc = meta.display_name || meta.location || "Temple Region";
        const humidity = cur.relative_humidity !== undefined ? `${cur.relative_humidity}%` : "--";
        const wind = cur.wind_speed !== undefined ? `${cur.wind_speed} km/h` : "--";
        const rain = cur.precipitation !== undefined ? `${cur.precipitation} mm` : "--";
        const cacheTag = meta.cache_hit ? "Cached" : "Fresh";

        return `
            <div class="structured-card weather-card">
                <div class="weather-header">
                    <div>
                        <div class="weather-loc">🌤️ ${escapeHtml(loc)}</div>
                        <div style="font-size: 11.5px; color: var(--text-secondary);">${escapeHtml(cond)} • <span style="color: var(--accent-emerald);">${cacheTag}</span></div>
                    </div>
                    <div class="weather-temp-badge">${temp}</div>
                </div>
                <div class="weather-grid">
                    <div class="weather-metric">
                        <div class="label">Humidity</div>
                        <div class="val">${humidity}</div>
                    </div>
                    <div class="weather-metric">
                        <div class="label">Wind Speed</div>
                        <div class="val">${wind}</div>
                    </div>
                    <div class="weather-metric">
                        <div class="label">Rain</div>
                        <div class="val">${rain}</div>
                    </div>
                </div>
            </div>
        `;
    }

    function renderBudgetCard(meta, isItineraryWithBudget = false) {
        const total = meta.calculated_total || {};
        const params = meta.parameters || {};
        const pax = params.pax || meta.pax || 1;
        const adults = params.adults !== undefined ? params.adults : meta.adults;
        const children = params.children !== undefined ? params.children : meta.children;
        const seniors = params.seniors !== undefined ? params.seniors : (params.senior_citizens !== undefined ? params.senior_citizens : (meta.seniors !== undefined ? meta.seniors : meta.senior_citizens));
        let paxParts = [];
        if (adults !== undefined && adults !== null) {
            paxParts.push(`${adults} Adult${adults === 1 ? '' : 's'}`);
        }
        if (children !== undefined && children !== null) {
            paxParts.push(`${children} ${children === 1 ? 'Child' : 'Children'}`);
        }
        if (seniors !== undefined && seniors !== null) {
            paxParts.push(`${seniors} Senior Citizen${seniors === 1 ? '' : 's'}`);
        }
        let paxStr = `${pax} Pilgrim(s)`;
        if (paxParts.length > 0) {
            paxStr = `${pax} Pilgrim(s) (${paxParts.join(', ')})`;
        }
        const days = params.days || 1;
        const nights = params.nights !== undefined ? params.nights : (params.is_day_trip ? 0 : Math.max(0, days - 1));
        const durStr = params.is_day_trip ? "1 Day (Day Trip)" : `${days} Day(s), ${nights} Night(s)`;
        const orig = params.origin || "Hyderabad";
        const dest = params.destination || "Temple";
        const blim = params.budget_limit;
        const status = meta.budget_status || "ESTIMATED_WITHIN_BUDGET";

        const selectedTotal = (meta.selected_plan && meta.selected_plan.total !== undefined)
            ? meta.selected_plan.total
            : (total.selected_feasible_total !== undefined ? total.selected_feasible_total : (total.min !== undefined ? total.min : 0));
        const remaining = (blim !== null && blim !== undefined) ? (blim - selectedTotal) : null;

        let totalStr = `₹${selectedTotal.toLocaleString()}`;
        let totalSubtext = "";
        if (meta.is_cross_limit && total.min !== undefined && total.max !== undefined) {
            totalSubtext = `<div style="font-size: 11px; color: #fbbf24; margin-top: 3px;">⚠️ Full Range: ₹${total.min.toLocaleString()} – ₹${total.max.toLocaleString()} (Higher-tier options reach up to ₹${total.max.toLocaleString()})</div>`;
        } else if (total.min !== undefined && total.max !== undefined && total.min !== total.max) {
            totalStr = `₹${total.min.toLocaleString()} – ₹${total.max.toLocaleString()}`;
        }

        let statusClass = "budget-status-within";
        let statusLabel = "✅ Within Budget";
        if (remaining !== null && remaining < 0) {
            statusClass = "budget-status-over";
            statusLabel = "⚠️ Over Budget";
        } else if (status === "OVER_BUDGET") {
            statusClass = "budget-status-over";
            statusLabel = "⚠️ Over Budget";
        } else if (status === "CANNOT_VERIFY") {
            statusClass = "budget-status-unverified";
            statusLabel = "❓ Unverified Pricing";
        } else if (meta.origin_assumed || meta.is_preliminary_origin) {
            statusClass = "budget-status-unverified";
            statusLabel = "ℹ️ Partial On-Site Baseline (Origin Pending)";
        } else if (meta.is_cross_limit) {
            statusClass = "budget-status-within";
            statusLabel = "✅ Within Budget (Low-Cost Option)";
        }

        const items = meta.line_items || [];
        let itemsRows = "";
        if (items.length > 0) {
            itemsRows = items.map(it => {
                let cVal = it.subtotal_min !== undefined ? it.subtotal_min : it.min_cost;
                let isPendingOrigin = (it.category === "travel" || (it.item_name && it.item_name.includes("Intercity"))) && (it.provenance === "UNAVAILABLE" || (it.item_name && it.item_name.includes("Starting Location Unspecified")));
                let isUnquantified = !isPendingOrigin && (it.provenance === "UNAVAILABLE" || it.is_quantified === false || cVal === null || (it.item_name && it.item_name.includes("Tariff Unlisted")));

                let isDayTripStay = it.category === "accommodation" && (
                    (it.quantity_desc && it.quantity_desc.toLowerCase().includes("no overnight stay")) ||
                    (it.notes && it.notes.toLowerCase().includes("single-day")) ||
                    (meta.canonical_nights === 0 || meta.nights === 0 || (meta.days === 1 && !meta.canonical_nights))
                );
                let isNotSelectedStay = it.category === "accommodation" && (
                    (it.notes && it.notes.toLowerCase().includes("not selected")) ||
                    (it.quantity_desc && it.quantity_desc.toLowerCase().includes("not selected"))
                );

                let cStr;
                if (isPendingOrigin) {
                    cStr = "Pending starting location";
                } else if (isDayTripStay) {
                    cStr = "Not required";
                } else if (isNotSelectedStay) {
                    cStr = "Not selected";
                } else if (isUnquantified) {
                    cStr = "Tariff unlisted";
                } else if (it.is_free) {
                    cStr = "FREE";
                } else if (it.subtotal_min === 0 && it.subtotal_max === 0) {
                    cStr = it.category === "accommodation" ? "Not required" : "FREE";
                } else if (it.subtotal_min !== it.subtotal_max && it.subtotal_max) {
                    cStr = `₹${Number(it.subtotal_min).toLocaleString()} <span style="font-size: 10px; color: var(--text-muted); font-weight: normal;">(Range: ₹${Number(it.subtotal_min).toLocaleString()} – ₹${Number(it.subtotal_max).toLocaleString()})</span>`;
                } else {
                    cStr = `₹${Number(cVal).toLocaleString()}`;
                }

                let cleanName = it.item_name || "Expense";
                if (cleanName.includes("Circuit Intercity Travel")) cleanName = "Intercity Transportation (Bus/Train)";
                else if (cleanName.includes("Local Sightseeing")) cleanName = "Local Transportation & Sightseeing Transit";
                else if (cleanName.includes("Meals")) cleanName = "Meals & Refreshments (Vegetarian)";
                else if (cleanName.includes("General Darshan") && !cleanName.includes(" — ")) cleanName = "General Darshan (Temple Sanctum)";
                else if (cleanName.includes("Accommodation") && meta.is_cross_limit) cleanName = `${cleanName} (Standard Non-AC)`;

                let evClass = "evidence-estimate";
                let evLabel = "Estimated";
                if (isPendingOrigin) {
                    evClass = "evidence-pending";
                    evLabel = "Pending confirmation";
                } else if (isUnquantified) {
                    evClass = "evidence-estimate";
                    evLabel = "Pending confirmation";
                } else if (it.is_free || (it.subtotal_min === 0 && it.subtotal_max === 0)) {
                    evClass = "evidence-listed";
                    evLabel = "Complimentary";
                } else if (it.provenance === "ESTIMATE") {
                    evClass = "evidence-estimate";
                    evLabel = "Estimated benchmark";
                } else if (it.subtotal_min !== it.subtotal_max && it.subtotal_max) {
                    evClass = "evidence-estimate";
                    evLabel = "Estimated (Fare range)";
                } else if (it.provenance === "LISTED_PRICE" || it.provenance === "VERIFIED") {
                    evClass = "evidence-listed";
                    evLabel = "Listed price";
                }

                return `
                    <tr>
                        <td><strong>${escapeHtml(cleanName)}</strong></td>
                        <td style="color: var(--text-muted);">${escapeHtml(it.quantity_desc || it.unit || "Standard option")}</td>
                        <td style="font-weight: 700; color: #f1f5f9;">${cStr}</td>
                    </tr>
                `;
            }).join("");
        }

        // Accommodation comparison & alternatives block
        let accomSectionHtml = "";
        const accomByTemple = meta.accommodation_options;
        if (accomByTemple && typeof accomByTemple === "object" && Object.keys(accomByTemple).length > 0) {
            let templeCards = "";
            for (const [tName, opts] of Object.entries(accomByTemple)) {
                const isBalkampet = tName.toLowerCase().includes("balkampet");
                let optListHtml = "";
                if (Array.isArray(opts) && opts.length > 0) {
                    optListHtml = opts.map(o => {
                        // 1. Price badge: green for numeric tariffs, subtle for unlisted
                        const rawPrice = o.price || "";
                        const hasNumericPrice = /\d/.test(rawPrice) && !rawPrice.toLowerCase().includes("contact") && !rawPrice.toLowerCase().includes("unlisted");
                        const priceHtml = hasNumericPrice
                            ? `<span style="color: #6ee7b7; font-weight: 700; font-size: 12px;">${escapeHtml(rawPrice)}</span>`
                            : `<span style="color: #94a3b8; font-weight: 600; font-size: 11px;">${escapeHtml(rawPrice || "Tariff on request")}</span>`;

                        // 2. Property & Management Pill
                        let mgmtBadge = "";
                        if (o.management_type === "TEMPLE_MANAGED") {
                            mgmtBadge = `<span style="background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 4px; padding: 1px 6px; font-size: 10px; font-weight: 600;">🛕 Temple Devasthanam</span>`;
                        } else if (o.management_type === "GOVERNMENT_TOURISM") {
                            mgmtBadge = `<span style="background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 4px; padding: 1px 6px; font-size: 10px; font-weight: 600;">🏛️ Telangana Tourism</span>`;
                        } else if (o.management_type === "COMMUNITY") {
                            mgmtBadge = `<span style="background: rgba(147, 51, 234, 0.15); color: #c084fc; border: 1px solid rgba(147, 51, 234, 0.3); border-radius: 4px; padding: 1px 6px; font-size: 10px; font-weight: 600;">🤝 Community Satram</span>`;
                        } else {
                            const subType = o.property_type || o.type || "Stay";
                            mgmtBadge = `<span style="background: rgba(148, 163, 184, 0.12); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 4px; padding: 1px 6px; font-size: 10px; font-weight: 500;">🏨 ${escapeHtml(subType)}</span>`;
                        }

                        // 3. Contact & Action Buttons
                        let actionButtons = [];
                        if (o.phone) {
                            const telClean = o.phone.replace(/[^0-9+]/g, '');
                            actionButtons.push(`<a href="tel:${escapeHtml(telClean)}" style="display: inline-flex; align-items: center; gap: 4px; background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 4px; padding: 3px 8px; font-size: 11px; text-decoration: none; font-weight: 600;" title="Call ${escapeHtml(o.phone)}">📞 ${escapeHtml(o.phone)}</a>`);
                        }
                        if (o.booking_url) {
                            actionButtons.push(`<a href="${escapeHtml(o.booking_url)}" target="_blank" rel="noopener noreferrer" style="display: inline-flex; align-items: center; gap: 4px; background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 4px; padding: 3px 8px; font-size: 11px; text-decoration: none; font-weight: 600;">🌐 Official Booking</a>`);
                        } else if (o.website_url) {
                            actionButtons.push(`<a href="${escapeHtml(o.website_url)}" target="_blank" rel="noopener noreferrer" style="display: inline-flex; align-items: center; gap: 4px; background: rgba(148, 163, 184, 0.15); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.3); border-radius: 4px; padding: 3px 8px; font-size: 11px; text-decoration: none; font-weight: 500;">🌐 Official Portal</a>`);
                        }
                        if (!o.phone && !o.booking_url && !o.website_url) {
                            if (o.management_type === "TEMPLE_MANAGED") {
                                actionButtons.push(`<span style="display: inline-flex; align-items: center; gap: 4px; background: rgba(245, 158, 11, 0.1); color: #fde68a; border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 4px; padding: 3px 8px; font-size: 11px;">🛕 Contact temple administration</span>`);
                            } else {
                                actionButtons.push(`<span style="display: inline-flex; align-items: center; gap: 4px; color: var(--text-muted); font-size: 11px;">ℹ️ Contact details unavailable</span>`);
                            }
                        }

                        // In-app map action
                        const mapTargetQuery = encodeURIComponent(`${o.name} ${tName || ""}`);
                        const mapUrl = `https://www.google.com/maps/search/?api=1&query=${mapTargetQuery}`;
                        actionButtons.push(`<a href="${mapUrl}" class="in-app-map-action" data-maps-url="${mapUrl}" onclick="handleInAppMapLinkClick(event, this)" style="display: inline-flex; align-items: center; gap: 4px; background: rgba(255, 255, 255, 0.05); color: #e2e8f0; border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 4px; padding: 3px 8px; font-size: 11px; text-decoration: none; cursor: pointer;">🗺️ In-App Map</a>`);

                        // 4. Availability alert
                        const availHtml = o.availability_status
                            ? `<div style="font-size: 11px; color: #f87171; margin-top: 3px; font-weight: 500;">⚠️ Availability: ${escapeHtml(o.availability_status)}</div>`
                            : "";

                        return `
                            <div class="accom-stay-card" style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 6px; padding: 8px 12px; margin-bottom: 6px;">
                                <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 6px;">
                                    <strong style="color: #f1f5f9; font-size: 12.5px;">${escapeHtml(o.name)}</strong>
                                    ${priceHtml}
                                </div>
                                <div style="font-size: 11px; color: var(--text-muted); margin-top: 3px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                                    <span>📍 ${escapeHtml(o.distance)}</span>
                                    ${mgmtBadge}
                                </div>
                                ${availHtml}
                                ${o.description ? `<div style="font-size: 11px; color: #94a3b8; margin-top: 4px; line-height: 1.4;">${escapeHtml(o.description)}</div>` : ""}
                                <div style="display: flex; gap: 6px; flex-wrap: wrap; margin-top: 6px; align-items: center;">
                                    ${actionButtons.join("")}
                                </div>
                            </div>
                        `;
                    }).join("");
                } else {
                    optListHtml = `<div style="font-size: 11.5px; color: var(--text-muted);">Standard budget lodges and private hotels available nearby; confirm tariffs at reception.</div>`;
                }
                templeCards += `
                    <div style="margin-top: 8px;">
                        <div style="font-size: 12px; font-weight: 700; color: #fde68a; margin-bottom: 4px;">🛕 ${escapeHtml(tName)} Stays:</div>
                        ${isBalkampet ? `<div style="font-size: 11px; color: #94a3b8; margin-bottom: 5px; font-style: italic;">(Note: Stays located in adjacent Ameerpet / SR Nagar; no on-site Devasthanam choultry exists at Balkampet shrine).</div>` : ""}
                        ${optListHtml}
                    </div>
                `;
            }
            accomSectionHtml = `
                <div class="budget-confs-box" style="margin-top: 12px; background: rgba(30, 41, 59, 0.4); border-left: 3px solid #38bdf8; padding: 10px 14px; border-radius: 6px;">
                    <strong style="color: #7dd3fc; font-size: 12.5px;">🏨 Accommodation Comparison & Suggestions:</strong>
                    ${templeCards}
                </div>
            `;
        }

        let broaderNoticeHtml = "";
        if (meta.is_cross_limit && meta.broader_range && meta.broader_range.exceeds_budget) {
            broaderNoticeHtml = `
                <div class="budget-confs-box" style="border-left: 3px solid #f59e0b; background: rgba(245, 158, 11, 0.08); margin-top: 12px; padding: 10px 14px; border-radius: 6px;">
                    <strong style="color: #fbbf24; font-size: 12.5px;">⚠️ Cost Range Advisory:</strong>
                    <div style="font-size: 11.5px; color: #fde68a; margin-top: 4px;">
                        The selected baseline plan of ₹${selectedTotal.toLocaleString()} ${(meta.origin_assumed || meta.is_preliminary_origin) ? `leaves an arithmetic remainder of ₹${remaining !== null ? remaining.toLocaleString() : ""} before intercity transport and other unconfirmed costs against your ₹${blim ? blim.toLocaleString() : ""} budget (this remainder is NOT guaranteed spendable surplus).` : `fits within your ₹${blim ? blim.toLocaleString() : ""} budget (Estimated remaining: ₹${remaining !== null ? remaining.toLocaleString() : ""}).`}
                        Higher-tier choices (such as AC rooms or private cabs) reach up to ₹${meta.broader_range.max.toLocaleString()}, exceeding this limit by ₹${meta.broader_range.excess_amount.toLocaleString()}. Choose the baseline options above to stay within budget.
                    </div>
                </div>
            `;
        }

        const confs = meta.confirmation_required || [];
        let confsHtml = "";
        if (confs.length > 0) {
            const confList = confs.map(c => `<li>${escapeHtml(c)}</li>`).join("");
            confsHtml = `
                <div class="budget-confs-box" style="margin-top: 10px; padding: 8px 12px; border-radius: 6px;">
                    <strong style="font-size: 11.5px; color: #fde68a;">ℹ️ Fares & Tariffs Requiring Live Confirmation Before Departure:</strong>
                    <ul style="margin: 4px 0 0 16px; padding: 0; font-size: 11px; color: #cbd5e1;">${confList}</ul>
                </div>
            `;
        }

        return `
            <div class="structured-card budget-card">
                <div class="budget-header-row">
                    <div>
                        <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px;">Your ${days}-Day Pilgrimage Budget</div>
                        <div class="budget-total-val">${totalStr}</div>
                        ${meta.broader_range && meta.broader_range.min !== meta.broader_range.max ? `<div style="font-size: 11px; color: #94a3b8; margin-top: 2px;">Consistent Cost Range: ₹${meta.broader_range.min.toLocaleString()} – ₹${meta.broader_range.max.toLocaleString()}</div>` : ""}
                        ${totalSubtext}
                    </div>
                    <div>
                        <span class="budget-status-pill ${statusClass}">${statusLabel}</span>
                    </div>
                </div>
                <div class="budget-metrics-grid">
                    ${blim ? `
                    <div class="budget-metric-box">
                        <div class="label">Budget Cap</div>
                        <div class="val">🎯 ₹${blim.toLocaleString()}</div>
                    </div>
                    ` : ""}
                    <div class="budget-metric-box">
                        <div class="label">Selected Low-Cost Estimate</div>
                        <div class="val" style="color: #6ee7b7; font-weight: 700;">₹${selectedTotal.toLocaleString()}</div>
                    </div>
                    ${remaining !== null ? `
                    <div class="budget-metric-box">
                        <div class="label">${(meta.origin_assumed || meta.is_preliminary_origin) ? "Arithmetic Remainder (Pre-Transit)" : (remaining >= 0 ? "Estimated Remaining" : "Estimated Shortfall")}</div>
                        <div class="val" style="color: ${remaining >= 0 ? '#34d399' : '#f87171'}; font-weight: 700;">
                            ${(meta.origin_assumed || meta.is_preliminary_origin) ? `₹${remaining.toLocaleString()} (Pending Transit)` : (remaining >= 0 ? `💰 ₹${remaining.toLocaleString()}` : `⚠️ ₹${Math.abs(remaining).toLocaleString()}`)}
                        </div>
                    </div>
                    ` : ""}
                    <div class="budget-metric-box">
                        <div class="label">Travellers</div>
                        <div class="val">👥 ${escapeHtml(paxStr)}</div>
                    </div>
                    <div class="budget-metric-box">
                        <div class="label">Duration</div>
                        <div class="val">📅 ${escapeHtml(durStr)}</div>
                    </div>
                    <div class="budget-metric-box">
                        <div class="label">Destinations</div>
                        <div class="val">🛕 ${escapeHtml(dest)}</div>
                    </div>
                </div>
                ${itemsRows ? `
                <div class="budget-table-wrapper">
                    <table class="budget-table">
                        <thead>
                            <tr>
                                <th>Expense Category</th>
                                <th>Selected Option & Basis</th>
                                <th>Estimated Amount</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${itemsRows}
                        </tbody>
                    </table>
                </div>
                ` : ""}
                ${isItineraryWithBudget ? "" : accomSectionHtml}
                ${broaderNoticeHtml}
                ${confsHtml}
            </div>
        `;
    }

    let latestNavigationMeta = null;

    function renderNavigationCard(meta) {
        latestNavigationMeta = meta;

        // P0: If origin is missing, render interactive starting-point prompt card
        if (meta.missing_origin) {
            const dest = meta.destination || "Temple";
            const formId = "navClarifyForm_" + Math.random().toString(36).substring(2, 7);
            const safeDest = escapeHtml(dest).replace(/'/g, "\\'");
            return `
                <div class="nav-clarification-card">
                    <div class="nav-clarification-header">
                        <span class="nav-clarification-icon">🧭</span>
                        <h4>Where are you starting from?</h4>
                    </div>
                    <p class="nav-clarification-text">Enter your starting location to see the route to <strong>${escapeHtml(dest)}</strong>.</p>
                    <form id="${formId}" class="nav-clarification-form" onsubmit="event.preventDefault(); submitNavOrigin('${formId}', '${safeDest}');">
                        <div class="nav-clarification-inputs">
                            <div class="nav-input-wrapper">
                                <input type="text" id="${formId}_origin" class="nav-clarification-input" placeholder="e.g. Hyderabad, Warangal, Secunderabad Station" autocomplete="off" required>
                                <button type="button" class="btn-clarify-gps" title="Use device location" onclick="useCurrentLocationForNav('${formId}_origin')">📍</button>
                            </div>
                            <select id="${formId}_mode" class="nav-clarification-select">
                                <option value="driving" selected>🚗 Driving</option>
                                <option value="transit">🚌 Transit</option>
                                <option value="two_wheeler">🛵 Two-Wheeler</option>
                                <option value="walking">🚶 Walking</option>
                            </select>
                            <button type="submit" class="btn-clarification-submit">
                                Get Route ➔
                            </button>
                        </div>
                    </form>
                </div>
            `;
        }

        const orig = meta.origin || "Origin";
        const dest = meta.destination || "Destination";
        const origCanon = meta.origin_canonical || "";
        const destCanon = meta.destination_canonical || "";

        const hasRoute = (meta.distance_km !== null && meta.distance_km !== undefined && meta.distance_km > 0) && (meta.polyline && meta.polyline.length > 5);
        const googleMapsDirUrl = `https://www.google.com/maps/dir/?api=1&origin=${encodeURIComponent(orig)}&destination=${encodeURIComponent(dest)}&travelmode=${encodeURIComponent(meta.travel_mode || 'driving')}`;

        if (!hasRoute) {
            return `
                <div class="structured-card nav-card" style="border-left: 3px solid #ef4444;">
                    <div class="nav-route-row">
                        <span>⚠️</span>
                        <span>Route Unavailable: ${escapeHtml(orig)} ➔ ${escapeHtml(dest)}</span>
                    </div>
                    <div style="font-size: 12px; color: var(--text-muted); margin: 6px 0;">
                        Live road route geometry could not be calculated between these locations. You can view directions directly in Google Maps:
                    </div>
                    <div style="margin-top: 6px;">
                        <a href="${googleMapsDirUrl}" target="_blank" rel="noopener noreferrer" class="nav-btn-external-map">
                            ↗️ Open Directions in Google Maps
                        </a>
                    </div>
                </div>
            `;
        }

        let dist = `${meta.distance_km} km`;
        let dur = "Unavailable";
        if (meta.duration_minutes !== null && meta.duration_minutes !== undefined && meta.duration_minutes > 0) {
            dur = formatDurationHoursMinutes(meta.duration_minutes) || "Unavailable";
        }

        let traffic = null;
        if (meta.traffic_duration_minutes !== null && meta.traffic_duration_minutes !== undefined && meta.traffic_duration_minutes > 0) {
            const formattedTraffic = formatDurationHoursMinutes(meta.traffic_duration_minutes);
            if (formattedTraffic) {
                traffic = `${formattedTraffic} (traffic)`;
            }
        }

        const modeIcon = meta.travel_mode === "transit" ? "🚌 Transit" : (meta.travel_mode === "two_wheeler" ? "🛵 Two-Wheeler" : (meta.travel_mode === "walking" ? "🚶 Walking" : "🚗 Driving"));

        const safeOrig = escapeHtml(orig).replace(/'/g, "\\'");
        const safeDest = escapeHtml(dest).replace(/'/g, "\\'");
        const safeOrigCanon = escapeHtml(origCanon).replace(/'/g, "\\'");
        const safeDestCanon = escapeHtml(destCanon).replace(/'/g, "\\'");

        return `
            <div class="structured-card nav-card">
                <div class="nav-route-row">
                    <span>🧭</span>
                    <span>${escapeHtml(orig)}</span>
                    <span style="color: var(--accent-gold);">➔</span>
                    <span>${escapeHtml(dest)}</span>
                    <span class="nav-mode-badge" style="margin-left: 8px; font-size: 11px; padding: 2px 7px; background: rgba(251, 191, 36, 0.15); border-radius: 6px; color: var(--accent-gold);">${modeIcon}</span>
                </div>
                <div class="nav-metrics">
                    <div class="nav-metric-item"><strong>Distance:</strong> ${dist}</div>
                    <div class="nav-metric-item"><strong>Duration:</strong> ${dur}</div>
                    ${traffic ? `<div class="nav-metric-item" style="color: var(--accent-gold);"><strong>Live Traffic:</strong> ${traffic}</div>` : ""}
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px; flex-wrap: wrap; gap: 8px;">
                    <span style="font-size: 10.5px; color: var(--text-muted);">Source: ${escapeHtml(meta.provider || 'Google Maps Routes API')}</span>
                    <div style="display: flex; gap: 6px; align-items: center;">
                        <a href="${googleMapsDirUrl}" target="_blank" rel="noopener noreferrer" class="nav-btn-external-map">
                            ↗️ Google Maps
                        </a>
                        <button class="nav-btn-view-map" 
                                id="viewRouteBtn"
                                data-origin="${escapeHtml(orig)}" 
                                data-destination="${escapeHtml(dest)}" 
                                data-origin-canonical="${escapeHtml(origCanon)}" 
                                data-destination-canonical="${escapeHtml(destCanon)}"
                                data-polyline="${escapeHtml(meta.polyline || '')}"
                                data-origin-coords="${escapeHtml(JSON.stringify(meta.origin_coordinates || []))}"
                                data-dest-coords="${escapeHtml(JSON.stringify(meta.destination_coordinates || []))}"
                                onclick="handleViewRouteClick(this, '${safeOrig}', '${safeDest}', '${safeOrigCanon}', '${safeDestCanon}')">
                            🗺️ View Route on Map
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    window.submitNavOrigin = function(formId, dest) {
        const input = document.getElementById(`${formId}_origin`);
        const modeSelect = document.getElementById(`${formId}_mode`);
        if (!input || !input.value.trim()) return;
        const origin = input.value.trim();
        const mode = modeSelect ? modeSelect.value : "driving";
        sendPrompt(`${origin} to ${dest} ${mode}`);
    };

    window.useCurrentLocationForNav = function(inputId) {
        const input = document.getElementById(inputId);
        if (!input) return;
        if (navigator.geolocation) {
            input.placeholder = "Detecting current location...";
            navigator.geolocation.getCurrentPosition(
                (pos) => {
                    const lat = pos.coords.latitude.toFixed(5);
                    const lng = pos.coords.longitude.toFixed(5);
                    input.value = `${lat}, ${lng}`;
                },
                (err) => {
                    input.placeholder = "Enter starting location";
                    alert("Location permission denied or unavailable. Please enter your starting city or landmark manually.");
                },
                { timeout: 5000 }
            );
        } else {
            alert("Geolocation is not supported by your browser. Please enter your starting city manually.");
        }
    };

    function renderItineraryCard(meta) {
        const temples = meta.temples || [];
        const days = meta.duration_days || 1;
        const templePills = temples.map(t => `<span class="src-chip" style="background: rgba(244, 63, 94, 0.15); color: #fda4af;">${escapeHtml(t)}</span>`).join(" ");

        return `
            <div class="structured-card" style="border-color: rgba(244, 63, 94, 0.3); background: linear-gradient(135deg, rgba(136, 19, 55, 0.2), rgba(15, 23, 42, 0.8));">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-weight: 700; color: #fff; font-size: 13.5px;">📅 ${days} Day Pilgrimage Schedule</span>
                    <span style="font-size: 10.5px; color: var(--accent-gold);">Verified Temples</span>
                </div>
                <div style="display: flex; gap: 6px; flex-wrap: wrap;">
                    ${templePills}
                </div>
            </div>
        `;
    }

    // -------------------------------------------------------------
    // Route Visualization & Location Resolution
    // -------------------------------------------------------------
    const KNOWN_REGIONAL_COORDINATES = {
        "hyderabad": { lat: 17.385044, lng: 78.486671, name: "Hyderabad" },
        "secunderabad": { lat: 17.4399, lng: 78.4983, name: "Secunderabad" },
        "warangal": { lat: 17.9689, lng: 79.5941, name: "Warangal" },
        "karimnagar": { lat: 18.4386, lng: 79.1288, name: "Karimnagar" },
        "nizamabad": { lat: 18.6725, lng: 78.0941, name: "Nizamabad" },
        "khammam": { lat: 17.2473, lng: 80.1514, name: "Khammam" },
        "nalgonda": { lat: 17.0575, lng: 79.2684, name: "Nalgonda" },
        "suryapet": { lat: 17.1439, lng: 79.6239, name: "Suryapet" },
        "mahabubnagar": { lat: 16.7488, lng: 77.9856, name: "Mahabubnagar" },
        "siddipet": { lat: 18.1018, lng: 78.8520, name: "Siddipet" },
        "ramagundam": { lat: 18.7551, lng: 79.5133, name: "Ramagundam" },
        "mancherial": { lat: 18.8679, lng: 79.4639, name: "Mancherial" },
        "vijayawada": { lat: 16.5062, lng: 80.6480, name: "Vijayawada" },
        "guntur": { lat: 16.3067, lng: 80.4365, name: "Guntur" },
        "visakhapatnam": { lat: 17.6868, lng: 83.2185, name: "Visakhapatnam" },
        "tirupati": { lat: 13.6288, lng: 79.4192, name: "Tirupati" },
        "kurnool": { lat: 15.8281, lng: 78.0373, name: "Kurnool" },
        "bengaluru": { lat: 12.9716, lng: 77.5946, name: "Bengaluru" },
        "bangalore": { lat: 12.9716, lng: 77.5946, name: "Bengaluru" },
        "chennai": { lat: 13.0827, lng: 80.2707, name: "Chennai" },
        "mumbai": { lat: 19.0760, lng: 72.8777, name: "Mumbai" },
        "delhi": { lat: 28.6139, lng: 77.2090, name: "Delhi" },
        "pune": { lat: 18.5204, lng: 73.8567, name: "Pune" },
        "palvancha": { lat: 17.5988, lng: 80.7064, name: "Palvancha" },
        "palwancha": { lat: 17.5988, lng: 80.7064, name: "Palvancha" },
        "kothagudem": { lat: 17.5487, lng: 80.6178, name: "Kothagudem" },
        "bhadrachalam": { lat: 17.6688, lng: 80.8936, name: "Bhadrachalam" },
        "kaleshwaram": { lat: 18.8122, lng: 79.9075, name: "Kaleshwaram" },
        "medaram": { lat: 18.2368, lng: 80.2858, name: "Medaram" },
        "basara": { lat: 18.9754, lng: 77.9542, name: "Basara" },
        "vemulawada": { lat: 18.4682, lng: 78.8687, name: "Vemulawada" },
        "yadagirigutta": { lat: 17.5886, lng: 78.9392, name: "Yadagirigutta" },
        "srisailam": { lat: 16.0745, lng: 78.8682, name: "Srisailam" },
        "alampur": { lat: 15.8797, lng: 78.1325, name: "Alampur" },
        "dharmapuri": { lat: 18.9486, lng: 79.0911, name: "Dharmapuri" }
    };

    const CANONICAL_TEMPLE_ALIASES = {
        "vemulawada": "Vemulavada",
        "vemulavada": "Vemulavada",
        "rajarajeshwara": "Vemulavada",
        "rajanna": "Vemulavada",
        "yadagiri": "Yadadri",
        "yadagirigutta": "Yadadri",
        "yadadri": "Yadadri",
        "basar": "Basara",
        "basara": "Basara",
        "gnana saraswathi": "Basara",
        "dharmapuri": "Dharmapuri",
        "bhadrachalam": "Badrachalam",
        "badrachalam": "Badrachalam",
        "srisailam": "Srisaila",
        "srisaila": "Srisaila"
    };

    function isLikelyTempleName(clean, canonicalHint) {
        if (!clean && !canonicalHint) return false;
        const c = String(clean || "").toLowerCase();
        const canon = String(canonicalHint || "").toLowerCase();

        // Explicit non-temple keywords
        const nonTempleWords = [
            "railway station", "railway", "station", "bus stand", "bus stop",
            "airport", "hotel", "resort", "restaurant", "cafe", "bhavan", "lodge",
            "hospital", "college", "school", "office", "street", "road", "colony",
            "nagar", "circle", "junction", "cross road", "crossroad", "bypass", "highway"
        ];
        if (nonTempleWords.some(w => c.includes(w))) return false;

        // Check if it's an exact known city/town
        if (KNOWN_REGIONAL_COORDINATES[c]) return false;

        // Explicit temple keywords
        const templeWords = [
            "temple", "mandir", "devasthanam", "gudi", "kovil", "shrine",
            "dargah", "devalayam", "sannidhi", "sanctuary", "jyotirlinga", "shakti peeth"
        ];
        if (templeWords.some(w => c.includes(w) || canon.includes(w))) return true;

        // Specific deity / distinctive temple identifiers
        const deityWords = [
            "saraswathi", "gnana saraswathi", "yellamma", "balkampet",
            "chilkur balaji", "chilkur", "jogulamba", "mallikarjuna",
            "bhramaramba", "rajarajeshwara", "rajanna", "mukteshwara",
            "kaleshwara", "sammakka", "saralamma", "ramalingeswara",
            "thousand pillar", "rudreswara", "veera satyanarayana",
            "annavaram", "surendrapuri", "karmanghat", "sanghi", "birla mandir", "ahobilam"
        ];
        if (deityWords.some(w => c.includes(w) || canon.includes(w))) return true;

        return false;
    }

    function resolveLocationForMap(locationStr, canonicalHint) {
        if (!locationStr && !canonicalHint) return null;
        if (typeof locationStr === "object" && locationStr.lat && locationStr.lng) {
            return locationStr;
        }

        const raw = String(locationStr || "").trim();
        const clean = raw.toLowerCase().replace(/^(the|a|an)\s+/i, "").replace(/[?!.,;]+$/, "").trim();
        const canonClean = String(canonicalHint || "").toLowerCase().trim();

        // 1. Check known regional cities (Priority 1 if input matches regional city and is not explicitly a temple)
        const isTempleIntent = isLikelyTempleName(clean, canonClean);
        if (!isTempleIntent) {
            if (KNOWN_REGIONAL_COORDINATES[clean]) {
                const city = KNOWN_REGIONAL_COORDINATES[clean];
                return { lat: city.lat, lng: city.lng, name: city.name, isTemple: false };
            }
        }

        // 2. Check canonicalHint in verified templeCoordinates (Priority 2)
        if (canonClean && Array.isArray(templeCoordinates) && templeCoordinates.length > 0) {
            const foundCanon = templeCoordinates.find(t => t.canonical_name && t.canonical_name.toLowerCase() === canonClean);
            if (foundCanon && isTempleIntent) {
                return { lat: foundCanon.latitude, lng: foundCanon.longitude, name: foundCanon.display_name, isTemple: true };
            }
        }

        // 3. Check alias map ONLY when temple intent is detected
        if (isTempleIntent) {
            for (const [alias, canonicalTarget] of Object.entries(CANONICAL_TEMPLE_ALIASES)) {
                if (clean === alias || clean.includes(alias)) {
                    const found = templeCoordinates.find(t => t.canonical_name && t.canonical_name.toLowerCase() === canonicalTarget.toLowerCase());
                    if (found) {
                        return { lat: found.latitude, lng: found.longitude, name: found.display_name, isTemple: true };
                    }
                }
            }

            // 4. Check clean in verified templeCoordinates
            if (Array.isArray(templeCoordinates) && templeCoordinates.length > 0) {
                // A. Exact canonical_name
                let found = templeCoordinates.find(t => t.canonical_name && t.canonical_name.toLowerCase() === clean);
                if (found) return { lat: found.latitude, lng: found.longitude, name: found.display_name, isTemple: true };

                // B. Exact display_name
                found = templeCoordinates.find(t => t.display_name && t.display_name.toLowerCase() === clean);
                if (found) return { lat: found.latitude, lng: found.longitude, name: found.display_name, isTemple: true };

                // C. Containment match: User input CONTAINS canonical name (e.g. "visit srisailam today")
                // NEVER check if temple display name contains user input (e.g. "Balkampet Yellamma Temple, Hyderabad" contains "hyderabad")
                found = templeCoordinates.find(t => {
                    const c = (t.canonical_name || "").toLowerCase();
                    return c && c.length > 3 && clean.includes(c);
                });
                if (found) return { lat: found.latitude, lng: found.longitude, name: found.display_name, isTemple: true };
            }
        }

        // 5. Fallback check for regional cities (if not matched yet)
        if (KNOWN_REGIONAL_COORDINATES[clean]) {
            const city = KNOWN_REGIONAL_COORDINATES[clean];
            return { lat: city.lat, lng: city.lng, name: city.name, isTemple: false };
        }

        // Return null so geocodeLocationAsync uses Google PlacesService for arbitrary places
        return null;
    }

    async function geocodeLocationAsync(locationStr, canonicalHint) {
        if (!locationStr && !canonicalHint) return null;
        if (typeof locationStr === "object" && locationStr.lat && locationStr.lng) {
            return locationStr;
        }

        // 1. Try local verified coordinates first (temples, regional cities)
        const localResolved = resolveLocationForMap(locationStr, canonicalHint);
        if (localResolved && typeof localResolved === "object" && localResolved.lat !== undefined) {
            return localResolved;
        }

        const raw = String(locationStr || "").trim();
        if (!raw) return null;

        // 2. Use browser PlacesService for arbitrary valid destinations (railway stations, hotels, restaurants, landmarks, addresses)
        if (window.google && google.maps && google.maps.places && google.maps.places.PlacesService) {
            try {
                const queryStr = raw.toLowerCase().includes("telangana") || raw.toLowerCase().includes("india")
                    ? raw
                    : `${raw}, Telangana, India`;

                const dummy = document.createElement("div");
                const svc = new google.maps.places.PlacesService(dummy);

                const placesResult = await new Promise((resolve) => {
                    const timer = setTimeout(() => resolve(null), 3500);
                    svc.textSearch({ query: queryStr }, (results, status) => {
                        clearTimeout(timer);
                        if (status === "OK" && results && results.length > 0) {
                            const top = results[0];
                            const loc = top.geometry && top.geometry.location;
                            if (loc) {
                                resolve({
                                    lat: typeof loc.lat === "function" ? loc.lat() : Number(loc.lat),
                                    lng: typeof loc.lng === "function" ? loc.lng() : Number(loc.lng),
                                    name: top.name || raw,
                                    address: top.formatted_address || queryStr,
                                    isTemple: isLikelyTempleName(top.name || raw)
                                });
                                return;
                            }
                        }
                        resolve(null);
                    });
                });

                if (placesResult) return placesResult;
            } catch (placesErr) {
                console.warn("[PilgrimAI] PlacesService textSearch error:", placesErr);
            }
        }

        return null;
    }

    function decodePolyline(encoded) {
        if (!encoded) return [];
        let points = [];
        let index = 0, len = encoded.length;
        let lat = 0, lng = 0;
        while (index < len) {
            let b, shift = 0, result = 0;
            do {
                b = encoded.charCodeAt(index++) - 63;
                result |= (b & 0x1f) << shift;
                shift += 5;
            } while (b >= 0x20);
            let dlat = ((result & 1) ? ~(result >> 1) : (result >> 1));
            lat += dlat;
            shift = 0;
            result = 0;
            do {
                b = encoded.charCodeAt(index++) - 63;
                result |= (b & 0x1f) << shift;
                shift += 5;
            } while (b >= 0x20);
            let dlng = ((result & 1) ? ~(result >> 1) : (result >> 1));
            lng += dlng;
            points.push({ lat: lat / 1e5, lng: lng / 1e5 });
        }
        return points;
    }

    function clearActiveRoute() {
        try {
            if (directionsRenderer) {
                directionsRenderer.setMap(null);
                directionsRenderer = null;
            }
        } catch (e) {
            console.warn("[PilgrimAI Routes Diagnostics] clearActiveRoute error:", e);
        }
        try {
            if (activeRoutePolyline) {
                activeRoutePolyline.setMap(null);
                activeRoutePolyline = null;
            }
        } catch (e) {}
        try {
            if (Array.isArray(activeRouteMarkers)) {
                activeRouteMarkers.forEach(m => {
                    try { m.setMap(null); } catch (e) {}
                });
                activeRouteMarkers = [];
            }
        } catch (e) {}

        const mapRouteStrip = document.getElementById("mapRouteStrip");
        if (mapRouteStrip) mapRouteStrip.style.display = "none";

        activeNavigationState = {
            origin: "",
            destination: "",
            travelMode: "driving",
            distance_km: null,
            duration_minutes: null,
            polyline: null,
            origin_coordinates: null,
            destination_coordinates: null
        };
        window.activeRoutePolyline = null;
        window.activeRouteMarkers = [];
        window.activeNavigationState = null;
    }

    function formatRoutesWaypoint(resolved) {
        if (!resolved) return null;
        if (typeof resolved === "object" && resolved.lat !== undefined && resolved.lng !== undefined) {
            return {
                location: {
                    lat: Number(resolved.lat),
                    lng: Number(resolved.lng)
                }
            };
        }
        if (typeof resolved === "string" && resolved.trim()) {
            return {
                address: resolved.trim()
            };
        }
        if (typeof resolved === "object" && resolved.name) {
            return {
                address: String(resolved.name).trim()
            };
        }
        return null;
    }

    async function tryModernRoutesCompute(origResolved, destResolved) {
        if (!window.google || !google.maps || typeof google.maps.importLibrary !== "function") {
            return null;
        }
        try {
            const routePromise = (async () => {
                try {
                    const { Route } = await google.maps.importLibrary("routes");
                    if (!Route || typeof Route.computeRoutes !== "function") return null;

                    const originWaypoint = formatRoutesWaypoint(origResolved);
                    const destWaypoint = formatRoutesWaypoint(destResolved);

                    if (!originWaypoint || !destWaypoint) return null;

                    const request = {
                        origin: originWaypoint,
                        destination: destWaypoint,
                        travelMode: "DRIVING",
                        fields: [
                            "routes.distanceMeters",
                            "routes.duration",
                            "routes.polyline.encodedPolyline",
                            "routes.viewport"
                        ]
                    };

                    const res = await Route.computeRoutes(request);

                    if (res && res.routes && res.routes.length > 0) {
                        const primary = res.routes[0];
                        const encoded = primary.polyline && primary.polyline.encodedPolyline;
                        if (encoded) {
                            const path = decodePolyline(encoded);
                            if (path && path.length > 0) {
                                return {
                                    path,
                                    viewport: primary.viewport,
                                    distanceMeters: primary.distanceMeters,
                                    duration: primary.duration
                                };
                            }
                        }
                    }
                    return null;
                } catch (innerErr) {
                    console.warn("[PilgrimAI Routes Diagnostics] Modern Route.computeRoutes error:", innerErr.message || innerErr);
                    return null;
                }
            })();

            return await Promise.race([
                routePromise,
                new Promise(resolve => setTimeout(() => resolve(null), 8000))
            ]);
        } catch (e) {
            console.warn("[PilgrimAI Routes Diagnostics] Modern Route.computeRoutes failed:", e.message || e);
        }
        return null;
    }

    function renderModernPolylineRoute(routeData, origResolved, destResolved) {
        clearActiveRoute();

        activeRoutePolyline = new google.maps.Polyline({
            path: routeData.path,
            geodesic: true,
            strokeColor: "#fbbf24",
            strokeOpacity: 0.85,
            strokeWeight: 6,
            map: googleMap
        });
        window.activeRoutePolyline = activeRoutePolyline;

        // Fit bounds across all points
        const bounds = new google.maps.LatLngBounds();
        routeData.path.forEach(pt => bounds.extend(pt));
        googleMap.fitBounds(bounds);

        activeRouteMarkers = [];

        // Origin and Destination Markers
        const startPos = (origResolved && typeof origResolved === "object" && origResolved.lat !== undefined)
            ? { lat: Number(origResolved.lat), lng: Number(origResolved.lng) }
            : (routeData.path && routeData.path.length > 0 ? routeData.path[0] : null);

        const endPos = (destResolved && typeof destResolved === "object" && destResolved.lat !== undefined)
            ? { lat: Number(destResolved.lat), lng: Number(destResolved.lng) }
            : (routeData.path && routeData.path.length > 0 ? routeData.path[routeData.path.length - 1] : null);

        const origLabel = (origResolved && origResolved.name)
            || (typeof origResolved === "string" ? origResolved.split(",")[0].trim() : null)
            || (latestNavigationMeta && latestNavigationMeta.origin)
            || (activeNavigationState && activeNavigationState.origin)
            || "Origin";

        const destLabel = (destResolved && destResolved.name)
            || (typeof destResolved === "string" ? destResolved.split(",")[0].trim() : null)
            || (latestNavigationMeta && latestNavigationMeta.destination)
            || (activeNavigationState && activeNavigationState.destination)
            || "Destination";

        if (startPos) {
            const startMarker = new google.maps.Marker({
                position: startPos,
                map: googleMap,
                title: origLabel,
                label: { text: "A", color: "#fff", fontWeight: "bold" },
                icon: {
                    path: google.maps.SymbolPath.CIRCLE,
                    scale: 8,
                    fillColor: "#3b82f6",
                    fillOpacity: 1,
                    strokeColor: "#ffffff",
                    strokeWeight: 2
                }
            });
            const origInfo = new google.maps.InfoWindow({
                content: `<div style="color:#0f172a;padding:4px;font-size:12px;"><strong>Starting Point (A):</strong><br>${escapeHtml(origLabel)}</div>`
            });
            startMarker.addListener("click", () => origInfo.open(googleMap, startMarker));
            activeRouteMarkers.push(startMarker);
        }

        if (endPos) {
            const endMarker = new google.maps.Marker({
                position: endPos,
                map: googleMap,
                title: destLabel,
                label: { text: "B", color: "#fff", fontWeight: "bold" },
                icon: {
                    path: google.maps.SymbolPath.CIRCLE,
                    scale: 8,
                    fillColor: "#10b981",
                    fillOpacity: 1,
                    strokeColor: "#ffffff",
                    strokeWeight: 2
                }
            });
            const destInfo = new google.maps.InfoWindow({
                content: `<div style="color:#0f172a;padding:4px;font-size:12px;"><strong>Destination (B):</strong><br>${escapeHtml(destLabel)}</div>`
            });
            endMarker.addListener("click", () => destInfo.open(googleMap, endMarker));
            activeRouteMarkers.push(endMarker);
        }
        window.activeRouteMarkers = activeRouteMarkers;

        // Update active route strip
        const mapRouteStrip = document.getElementById("mapRouteStrip");
        const mapRouteSummaryText = document.getElementById("mapRouteSummaryText");
        if (mapRouteStrip && mapRouteSummaryText) {
            mapRouteStrip.style.display = "flex";
            let metrics = "";
            if (latestNavigationMeta && latestNavigationMeta.distance_km) {
                metrics += ` • ${latestNavigationMeta.distance_km} km`;
            }
            if (latestNavigationMeta && latestNavigationMeta.duration_minutes) {
                const formattedDur = formatDurationHoursMinutes(latestNavigationMeta.duration_minutes);
                if (formattedDur) {
                    metrics += ` (${formattedDur})`;
                }
            }
            mapRouteSummaryText.innerHTML = `<strong>${escapeHtml(origLabel)} ➔ ${escapeHtml(destLabel)}</strong>${metrics}`;
        }

        // Update route planner inputs
        const mapOriginInput = document.getElementById("mapOriginInput");
        const mapDestInput = document.getElementById("mapDestInput");
        if (mapOriginInput && origResolved && origResolved.name) {
            const manualOrigin = window.getUserManualOrigin ? window.getUserManualOrigin() : null;
            if (!manualOrigin || !mapOriginInput.value.trim()) {
                mapOriginInput.value = origResolved.name;
            }
        }
        if (mapDestInput && destResolved && destResolved.name) {
            const manualDest = window.getUserManualDestination ? window.getUserManualDestination() : null;
            if (!manualDest || !mapDestInput.value.trim()) {
                mapDestInput.value = destResolved.name;
            }
        }

        highlightDestinationTemple(destResolved);
        return true;
    }

    function highlightDestinationTemple(destResolved) {
        if (!destResolved || !destResolved.isTemple) return;
        const match = templeCoordinates.find(t => 
            (destResolved.name && t.display_name === destResolved.name) ||
            (destResolved.lat && Math.abs(t.latitude - destResolved.lat) < 0.001 && Math.abs(t.longitude - destResolved.lng) < 0.001)
        );
        if (match) {
            currentSelectedTemple = match.display_name;
            currentLat = match.latitude;
            currentLng = match.longitude;
        }
    }

    // -------------------------------------------------------------
    // Map & Route Side-Effects
    // -------------------------------------------------------------
    function handleMapSideEffects(data) {
        const meta = data.metadata || {};

        // 1. Navigation query: highlight route or handle missing origin
        if (data.agent_name === "live_navigation" || meta.distance_km !== undefined || meta.missing_origin) {
            if (meta.missing_origin) {
                const mapDestInput = document.getElementById("mapDestInput");
                const mapOriginInput = document.getElementById("mapOriginInput");
                if (mapDestInput && meta.destination) {
                    mapDestInput.value = meta.destination;
                }
                if (mapOriginInput) {
                    mapOriginInput.placeholder = "Enter starting location to calculate route";
                }
            } else if (meta.origin && meta.destination) {
                const mapOriginInput = document.getElementById("mapOriginInput");
                const mapDestInput = document.getElementById("mapDestInput");
                const mapModeSelect = document.getElementById("mapModeSelect");
                const manualOrigin = window.getUserManualOrigin ? window.getUserManualOrigin() : null;
                const manualDest = window.getUserManualDestination ? window.getUserManualDestination() : null;
                if (mapOriginInput && (!manualOrigin || !mapOriginInput.value.trim())) mapOriginInput.value = meta.origin;
                if (mapDestInput && (!manualDest || !mapDestInput.value.trim())) mapDestInput.value = meta.destination;
                if (mapModeSelect && meta.travel_mode) mapModeSelect.value = meta.travel_mode;

                activeNavigationState = {
                    origin: meta.origin,
                    destination: meta.destination,
                    travelMode: meta.travel_mode || "driving",
                    distance_km: meta.distance_km || null,
                    duration_minutes: meta.duration_minutes || null,
                    polyline: meta.polyline || null,
                    origin_coordinates: meta.origin_coordinates || null,
                    destination_coordinates: meta.destination_coordinates || null
                };

                setTimeout(() => {
                    const btn = document.getElementById("viewRouteBtn");
                    focusRouteOnMap(
                        meta.origin,
                        meta.destination,
                        btn,
                        meta.origin_canonical,
                        meta.destination_canonical,
                        meta.polyline,
                        meta.origin_coordinates,
                        meta.destination_coordinates
                    );
                }, 100);
            }
        }

        // 2. Weather query: pan to temple
        if (data.agent_name === "live_weather" && meta.latitude && meta.longitude) {
            currentSelectedTemple = meta.display_name || meta.location || currentSelectedTemple;
            currentLat = parseFloat(meta.latitude);
            currentLng = parseFloat(meta.longitude);
            panToCoordinates(meta.latitude, meta.longitude, meta.display_name || meta.location);
            fetchNearbyPlaces(meta.display_name || meta.location, meta.latitude, meta.longitude);
        }

        // 3. Temple Information query / Itinerary / Budget: focus on temple pin and set context
        let detectedTempleName = meta.detected_temple;
        if (!detectedTempleName && meta.temple) {
            detectedTempleName = meta.temple;
        } else if (!detectedTempleName && Array.isArray(meta.temples) && meta.temples.length > 0) {
            detectedTempleName = meta.temples[0];
        } else if (!detectedTempleName && meta.destination_canonical) {
            detectedTempleName = meta.destination_canonical;
        }

        if (detectedTempleName) {
            const cleanTarget = detectedTempleName.toLowerCase();
            let found = templeCoordinates.find(t => 
                t.canonical_name.toLowerCase() === cleanTarget ||
                t.display_name.toLowerCase().includes(cleanTarget)
            );
            if (!found) {
                for (const [alias, canon] of Object.entries(CANONICAL_TEMPLE_ALIASES)) {
                    if (cleanTarget.includes(alias)) {
                        found = templeCoordinates.find(t => t.canonical_name.toLowerCase() === canon.toLowerCase());
                        if (found) break;
                    }
                }
            }
            if (found) {
                currentSelectedTemple = found.display_name;
                currentLat = found.latitude;
                currentLng = found.longitude;
                if (data.agent_name !== "live_navigation") {
                    panToCoordinates(found.latitude, found.longitude, found.display_name);
                }
                fetchNearbyPlaces(found.display_name, found.latitude, found.longitude);
            }
        }
    }

    window.getMapInstance = () => googleMap;
    window.getActivePolyline = () => activeRoutePolyline;
    window.getActiveMarkers = () => activeRouteMarkers;
    window.getActiveNavigationState = () => activeNavigationState;

    window.focusRouteOnMap = async function(origin, destination, btn, originCanonical, destCanonical, polyline, origCoords, destCoords) {
        if (!origin || !destination) {
            console.warn("[PilgrimAI Routes Diagnostics] focusRouteOnMap called without origin or destination", { origin, destination });
            return false;
        }

        // Wait up to 1500ms if googleMap instance is still initializing
        if (!googleMap) {
            for (let i = 0; i < 15; i++) {
                await new Promise(r => setTimeout(r, 100));
                if (googleMap) break;
            }
        }

        // Failure 1: Map unavailable
        if (!googleMap) {
            console.warn("[PilgrimAI Routes Diagnostics] [Failure 1: Map unavailable] Google Map instance not initialized yet.");
            return false;
        }

        // 1. Ensure map panel is open and visible
        if (mapPanel && mapPanel.classList.contains("collapsed")) {
            mapPanel.classList.remove("collapsed");
        }
        if (tabMapBtn && !tabMapBtn.classList.contains("active")) {
            tabMapBtn.classList.add("active");
            if (tabPlacesBtn) tabPlacesBtn.classList.remove("active");
        }
        google.maps.event.trigger(googleMap, "resize");

        // 2. Clear any previously displayed route (single active route visualization)
        clearActiveRoute();

        // 3. Resolve origin and destination
        let origResolved = null;
        if (origCoords && Array.isArray(origCoords) && origCoords.length >= 2) {
            origResolved = { lat: Number(origCoords[0]), lng: Number(origCoords[1]), name: origin, isTemple: false };
        } else {
            origResolved = resolveLocationForMap(origin, originCanonical);
        }

        let destResolved = null;
        if (destCoords && Array.isArray(destCoords) && destCoords.length >= 2) {
            const isTempleCheck = isLikelyTempleName(destination, destCanonical);
            destResolved = { lat: Number(destCoords[0]), lng: Number(destCoords[1]), name: destination, isTemple: isTempleCheck };
        } else {
            destResolved = resolveLocationForMap(destination, destCanonical);
        }

        // Failure 2: Origin coordinates unavailable
        if (!origResolved) {
            console.warn("[PilgrimAI Routes Diagnostics] [Failure 2: Origin coordinates unavailable] Unable to resolve coordinates for origin: " + origin);
            return false;
        }

        // Failure 3: Destination coordinates unavailable
        if (!destResolved) {
            console.warn("[PilgrimAI Routes Diagnostics] [Failure 3: Destination coordinates unavailable] Unable to resolve coordinates for destination: " + destination);
            return false;
        }

        // 4. Primary: Directly decode and render polyline from backend Routes API response
        const activePolyline = polyline || (latestNavigationMeta && latestNavigationMeta.polyline);
        if (activePolyline && typeof activePolyline === "string" && activePolyline.length > 5) {
            try {
                const path = decodePolyline(activePolyline);
                if (path && path.length > 0) {
                    const rendered = renderModernPolylineRoute({ path }, origResolved, destResolved);
                    if (rendered) {
                        activeNavigationState = {
                            origin: origin,
                            destination: destination,
                            travelMode: (latestNavigationMeta && latestNavigationMeta.travel_mode) || "driving",
                            distance_km: (latestNavigationMeta && latestNavigationMeta.distance_km) || null,
                            duration_minutes: (latestNavigationMeta && latestNavigationMeta.duration_minutes) || null,
                            polyline: activePolyline,
                            origin_coordinates: origCoords,
                            destination_coordinates: destCoords
                        };
                        if (btn) {
                            btn.classList.add("success");
                            btn.innerText = "✓ Route shown on map";
                            setTimeout(() => {
                                btn.classList.remove("success");
                                btn.innerText = "🗺️ View Route on Map";
                            }, 4000);
                        }
                        return true;
                    }
                }
            } catch (polyErr) {
                console.warn("[PilgrimAI Routes Diagnostics] Failed to decode backend polyline:", polyErr);
            }
        }

        // 5. Try modern Route.computeRoutes first if available (strict 300ms timeout)
        const modernRoute = await tryModernRoutesCompute(origResolved, destResolved);
        if (modernRoute) {
            // Failure 6: Route geometry unavailable check
            if (!modernRoute.path || modernRoute.path.length === 0) {
                console.warn("[PilgrimAI Routes Diagnostics] [Failure 6: Route geometry unavailable] Modern routes returned no polyline points.");
            } else {
                const rendered = renderModernPolylineRoute(modernRoute, origResolved, destResolved);
                if (rendered) {
                    if (btn) {
                        btn.classList.add("success");
                        btn.innerText = "✓ Route shown on map";
                        setTimeout(() => {
                            btn.classList.remove("success");
                            btn.innerText = "🗺️ View Route on Map";
                        }, 4000);
                    }
                    return true;
                }
            }
        }

        // 6. DirectionsService & DirectionsRenderer fallback
        return new Promise((resolve) => {
            try {
                // Failure 4: Routes JS API unavailable
                if (!window.google || !google.maps) {
                    console.warn("[PilgrimAI Routes Diagnostics] [Failure 4: Routes JS API unavailable] Google Maps JavaScript API is not loaded.");
                    resolve(false);
                    return;
                }

                if (!directionsService) {
                    directionsService = new google.maps.DirectionsService();
                }
                if (!directionsRenderer) {
                    directionsRenderer = new google.maps.DirectionsRenderer({
                        map: googleMap,
                        suppressMarkers: false,
                        polylineOptions: {
                            strokeColor: "#fbbf24",
                            strokeWeight: 6,
                            strokeOpacity: 0.9
                        }
                    });
                }
                directionsRenderer.setMap(googleMap);

                const origWaypoint = (typeof origResolved === "object" && origResolved.lat && origResolved.lng)
                    ? new google.maps.LatLng(origResolved.lat, origResolved.lng)
                    : origResolved;
                const destWaypoint = (typeof destResolved === "object" && destResolved.lat && destResolved.lng)
                    ? new google.maps.LatLng(destResolved.lat, destResolved.lng)
                    : destResolved;

                directionsService.route({
                    origin: origWaypoint,
                    destination: destWaypoint,
                    travelMode: google.maps.TravelMode.DRIVING
                }, (result, status) => {
                    if (status === google.maps.DirectionsStatus.OK && result) {
                        // Failure 6 check: geometry check
                        if (!result.routes || result.routes.length === 0 || !result.routes[0].overview_path || result.routes[0].overview_path.length === 0) {
                            console.warn("[PilgrimAI Routes Diagnostics] [Failure 6: Route geometry unavailable] DirectionsService returned OK but empty overview_path.");
                            resolve(false);
                            return;
                        }

                        directionsRenderer.setDirections(result);

                        // Fit bounds to entire route
                        if (result.routes[0].bounds) {
                            googleMap.fitBounds(result.routes[0].bounds);
                        }

                        // Keep destination temple marker visible
                        highlightDestinationTemple(destResolved);

                        if (btn) {
                            btn.classList.add("success");
                            btn.innerText = "✓ Route shown on map";
                            setTimeout(() => {
                                btn.classList.remove("success");
                                btn.innerText = "🗺️ View Route on Map";
                            }, 4000);
                        }
                        resolve(true);
                    } else {
                        // Failure 5: Route calculation rejected (e.g. REQUEST_DENIED due to key restrictions)
                        console.warn("[PilgrimAI Routes Diagnostics] [Failure 5: Route calculation rejected] DirectionsService status: " + status + ". (Browser key lacks Directions API / Routes API capability in Google Cloud API restrictions)");
                        resolve(false);
                    }
                });
            } catch (err) {
                // Failure 7: Route rendering failed
                console.error("[PilgrimAI Routes Diagnostics] [Failure 7: Route rendering failed] DirectionsService execution error: " + (err.message || err));
                resolve(false);
            }
        });
    };

    window.handleViewRouteClick = async function(btn, orig, dest, origCanon, destCanon) {
        const meta = latestNavigationMeta || {};
        const mapOriginInput = document.getElementById("mapOriginInput");
        const currentManualOrigin = (window.getUserManualOrigin && window.getUserManualOrigin()) ||
            (mapOriginInput && mapOriginInput.value ? mapOriginInput.value.trim() : "");

        if (!dest) {
            dest = (btn && btn.dataset && btn.dataset.destination) || meta.destination;
            destCanon = destCanon || (btn && btn.dataset && btn.dataset.destinationCanonical) || meta.destination_canonical;
        }

        // If user has a manually entered origin in the route planner, preserve it!
        if (currentManualOrigin) {
            orig = currentManualOrigin;
            origCanon = null;
        } else if (!orig) {
            orig = (btn && btn.dataset && btn.dataset.origin) || meta.origin;
            origCanon = origCanon || (btn && btn.dataset && btn.dataset.originCanonical) || meta.origin_canonical;
        }

        let polyline = (btn && btn.dataset && btn.dataset.polyline) || (meta && meta.polyline) || null;
        let origCoords = (meta && meta.origin_coordinates) || null;
        let destCoords = (meta && meta.destination_coordinates) || null;
        if (btn && btn.dataset && btn.dataset.originCoords) {
            try { const p = JSON.parse(btn.dataset.originCoords); if (Array.isArray(p) && p.length >= 2) origCoords = p; } catch (e) {}
        }
        if (btn && btn.dataset && btn.dataset.destCoords) {
            try { const p = JSON.parse(btn.dataset.destCoords); if (Array.isArray(p) && p.length >= 2) destCoords = p; } catch (e) {}
        }

        if (!orig || !dest) {
            if (btn) btn.innerText = "Route coordinates missing";
            return;
        }

        const originalText = "🗺️ View Route on Map";

        // If user manually entered an origin and it differs from the card's origin,
        // calculate fresh route directly using the resolved manual origin!
        const btnOrigin = (btn && btn.dataset && btn.dataset.origin) || "";
        if (currentManualOrigin && btnOrigin && currentManualOrigin.toLowerCase() !== btnOrigin.toLowerCase()) {
            if (btn) {
                btn.disabled = true;
                btn.classList.add("loading");
                btn.innerText = "Loading route...";
            }
            try {
                const mode = (latestNavigationMeta && latestNavigationMeta.travel_mode) || "driving";
                const success = await calculateDirectRoute(currentManualOrigin, dest, mode);
                if (btn) {
                    btn.disabled = false;
                    btn.classList.remove("loading");
                    if (success) {
                        btn.classList.add("success");
                        btn.innerText = "✓ Route shown on map";
                        setTimeout(() => {
                            btn.classList.remove("success");
                            btn.innerText = originalText;
                        }, 4000);
                    } else {
                        btn.classList.add("error");
                        btn.innerText = "Unable to display this route on the map.";
                        setTimeout(() => {
                            btn.classList.remove("error");
                            btn.innerText = originalText;
                        }, 4000);
                    }
                }
            } catch (err) {
                console.warn("[PilgrimAI] handleViewRouteClick recalculation error:", err);
                if (btn) {
                    btn.disabled = false;
                    btn.classList.remove("loading");
                    btn.classList.add("error");
                    btn.innerText = "Unable to display this route on the map.";
                    setTimeout(() => {
                        btn.classList.remove("error");
                        btn.innerText = originalText;
                    }, 4000);
                }
            }
            return;
        }

        if (btn) {
            btn.disabled = true;
            btn.classList.add("loading");
            btn.innerText = "Loading route...";
        }

        try {
            const success = await focusRouteOnMap(orig, dest, btn, origCanon, destCanon, polyline, origCoords, destCoords);
            if (btn) {
                btn.disabled = false;
                btn.classList.remove("loading");
                if (success) {
                    btn.classList.add("success");
                    btn.innerText = "✓ Route shown on map";
                    setTimeout(() => {
                        btn.classList.remove("success");
                        btn.innerText = originalText;
                    }, 4000);
                } else {
                    btn.classList.add("error");
                    btn.innerText = "Unable to display this route on the map.";
                    setTimeout(() => {
                        btn.classList.remove("error");
                        btn.innerText = originalText;
                    }, 4000);
                }
            }
        } catch (err) {
            console.error("[PilgrimAI Routes Diagnostics] handleViewRouteClick caught error:", err);
            if (btn) {
                btn.disabled = false;
                btn.classList.remove("loading");
                btn.classList.add("error");
                btn.innerText = "Unable to display this route on the map.";
                setTimeout(() => {
                    btn.classList.remove("error");
                    btn.innerText = originalText;
                }, 4000);
            }
        }
    };

    function panToCoordinates(lat, lng, title) {
        if (!googleMap) return;
        const pos = { lat: parseFloat(lat), lng: parseFloat(lng) };
        googleMap.panTo(pos);
        googleMap.setZoom(13);
    }

    // -------------------------------------------------------------
    // Google Maps JavaScript API Loader
    // -------------------------------------------------------------
    async function initGoogleMaps() {
        try {
            const res = await fetch("/api/maps/config");
            if (!res.ok) {
                showMapFallback();
                return;
            }
            const config = await res.json();
            if (!config.enabled || !config.api_key) {
                showMapFallback();
                return;
            }

            // Expose global callback
            window.initGoogleMap = onGoogleMapsLoaded;

            // Handle potential auth failure gracefully
            window.gm_authFailure = function() {
                console.warn("Google Maps JavaScript API authentication failed. Check referrer/API restrictions.");
                showMapFallback();
            };

            const script = document.createElement("script");
            script.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(config.api_key)}&libraries=places&callback=initGoogleMap`;
            script.async = true;
            script.defer = true;
            script.onerror = () => {
                showMapFallback();
            };
            document.head.appendChild(script);

        } catch (e) {
            showMapFallback();
        }
    }

    function showMapFallback() {
        if (mapFallback) {
            mapFallback.style.display = "flex";
        }
    }

    function onGoogleMapsLoaded() {
        const container = document.getElementById("googleMapContainer");
        if (!container) return;

        // Center on Telangana geographic centroid
        const telanganaCenter = { lat: 17.8495, lng: 79.1151 };

        googleMap = new google.maps.Map(container, {
            center: telanganaCenter,
            zoom: 7,
            mapId: "PILGRIM_AI_MAP",
            disableDefaultUI: false,
            zoomControl: true,
            styles: [
                { elementType: "geometry", stylers: [{ color: "#17263c" }] },
                { elementType: "labels.text.stroke", stylers: [{ color: "#242f3e" }] },
                { elementType: "labels.text.fill", stylers: [{ color: "#746855" }] },
                { featureType: "administrative.locality", elementType: "labels.text.fill", stylers: [{ color: "#d59563" }] },
                { featureType: "poi", elementType: "labels.text.fill", stylers: [{ color: "#d59563" }] },
                { featureType: "poi.park", elementType: "geometry", stylers: [{ color: "#263c3f" }] },
                { featureType: "road", elementType: "geometry", stylers: [{ color: "#38414e" }] },
                { featureType: "water", elementType: "geometry", stylers: [{ color: "#17263c" }] }
            ]
        });

        directionsService = new google.maps.DirectionsService();
        directionsRenderer = new google.maps.DirectionsRenderer({
            map: googleMap,
            suppressMarkers: false,
            polylineOptions: {
                strokeColor: "#fbbf24",
                strokeWeight: 5,
                strokeOpacity: 0.8
            }
        });
        window.googleMap = googleMap;
        window.directionsRenderer = directionsRenderer;

        // Load 20 temple coordinates from backend registry
        loadTemplePins();

        // Restore pending saved route if initialized before map loaded
        if (pendingRouteToRestore) {
            const pr = pendingRouteToRestore;
            pendingRouteToRestore = null;
            focusRouteOnMap(pr.origin, pr.destination, pr.btn, pr.originCanonical, pr.destCanonical, pr.polyline, pr.origCoords, pr.destCoords);
        }
    }

    async function loadTemplePins() {
        try {
            const res = await fetch("/api/maps/temples");
            if (!res.ok) return;
            templeCoordinates = await res.json();

            // Populate temple markers
            templeCoordinates.forEach(t => {
                const marker = new google.maps.Marker({
                    position: { lat: t.latitude, lng: t.longitude },
                    map: googleMap,
                    title: t.display_name,
                    icon: {
                        path: google.maps.SymbolPath.CIRCLE,
                        scale: 7,
                        fillColor: "#fbbf24",
                        fillOpacity: 1,
                        strokeColor: "#000",
                        strokeWeight: 2
                    }
                });

                const infoWindow = new google.maps.InfoWindow({
                    content: `
                        <div style="color: #000; padding: 6px; font-family: sans-serif;">
                            <strong style="font-size: 14px;">${escapeHtml(t.display_name)}</strong><br>
                            <span style="font-size: 12px; color: #555;">District: ${escapeHtml(t.district)}, ${escapeHtml(t.state)}</span><br>
                            <div style="margin-top: 6px;">
                                <button onclick="discoverPlacesForTemple('${escapeHtml(t.display_name)}', ${t.latitude}, ${t.longitude})" style="background: #fbbf24; border: none; padding: 4px 8px; border-radius: 4px; font-weight: 600; cursor: pointer;">
                                    Find Nearby Places
                                </button>
                            </div>
                        </div>
                    `
                });

                marker.addListener("click", () => {
                    infoWindow.open(googleMap, marker);
                    fetchNearbyPlaces(t.display_name, t.latitude, t.longitude);
                });

                mapMarkers.push(marker);
            });

        } catch (e) {
            console.warn("Could not load temple pins:", e);
        }
    }

    // -------------------------------------------------------------
    // Google Places API Discovery (Segregated & Safe)
    // -------------------------------------------------------------
    window.discoverPlacesForTemple = function(name, lat, lng) {
        fetchNearbyPlaces(name, lat, lng);
    };

    // -------------------------------------------------------------
    // Google Places API Discovery (Browser-Side & Safe)
    // -------------------------------------------------------------
    window.discoverPlacesForTemple = function(name, lat, lng) {
        fetchNearbyPlaces(name, lat, lng);
    };

    window.getPlacesDiagnostics = function() {
        const googleLoaded = typeof window.google !== "undefined";
        const mapsLoaded = !!(googleLoaded && window.google.maps);
        const placesLibraryLoaded = !!(mapsLoaded && window.google.maps.places);
        const placesServiceAvailable = !!(placesLibraryLoaded && window.google.maps.places.PlacesService);

        return {
            googleLoaded: googleLoaded,
            mapsLoaded: mapsLoaded,
            placesLibraryLoaded: placesLibraryLoaded,
            placesServiceAvailable: placesServiceAvailable,
            selectedTemple: currentSelectedTemple,
            selectedLatitude: currentLat,
            selectedLongitude: currentLng,
            latitude: currentLat,
            longitude: currentLng,
            requestStarted: placesRequestStarted,
            lastStatus: placesLastStatus,
            lastResultCount: placesLastResultCount,
            resultsCount: placesLastResultCount,
            lastError: placesLastError
        };
    };

    function fetchNearbyPlaces(templeName, lat, lng, onComplete) {
        placesRequestStarted = true;

        if (!templeName) {
            if (currentSelectedTemple) {
                templeName = currentSelectedTemple;
            } else {
                if (placesTargetTitle) placesTargetTitle.innerText = "Nearby Discoveries";
                if (placesList) {
                    placesList.innerHTML = `<div style="font-size: 12px; color: var(--text-muted); text-align: center; padding: 20px;">Please select a temple from the covered temples list or ask a question to discover nearby places.</div>`;
                }
                finish("NO_TEMPLE_SELECTED", 0, null);
                return;
            }
        }

        const finish = (status, count, err) => {
            placesLastStatus = status;
            placesLastResultCount = count || 0;
            placesLastError = err || null;
            if (typeof onComplete === "function") {
                try { onComplete(); } catch (e) {}
            }
        };

        let parsedLat = parseFloat(lat);
        let parsedLng = parseFloat(lng);

        if (lat === undefined || lat === null || lat === "") {
            parsedLat = parseFloat(currentLat);
        }
        if (lng === undefined || lng === null || lng === "") {
            parsedLng = parseFloat(currentLng);
        }

        // Validate Coordinates (-90 to 90 for lat, -180 to 180 for lng)
        if (isNaN(parsedLat) || isNaN(parsedLng) || parsedLat < -90 || parsedLat > 90 || parsedLng < -180 || parsedLng > 180) {
            const err = "Selected temple does not have valid geographic coordinates.";
            finish("INVALID_COORDINATES", 0, err);
            renderPlacesError("Invalid Coordinates", err);
            return;
        }

        currentSelectedTemple = templeName;
        currentLat = parsedLat;
        currentLng = parsedLng;

        if (placesTargetTitle) {
            placesTargetTitle.innerText = `Places near ${templeName}`;
        }
        if (placesList) {
            placesList.innerHTML = `<div style="font-size: 12px; color: var(--text-muted); text-align: center; padding: 14px;">Searching live Google Places for hotels & restaurants near ${escapeHtml(templeName)}...</div>`;
        }

        const googleLoaded = typeof window.google !== "undefined";
        const mapsLoaded = !!(googleLoaded && window.google.maps);
        const placesLibraryLoaded = !!(mapsLoaded && window.google.maps.places);
        const placesServiceAvailable = !!(placesLibraryLoaded && window.google.maps.places.PlacesService);

        // PRIMARY PATH: Browser-side PlacesService using browser HTTP referrer restricted key
        if (placesServiceAvailable) {
            try {
                const mapContainer = googleMap || document.getElementById("googleMapContainer") || document.createElement("div");
                const service = new google.maps.places.PlacesService(mapContainer);
                const pos = new google.maps.LatLng(currentLat, currentLng);
                const request = {
                    location: pos,
                    radius: 5000,
                    keyword: "hotels and restaurants"
                };

                service.nearbySearch(request, (results, status) => {
                    const statusStr = String(status);
                    console.log(`Places status: ${statusStr}, results count: ${results ? results.length : 0}, selected temple: ${templeName}, coordinates: (${currentLat}, ${currentLng})`);

                    if (status === google.maps.places.PlacesServiceStatus.OK && results && results.length > 0) {
                        finish("OK", results.length, null);
                        renderPlacesResults(results.map(r => ({
                            name: r.name || "Unknown Place",
                            address: r.vicinity || r.formatted_address || "Address unavailable",
                            rating: r.rating,
                            user_ratings_total: r.user_ratings_total,
                            open_now: r.opening_hours ? r.opening_hours.isOpen() : null
                        })));
                    } else if (status === google.maps.places.PlacesServiceStatus.ZERO_RESULTS || (status === google.maps.places.PlacesServiceStatus.OK && (!results || results.length === 0))) {
                        finish("ZERO_RESULTS", 0, null);
                        renderPlacesZeroResults(`No nearby places were found within 5 km of ${escapeHtml(templeName)}.`);
                    } else if (status === google.maps.places.PlacesServiceStatus.REQUEST_DENIED) {
                        const errMsg = "Google Places access was denied. Check that the Places API is enabled for the browser API key in Google Cloud Console.";
                        finish("REQUEST_DENIED", 0, errMsg);
                        renderPlacesError("Google Places Access Denied (REQUEST_DENIED)", errMsg, "In Google Cloud Console under 'APIs & Services > Credentials', verify that your API key's API Restrictions include 'Places API' (in addition to 'Maps JavaScript API').");
                    } else if (status === google.maps.places.PlacesServiceStatus.OVER_QUERY_LIMIT) {
                        const errMsg = "Google Places request quota has been reached. Please try again later.";
                        finish("OVER_QUERY_LIMIT", 0, errMsg);
                        renderPlacesError("Query Limit Exceeded (OVER_QUERY_LIMIT)", errMsg);
                    } else if (status === google.maps.places.PlacesServiceStatus.INVALID_REQUEST) {
                        const errMsg = "Google Places request was invalid. Check request coordinates and radius.";
                        finish("INVALID_REQUEST", 0, errMsg);
                        renderPlacesError("Invalid Request (INVALID_REQUEST)", errMsg);
                    } else if (status === google.maps.places.PlacesServiceStatus.UNKNOWN_ERROR) {
                        const errMsg = "Google Places service encountered a temporary server error. Please retry.";
                        finish("UNKNOWN_ERROR", 0, errMsg);
                        renderPlacesError("Google Places Server Error (UNKNOWN_ERROR)", errMsg);
                    } else {
                        const errMsg = `Google Places request failed: ${statusStr}`;
                        finish(statusStr, 0, errMsg);
                        renderPlacesError(`Google Places Unavailable (${statusStr})`, errMsg);
                    }
                });
                return;
            } catch (err) {
                console.warn("Browser PlacesService exception:", err);
                const errMsg = err.message || "PlacesService invocation exception";
                finish("EXCEPTION", 0, errMsg);
                renderPlacesError("PlacesService Error", errMsg);
                return;
            }
        }

        // Fallback: Check server endpoint if browser places library is not loaded
        if (!placesLibraryLoaded) {
            fetch(`/api/maps/places?query=${encodeURIComponent("hotels and restaurants near " + templeName)}&latitude=${currentLat}&longitude=${currentLng}&radius=5000`)
                .then(res => res.json())
                .then(data => {
                    if (data.results && data.results.length > 0) {
                        finish("OK", data.results.length, null);
                        renderPlacesResults(data.results);
                    } else if (data.status === "BROWSER_SEARCH_REQUIRED") {
                        const errMsg = "Google Places is not available in this browser session. The Places JavaScript library is not loaded, and the server-side Places API credential is not configured.";
                        finish("BROWSER_SEARCH_REQUIRED", 0, errMsg);
                        renderPlacesError("Google Places Unavailable", errMsg, "Ensure the Google Maps JavaScript API script is loaded with &libraries=places and that the Places API is enabled.");
                    } else if (data.status === "ZERO_RESULTS") {
                        finish("ZERO_RESULTS", 0, null);
                        renderPlacesZeroResults(`No nearby places were found within 5 km of ${escapeHtml(templeName)}.`);
                    } else {
                        const errMsg = data.message || `Google Places request failed: ${data.status}`;
                        finish(data.status || "FAILED", 0, errMsg);
                        renderPlacesError("Google Places Request Failed", errMsg);
                    }
                })
                .catch(() => {
                    const errMsg = "Google Places is not available in this browser session.";
                    finish("NETWORK_ERROR", 0, errMsg);
                    renderPlacesError("Google Places Unavailable", errMsg);
                });
        } else {
            const errMsg = "Google Places is not available in this browser session.";
            finish("NOT_AVAILABLE", 0, errMsg);
            renderPlacesError("Google Places Unavailable", errMsg);
        }
    }

    function renderPlacesResults(items) {
        if (!placesList) return;
        if (!items || items.length === 0) {
            renderPlacesZeroResults("No nearby places returned.");
            return;
        }

        let cardsHtml = "";
        items.slice(0, 10).forEach(p => {
            const rating = p.rating ? `⭐ ${p.rating} (${p.user_ratings_total || 0})` : "Rating: N/A";
            cardsHtml += `
                <div class="place-card">
                    <div class="place-title">${escapeHtml(p.name)}</div>
                    <div class="place-address">${escapeHtml(p.address || "Address unavailable")}</div>
                    <div class="place-meta">
                        <span class="place-rating">${rating}</span>
                        ${p.open_now !== null && p.open_now !== undefined ? `<span>${p.open_now ? "🟢 Open" : "🔴 Closed"}</span>` : ""}
                    </div>
                </div>
            `;
        });
        placesList.innerHTML = cardsHtml;
    }

    function renderPlacesError(title, message, hint) {
        if (!placesList) return;
        placesList.innerHTML = `
            <div class="places-error-banner">
                <div class="places-error-title">⚠️ ${escapeHtml(title)}</div>
                <div class="places-error-desc">${escapeHtml(message)}</div>
                ${hint ? `<div class="places-config-hint">💡 ${escapeHtml(hint)}</div>` : ""}
            </div>
        `;
    }

    function renderPlacesZeroResults(message) {
        if (!placesList) return;
        placesList.innerHTML = `
            <div class="places-zero-results">
                ${escapeHtml(message || "No nearby places found.")}
            </div>
        `;
    }

    function renderPlacesOffline(msg) {
        renderPlacesError("Google Places Unavailable", msg || "Google Places discovery currently offline or unconfigured. Verified temple guide remains active.");
    }

    // Sidebar temple items click listener
    if (templeList) {
        templeList.addEventListener("click", (e) => {
            const item = e.target.closest(".temple-item");
            if (!item) return;
            const tName = item.getAttribute("data-temple");
            if (!tName) return;

            document.querySelectorAll(".temple-item").forEach(i => i.classList.remove("active"));
            item.classList.add("active");

            // Pan map if loaded
            const found = templeCoordinates.find(t => t.canonical_name.toLowerCase() === tName.toLowerCase() || t.display_name.toLowerCase().includes(tName.toLowerCase()));
            if (found) {
                currentSelectedTemple = found.display_name;
                currentLat = found.latitude;
                currentLng = found.longitude;
                panToCoordinates(found.latitude, found.longitude, found.display_name);
                fetchNearbyPlaces(found.display_name, found.latitude, found.longitude);
            }
        });
    }

    // -------------------------------------------------------------
    // Native Browser Voice UI (Speech-to-Text & Text-to-Speech)
    // -------------------------------------------------------------
    let cachedVoices = [];

    function populateVoices() {
        if (!window.speechSynthesis) return;
        cachedVoices = window.speechSynthesis.getVoices();
    }

    if (window.speechSynthesis) {
        populateVoices();
        if (speechSynthesis.onvoiceschanged !== undefined) {
            speechSynthesis.onvoiceschanged = populateVoices;
        }
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition && micBtn) {
        speechRecognition = new SpeechRecognition();
        speechRecognition.continuous = false;
        speechRecognition.interimResults = false;
        speechRecognition.lang = "en-IN";

        speechRecognition.onstart = () => {
            isListening = true;
            micBtn.classList.add("listening");
        };

        speechRecognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            userQuery.value = transcript;
            isListening = false;
            micBtn.classList.remove("listening");
            chatForm.dispatchEvent(new Event("submit"));
        };

        speechRecognition.onerror = () => {
            isListening = false;
            micBtn.classList.remove("listening");
        };

        speechRecognition.onend = () => {
            isListening = false;
            micBtn.classList.remove("listening");
        };

        micBtn.addEventListener("click", () => {
            if (!speechRecognition) return;
            if (isListening) {
                speechRecognition.stop();
            } else {
                speechRecognition.lang = currentLang === "te" ? "te-IN" : (currentLang === "hi" ? "hi-IN" : "en-IN");
                speechRecognition.start();
            }
        });
    } else if (micBtn) {
        micBtn.style.display = "none";
    }

    // -------------------------------------------------------------
    // Markdown Stripping for Natural Speech Output
    // -------------------------------------------------------------
    function stripMarkdownForSpeech(text) {
        if (!text) return "";
        let clean = text;

        // 1. Remove fenced code blocks ```...```
        clean = clean.replace(/```[\s\S]*?```/g, " ");

        // 2. Remove inline code `...`
        clean = clean.replace(/`([^`]+)`/g, "$1");

        // 3. Remove UI note disclaimers e.g. *(Note: ...)* or *(Note: ... )*
        clean = clean.replace(/\*\s*\(\s*Note:[\s\S]*?\)\s*\*/gi, " ");

        // 4. Convert markdown links [Label](url) -> Label
        clean = clean.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");

        // 5. Remove citation chips e.g. [1], [2], [12]
        clean = clean.replace(/\[\d+\]/g, " ");

        // 6. Replace headers (e.g. ### Header) with Header followed by period for pause
        clean = clean.replace(/^#{1,6}\s+(.*$)/gm, "$1. ");

        // 7. Remove blockquotes (> ...)
        clean = clean.replace(/^>\s+/gm, " ");

        // 8. Replace bullet lists and numbered lists with punctuation pauses
        clean = clean.replace(/^\s*[-*+]\s+/gm, " ");
        clean = clean.replace(/^\s*\d+\.\s+/gm, " ");

        // 9. Remove bold and italic markers (* and _)
        clean = clean.replace(/[*_]{1,3}([^*_]+)[*_]{1,3}/g, "$1");
        clean = clean.replace(/[*_]/g, "");

        // 10. Replace table pipes and separators
        clean = clean.replace(/\|/g, " ");

        // 11. Normalize newlines to sentence periods if missing punctuation
        clean = clean.replace(/([^\w\s.?!])\n+/g, " ");
        clean = clean.replace(/([a-zA-Z0-9])\n+/g, "$1. ");
        clean = clean.replace(/\n+/g, " ");

        // 12. Normalize multiple spaces and multiple periods
        clean = clean.replace(/\s+/g, " ");
        clean = clean.replace(/\.+/g, ".");
        clean = clean.replace(/\s+([.,?!])/g, "$1");

        return clean.trim();
    }
    window.stripMarkdownForSpeech = stripMarkdownForSpeech;

    function showVoiceNotification(message) {
        let toast = document.getElementById("voiceToast");
        if (!toast) {
            toast = document.createElement("div");
            toast.id = "voiceToast";
            toast.style.cssText = "position: fixed; bottom: 80px; left: 50%; transform: translateX(-50%); background: #1e293b; color: #fbbf24; border: 1px solid #d97706; padding: 10px 18px; border-radius: 8px; font-size: 13px; font-weight: 500; z-index: 9999; box-shadow: 0 4px 12px rgba(0,0,0,0.5); display: flex; align-items: center; gap: 8px; transition: opacity 0.3s ease;";
            document.body.appendChild(toast);
        }
        toast.innerHTML = `<span>⚠️ ${escapeHtml(message)}</span>`;
        toast.style.opacity = "1";
        toast.style.display = "flex";
        setTimeout(() => {
            toast.style.opacity = "0";
            setTimeout(() => { toast.style.display = "none"; }, 300);
        }, 4000);
    }

    function selectBestVoice(targetLang) {
        const voices = cachedVoices.length > 0 ? cachedVoices : (window.speechSynthesis ? window.speechSynthesis.getVoices() : []);
        if (!voices || voices.length === 0) return null;

        if (targetLang === "te") {
            // Target Telugu
            const teVoice = voices.find(v => v.lang.toLowerCase() === "te-in" || v.lang.toLowerCase().startsWith("te"));
            return teVoice || null;
        } else if (targetLang === "hi") {
            // Target Hindi
            const hiVoice = voices.find(v => v.lang.toLowerCase() === "hi-in" || v.lang.toLowerCase().startsWith("hi"));
            return hiVoice || null;
        } else {
            // Target English: en-IN -> en-US -> en-GB -> system default
            const enIn = voices.find(v => v.lang.toLowerCase() === "en-in");
            if (enIn) return enIn;
            const enUs = voices.find(v => v.lang.toLowerCase() === "en-us");
            if (enUs) return enUs;
            const enGb = voices.find(v => v.lang.toLowerCase() === "en-gb");
            if (enGb) return enGb;
            const anyEn = voices.find(v => v.lang.toLowerCase().startsWith("en"));
            if (anyEn) return anyEn;
            return voices[0] || null;
        }
    }

    // -------------------------------------------------------------
    // Browser-Native Text-to-Speech Playback Controls (Play / Pause / Resume / Stop)
    // -------------------------------------------------------------
    const SPEECH_ICONS = {
        play: '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>',
        pause: '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>',
        resume: '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>',
        stop: '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect></svg>'
    };

    let activeSpeechState = "idle"; // "idle" | "playing" | "paused"
    let activeUtterance = null;
    let activeSpeechContainer = null;
    let speechKeepAliveTimer = null;

    function renderSpeechControlsHtml(state) {
        if (state === "playing") {
            return `
                <button class="btn-speech-control btn-speech-pause" title="Pause speech" onclick="handleSpeechAction(this, 'pause')">
                    ${SPEECH_ICONS.pause}
                    <span class="btn-speech-label">Pause</span>
                </button>
                <button class="btn-speech-control btn-speech-stop" title="Stop speech" onclick="handleSpeechAction(this, 'stop')">
                    ${SPEECH_ICONS.stop}
                    <span class="btn-speech-label">Stop</span>
                </button>
            `;
        } else if (state === "paused") {
            return `
                <button class="btn-speech-control btn-speech-resume" title="Resume speech" onclick="handleSpeechAction(this, 'resume')">
                    ${SPEECH_ICONS.resume}
                    <span class="btn-speech-label">Resume</span>
                </button>
                <button class="btn-speech-control btn-speech-stop" title="Stop speech" onclick="handleSpeechAction(this, 'stop')">
                    ${SPEECH_ICONS.stop}
                    <span class="btn-speech-label">Stop</span>
                </button>
            `;
        } else {
            // idle
            return `
                <button class="btn-speech-control btn-speech-play" title="Read Aloud" onclick="handleSpeechAction(this, 'play')">
                    ${SPEECH_ICONS.play}
                    <span class="btn-speech-label">Read</span>
                </button>
            `;
        }
    }

    function updateSpeechControlsUi(container, state) {
        if (!container) return;
        container.setAttribute("data-speech-state", state);
        container.innerHTML = renderSpeechControlsHtml(state);
    }

    function resetActiveSpeechUi() {
        clearSpeechKeepAlive();
        if (activeSpeechContainer) {
            updateSpeechControlsUi(activeSpeechContainer, "idle");
            activeSpeechContainer = null;
        }
        activeSpeechState = "idle";
        activeUtterance = null;
    }

    function startSpeechKeepAlive() {
        clearSpeechKeepAlive();
        // Periodically pulse pause/resume while playing to prevent Chromium 15s engine timeout during long speech
        speechKeepAliveTimer = setInterval(() => {
            if (!window.speechSynthesis) return;
            if (activeSpeechState === "playing" && window.speechSynthesis.speaking && !window.speechSynthesis.paused) {
                window.speechSynthesis.pause();
                window.speechSynthesis.resume();
            }
        }, 10000);
    }

    function clearSpeechKeepAlive() {
        if (speechKeepAliveTimer) {
            clearInterval(speechKeepAliveTimer);
            speechKeepAliveTimer = null;
        }
    }

    window.handleSpeechAction = function(btn, action) {
        if (!window.speechSynthesis) {
            showVoiceNotification("Speech synthesis is not supported in this browser.");
            return;
        }

        const container = btn.closest(".speech-controls") || btn.parentElement;
        if (!container) return;

        if (action === "pause") {
            if (activeSpeechState === "playing") {
                window.speechSynthesis.pause();
                activeSpeechState = "paused";
                clearSpeechKeepAlive();
                updateSpeechControlsUi(container, "paused");
            }
            return;
        }

        if (action === "resume") {
            if (activeSpeechState === "paused" && activeUtterance) {
                window.speechSynthesis.resume();
                activeSpeechState = "playing";
                startSpeechKeepAlive();
                updateSpeechControlsUi(container, "playing");
            }
            return;
        }

        if (action === "stop") {
            window.speechSynthesis.cancel();
            resetActiveSpeechUi();
            return;
        }

        if (action === "play") {
            // If already speaking this message, ignore duplicate rapid clicks
            if (activeSpeechContainer === container && activeSpeechState === "playing") {
                return;
            }

            // Cancel any previous speech
            window.speechSynthesis.cancel();
            resetActiveSpeechUi();

            const bubble = btn.closest(".msg-bubble-wrapper").querySelector(".msg-bubble");
            if (!bubble) return;

            const rawText = bubble.getAttribute("data-raw-answer") || bubble.innerText;
            const cleanText = stripMarkdownForSpeech(rawText);
            if (!cleanText) return;

            // Determine language from content script or current selection
            let targetLang = "en";
            if (/[\u0C00-\u0C7F]/.test(cleanText)) {
                targetLang = "te";
            } else if (/[\u0900-\u097F]/.test(cleanText)) {
                targetLang = "hi";
            } else if (currentLang === "te" || currentLang === "hi") {
                targetLang = currentLang;
            }

            const voice = selectBestVoice(targetLang);

            // Regional voice fallback handling
            if (targetLang === "te" && !voice) {
                showVoiceNotification("Telugu voice is not available in this browser.");
                return;
            }

            const utterance = new SpeechSynthesisUtterance(cleanText);
            if (voice) {
                utterance.voice = voice;
                utterance.lang = voice.lang;
            } else {
                utterance.lang = targetLang === "hi" ? "hi-IN" : "en-IN";
            }
            utterance.rate = 1.0;

            utterance.onstart = () => {
                activeSpeechState = "playing";
                updateSpeechControlsUi(container, "playing");
                startSpeechKeepAlive();
            };

            utterance.onpause = () => {
                if (activeSpeechState !== "paused") {
                    activeSpeechState = "paused";
                    clearSpeechKeepAlive();
                    updateSpeechControlsUi(container, "paused");
                }
            };

            utterance.onresume = () => {
                if (activeSpeechState !== "playing") {
                    activeSpeechState = "playing";
                    startSpeechKeepAlive();
                    updateSpeechControlsUi(container, "playing");
                }
            };

            utterance.onend = () => {
                if (activeUtterance === utterance) {
                    resetActiveSpeechUi();
                }
            };

            utterance.onerror = (e) => {
                if (activeUtterance === utterance) {
                    resetActiveSpeechUi();
                }
            };

            activeUtterance = utterance;
            activeSpeechContainer = container;
            activeSpeechState = "playing";
            updateSpeechControlsUi(container, "playing");

            if (window.speechSynthesis.paused) {
                window.speechSynthesis.resume();
            }
            window.speechSynthesis.speak(utterance);
        }
    };

    window.playSpeechText = function(btn) {
        window.handleSpeechAction(btn, "play");
    };

    // -------------------------------------------------------------
    // Voice Diagnostics (Safe & Credential-Free)
    // -------------------------------------------------------------
    window.getVoiceDiagnostics = function(lang = "en") {
        const supported = !!window.speechSynthesis;
        const voices = supported ? (cachedVoices.length > 0 ? cachedVoices : window.speechSynthesis.getVoices()) : [];
        const matched = selectBestVoice(lang);

        return {
            speechSynthesisSupported: supported,
            availableVoicesCount: voices.length,
            selectedVoice: matched ? matched.name : (voices.length > 0 ? "System Default" : "None"),
            selectedLanguage: lang
        };
    };

    // -------------------------------------------------------------
    // Utility Helpers
    // -------------------------------------------------------------
    function scrollToBottom() {
        messageArea.scrollTop = messageArea.scrollHeight;
    }

    function escapeHtml(str) {
        if (!str) return "";
        return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    }

    let activeInAppPlaceMarker = null;
    let activeInAppInfoWindow = null;

    function resolveAndShowInAppPlace(query, fallbackUrl) {
        if (!googleMap) return;

        const qClean = (query || "").toLowerCase();
        let foundTemple = templeCoordinates.find(t => 
            (t.canonical_name && qClean.includes(t.canonical_name.toLowerCase())) ||
            (t.display_name && qClean.includes(t.display_name.toLowerCase()))
        );

        if (!foundTemple) {
            for (const [alias, canon] of Object.entries(CANONICAL_TEMPLE_ALIASES)) {
                if (qClean.includes(alias)) {
                    foundTemple = templeCoordinates.find(t => t.canonical_name.toLowerCase() === canon.toLowerCase());
                    if (foundTemple) break;
                }
            }
        }

        const hasPlaces = !!(window.google && google.maps && google.maps.places && google.maps.places.PlacesService);
        if (hasPlaces) {
            try {
                const placesService = new google.maps.places.PlacesService(googleMap);
                placesService.findPlaceFromQuery({
                    query: query,
                    fields: ["name", "geometry", "formatted_address"]
                }, (results, status) => {
                    if (status === google.maps.places.PlacesServiceStatus.OK && results && results.length > 0) {
                        const r = results[0];
                        const pos = { lat: r.geometry.location.lat(), lng: r.geometry.location.lng() };
                        displayInAppMarker(pos, r.name || query, r.formatted_address, fallbackUrl);
                    } else if (foundTemple) {
                        displayInAppMarker({ lat: foundTemple.latitude, lng: foundTemple.longitude }, query, `Near ${foundTemple.display_name}`, fallbackUrl);
                    } else {
                        displayInAppMarker({ lat: currentLat, lng: currentLng }, query, "Near pilgrimage shrine", fallbackUrl);
                    }
                });
                return;
            } catch (e) {
                console.warn("[PilgrimAI] PlacesService findPlace error:", e);
            }
        }

        if (foundTemple) {
            displayInAppMarker({ lat: foundTemple.latitude, lng: foundTemple.longitude }, query, `Near ${foundTemple.display_name}`, fallbackUrl);
        } else {
            displayInAppMarker({ lat: currentLat, lng: currentLng }, query, "Near pilgrimage shrine", fallbackUrl);
        }
    }

    function displayInAppMarker(pos, title, address, externalUrl) {
        if (!googleMap) return;

        if (activeInAppPlaceMarker) {
            try { activeInAppPlaceMarker.setMap(null); } catch (e) {}
        }
        if (activeInAppInfoWindow) {
            try { activeInAppInfoWindow.close(); } catch (e) {}
        }

        activeInAppPlaceMarker = new google.maps.Marker({
            position: pos,
            map: googleMap,
            title: title,
            animation: google.maps.Animation.DROP,
            icon: {
                path: google.maps.SymbolPath.BACKWARD_CLOSED_ARROW,
                scale: 6,
                fillColor: "#f59e0b",
                fillOpacity: 1,
                strokeColor: "#ffffff",
                strokeWeight: 2
            }
        });

        const infoContent = `
            <div style="color: #0f172a; padding: 6px 8px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 240px;">
                <div style="font-weight: 700; font-size: 13px; color: #0f172a; margin-bottom: 2px;">${escapeHtml(title)}</div>
                <div style="font-size: 11.5px; color: #475569; line-height: 1.4;">${escapeHtml(address || "Located near shrine")}</div>
                <div style="margin-top: 8px; padding-top: 6px; border-top: 1px solid #e2e8f0; display: flex; justify-content: space-between; align-items: center; gap: 8px;">
                    <span style="font-size: 10px; color: #059669; font-weight: 600; display: inline-flex; align-items: center; gap: 3px;">
                        <span>✓</span> In-App Map
                    </span>
                    <a href="${escapeHtml(externalUrl)}" target="_blank" rel="noopener noreferrer" style="font-size: 11px; color: #d97706; text-decoration: underline; font-weight: 500;">
                        External Fallback ↗
                    </a>
                </div>
            </div>
        `;

        activeInAppInfoWindow = new google.maps.InfoWindow({
            content: infoContent
        });

        activeInAppPlaceMarker.addListener("click", () => {
            activeInAppInfoWindow.open(googleMap, activeInAppPlaceMarker);
        });

        activeInAppInfoWindow.open(googleMap, activeInAppPlaceMarker);
        googleMap.panTo(pos);
        googleMap.setZoom(15);
    }

    async function calculateDirectRoute(orig, dest, mode = "driving", explicitOriginCoords = null, explicitDestCoords = null) {
        const btn = document.getElementById("btnCalcRoute") || document.getElementById("mapCalcRouteBtn");
        if (btn) {
            btn.disabled = true;
            btn.classList.add("loading");
            btn.innerText = "Calculating...";
        }

        try {
            // Geocode origin and destination to precise coordinates
            let resolvedOriginGeo = explicitOriginCoords ? { lat: explicitOriginCoords[0], lng: explicitOriginCoords[1], name: orig } : await geocodeLocationAsync(orig);
            let resolvedDestGeo = explicitDestCoords ? { lat: explicitDestCoords[0], lng: explicitDestCoords[1], name: dest } : await geocodeLocationAsync(dest);

            if (!resolvedOriginGeo || resolvedOriginGeo.lat === undefined) {
                clearActiveRoute();
                showVoiceNotification(`Could not find starting location "${orig}". Please check the spelling or enter a valid location.`);
                return false;
            }

            if (!resolvedDestGeo || resolvedDestGeo.lat === undefined) {
                clearActiveRoute();
                showVoiceNotification(`Could not find destination "${dest}". Please check the spelling or enter a valid location.`);
                return false;
            }

            const origCoords = [Number(resolvedOriginGeo.lat), Number(resolvedOriginGeo.lng)];
            const destCoords = [Number(resolvedDestGeo.lat), Number(resolvedDestGeo.lng)];

            const reqBody = {
                origin: (resolvedOriginGeo && resolvedOriginGeo.name) || orig,
                destination: (resolvedDestGeo && resolvedDestGeo.name) || dest,
                travel_mode: mode
            };
            if (origCoords) reqBody.origin_coordinates = origCoords;
            if (destCoords) reqBody.destination_coordinates = destCoords;

            const res = await fetch("/api/maps/route", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(reqBody)
            });

            if (!res.ok) {
                clearActiveRoute();
                showVoiceNotification("Route could not be calculated. Please check the locations.");
                return false;
            }

            const data = await res.json();
            if (data.success && data.polyline) {
                // Ensure map panel is open and visible
                if (mapPanel && mapPanel.classList.contains("collapsed")) {
                    mapPanel.classList.remove("collapsed");
                }
                if (tabMapBtn && !tabMapBtn.classList.contains("active")) {
                    tabMapBtn.classList.add("active");
                    if (tabPlacesBtn) tabPlacesBtn.classList.remove("active");
                }
                if (googleMap) {
                    google.maps.event.trigger(googleMap, "resize");
                }

                const resolvedOriginName = orig || data.origin;
                const resolvedDestName = (resolvedDestGeo && !resolvedDestGeo.isTemple) ? (resolvedDestGeo.name || dest) : (data.destination || dest);

                // Update latestNavigationMeta for metrics display
                latestNavigationMeta = {
                    origin: resolvedOriginName,
                    destination: resolvedDestName,
                    origin_canonical: data.origin_canonical,
                    destination_canonical: data.destination_canonical,
                    travel_mode: data.travel_mode || mode,
                    distance_km: data.distance_km,
                    duration_minutes: data.duration_minutes,
                    traffic_duration_minutes: data.traffic_duration_minutes,
                    polyline: data.polyline,
                    origin_coordinates: data.origin_coordinates,
                    destination_coordinates: data.destination_coordinates
                };

                const success = await focusRouteOnMap(
                    resolvedOriginName,
                    resolvedDestName,
                    btn,
                    data.origin_canonical,
                    data.destination_canonical,
                    data.polyline,
                    data.origin_coordinates,
                    data.destination_coordinates
                );

                if (success) {
                    activeNavigationState = {
                        origin: resolvedOriginName,
                        destination: resolvedDestName,
                        travelMode: data.travel_mode || mode,
                        distance_km: data.distance_km,
                        duration_minutes: data.duration_minutes,
                        polyline: data.polyline,
                        origin_coordinates: data.origin_coordinates,
                        destination_coordinates: data.destination_coordinates
                    };
                    window.activeNavigationState = activeNavigationState;
                    window.latestNavigationMeta = latestNavigationMeta;
                    return true;
                }
            } else {
                // Calculation failed: Clear any previous route and do not present stale route
                clearActiveRoute();
                showVoiceNotification(data.error || data.message || "Route could not be calculated for this journey.");
                return false;
            }
        } catch (err) {
            console.warn("[PilgrimAI] Direct route calculation error:", err);
            clearActiveRoute();
            showVoiceNotification("Route request failed. Please check network connection.");
            return false;
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.classList.remove("loading");
                btn.innerText = "Get Route";
            }
        }
        return false;
    }

    window.calculateDirectRoute = calculateDirectRoute;
    window.getActiveRoutePolyline = () => activeRoutePolyline;
    window.getActiveRouteMarkers = () => activeRouteMarkers;
    window.getActiveNavigationState = () => activeNavigationState;
    window.getLatestNavigationMeta = () => latestNavigationMeta;
    window.getGoogleMap = () => googleMap;

    function initMapRoutePlanner() {
        const mapOriginInput = document.getElementById("mapOriginInput");
        const mapDestInput = document.getElementById("mapDestInput");
        const mapModeSelect = document.getElementById("mapModeSelect");
        const mapSwapBtn = document.getElementById("mapSwapBtn");
        const mapUseGpsBtn = document.getElementById("mapUseGpsBtn");
        const mapCalcRouteBtn = document.getElementById("btnCalcRoute") || document.getElementById("mapCalcRouteBtn");
        const mapClearRouteBtn = document.getElementById("mapClearRouteBtn");

        let userHasManuallyEditedOrigin = false;
        let userManualOriginValue = "";
        let userHasManuallyEditedDest = false;
        let userManualDestValue = "";

        if (mapOriginInput) {
            const updateOriginTracking = () => {
                userManualOriginValue = mapOriginInput.value.trim();
                userHasManuallyEditedOrigin = Boolean(userManualOriginValue);
            };
            mapOriginInput.addEventListener("input", updateOriginTracking);
            mapOriginInput.addEventListener("change", updateOriginTracking);
        }
        window.getUserManualOrigin = () => {
            const currentVal = mapOriginInput ? mapOriginInput.value.trim() : "";
            return currentVal || (userHasManuallyEditedOrigin ? userManualOriginValue : null);
        };

        if (mapDestInput) {
            const updateDestTracking = () => {
                userManualDestValue = mapDestInput.value.trim();
                userHasManuallyEditedDest = Boolean(userManualDestValue);
            };
            mapDestInput.addEventListener("input", updateDestTracking);
            mapDestInput.addEventListener("change", updateDestTracking);
        }
        window.getUserManualDestination = () => {
            const currentVal = mapDestInput ? mapDestInput.value.trim() : "";
            return currentVal || (userHasManuallyEditedDest ? userManualDestValue : null);
        };

        if (mapModeSelect) {
            mapModeSelect.addEventListener("change", async () => {
                const selectedMode = mapModeSelect.value;
                const orig = mapOriginInput ? mapOriginInput.value.trim() : "";
                const dest = mapDestInput ? mapDestInput.value.trim() : "";

                const targetOrig = orig || (activeNavigationState && activeNavigationState.origin);
                const targetDest = dest || (activeNavigationState && activeNavigationState.destination);

                if (targetOrig && targetDest) {
                    await calculateDirectRoute(targetOrig, targetDest, selectedMode);
                } else if (activeNavigationState) {
                    activeNavigationState.travelMode = selectedMode;
                }
            });
        }

        if (mapSwapBtn) {
            mapSwapBtn.addEventListener("click", (e) => {
                if (e) {
                    e.preventDefault();
                    e.stopPropagation();
                }
                if (!mapOriginInput || !mapDestInput) return;
                const temp = mapOriginInput.value;
                mapOriginInput.value = mapDestInput.value;
                mapDestInput.value = temp;

                // Swap manual tracking states
                const tempVal = userManualOriginValue;
                userManualOriginValue = userManualDestValue;
                userManualDestValue = tempVal;

                const tempEdited = userHasManuallyEditedOrigin;
                userHasManuallyEditedOrigin = userHasManuallyEditedDest;
                userHasManuallyEditedDest = tempEdited;

                if (mapOriginInput.value && mapDestInput.value) {
                    const mode = mapModeSelect ? mapModeSelect.value : "driving";
                    calculateDirectRoute(mapOriginInput.value, mapDestInput.value, mode);
                }
            });
        }

        if (mapUseGpsBtn) {
            mapUseGpsBtn.addEventListener("click", (e) => {
                if (e) {
                    e.preventDefault();
                    e.stopPropagation();
                }
                if (!mapOriginInput) return;
                if (navigator.geolocation) {
                    mapOriginInput.placeholder = "Detecting device location...";
                    navigator.geolocation.getCurrentPosition(
                        (pos) => {
                            mapOriginInput.value = `${pos.coords.latitude.toFixed(5)}, ${pos.coords.longitude.toFixed(5)}`;
                        },
                        () => {
                            mapOriginInput.placeholder = "Starting city or landmark";
                            alert("Location access was denied or timed out. Please enter your starting city or landmark manually.");
                        },
                        { timeout: 5000 }
                    );
                } else {
                    alert("Geolocation is not supported by your browser. Please enter your starting city manually.");
                }
            });
        }

        if (mapCalcRouteBtn) {
            mapCalcRouteBtn.addEventListener("click", (e) => {
                if (e) {
                    e.preventDefault();
                    e.stopPropagation();
                }
                const orig = mapOriginInput ? mapOriginInput.value.trim() : "";
                const dest = mapDestInput ? mapDestInput.value.trim() : "";
                const mode = mapModeSelect ? mapModeSelect.value : "driving";

                if (!orig) {
                    if (mapOriginInput) {
                        mapOriginInput.focus();
                        mapOriginInput.style.borderColor = "var(--accent-gold)";
                        setTimeout(() => { mapOriginInput.style.borderColor = ""; }, 2000);
                    }
                    showVoiceNotification("Where are you starting from? Enter your starting location to see the route.");
                    return;
                }

                if (!dest) {
                    if (mapDestInput) {
                        mapDestInput.focus();
                        mapDestInput.style.borderColor = "var(--accent-gold)";
                        setTimeout(() => { mapDestInput.style.borderColor = ""; }, 2000);
                    }
                    showVoiceNotification("Please enter a destination to calculate route.");
                    return;
                }

                calculateDirectRoute(orig, dest, mode);
            });
        }

        if (mapClearRouteBtn) {
            mapClearRouteBtn.addEventListener("click", (e) => {
                if (e) {
                    e.preventDefault();
                    e.stopPropagation();
                }
                userHasManuallyEditedOrigin = false;
                userManualOriginValue = "";
                userHasManuallyEditedDest = false;
                userManualDestValue = "";
                clearActiveRoute();
            });
        }

        [mapOriginInput, mapDestInput].forEach(inp => {
            if (inp) {
                inp.addEventListener("keydown", (e) => {
                    if (e.key === "Enter") {
                        e.preventDefault();
                        e.stopPropagation();
                        if (mapCalcRouteBtn) mapCalcRouteBtn.click();
                    }
                });
            }
        });
    }

    window.handleInAppMapLinkClick = async function(event, el) {
        if (event) {
            event.preventDefault();
            event.stopPropagation();
        }
        const rawUrl = el ? (el.getAttribute("data-maps-url") || el.getAttribute("href") || "") : "";
        if (!rawUrl) return;

        // 1. Ensure Map Panel is expanded & interactive map tab is active
        if (mapPanel && mapPanel.classList.contains("collapsed")) {
            mapPanel.classList.remove("collapsed");
        }
        if (tabMapBtn && !tabMapBtn.classList.contains("active")) {
            tabMapBtn.classList.add("active");
            if (tabPlacesBtn) tabPlacesBtn.classList.remove("active");
        }
        if (googleMap) {
            google.maps.event.trigger(googleMap, "resize");
        }

        // 2. Wait up to 1500ms if googleMap is still initializing
        if (!googleMap) {
            for (let i = 0; i < 15; i++) {
                await new Promise(r => setTimeout(r, 100));
                if (googleMap) break;
            }
        }

        // If map is still unavailable, notify user and provide external link fallback
        if (!googleMap) {
            showVoiceNotification("In-app map visualization is currently offline. Opening external Google Maps...");
            window.open(rawUrl, "_blank", "noopener,noreferrer");
            return;
        }

        try {
            let urlObj = null;
            try {
                urlObj = new URL(rawUrl);
            } catch (e) {
                urlObj = null;
            }

            const isDir = urlObj && (urlObj.pathname.includes("/dir") || urlObj.searchParams.has("destination"));
            let query = "";
            let dest = "";
            let orig = "";

            if (isDir) {
                orig = urlObj.searchParams.get("origin") || "";
                dest = urlObj.searchParams.get("destination") || "";
            }
            if (urlObj && urlObj.searchParams.has("query")) {
                query = urlObj.searchParams.get("query");
            } else if (urlObj && urlObj.searchParams.has("destination")) {
                query = urlObj.searchParams.get("destination");
            } else if (el && el.innerText) {
                query = el.innerText.replace(/^[🗺️📍\s]+/, "").trim();
            }

            query = decodeURIComponent(query || dest || "").replace(/\+/g, " ").trim();

            // P3: If an origin is available, offer a route to that place
            const knownOrigin = orig || activeNavigationState.origin;
            if (knownOrigin && query) {
                const mapOriginInput = document.getElementById("mapOriginInput");
                const mapDestInput = document.getElementById("mapDestInput");
                if (mapOriginInput) mapOriginInput.value = knownOrigin;
                if (mapDestInput) mapDestInput.value = query;

                const routeSuccess = await focusRouteOnMap(knownOrigin, query, null, null, null, null, null, null);
                if (routeSuccess) {
                    return;
                }
            }

            // P3: If origin is missing, pre-fill destination in route planner and focus origin input
            const mapDestInput = document.getElementById("mapDestInput");
            const mapOriginInput = document.getElementById("mapOriginInput");
            if (mapDestInput && query) mapDestInput.value = query;
            if (mapOriginInput && !mapOriginInput.value) {
                mapOriginInput.focus();
                mapOriginInput.placeholder = "Enter starting location to calculate route";
            }

            if (!query) {
                query = currentSelectedTemple || "Temple";
            }

            resolveAndShowInAppPlace(query, rawUrl);

        } catch (err) {
            console.warn("[PilgrimAI] handleInAppMapLinkClick error:", err);
            window.open(rawUrl, "_blank", "noopener,noreferrer");
        }
    };

    function formatMarkdownText(md) {
        if (!md) return "";

        // Extract <details>...</details> blocks before HTML escaping to render genuine expandable UI component
        const detailsPlaceholders = [];
        const detailsBlockRegex = /<details>[\s\r\n]*<summary>(?:<b>)?([\s\S]*?)(?:<\/b>)?<\/summary>([\s\S]*?)<\/details>/gi;
        let preprocessed = md.replace(detailsBlockRegex, function(match, summaryRaw, bodyRaw) {
            const cleanSummary = summaryRaw.replace(/<[^>]+>/g, "").trim();
            const placeholder = `___DETAILS_BLOCK_${detailsPlaceholders.length}___`;
            detailsPlaceholders.push({
                summary: cleanSummary || "Optional Seva Tariffs & Counter Offerings",
                body: bodyRaw.trim()
            });
            return `\n\n${placeholder}\n\n`;
        });

        let html = escapeHtml(preprocessed);

        // Strip raw internal source references e.g. (Source: .../....json) or (Source: ...)
        html = html.replace(/\s*\(\s*Source:\s*[^)]+\.json\s*\)/gi, "");
        html = html.replace(/\s*\*\s*\(\s*Source:\s*[^)]+\.json\s*\)\s*\*/gi, "");
        // Strip trailing raw **Sources:** markdown block from answer body
        html = html.replace(/\n+\*\*Sources:\*\*\s*(?:\n+-\s*\*\*[^*]+\*\*:[^\n]+)+/gi, "");

        // Unescape escaped hashes like \#### or \###
        html = html.replace(/\\(#+)/g, "$1");

        // Clean any raw scraping footnote citations e.g. [[1](http...)] or [[1]http...]
        html = html.replace(/:?\s*\[\[?\d+\]?\(?https?:\/\/[^\s\])]+\)?\]?/g, ".");
        html = html.replace(/\.\s*\./g, ".");

        // Parse and extract Markdown Tables before newline-to-break conversions
        const tablePlaceholders = [];
        const tableBlockRegex = /(?:^[ \t]*\|.+?\|[ \t]*(?:\r?\n|$)){2,}/gm;
        html = html.replace(tableBlockRegex, function(match) {
            const rawLines = match.trim().split(/\r?\n/).map(l => l.trim()).filter(l => l.startsWith("|") && l.endsWith("|"));
            if (rawLines.length < 2) return match;

            const headerCells = rawLines[0].split("|").slice(1, -1).map(c => c.trim());
            const sepLine = rawLines[1];
            // Verify separator row e.g. | :--- | :--- | or | --- | --- |
            if (!/^\|(?:\s*:?-+:?\s*\|)+$/.test(sepLine)) {
                return match;
            }

            let tableHtml = '<div class="budget-table-wrapper"><table class="budget-table"><thead><tr>';
            headerCells.forEach(cell => {
                let cellFormatted = cell.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>").replace(/\*(.*?)\*/g, "<em>$1</em>");
                tableHtml += `<th>${cellFormatted}</th>`;
            });
            tableHtml += '</tr></thead><tbody>';

            for (let i = 2; i < rawLines.length; i++) {
                const rowCells = rawLines[i].split("|").slice(1, -1).map(c => c.trim());
                tableHtml += '<tr>';
                rowCells.forEach((cell, cellIdx) => {
                    let cellFormatted = cell.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>").replace(/\*(.*?)\*/g, "<em>$1</em>");
                    if (cellIdx === rowCells.length - 1 && (cell.toLowerCase().includes("listed") || cell.toLowerCase().includes("estimated") || cell.toLowerCase().includes("complimentary"))) {
                        const isListed = cell.toLowerCase().includes("listed") || cell.toLowerCase().includes("complimentary");
                        cellFormatted = `<span class="evidence-pill ${isListed ? 'evidence-listed' : 'evidence-estimate'}">${cellFormatted}</span>`;
                    }
                    tableHtml += `<td>${cellFormatted}</td>`;
                });
                tableHtml += '</tr>';
            }
            tableHtml += '</tbody></table></div>';

            const idx = tablePlaceholders.length;
            tablePlaceholders.push(tableHtml);
            return `\n\n___TABLE_BLOCK_${idx}___\n\n`;
        });

        // Bold
        html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
        // Italic
        html = html.replace(/\*(.*?)\*/g, "<em>$1</em>");
        // Headers (apply itinerary-day-heading class specifically to Day headings for prominent typography)
        html = html.replace(/^(?:####|###)\s*(Day\s+\d+.*$)/gim, '<h4 class="itinerary-day-heading">$1</h4>');
        html = html.replace(/^#### (.*$)/gim, "<h4>$1</h4>");
        html = html.replace(/^### (.*$)/gim, "<h3>$1</h3>");
        html = html.replace(/^## (.*$)/gim, "<h2>$1</h2>");
        html = html.replace(/^# (.*$)/gim, "<h1>$1</h1>");
        // Blockquotes
        html = html.replace(/^> (.*$)/gim, "<blockquote>$1</blockquote>");
        // Bullet lists (including indented sub-items)
        html = html.replace(/^[\s\t]*\-\s+(.*$)/gim, "<li>$1</li>");
        // Wrap adjacent li elements into ul
        html = html.replace(/(<li>.*?<\/li>)/gs, "<ul>$1</ul>");
        html = html.replace(/<\/ul>\s*<ul>/g, "");

        // Links [Label](url) - transform navigation / map links into in-app interactive actions
        html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s\)]+)\)/g, function(match, label, url) {
            const isMapLink = /directions|route|map/i.test(label) || /google\.com\/maps|maps\.google/i.test(url);
            if (isMapLink) {
                return `<a href="${url}" class="in-app-map-action" data-maps-url="${url}" onclick="handleInAppMapLinkClick(event, this)">🗺️ ${label}</a>`;
            }
            return `<a href="${url}" target="_blank" rel="noopener noreferrer" style="color:var(--accent-gold);text-decoration:underline;">${label}</a>`;
        });

        // Newlines to line breaks (outside lists)
        html = html.replace(/\n\n/g, "<br><br>");
        html = html.replace(/\n/g, "<br>");

        // Restore parsed HTML tables
        for (let idx = 0; idx < tablePlaceholders.length; idx++) {
            html = html.replace(new RegExp(`<br>\\s*___TABLE_BLOCK_${idx}___\\s*<br>`, "g"), tablePlaceholders[idx]);
            html = html.replace(new RegExp(`___TABLE_BLOCK_${idx}___`, "g"), tablePlaceholders[idx]);
        }

        // Restore parsed Details Expandable Drawers (P1 fix)
        for (let idx = 0; idx < detailsPlaceholders.length; idx++) {
            const item = detailsPlaceholders[idx];
            let bodyHtml = escapeHtml(item.body);
            bodyHtml = bodyHtml.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
            bodyHtml = bodyHtml.replace(/\*(.*?)\*/g, "<em>$1</em>");
            bodyHtml = bodyHtml.replace(/^[\s\t]*\-\s+(.*$)/gim, "<li>$1</li>");
            bodyHtml = bodyHtml.replace(/(<li>.*?<\/li>)/gs, "<ul>$1</ul>");
            bodyHtml = bodyHtml.replace(/<\/ul>\s*<ul>/g, "");
            bodyHtml = bodyHtml.replace(/\n\n/g, "<br>");
            bodyHtml = bodyHtml.replace(/\n/g, "<br>");

            const drawerHtml = `
                <details class="seva-expandable-drawer">
                    <summary class="seva-drawer-summary">
                        <span class="seva-drawer-left">
                            <span class="seva-drawer-icon">✨</span>
                            <strong class="seva-drawer-title">${escapeHtml(item.summary)}</strong>
                        </span>
                        <span class="seva-drawer-badge">Click to Expand</span>
                    </summary>
                    <div class="seva-drawer-content">
                        ${bodyHtml}
                    </div>
                </details>
            `;

            html = html.replace(new RegExp(`<br>\\s*___DETAILS_BLOCK_${idx}___\\s*<br>`, "g"), drawerHtml);
            html = html.replace(new RegExp(`___DETAILS_BLOCK_${idx}___`, "g"), drawerHtml);
        }

        // Fallback: cleanup any stray or unparsed details/summary escaped tags
        html = html.replace(/&lt;details&gt;/gi, '<details class="seva-expandable-drawer">');
        html = html.replace(/&lt;\/details&gt;/gi, '</details>');
        html = html.replace(/&lt;summary&gt;(?:&lt;b&gt;)?([\s\S]*?)(?:&lt;\/b&gt;)?&lt;\/summary&gt;/gi, '<summary class="seva-drawer-summary"><span class="seva-drawer-left"><span class="seva-drawer-icon">✨</span><strong class="seva-drawer-title">$1</strong></span><span class="seva-drawer-badge">Click to Expand</span></summary>');

        return html;
    }

    // -------------------------------------------------------------
    // Full Page Authentication State & View Controller
    // -------------------------------------------------------------
    const authPage = document.getElementById("authPage");
    const appLayout = document.getElementById("appLayout");
    const headerUserName = document.getElementById("headerUserName");
    const logoutBtn = document.getElementById("logoutBtn");

    // Views
    const viewLogin = document.getElementById("viewLogin");
    const viewRegister = document.getElementById("viewRegister");
    const viewForgotPassword = document.getElementById("viewForgotPassword");
    const viewResetPassword = document.getElementById("viewResetPassword");

    // Navigation Switch Links
    const linkToRegister = document.getElementById("linkToRegister");
    const linkToLoginFromReg = document.getElementById("linkToLoginFromReg");
    const linkToForgotPassword = document.getElementById("linkToForgotPassword");
    const linkToLoginFromForgot = document.getElementById("linkToLoginFromForgot");
    const linkToLoginFromReset = document.getElementById("linkToLoginFromReset");

    // Forms
    const loginForm = document.getElementById("loginForm");
    const registerForm = document.getElementById("registerForm");
    const forgotPasswordForm = document.getElementById("forgotPasswordForm");
    const resetPasswordForm = document.getElementById("resetPasswordForm");

    // Message Elements
    const loginError = document.getElementById("loginError");
    const registerError = document.getElementById("registerError");
    const registerSuccess = document.getElementById("registerSuccess");
    const forgotError = document.getElementById("forgotError");
    const forgotSuccess = document.getElementById("forgotSuccess");
    const resetError = document.getElementById("resetError");
    const resetSuccess = document.getElementById("resetSuccess");
    const resetTokenValue = document.getElementById("resetTokenValue");

    function clearAuthMessages() {
        if (loginError) loginError.style.display = "none";
        if (registerError) registerError.style.display = "none";
        if (registerSuccess) registerSuccess.style.display = "none";
        if (forgotError) forgotError.style.display = "none";
        if (forgotSuccess) forgotSuccess.style.display = "none";
        if (resetError) resetError.style.display = "none";
        if (resetSuccess) resetSuccess.style.display = "none";
    }

    function switchAuthView(viewName) {
        clearAuthMessages();
        if (viewName === "AUTHENTICATED_APP") {
            if (authPage) authPage.style.display = "none";
            if (appLayout) appLayout.style.display = "flex";
            if (window.lucide) lucide.createIcons();
            if (googleMap) google.maps.event.trigger(googleMap, "resize");
            loadChatHistory(true);
            return;
        }

        // Show Auth Page, hide main application
        if (authPage) authPage.style.display = "flex";
        if (appLayout) appLayout.style.display = "none";

        if (viewLogin) viewLogin.style.display = (viewName === "AUTH_LOGIN") ? "block" : "none";
        if (viewRegister) viewRegister.style.display = (viewName === "AUTH_REGISTER") ? "block" : "none";
        if (viewForgotPassword) viewForgotPassword.style.display = (viewName === "AUTH_FORGOT_PASSWORD") ? "block" : "none";
        if (viewResetPassword) viewResetPassword.style.display = (viewName === "AUTH_RESET_PASSWORD") ? "block" : "none";

        if (window.lucide) lucide.createIcons();
    }

    // Link Click Event Listeners
    if (linkToRegister) {
        linkToRegister.addEventListener("click", () => switchAuthView("AUTH_REGISTER"));
    }
    if (linkToLoginFromReg) {
        linkToLoginFromReg.addEventListener("click", () => switchAuthView("AUTH_LOGIN"));
    }
    if (linkToForgotPassword) {
        linkToForgotPassword.addEventListener("click", () => switchAuthView("AUTH_FORGOT_PASSWORD"));
    }
    if (linkToLoginFromForgot) {
        linkToLoginFromForgot.addEventListener("click", () => switchAuthView("AUTH_LOGIN"));
    }
    if (linkToLoginFromReset) {
        linkToLoginFromReset.addEventListener("click", () => {
            // Remove reset_token query parameter from address bar
            window.history.replaceState({}, document.title, window.location.pathname);
            switchAuthView("AUTH_LOGIN");
        });
    }

    // 1. Login Form Submit
    if (loginForm) {
        loginForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            clearAuthMessages();
            const email = document.getElementById("loginEmail").value.trim();
            const password = document.getElementById("loginPassword").value;

            try {
                const res = await fetch("/api/auth/login", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ email, password })
                });
                const data = await res.json();
                if (!res.ok) {
                    if (loginError) {
                        loginError.textContent = data.detail || "Invalid email or password.";
                        loginError.style.display = "block";
                    }
                    return;
                }
                localStorage.setItem("pilgrim_jwt", data.access_token);
                if (headerUserName) {
                    headerUserName.textContent = data.user ? (data.user.name || data.user.email) : "Pilgrim";
                }
                switchAuthView("AUTHENTICATED_APP");
            } catch (err) {
                if (loginError) {
                    loginError.textContent = "Unable to connect to server. Please try again.";
                    loginError.style.display = "block";
                }
            }
        });
    }

    // 2. Register Form Submit
    if (registerForm) {
        registerForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            clearAuthMessages();
            const name = document.getElementById("regName").value.trim();
            const email = document.getElementById("regEmail").value.trim();
            const password = document.getElementById("regPassword").value;
            const confirmPassword = document.getElementById("regPasswordConfirm").value;

            if (password !== confirmPassword) {
                if (registerError) {
                    registerError.textContent = "Passwords do not match.";
                    registerError.style.display = "block";
                }
                return;
            }

            try {
                const res = await fetch("/api/auth/register", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ name, email, password })
                });
                const data = await res.json();
                if (!res.ok) {
                    if (registerError) {
                        registerError.textContent = data.detail || "Registration failed.";
                        registerError.style.display = "block";
                    }
                    return;
                }
                if (registerSuccess) {
                    registerSuccess.textContent = "Registration successful! A verification email has been sent. Redirecting to login...";
                    registerSuccess.style.display = "block";
                }
                setTimeout(() => {
                    switchAuthView("AUTH_LOGIN");
                    const loginEmailInput = document.getElementById("loginEmail");
                    if (loginEmailInput) loginEmailInput.value = email;
                }, 2000);
            } catch (err) {
                if (registerError) {
                    registerError.textContent = "Unable to connect to server. Please try again.";
                    registerError.style.display = "block";
                }
            }
        });
    }

    // 3. Forgot Password Form Submit
    if (forgotPasswordForm) {
        forgotPasswordForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            clearAuthMessages();
            const email = document.getElementById("forgotEmail").value.trim();

            try {
                const res = await fetch("/api/auth/forgot-password", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ email })
                });
                const data = await res.json();
                if (forgotSuccess) {
                    forgotSuccess.textContent = data.message || "If an account exists for this email, a password reset link has been sent.";
                    forgotSuccess.style.display = "block";
                }
            } catch (err) {
                if (forgotError) {
                    forgotError.textContent = "Unable to connect to server. Please try again.";
                    forgotError.style.display = "block";
                }
            }
        });
    }

    // 4. Reset Password Form Submit
    if (resetPasswordForm) {
        resetPasswordForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            clearAuthMessages();
            const token = resetTokenValue ? resetTokenValue.value : "";
            const new_password = document.getElementById("resetPasswordInput").value;
            const confirm_password = document.getElementById("resetPasswordConfirmInput").value;

            if (new_password !== confirm_password) {
                if (resetError) {
                    resetError.textContent = "New passwords do not match.";
                    resetError.style.display = "block";
                }
                return;
            }

            try {
                const res = await fetch("/api/auth/reset-password", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ token, new_password })
                });
                const data = await res.json();
                if (!res.ok) {
                    if (resetError) {
                        resetError.textContent = data.detail || "Failed to reset password.";
                        resetError.style.display = "block";
                    }
                    return;
                }
                if (resetSuccess) {
                    resetSuccess.textContent = "Password reset successfully! Redirecting to sign in...";
                    resetSuccess.style.display = "block";
                }
                setTimeout(() => {
                    window.history.replaceState({}, document.title, window.location.pathname);
                    switchAuthView("AUTH_LOGIN");
                }, 2000);
            } catch (err) {
                if (resetError) {
                    resetError.textContent = "Unable to connect to server. Please try again.";
                    resetError.style.display = "block";
                }
            }
        });
    }

    // 5. Logout
    if (logoutBtn) {
        logoutBtn.addEventListener("click", () => {
            localStorage.removeItem("pilgrim_jwt");
            createFreshSession();
            if (chatHistoryList) {
                chatHistoryList.innerHTML = '<div class="history-empty-msg">No conversations yet</div>';
            }
            switchAuthView("AUTH_LOGIN");
        });
    }

    // 6. Startup Authentication / URL Token Check
    async function initAuthFlow() {
        // Check for URL query parameter ?reset_token=...
        const urlParams = new URLSearchParams(window.location.search);
        const resetToken = urlParams.get("reset_token");
        if (resetToken) {
            if (resetTokenValue) resetTokenValue.value = resetToken;
            switchAuthView("AUTH_RESET_PASSWORD");
            return;
        }

        const token = localStorage.getItem("pilgrim_jwt");
        if (!token) {
            switchAuthView("AUTH_LOGIN");
            return;
        }

        try {
            const res = await fetch("/api/auth/me", {
                headers: { "Authorization": `Bearer ${token}` }
            });
            if (res.ok) {
                const user = await res.json();
                if (headerUserName) {
                    headerUserName.textContent = user.name || user.email;
                }
                switchAuthView("AUTHENTICATED_APP");
            } else {
                localStorage.removeItem("pilgrim_jwt");
                switchAuthView("AUTH_LOGIN");
            }
        } catch (e) {
            console.warn("Auth startup verification error:", e);
            switchAuthView("AUTH_LOGIN");
        }
    }

    // Run startup authentication check
    initAuthFlow();

    // Initialize Maps
    initGoogleMaps();
});


