/* ==========================================================================
   PulmoScan AI Platform - Interactive Frontend Controller
   ========================================================================== */

// --- GLOBAL WORKSPACE STATE ---
let activeScanImg = null;       // Original CT scan Image object
let activeMaskImg = null;       // Base64-derived Tumor Mask Image object
let offscreenMaskCanvas = null; // Temp canvas for real-time colormap pixel changes
let activeSamplePath = null;    // Tracks currently loaded archive sample
let loadedFileName = "";        // Current filename display

// Synchronized Zoom/Pan transform variables
let zoomScale = 1.0;
let panX = 0;
let panY = 0;
let isDragging = false;
let startDragX = 0;
let startDragY = 0;

// PACS configuration state
let maskOpacity = 0.7;          // Default 70% opacity
let activeColorMap = "ruby";    // Default color map is Ruby Red

// Animation parameters
let targetConfidence = 0;
let currentConfidence = 0;
let confidenceAnimTimer = null;

// DOM Elements cache
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const btnBrowse = document.getElementById("btn-browse-file");
const pacsContainer = document.getElementById("pacs-view-container");
const diagEmptyCard = document.getElementById("diag-empty-card");
const diagResultsCard = document.getElementById("diag-results-card");
const loadedFilenameLabel = document.getElementById("loaded-filename");
const opacitySlider = document.getElementById("mask-opacity-slider");
const opacityLabel = document.getElementById("mask-opacity-label");
const samplesGrid = document.getElementById("samples-container");

// API Badges & Controls
const apiStatusBadge = document.getElementById("api-status-badge");
const activeEngineBadge = document.getElementById("active-engine-badge");
const dlOption = document.getElementById("dl-option");
const sensitivitySlider = document.getElementById("sensitivity-slider");
const sensitivityValue = document.getElementById("sensitivity-value");

// PACS Canvas Setup
const canvasOriginal = document.getElementById("canvas-original");
const canvasOverlay = document.getElementById("canvas-overlay");
const ctxOriginal = canvasOriginal.getContext("2d");
const ctxOverlay = canvasOverlay.getContext("2d");

// --- INITIALIZATION ---
document.addEventListener("DOMContentLoaded", () => {
    // 1. Fetch system status & preloaded clinical archives
    checkBackendStatus();
    loadClinicalSamples();

    // 2. Register Upload & Drag-and-Drop listeners
    setupUploadHandlers();

    // 3. Register PACS view manipulation handlers (Pan/Zoom/Toggles)
    setupPacsControlHandlers();

    // 4. Register PDF Exporter
    setupPdfExportHandler();
});

// --- API COMMUNICATIONS & STATUS TASKS ---
async function checkBackendStatus() {
    try {
        // Test API readiness
        const response = await fetch("/api/samples");
        if (response.ok) {
            apiStatusBadge.textContent = "ONLINE";
            apiStatusBadge.className = "pill-value green";
        }
    } catch (e) {
        apiStatusBadge.textContent = "OFFLINE";
        apiStatusBadge.className = "pill-value text-accent"; // Warn color
        console.error("API connection failed", e);
    }
}

async function loadClinicalSamples() {
    try {
        const response = await fetch("/api/samples");
        if (!response.ok) throw new Error("Could not fetch samples");
        
        const samples = await response.json();
        samplesGrid.innerHTML = "";
        
        if (samples.length === 0) {
            samplesGrid.innerHTML = `<div class="samples-loading">No sample scans found. Copy images into test/ folders to populate.</div>`;
            return;
        }

        samples.forEach(sample => {
            const card = document.createElement("div");
            card.className = `sample-thumb-card ${sample.category.toLowerCase().includes('normal') ? 'normal' : ''}`;
            card.dataset.path = sample.path;
            card.dataset.name = sample.name;

            // Stream thumbnail dynamically using our dynamic endpoint
            const thumbUrl = `/api/sample-image?path=${encodeURIComponent(sample.path)}`;
            
            card.innerHTML = `
                <div class="thumb-img-placeholder">
                    <img src="${thumbUrl}" alt="Clinical thumbnail" onerror="this.style.display='none'">
                    <span class="thumb-icon">🩻</span>
                </div>
                <p class="sample-cat">${sample.category}</p>
                <p class="sample-name">${sample.name}</p>
            `;

            card.addEventListener("click", () => {
                // Clear active states and set this card active
                document.querySelectorAll(".sample-thumb-card").forEach(c => c.classList.remove("active"));
                card.classList.add("active");
                
                loadedFileName = sample.name;
                activeSamplePath = sample.path;
                
                // Run diagnostic pipeline directly
                analyzeScan(null, sample.path);
            });

            samplesGrid.appendChild(card);
        });
    } catch (e) {
        samplesGrid.innerHTML = `<div class="samples-loading">Failed to load archive database.</div>`;
        console.error(e);
    }
}

// --- ANALYSIS PIPELINE EXECUTION ---
async function analyzeScan(fileObj, samplePath) {
    // Show Loading Overlay state
    showWorkspaceLoading(true);

    const formData = new FormData();
    formData.append("sensitivity", sensitivitySlider.value);

    if (fileObj) {
        formData.append("file", fileObj);
        activeSamplePath = null; // Reset sample tracker
    } else if (samplePath) {
        formData.append("sample_path", samplePath);
    }

    try {
        const response = await fetch("/api/analyze", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || "Diagnostic analysis failed");
        }

        const data = await response.json();
        renderDiagnosticResults(data, fileObj, samplePath);

    } catch (e) {
        alert("Clinical diagnostic pipeline encountered an error: " + e.message);
        showWorkspaceLoading(false);
    }
}

function showWorkspaceLoading(isLoading) {
    if (isLoading) {
        dropZone.querySelector("h3").textContent = "Analyzing Scan Slice...";
        dropZone.querySelector("p").textContent = "Running neural segmentations & clinical metrics. Please wait...";
        dropZone.querySelector(".upload-circle").style.animation = "rotateLoading 1s linear infinite";
        dropZone.style.pointerEvents = "none";
        document.querySelectorAll(".sample-thumb-card").forEach(c => c.style.pointerEvents = "none");
    } else {
        dropZone.querySelector("h3").textContent = "Upload Chest CT Scan";
        dropZone.querySelector("p").textContent = "Drag & Drop chest scan slice (PNG, JPG, TIFF) or browse computer";
        dropZone.querySelector(".upload-circle").style.animation = "rotateLoading 3s linear infinite";
        dropZone.style.pointerEvents = "auto";
        document.querySelectorAll(".sample-thumb-card").forEach(c => c.style.pointerEvents = "auto");
    }
}

// --- RESULTS RENDERING & VISUALIZATION SETUP ---
function renderDiagnosticResults(data, fileObj, samplePath) {
    // 1. Update Badge status
    activeEngineBadge.textContent = data.engine;
    if (data.engine.includes("Deep Learning")) {
        activeEngineBadge.className = "pill-value text-accent";
        dlOption.disabled = false;
        dlOption.textContent = "Keras DL Engine (Active)";
    } else {
        activeEngineBadge.className = "pill-value";
        dlOption.disabled = true;
        dlOption.textContent = "Keras DL Engine (Locked)";
    }

    // 2. Load CT scan image into browser Image memory
    activeScanImg = new Image();
    
    if (fileObj) {
        activeScanImg.src = URL.createObjectURL(fileObj);
    } else if (samplePath) {
        activeScanImg.src = `/api/sample-image?path=${encodeURIComponent(samplePath)}`;
    }

    // 3. Load Base64 mask image
    activeMaskImg = new Image();
    activeMaskImg.src = "data:image/png;base64," + data.mask_base64;

    // Wait until both scans load before drawing
    let imagesLoaded = 0;
    const onImgLoad = () => {
        imagesLoaded++;
        if (imagesLoaded === 2) {
            setupOffscreenMask();
            resetViewport();
            showWorkstationUI();
            showWorkspaceLoading(false);
        }
    };

    activeScanImg.onload = onImgLoad;
    activeMaskImg.onload = onImgLoad;

    // 4. Update HUD Stats & Insights Panel
    updateHUDPanels(data);
}

function updateHUDPanels(data) {
    // Reset any ongoing gauge animations
    clearInterval(confidenceAnimTimer);
    
    // Animate radial confidence gauge
    targetConfidence = Math.round(data.confidence * 100);
    currentConfidence = 0;
    
    const gaugeFill = document.getElementById("gauge-confidence-fill");
    const gaugeText = document.getElementById("result-confidence-text");
    
    confidenceAnimTimer = setInterval(() => {
        if (currentConfidence >= targetConfidence) {
            clearInterval(confidenceAnimTimer);
        } else {
            currentConfidence++;
            gaugeText.textContent = `${currentConfidence}%`;
            
            // stroke-dasharray is 251.2 representing full circle circumference (2 * pi * r, r=40)
            const offset = 251.2 - (251.2 * currentConfidence) / 100;
            gaugeFill.style.strokeDashoffset = offset;
        }
    }, 15);

    // Primary assessment
    const label = data.label.replace(".", " ").toUpperCase();
    document.getElementById("result-class-name").textContent = label;
    document.getElementById("result-summary-text").textContent = data.insights.summary;

    // Set Urgency severity badge
    const badge = document.getElementById("result-severity-badge");
    badge.textContent = data.insights.urgency.toUpperCase();
    if (data.insights.urgency.toLowerCase() === "normal" || data.insights.urgency.toLowerCase() === "low") {
        badge.className = "severity-badge normal";
    } else if (data.insights.urgency.toLowerCase() === "high") {
        badge.className = "severity-badge suspicious";
    } else {
        badge.className = "severity-badge critical";
    }

    // Update Probability bars
    const probs = data.confidences;
    updateProbabilityBar("adenocarcinoma", probs.adenocarcinoma || 0);
    updateProbabilityBar("squamous", probs["squamous.cell.carcinoma"] || 0);
    updateProbabilityBar("largecell", probs["large.cell.carcinoma"] || 0);
    updateProbabilityBar("normal", probs.normal || 0);

    // Update Volumetric sizing metrics
    document.getElementById("metric-diameter").innerHTML = `${data.stats.tumor_diameter_mm.toFixed(1)} <span class="unit">mm</span>`;
    document.getElementById("metric-percentage").innerHTML = `${data.stats.tumor_percentage.toFixed(2)} <span class="unit">%</span>`;
    document.getElementById("metric-circularity").textContent = data.stats.circularity.toFixed(2);
    document.getElementById("metric-nodules-count").textContent = data.stats.nodules_found;

    // Update Clinical narrative text
    document.getElementById("insight-markers").textContent = data.insights.radiology_signs;
    document.getElementById("insight-recommendation").textContent = data.insights.recommendation;

    // Save state for PDF reports
    document.getElementById("print-diag-label").textContent = data.label.replace(".", " ");
    document.getElementById("print-diag-conf").textContent = `${Math.round(data.confidence * 100)}%`;
    document.getElementById("print-diag-summary").textContent = data.insights.summary;
    document.getElementById("print-diag-diameter").textContent = `${data.stats.tumor_diameter_mm.toFixed(1)} mm`;
    document.getElementById("print-diag-fraction").textContent = `${data.stats.tumor_percentage.toFixed(2)} %`;
    document.getElementById("print-diag-circularity").textContent = data.stats.circularity.toFixed(2);
    document.getElementById("print-diag-count").textContent = data.stats.nodules_found;
    document.getElementById("print-diag-markers").textContent = data.insights.radiology_signs;
    document.getElementById("print-diag-recommendation").textContent = data.insights.recommendation;
    document.getElementById("print-engine").textContent = data.engine;
    document.getElementById("print-urgency").textContent = data.insights.urgency.toUpperCase();
}

function updateProbabilityBar(classId, fractionValue) {
    const percentage = Math.round(fractionValue * 100);
    document.getElementById(`prob-val-${classId}`).textContent = `${percentage}%`;
    document.getElementById(`prob-bar-${classId}`).style.width = `${percentage}%`;
}

function showWorkstationUI() {
    dropZone.classList.add("hide");
    pacsContainer.classList.remove("hide");
    diagEmptyCard.classList.add("hide");
    diagResultsCard.classList.remove("hide");
    loadedFilenameLabel.textContent = loadedFileName;
}

function hideWorkstationUI() {
    dropZone.classList.remove("hide");
    pacsContainer.classList.add("hide");
    diagEmptyCard.classList.remove("hide");
    diagResultsCard.classList.add("hide");
}

// --- DUAL CANVAS SYNC VISUALIZATION ENGINE ---
function setupOffscreenMask() {
    // Offscreen canvas is used to change monochrome white mask to custom medical HUD overlay colors on the fly
    if (!offscreenMaskCanvas) {
        offscreenMaskCanvas = document.createElement("canvas");
    }
    
    offscreenMaskCanvas.width = activeMaskImg.width;
    offscreenMaskCanvas.height = activeMaskImg.height;
    
    const osCtx = offscreenMaskCanvas.getContext("2d");
    osCtx.drawImage(activeMaskImg, 0, 0);

    // Apply color maps directly to the mask pixels
    const imgData = osCtx.getImageData(0, 0, offscreenMaskCanvas.width, offscreenMaskCanvas.height);
    const pixels = imgData.data;

    let rVal = 244, gVal = 63, bVal = 94; // Ruby red
    if (activeColorMap === "cyan") {
        rVal = 6; gVal = 182; bVal = 212; // Cyan
    } else if (activeColorMap === "emerald") {
        rVal = 16; gVal = 185; bVal = 129; // Toxic Emerald
    }

    for (let i = 0; i < pixels.length; i += 4) {
        const grayscale = pixels[i]; // Since mask is monochrome (R=G=B)
        if (grayscale > 127) {
            // Apply colormap RGB values
            pixels[i] = rVal;     // R
            pixels[i+1] = gVal;   // G
            pixels[i+2] = bVal;   // B
            pixels[i+3] = 255;    // Opacity
        } else {
            // Make black background fully transparent
            pixels[i+3] = 0;
        }
    }
    osCtx.putImageData(imgData, 0, 0);
}

function resetViewport() {
    zoomScale = 1.0;
    panX = 0;
    panY = 0;
    drawBothCanvases();
}

function drawBothCanvases() {
    if (!activeScanImg) return;

    // Sync canvas layouts
    resizeCanvas(canvasOriginal);
    resizeCanvas(canvasOverlay);

    // Draw original scan (Left)
    drawSingleCanvas(ctxOriginal, canvasOriginal, activeScanImg, null, 0);
    
    // Draw Overlay scan (Right)
    drawSingleCanvas(ctxOverlay, canvasOverlay, activeScanImg, offscreenMaskCanvas, maskOpacity);
}

function resizeCanvas(canvas) {
    const parent = canvas.parentElement;
    if (canvas.width !== parent.clientWidth || canvas.height !== parent.clientHeight) {
        canvas.width = parent.clientWidth;
        canvas.height = parent.clientHeight;
    }
}

function drawSingleCanvas(ctx, canvas, baseImg, overlayCanvas, overlayAlpha) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = false; // Keep diagnostic pixel lines sharp

    ctx.save();
    
    // Apply panning offsets & centering transformations
    ctx.translate(canvas.width / 2 + panX, canvas.height / 2 + panY);
    ctx.scale(zoomScale, zoomScale);

    // Draw the main axial slice CT scan
    const baseW = baseImg.width;
    const baseH = baseImg.height;
    ctx.drawImage(baseImg, -baseW / 2, -baseH / 2);

    // Overlay the colored tumor mask (Right side only)
    if (overlayCanvas && overlayAlpha > 0) {
        ctx.globalAlpha = overlayAlpha;
        ctx.drawImage(overlayCanvas, -baseW / 2, -baseH / 2);
    }

    ctx.restore();
}

// --- SYNCHRONIZED MOUSE VIEW MANIPULATION HANDLERS ---
function setupPacsControlHandlers() {
    
    // Opacity blending slider listener
    opacitySlider.addEventListener("input", (e) => {
        maskOpacity = parseFloat(e.target.value) / 100;
        opacityLabel.textContent = `${e.target.value}%`;
        drawBothCanvases();
    });

    // Colormap buttons click listeners
    document.querySelectorAll(".colormap-buttons button").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".colormap-buttons button").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            activeColorMap = btn.dataset.cmap;
            
            if (activeMaskImg) {
                setupOffscreenMask();
                drawBothCanvases();
            }
        });
    });

    // Reset View button listener
    document.getElementById("btn-reset-view").addEventListener("click", resetViewport);
    document.getElementById("btn-upload-new").addEventListener("click", hideWorkstationUI);

    // Dual mouse event bindings to synchronize transforms
    bindSyncPanZoomEvents(canvasOriginal);
    bindSyncPanZoomEvents(canvasOverlay);
    
    // Adapt to window resize
    window.addEventListener("resize", drawBothCanvases);
}

function bindSyncPanZoomEvents(canvas) {
    
    // Zoom controller (scroll wheel)
    canvas.addEventListener("wheel", (e) => {
        e.preventDefault();
        
        const zoomIntensity = 0.1;
        const wheelValue = e.deltaY;
        
        if (wheelValue < 0) {
            zoomScale += zoomIntensity;
        } else {
            zoomScale -= zoomIntensity;
        }
        
        // Boundaries
        zoomScale = Math.max(0.4, Math.min(6.0, zoomScale));
        
        // Update coord UI
        const percent = Math.round(zoomScale * 100);
        document.getElementById("coords-left").textContent = `Zoom: ${percent}%`;
        
        drawBothCanvases();
    });

    // Drag-to-pan controller
    canvas.addEventListener("mousedown", (e) => {
        isDragging = true;
        canvas.style.cursor = "grabbing";
        startDragX = e.clientX - panX;
        startDragY = e.clientY - panY;
    });

    window.addEventListener("mouseup", () => {
        if (isDragging) {
            isDragging = false;
            canvasOriginal.style.cursor = "grab";
            canvasOverlay.style.cursor = "grab";
        }
    });

    canvas.addEventListener("mousemove", (e) => {
        if (!isDragging) return;
        
        panX = e.clientX - startDragX;
        panY = e.clientY - startDragY;
        
        drawBothCanvases();
    });
}

// --- FILE UPLOADS & DRAG HANDLERS ---
function setupUploadHandlers() {
    
    btnBrowse.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            loadedFileName = file.name;
            analyzeScan(file, null);
        }
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "var(--accent-teal)";
        dropZone.style.background = "rgba(14, 165, 233, 0.05)";
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.style.borderColor = "rgba(14, 165, 233, 0.2)";
        dropZone.style.background = "rgba(6, 9, 19, 0.3)";
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "rgba(14, 165, 233, 0.2)";
        dropZone.style.background = "rgba(6, 9, 19, 0.3)";
        
        if (e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];
            loadedFileName = file.name;
            analyzeScan(file, null);
        }
    });

    sensitivitySlider.addEventListener("input", (e) => {
        sensitivityValue.textContent = `${parseFloat(e.target.value).toFixed(1)}x`;
    });
}

// --- CLINICAL PDF EXPORT EXECUTOR ---
function setupPdfExportHandler() {
    document.getElementById("btn-export-report").addEventListener("click", () => {
        // Set Patient metadata dynamically for printing
        const randomID = "PAT-" + (2026) + "-" + Math.floor(1000 + Math.random() * 9000);
        document.getElementById("print-patient-id").textContent = randomID;
        
        const now = new Date();
        document.getElementById("print-exam-date").textContent = now.toLocaleString();

        // Trigger system print window
        window.print();
    });
}
