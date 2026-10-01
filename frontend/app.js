/**
 * VeritasAI - Frontend Controller
 * Handles user interaction, live health checking, and REST API communication with FastAPI backend.
 */

// Configuration: Dynamic Backend URL resolution
const isLocalhost = Boolean(
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1" ||
  window.location.hostname === ""
);

// Default Render production backend URL
const DEFAULT_PRODUCTION_BACKEND_URL = "https://fake-news-backend.onrender.com";

function getApiBaseUrl() {
  const savedUrl = localStorage.getItem("VERITAS_API_URL");
  if (savedUrl && savedUrl.trim()) {
    return savedUrl.trim().replace(/\/+$/, "");
  }
  if (isLocalhost) {
    return "http://127.0.0.1:8000";
  }
  return DEFAULT_PRODUCTION_BACKEND_URL;
}

let API_BASE_URL = getApiBaseUrl();

// Pre-packaged Authentic Samples for Quick College Demonstrations
const SAMPLES = {
  real: {
    title: "U.S. military to accept transgender recruits on Monday: Pentagon",
    text: "WASHINGTON (Reuters) - Transgender people will be allowed for the first time to enlist in the U.S. military on Monday, the Pentagon said on Friday, after President Donald Trump's administration decided not to appeal court rulings barring his proposed ban. In August, Trump had issued a memorandum directing the military to ban transgender recruits. Multiple federal courts subsequently ruled against the policy, finding it likely violated the U.S. Constitution's guarantee of equal protection under the law."
  },
  fake: {
    title: "SHOCKING: Secret Documents Prove Alien Spaceship Landed at the Pentagon",
    text: "BREAKING BOMBSHELL: High-level anonymous whistleblowers have leaked classified footage revealing that an extraterrestrial flying disc landed directly on the Pentagon lawn. Deep state operatives immediately scrambled Black Ops helicopters to conceal the wreckage while mainstream media networks coordinated an unprecedented news blackout. Share this explosive report everywhere before it gets banned by the corrupt establishment!"
  }
};

// DOM Element Selectors
const elements = {
  form: document.getElementById("detection-form"),
  titleInput: document.getElementById("news-title"),
  textInput: document.getElementById("news-text"),
  charCount: document.getElementById("char-count"),
  wordCount: document.getElementById("word-count"),
  btnSubmit: document.getElementById("btn-submit"),
  btnText: document.getElementById("btn-text"),
  btnLoader: document.getElementById("btn-loader"),

  // Sample Buttons
  btnSampleReal: document.getElementById("btn-sample-real"),
  btnSampleFake: document.getElementById("btn-sample-fake"),
  btnClear: document.getElementById("btn-clear"),
  btnRetest: document.getElementById("btn-retest"),

  // State Views
  placeholderView: document.getElementById("state-placeholder"),
  loadingView: document.getElementById("state-loading"),
  resultView: document.getElementById("state-result"),
  loadingStatusText: document.getElementById("loading-status-text"),

  // Result Elements
  predictionBadge: document.getElementById("prediction-badge"),
  predictionIcon: document.getElementById("prediction-icon"),
  predictionTitle: document.getElementById("prediction-title"),
  confidenceVal: document.getElementById("confidence-percentage"),
  confidenceBar: document.getElementById("confidence-bar-fill"),
  decisionScoreVal: document.getElementById("decision-score-val"),
  inferenceTimeVal: document.getElementById("inference-time-val"),
  explanationText: document.getElementById("explanation-text"),

  // Alert Box
  errorAlert: document.getElementById("error-alert"),
  errorTitle: document.getElementById("error-title"),
  errorMessage: document.getElementById("error-message"),
  errorCloseBtn: document.getElementById("error-close-btn"),

  // Backend Status Indicator
  statusBadge: document.getElementById("backend-status-badge"),
  statusText: document.getElementById("backend-status-text"),

  // Modal Elements
  infoBtn: document.getElementById("project-info-btn"),
  modal: document.getElementById("specs-modal"),
  modalCloseBtn: document.getElementById("modal-close-btn"),
  modalOkBtn: document.getElementById("modal-ok-btn"),

  // API Configuration Elements
  apiConfigBtn: document.getElementById("api-config-btn"),
  apiModal: document.getElementById("api-modal"),
  apiModalCloseBtn: document.getElementById("api-modal-close-btn"),
  apiUrlInput: document.getElementById("api-url-input"),
  btnSaveApi: document.getElementById("btn-save-api"),
  btnResetApi: document.getElementById("btn-reset-api"),

  // Dynamic Footer Links
  footerLinkDocs: document.getElementById("footer-link-docs"),
  footerLinkHealth: document.getElementById("footer-link-health")
};

// =========================================================
// Initialization & Event Listeners
// =========================================================
document.addEventListener("DOMContentLoaded", () => {
  // Sync API URL and footer links
  syncApiEndpoints();

  // Check backend connectivity on startup
  checkBackendHealth();
  setInterval(checkBackendHealth, 15000); // Check every 15s

  // Textarea input monitoring
  elements.textInput.addEventListener("input", updateCounters);

  // Form submission
  elements.form.addEventListener("submit", handleFormSubmit);

  // Sample loader buttons
  elements.btnSampleReal.addEventListener("click", () => loadSample("real"));
  elements.btnSampleFake.addEventListener("click", () => loadSample("fake"));
  elements.btnClear.addEventListener("click", clearForm);
  elements.btnRetest.addEventListener("click", () => {
    elements.textInput.focus();
    elements.textInput.scrollIntoView({ behavior: "smooth", block: "center" });
  });

  // Alert dismiss
  elements.errorCloseBtn.addEventListener("click", hideError);

  // Specs Modal handlers
  elements.infoBtn.addEventListener("click", () => elements.modal.classList.remove("hidden"));
  elements.modalCloseBtn.addEventListener("click", () => elements.modal.classList.add("hidden"));
  elements.modalOkBtn.addEventListener("click", () => elements.modal.classList.add("hidden"));
  elements.modal.addEventListener("click", (e) => {
    if (e.target === elements.modal) elements.modal.classList.add("hidden");
  });

  // API Config Modal handlers
  if (elements.apiConfigBtn && elements.apiModal) {
    elements.apiConfigBtn.addEventListener("click", () => {
      elements.apiUrlInput.value = API_BASE_URL;
      elements.apiModal.classList.remove("hidden");
    });
    elements.apiModalCloseBtn.addEventListener("click", () => elements.apiModal.classList.add("hidden"));
    elements.apiModal.addEventListener("click", (e) => {
      if (e.target === elements.apiModal) elements.apiModal.classList.add("hidden");
    });
    elements.btnSaveApi.addEventListener("click", () => {
      const val = elements.apiUrlInput.value.trim().replace(/\/+$/, "");
      if (val) {
        localStorage.setItem("VERITAS_API_URL", val);
        API_BASE_URL = val;
        syncApiEndpoints();
        checkBackendHealth();
        elements.apiModal.classList.add("hidden");
      }
    });
    elements.btnResetApi.addEventListener("click", () => {
      localStorage.removeItem("VERITAS_API_URL");
      API_BASE_URL = isLocalhost ? "http://127.0.0.1:8000" : DEFAULT_PRODUCTION_BACKEND_URL;
      elements.apiUrlInput.value = API_BASE_URL;
      syncApiEndpoints();
      checkBackendHealth();
      elements.apiModal.classList.add("hidden");
    });
  }
});

function syncApiEndpoints() {
  if (elements.footerLinkDocs) elements.footerLinkDocs.href = `${API_BASE_URL}/docs`;
  if (elements.footerLinkHealth) elements.footerLinkHealth.href = `${API_BASE_URL}/health`;
}

// =========================================================
// Live Backend Health Monitor
// =========================================================
async function checkBackendHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, { method: "GET" });
    if (response.ok) {
      const data = await response.json();
      if (data.status === "ok") {
        setBackendStatus(true, "API Online (Model Ready)");
        return;
      }
    }
    setBackendStatus(false, "API Error");
  } catch (error) {
    setBackendStatus(false, "API Offline");
  }
}

function setBackendStatus(isOnline, message) {
  elements.statusBadge.className = `status-badge ${isOnline ? "online" : "offline"}`;
  elements.statusText.textContent = message;
}

// =========================================================
// Input Text Counters
// =========================================================
function updateCounters() {
  const text = elements.textInput.value;
  const charLength = text.length;
  const wordCount = text.trim() === "" ? 0 : text.trim().split(/\s+/).length;

  elements.charCount.textContent = `${charLength.toLocaleString()} characters`;
  elements.wordCount.textContent = `${wordCount.toLocaleString()} words`;
}

// =========================================================
// Sample Data Loaders
// =========================================================
function loadSample(type) {
  hideError();
  const sample = SAMPLES[type];
  if (!sample) return;

  elements.titleInput.value = sample.title;
  elements.textInput.value = sample.text;
  updateCounters();

  // Scroll smoothly to form
  elements.textInput.scrollIntoView({ behavior: "smooth", block: "center" });
}

function clearForm() {
  hideError();
  elements.titleInput.value = "";
  elements.textInput.value = "";
  updateCounters();
  showState("placeholder");
  elements.textInput.focus();
}

// =========================================================
// State View Management (Placeholder -> Loading -> Result)
// =========================================================
function showState(state) {
  elements.placeholderView.classList.add("hidden");
  elements.loadingView.classList.add("hidden");
  elements.resultView.classList.add("hidden");

  if (state === "placeholder") elements.placeholderView.classList.remove("hidden");
  if (state === "loading") elements.loadingView.classList.remove("hidden");
  if (state === "result") elements.resultView.classList.remove("hidden");
}

// =========================================================
// Form Submit & Prediction Pipeline
// =========================================================
async function handleFormSubmit(e) {
  e.preventDefault();
  hideError();

  const text = elements.textInput.value.trim();
  const title = elements.titleInput.value.trim();

  // 1. Frontend validation
  if (!text) {
    showError("Missing Article Text", "Please paste or type article content to evaluate.");
    elements.textInput.focus();
    return;
  }

  if (text.length < 20) {
    showError("Text Too Short", "Articles shorter than 20 characters do not contain enough linguistic context for reliable evaluation.");
    return;
  }

  // 2. Prepare Loading UI
  setButtonLoading(true);
  showState("loading");

  // Animate dynamic loader messages
  const statusCycle = [
    "Vectorizing text with 100,000 TF-IDF features...",
    "Computing LinearSVC hyperplane distance...",
    "Estimating calibrated posterior confidence..."
  ];
  let cycleIdx = 0;
  const statusInterval = setInterval(() => {
    cycleIdx = (cycleIdx + 1) % statusCycle.length;
    elements.loadingStatusText.textContent = statusCycle[cycleIdx];
  }, 350);

  const startTime = performance.now();

  try {
    // 3. Send payload to FastAPI endpoint
    const payload = { text };
    if (title) payload.title = title;

    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const elapsed = Math.round(performance.now() - startTime);
    clearInterval(statusInterval);

    // 4. Handle Server Response
    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      const msg = errData.detail || errData.error || `Server returned status ${response.status}`;
      throw new Error(msg);
    }

    const data = await response.json();
    renderPredictionResult(data, elapsed);

  } catch (error) {
    clearInterval(statusInterval);
    showState("placeholder");
    showError(
      "Classification Failed",
      error.message.includes("Failed to fetch")
        ? `Cannot connect to backend API at ${API_BASE_URL}. Ensure the FastAPI server is running with 'python -m uvicorn backend.main:app --port 8000'.`
        : error.message
    );
  } finally {
    setButtonLoading(false);
  }
}

// =========================================================
// Render Prediction Results
// =========================================================
function renderPredictionResult(data, latencyMs) {
  showState("result");

  const isReal = data.prediction.toLowerCase() === "real";
  const confidencePercent = (data.confidence * 100).toFixed(2);
  const decisionScore = data.decision_score !== undefined ? (data.decision_score > 0 ? `+${data.decision_score.toFixed(4)}` : data.decision_score.toFixed(4)) : "N/A";

  // Update Classification Badge
  if (isReal) {
    elements.predictionBadge.className = "prediction-badge badge-real";
    elements.predictionIcon.className = "fa-solid fa-circle-check";
    elements.predictionTitle.textContent = "REAL NEWS VERIFIED";

    elements.confidenceBar.className = "confidence-bar-fill fill-real";
    elements.confidenceVal.style.color = "var(--color-real)";

    elements.explanationText.textContent =
      "The language displays standard journalistic syntax, formal factual attribution, and balanced vocabulary strongly consistent with verified, reputable news wire coverage.";
  } else {
    elements.predictionBadge.className = "prediction-badge badge-fake";
    elements.predictionIcon.className = "fa-solid fa-triangle-exclamation";
    elements.predictionTitle.textContent = "FAKE NEWS DETECTED";

    elements.confidenceBar.className = "confidence-bar-fill fill-fake";
    elements.confidenceVal.style.color = "var(--color-fake)";

    elements.explanationText.textContent =
      "The text exhibits high frequencies of emotive vocabulary, sensationalized phrasing, or informal narrative framing characteristic of clickbait and manufactured disinformation.";
  }

  // Update Metric Figures
  elements.confidenceVal.textContent = `${confidencePercent}%`;
  elements.confidenceBar.style.width = "0%";
  setTimeout(() => {
    elements.confidenceBar.style.width = `${confidencePercent}%`;
  }, 50);

  elements.decisionScoreVal.textContent = decisionScore;
  elements.inferenceTimeVal.textContent = `${latencyMs}ms`;

  // Scroll to results on mobile devices
  if (window.innerWidth <= 992) {
    elements.resultView.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

// =========================================================
// UI Helpers
// =========================================================
function setButtonLoading(isLoading) {
  elements.btnSubmit.disabled = isLoading;
  if (isLoading) {
    elements.btnText.parentElement.classList.add("hidden");
    elements.btnLoader.classList.remove("hidden");
  } else {
    elements.btnText.parentElement.classList.remove("hidden");
    elements.btnLoader.classList.add("hidden");
  }
}

function showError(title, message) {
  elements.errorTitle.textContent = title;
  elements.errorMessage.textContent = message;
  elements.errorAlert.classList.remove("hidden");
  elements.errorAlert.scrollIntoView({ behavior: "smooth", block: "center" });
}

function hideError() {
  elements.errorAlert.classList.add("hidden");
}
