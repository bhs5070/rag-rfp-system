const questionInput = document.getElementById("question");
const askButton = document.getElementById("ask-button");
const topKInput = document.getElementById("top-k");
const topKValue = document.getElementById("top-k-value");
const statusMessage = document.getElementById("status-message");
const answerBox = document.getElementById("answer-box");
const sourcesList = document.getElementById("sources-list");
const sourcesCount = document.getElementById("sources-count");
const chatModelBadge = document.getElementById("chat-model-badge");

const healthStatus = document.getElementById("health-status");
const healthCollection = document.getElementById("health-collection");
const healthCount = document.getElementById("health-count");

topKInput.addEventListener("input", () => {
  topKValue.textContent = topKInput.value;
});

function renderSources(contexts) {
  sourcesList.innerHTML = "";
  sourcesCount.textContent = `${contexts.length} chunks`;

  for (const context of contexts) {
    const card = document.createElement("article");
    card.className = "source-card";
    card.innerHTML = `
      <div class="source-top">
        <strong>${context.doc_id ?? "-"}</strong>
        <span class="source-meta">chunk ${context.chunk_index ?? "-"} | score ${Number(context.score ?? 0).toFixed(4)}</span>
      </div>
      <div class="source-text">${context.text ?? ""}</div>
    `;
    sourcesList.appendChild(card);
  }
}

async function loadHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    healthStatus.textContent = data.status;
    healthCollection.textContent = data.collection_name;
    healthCount.textContent = `${data.chunk_count} chunks`;
    chatModelBadge.textContent = `model: ${data.chat_model}`;
  } catch (error) {
    healthStatus.textContent = "error";
    healthCollection.textContent = "-";
    healthCount.textContent = "-";
  }
}

async function askQuestion() {
  const query = questionInput.value.trim();
  if (!query) {
    statusMessage.textContent = "질문을 입력하세요.";
    questionInput.focus();
    return;
  }

  askButton.disabled = true;
  statusMessage.textContent = "질문을 처리하는 중입니다...";
  answerBox.textContent = "";
  sourcesList.innerHTML = "";
  sourcesCount.textContent = "0 chunks";

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        top_k: Number(topKInput.value),
      }),
    });

    if (!response.ok) {
      throw new Error(`Request failed: ${response.status}`);
    }

    const data = await response.json();
    answerBox.textContent = data.answer ?? "";
    chatModelBadge.textContent = `model: ${data.model}`;
    renderSources(data.contexts ?? []);
    statusMessage.textContent = "응답 생성 완료.";
  } catch (error) {
    statusMessage.textContent = "요청 처리 중 오류가 발생했습니다.";
    answerBox.textContent = String(error);
  } finally {
    askButton.disabled = false;
  }
}

askButton.addEventListener("click", askQuestion);
questionInput.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    askQuestion();
  }
});

loadHealth();
