import { useState } from "react";
import axios from "axios";
import "./index.css";

import DomainSelector  from "./components/DomainSelector";
import QueryBox        from "./components/QueryBox";
import ProcessingTags  from "./components/ProcessingTags";
import AnswerCard      from "./components/AnswerCard";
import SourceCitations from "./components/SourceCitations";
import ConfidenceScore from "./components/ConfidenceScore";
import DeadlineTimeline from "./components/DeadlineTimeline";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [selectedDomain, setSelectedDomain] = useState("all");
  const [isLoading, setIsLoading]           = useState(false);
  const [result, setResult]                 = useState(null);
  const [error, setError]                   = useState(null);
  const [recentQueries, setRecentQueries]   = useState([]);

  const handleQuery = async (query) => {
    if (!query.trim()) return;

    setIsLoading(true);
    setResult(null);
    setError(null);

    try {
      const response = await axios.post(`${API_BASE}/query`, {
        query,
        domain: selectedDomain,
      });

      setResult(response.data);
      setRecentQueries((prev) => [
        { query, domain: selectedDomain, time: new Date() },
        ...prev.slice(0, 9),
      ]);
    } catch (err) {
      const msg =
        err.response?.data?.detail ||
        err.message ||
        "Something went wrong. Please try again.";
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  };

  const answer          = result?.answer;
  const processingSteps = result?.processing_steps || [];

  return (
    <div style={{ minHeight: "100vh", backgroundColor: "#E4F0EE", display: "flex", flexDirection: "column" }}>

      {/* ── Header ── */}
      <header style={{ backgroundColor: "#0F766E", padding: "12px 32px", display: "flex", alignItems: "center", gap: "12px" }}>
        <span style={{ fontSize: "22px" }}>⚖️</span>
        <h1 style={{ fontFamily: "Inter, sans-serif", fontSize: "1.25rem", fontWeight: 800, color: "#fff", letterSpacing: "-0.02em" }}>
          FinComply AI
        </h1>
        <span style={{ fontFamily: "DM Sans, sans-serif", fontSize: "0.75rem", color: "#A7F3D0", backgroundColor: "rgba(255,255,255,0.15)", padding: "2px 8px", borderRadius: "99px" }}>
          3-Agent · Fine-Tuned Mistral · India Regulatory Intelligence
        </span>
      </header>

      {/* ── 3-Column Layout ── */}
      <div style={{ display: "grid", gridTemplateColumns: "260px 1fr 300px", flex: 1, height: "calc(100vh - 54px)", overflow: "hidden" }}>

        {/* ── Left Sidebar ── */}
        <aside style={{ backgroundColor: "#C8E0DB", borderRight: "1px solid #A8CCCC", padding: "20px 16px", overflowY: "auto" }}>
          <p style={{ fontFamily: "DM Mono, monospace", fontSize: "0.7rem", color: "#6B8E89", letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: "12px" }}>
            Regulatory Domain
          </p>
          <DomainSelector selected={selectedDomain} onChange={setSelectedDomain} />

          {recentQueries.length > 0 && (
            <div style={{ marginTop: "28px" }}>
              <p style={{ fontFamily: "DM Mono, monospace", fontSize: "0.7rem", color: "#6B8E89", letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: "10px" }}>
                Recent Queries
              </p>
              {recentQueries.map((q, i) => (
                <button key={i} onClick={() => handleQuery(q.query)}
                  style={{ display: "block", width: "100%", textAlign: "left", background: "rgba(255,255,255,0.4)", border: "none", borderRadius: "8px", padding: "8px 10px", marginBottom: "6px", cursor: "pointer", fontSize: "0.8rem", color: "#1A2E2B", fontFamily: "DM Sans, sans-serif" }}>
                  {q.query.slice(0, 50)}{q.query.length > 50 ? "…" : ""}
                </button>
              ))}
            </div>
          )}
        </aside>

        {/* ── Main Center ── */}
        <main style={{ padding: "28px 32px", overflowY: "auto" }}>
          <div style={{ marginBottom: "24px" }}>
            <h2 style={{ fontFamily: "Inter, sans-serif", fontSize: "1.75rem", fontWeight: 800, color: "#1A2E2B", marginBottom: "6px" }}>
              India Financial Regulatory Intelligence
            </h2>
            <p style={{ color: "#3D5A56", fontSize: "0.9375rem" }}>
              Ask any GST, RBI, SEBI, or MCA compliance question. Answers verified against official government circulars only.
            </p>
          </div>

          <QueryBox onSubmit={handleQuery} isLoading={isLoading} />

          {(isLoading || processingSteps.length > 0) && (
            <ProcessingTags steps={processingSteps} isLoading={isLoading} />
          )}

          {result?.rejected && (
            <div style={{ backgroundColor: "#FEE2E2", border: "1px solid #FCA5A5", borderRadius: "10px", padding: "14px 18px", marginTop: "16px", color: "#991B1B", fontSize: "0.9rem" }}>
              ⚠️ {result.rejection_reason}
            </div>
          )}

          {error && (
            <div style={{ backgroundColor: "#FEE2E2", border: "1px solid #FCA5A5", borderRadius: "10px", padding: "14px 18px", marginTop: "16px", color: "#991B1B", fontSize: "0.9rem" }}>
              ✗ Error: {error}
            </div>
          )}

          {answer && !result?.rejected && (
            <>
              <AnswerCard answer={answer} responseTimeMs={result?.response_time_ms} />
              <SourceCitations answer={answer} />
            </>
          )}
        </main>

        {/* ── Right Panel ── */}
        <aside style={{ backgroundColor: "#C8E0DB", borderLeft: "1px solid #A8CCCC", padding: "20px 16px", overflowY: "auto" }}>
          {answer && !result?.rejected ? (
            <>
              <ConfidenceScore answer={answer} />
              <DeadlineTimeline deadlines={answer.deadlines || []} />
              <div style={{ marginTop: "20px" }}>
                <p style={{ fontFamily: "DM Mono, monospace", fontSize: "0.7rem", color: "#6B8E89", letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: "10px" }}>
                  Query Stats
                </p>
                <StatRow label="Domain"        value={(answer.domain || "—").toUpperCase()} />
                <StatRow label="Response time" value={`${result?.response_time_ms?.toFixed(0)} ms`} />
                <StatRow label="Agents run"    value="3 (Research · Critic · Supervisor)" />
                <StatRow label="Gov verified"  value={answer.is_gov_verified ? "✓ Yes" : "✗ No"} />
              </div>
            </>
          ) : (
            <div style={{ color: "#6B8E89", fontSize: "0.875rem", paddingTop: "8px" }}>
              <p style={{ fontFamily: "Syne, sans-serif", fontWeight: 600, color: "#3D5A56", marginBottom: "8px" }}>How it works</p>
              <p style={{ marginBottom: "8px" }}>1. Ask a compliance question</p>
              <p style={{ marginBottom: "8px" }}>2. Research Agent fetches live circulars</p>
              <p style={{ marginBottom: "8px" }}>3. Critic Agent verifies sources</p>
              <p>4. Supervisor delivers verified answer</p>
            </div>
          )}
        </aside>

      </div>
    </div>
  );
}

function StatRow({ label, value }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", padding: "6px 0", borderBottom: "1px solid rgba(168,204,204,0.4)", gap: "8px" }}>
      <span style={{ fontSize: "0.78rem", color: "#6B8E89", flexShrink: 0 }}>{label}</span>
      <span style={{ fontSize: "0.78rem", color: "#1A2E2B", fontWeight: 500, textAlign: "right" }}>{value}</span>
    </div>
  );
}