const DOMAINS = [
  { id: "gst",        label: "GST",         dot: "#2563EB", desc: "cbic.gov.in" },
  { id: "rbi",        label: "RBI",         dot: "#7C3AED", desc: "rbi.org.in" },
  { id: "sebi",       label: "SEBI",        dot: "#DB2777", desc: "sebi.gov.in" },
  { id: "mca",        label: "MCA",         dot: "#D97706", desc: "mca.gov.in" },
  { id: "income_tax", label: "Income Tax",  dot: "#059669", desc: "incometaxindia.gov.in" },
];

export default function DomainSelector({ selected, onChange }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
      {DOMAINS.map((d) => (
        <button key={d.id} onClick={() => onChange(d.id)}
          style={{ display: "flex", alignItems: "center", gap: "10px", padding: "9px 12px", borderRadius: "9px", border: selected === d.id ? "1.5px solid #0F766E" : "1.5px solid transparent", backgroundColor: selected === d.id ? "rgba(15,118,110,0.12)" : "rgba(255,255,255,0.35)", cursor: "pointer", textAlign: "left", width: "100%" }}>
          <span style={{ width: "8px", height: "8px", borderRadius: "50%", backgroundColor: d.dot, flexShrink: 0 }} />
          <div>
            <div style={{ fontFamily: "DM Sans, sans-serif", fontWeight: selected === d.id ? 600 : 400, fontSize: "0.875rem", color: selected === d.id ? "#0F766E" : "#1A2E2B" }}>
              {d.label}
            </div>
            <div style={{ fontFamily: "DM Mono, monospace", fontSize: "0.68rem", color: "#6B8E89" }}>
              {d.desc}
            </div>
          </div>
        </button>
      ))}
    </div>
  );
}