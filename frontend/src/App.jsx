import { useState, useCallback } from 'react'
import './index.css'
import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || ''

const TYPE_ICONS = {
  K: '🧠',
  P: '💬',
  A: '🎯',
  B: '📊',
  S: '🗣️',
  default: '📋',
}

function getBadgeClass(type) {
  if (type === 'K') return 'badge badge-k'
  if (type === 'P') return 'badge badge-p'
  return 'badge badge-other'
}

function getCardIcon(testTypes = []) {
  if (!testTypes.length) return '📋'
  const t = testTypes[0]
  return TYPE_ICONS[t] || TYPE_ICONS.default
}

function AssessmentCard({ item, rank }) {
  const delay = `${rank * 60}ms`
  const scorePercent = item.score ? Math.round(item.score * 100) : null
  const testTypes = typeof item.test_type === 'string' ? item.test_type.split(',').map(t => t.trim()) : (item.test_type || [])

  return (
    <div className="assessment-card animate-in" style={{ animationDelay: delay }}>
      <span className="card-rank">#{rank}</span>

      <div className="card-top">
        <div className="card-icon">{getCardIcon(testTypes)}</div>
        <div style={{ flex: 1 }}>
          <div className="card-name">{item.assessment_name}</div>
          <a
            className="card-url"
            href={item.assessment_url}
            target="_blank"
            rel="noopener noreferrer"
          >
            {item.assessment_url}
          </a>
        </div>
        {scorePercent !== null && (
          <div style={{
            background: 'rgba(99,102,241,0.15)',
            border: '1px solid rgba(99,102,241,0.3)',
            borderRadius: '8px',
            padding: '4px 10px',
            fontSize: '13px',
            fontWeight: 700,
            color: '#a5b4fc',
            whiteSpace: 'nowrap',
          }}>
            {scorePercent}%
          </div>
        )}
      </div>

      {item.explanation && (
        <p style={{
          fontSize: '13px',
          color: 'var(--text-secondary)',
          lineHeight: 1.5,
          margin: '8px 0',
          paddingLeft: '54px',
          fontStyle: 'italic',
        }}>
          {item.explanation}
        </p>
      )}

      <div className="card-meta">
        {testTypes.map((t) => (
          <span key={t} className={getBadgeClass(t)}>
            {t === 'K' ? '🧠 Knowledge & Skills' : t === 'P' ? '💬 Personality' : `📌 ${t}`}
          </span>
        ))}

        {item.duration_minutes && (
          <span className="badge badge-duration">
            ⏱ {item.duration_minutes} min
          </span>
        )}

        {item.remote_testing && (
          <span className="badge badge-remote">✅ Remote Testing</span>
        )}

        {item.adaptive_irt && (
          <span className="badge badge-remote">⚡ Adaptive IRT</span>
        )}
      </div>
    </div>
  )
}

const SAMPLE_QUERIES = [
  'I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment completable in 40 minutes.',
  'Looking to hire mid-level professionals proficient in Python, SQL and JavaScript. Max 60 minutes.',
  'I want to hire new graduates for a sales role. Budget is about an hour per test.',
  'Hiring a Senior Data Analyst with 5 years of experience in SQL, Excel and Python. Assessment can be 1-2 hours.',
]

export default function App() {
  const [tab, setTab] = useState('query')   // 'query' | 'jd' | 'url'
  const [query, setQuery] = useState('')
  const [jdText, setJdText] = useState('')
  const [jdUrl, setJdUrl] = useState('')
  const [balance, setBalance] = useState(50)
  const [loading, setLoading] = useState(false)
  const [loadingMsg, setLoadingMsg] = useState('Analyzing...')
  const [results, setResults] = useState(null)
  const [error, setError] = useState('')

  const inputLength = tab === 'query' ? query.length : tab === 'jd' ? jdText.length : jdUrl.length
  const canSubmit = !loading && (
    (tab === 'query' && query.trim().length > 3) ||
    (tab === 'jd' && jdText.trim().length > 10) ||
    (tab === 'url' && jdUrl.trim().startsWith('http'))
  )

  const handleSubmit = useCallback(async () => {
    setLoading(true)
    setError('')
    setResults(null)

    const body = {}
    if (tab === 'query') body.query = query.trim()
    else if (tab === 'jd') body.jd_text = jdText.trim()
    else if (tab === 'url') body.jd_url = jdUrl.trim()

    try {
      // Pre-ping health to wake Render free tier (cold start can take 30-60s)
      setLoadingMsg('Waking up server...')
      try { await axios.get(`${API_BASE}/health`, { timeout: 30000 }) } catch (_) {}

      setLoadingMsg('Analyzing...')
      const { data } = await axios.post(`${API_BASE}/recommend`, body, {
        headers: { 'Content-Type': 'application/json' },
        timeout: 240000,
      })
      setResults(data.recommendations || [])
    } catch (err) {
      const msg = err?.response?.data?.detail || err.message || 'Unknown error'
      setError(`Request failed: ${msg}`)
    } finally {
      setLoading(false)
      setLoadingMsg('Analyzing...')
    }
  }, [tab, query, jdText, jdUrl])

  const loadSample = () => {
    setTab('query')
    setQuery(SAMPLE_QUERIES[Math.floor(Math.random() * SAMPLE_QUERIES.length)])
  }

  return (
    <div className="app">
      <header className="hero">
        <div className="hero-badge">
          <span className="hero-badge-dot" />
          SHL AI Recommendation Engine
        </div>
        <h1>Find the Perfect<br />Assessment, Instantly</h1>
        <p className="hero-sub">
          Paste a job description, plain query, or JD URL — get the most relevant
          SHL Individual Test Solutions ranked by AI.
        </p>
      </header>

      <div className="form-card">
        <div className="tab-row">
          {[
            { id: 'query', label: '✍️ Quick Query' },
            { id: 'jd',    label: '📄 Job Description' },
            { id: 'url',   label: '🔗 JD URL' },
          ].map(({ id, label }) => (
            <button
              key={id}
              className={`tab-btn ${tab === id ? 'active' : ''}`}
              onClick={() => { setTab(id); setResults(null); setError('') }}
            >
              {label}
            </button>
          ))}
        </div>

        {tab === 'query' && (
          <>
            <label className="form-label">Hiring Query</label>
            <textarea
              className="form-textarea"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. I am hiring Java developers who can collaborate with business teams. Max 40 minutes."
              rows={5}
            />
          </>
        )}

        {tab === 'jd' && (
          <>
            <label className="form-label">Job Description Text</label>
            <textarea
              className="form-textarea"
              value={jdText}
              onChange={(e) => setJdText(e.target.value)}
              placeholder="Paste the full job description here..."
              rows={8}
              style={{ minHeight: '200px' }}
            />
          </>
        )}

        {tab === 'url' && (
          <>
            <label className="form-label">Job Description URL</label>
            <input
              className="form-input"
              type="url"
              value={jdUrl}
              onChange={(e) => setJdUrl(e.target.value)}
              placeholder="https://example.com/jobs/senior-developer"
            />
          </>
        )}

        <div className="balance-row">
          <label className="form-label">Assessment Balance</label>
          <input
            type="range"
            min={0} max={100}
            value={balance}
            onChange={(e) => setBalance(Number(e.target.value))}
            title="Slide to weight technical vs behavioral assessments"
          />
          <div className="balance-labels">
            <span>🧠 Pure Technical (K)</span>
            <span style={{ color: balance > 40 && balance < 60 ? '#a5b4fc' : 'inherit' }}>⚖️ Balanced</span>
            <span>💬 Pure Behavioral (P)</span>
          </div>
        </div>

        <div className="submit-row">
          <button
            style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '13px' }}
            onClick={loadSample}
          >
            Try a sample query →
          </button>
          <button
            className="btn-primary"
            onClick={handleSubmit}
            disabled={!canSubmit}
            id="recommend-btn"
          >
            {loading ? (
              <>
                <div className="spinner" />
                {loadingMsg}
              </>
            ) : (
              <>
                ✨ Get Recommendations
              </>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="error-banner">
          ⚠️ {error}
        </div>
      )}

      {results !== null && (
        results.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">🔍</div>
            <h3>No assessments found</h3>
            <p>Try rephrasing your query or using a more detailed job description.</p>
          </div>
        ) : (
          <section>
            <div className="results-header">
              <h2 className="results-title">Recommended Assessments</h2>
              <span className="results-count">{results.length} results</span>
            </div>
            <div className="assessment-grid">
              {results.map((item, i) => (
                <AssessmentCard key={item.assessment_url || i} item={item} rank={i + 1} />
              ))}
            </div>
          </section>
        )
      )}

      {results === null && !loading && (
        <div className="empty-state">
          <div className="empty-icon">🎯</div>
          <h3>Ready to recommend</h3>
          <p>Enter a hiring query above and click <strong>Get Recommendations</strong>.</p>
        </div>
      )}

      <footer className="footer">
        <p>Powered by Gemini 2.5 Flash · FAISS · BM25 · FastAPI · React</p>
        <p style={{ marginTop: '6px' }}>SHL Assessment Recommendation System &copy; 2025</p>
      </footer>
    </div>
  )
}
