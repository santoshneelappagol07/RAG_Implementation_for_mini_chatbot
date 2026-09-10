import React, { useState, useEffect, useRef } from 'react';
import { 
  FileText, Mail, KeyRound, ArrowRight, CheckCircle2, AlertCircle, 
  Clock, LogOut, Send, Sparkles, Copy, Check, UploadCloud, RefreshCw 
} from 'lucide-react';

const API_BASE = 'http://localhost:8000';

export default function App() {
  // ── Authentication state ──────────────────────────────────────────────
  const [token, setToken] = useState(() => localStorage.getItem('pdf_token') || '');
  const [userEmail, setUserEmail] = useState(() => localStorage.getItem('pdf_user_email') || '');
  const [sessionExpiryTime, setSessionExpiryTime] = useState(() => {
    const saved = localStorage.getItem('pdf_session_expiry');
    return saved ? parseInt(saved, 10) : 0;
  });
  const [sessionRemaining, setSessionRemaining] = useState(3600); // 1 hour default

  // ── Step 1 OTP Flow state ─────────────────────────────────────────────
  const [emailInput, setEmailInput] = useState('');
  const [otpSent, setOtpSent] = useState(false);
  const [otpDigits, setOtpDigits] = useState(['', '', '', '', '', '']);
  const [otpRemaining, setOtpRemaining] = useState(600); // 10 minutes default
  const [isSendingOtp, setIsSendingOtp] = useState(false);
  const [isVerifyingOtp, setIsVerifyingOtp] = useState(false);
  const [authError, setAuthError] = useState('');
  const [authSuccess, setAuthSuccess] = useState('');
  const [showManualToken, setShowManualToken] = useState(false);
  const [manualTokenInput, setManualTokenInput] = useState('');

  // ── Step 2 Document Q&A state ─────────────────────────────────────────
  const [documentId, setDocumentId] = useState(() => localStorage.getItem('last_doc_id') || '');
  const [question, setQuestion] = useState('');
  const [useStreaming, setUseStreaming] = useState(true);
  const [isAsking, setIsAsking] = useState(false);
  const [answerData, setAnswerData] = useState(null);
  const [streamedText, setStreamedText] = useState('');
  const [qaError, setQaError] = useState('');
  const [copied, setCopied] = useState(false);

  // ── PDF Upload modal/inline state ─────────────────────────────────────
  const [showUploader, setShowUploader] = useState(false);
  const [uploadFile, setUploadFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState('');

  const otpInputRefs = useRef([]);

  // ── Session Countdown Timer (1 Hour) ──────────────────────────────────
  useEffect(() => {
    if (!token) return;

    const timer = setInterval(() => {
      const now = Math.floor(Date.now() / 1000);
      if (sessionExpiryTime > 0) {
        const remaining = sessionExpiryTime - now;
        if (remaining <= 0) {
          handleLogout('Session expired (1 hour limit reached). Please verify OTP again.');
        } else {
          setSessionRemaining(remaining);
        }
      }
    }, 1000);

    return () => clearInterval(timer);
  }, [token, sessionExpiryTime]);

  // ── OTP Countdown Timer (10 Minutes) ──────────────────────────────────
  useEffect(() => {
    let interval = null;
    if (otpSent && otpRemaining > 0) {
      interval = setInterval(() => {
        setOtpRemaining((prev) => (prev > 0 ? prev - 1 : 0));
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [otpSent, otpRemaining]);

  // ── Handlers: OTP Flow ────────────────────────────────────────────────
  const handleSendOtp = async (e) => {
    e?.preventDefault();
    if (!emailInput || !emailInput.includes('@')) {
      setAuthError('Please enter a valid email address.');
      return;
    }

    setAuthError('');
    setAuthSuccess('');
    setIsSendingOtp(true);

    try {
      const res = await fetch(`${API_BASE}/auth/send-otp`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: emailInput.trim() }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Failed to send OTP email.');
      }

      setOtpSent(true);
      setOtpRemaining(data.expires_in_seconds || 600); // 10 minutes
      setAuthSuccess(`6-digit code sent to ${emailInput}. Valid for 10 minutes.`);
      setTimeout(() => otpInputRefs.current[0]?.focus(), 100);
    } catch (err) {
      setAuthError(err.message || 'Error communicating with server.');
    } finally {
      setIsSendingOtp(false);
    }
  };

  const handleOtpChange = (index, value) => {
    if (!/^\d*$/.test(value)) return;
    const newDigits = [...otpDigits];
    newDigits[index] = value.slice(-1);
    setOtpDigits(newDigits);

    // Auto-advance
    if (value && index < 5) {
      otpInputRefs.current[index + 1]?.focus();
    }

    // Auto-submit if all 6 filled
    if (newDigits.every((d) => d !== '') && index === 5) {
      handleVerifyOtp(newDigits.join(''));
    }
  };

  const handleOtpKeyDown = (index, e) => {
    if (e.key === 'Backspace' && !otpDigits[index] && index > 0) {
      otpInputRefs.current[index - 1]?.focus();
    }
  };

  const handleOtpPaste = (e) => {
    e.preventDefault();
    const pasted = e.clipboardData.getData('text').trim();
    if (/^\d{6}$/.test(pasted)) {
      const digits = pasted.split('');
      setOtpDigits(digits);
      handleVerifyOtp(pasted);
    }
  };

  const handleVerifyOtp = async (codeToVerify) => {
    const code = codeToVerify || otpDigits.join('');
    if (code.length !== 6) {
      setAuthError('Please enter the full 6-digit OTP code.');
      return;
    }

    setAuthError('');
    setIsVerifyingOtp(true);

    try {
      const res = await fetch(`${API_BASE}/auth/verify-otp`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: emailInput.trim(),
          otp: code,
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Invalid or expired OTP code.');
      }

      // Successful verification -> Store session (1 hour)
      const tokenReceived = data.token;
      const expiry = Math.floor(Date.now() / 1000) + (data.session_expires_in_seconds || 3600);

      localStorage.setItem('pdf_token', tokenReceived);
      localStorage.setItem('pdf_user_email', data.email);
      localStorage.setItem('pdf_session_expiry', expiry.toString());

      setToken(tokenReceived);
      setUserEmail(data.email);
      setSessionExpiryTime(expiry);
      setSessionRemaining(3600);
      setAuthSuccess('Authentication successful! Unlocking application...');
    } catch (err) {
      setAuthError(err.message || 'OTP verification failed.');
    } finally {
      setIsVerifyingOtp(false);
    }
  };

  const handleManualTokenSubmit = (e) => {
    e.preventDefault();
    if (!manualTokenInput.trim()) return;
    const expiry = Math.floor(Date.now() / 1000) + 3600;
    localStorage.setItem('pdf_token', manualTokenInput.trim());
    localStorage.setItem('pdf_user_email', 'token-user@app');
    localStorage.setItem('pdf_session_expiry', expiry.toString());
    setToken(manualTokenInput.trim());
    setUserEmail('authenticated-user');
    setSessionExpiryTime(expiry);
    setSessionRemaining(3600);
  };

  const handleLogout = (msg) => {
    localStorage.removeItem('pdf_token');
    localStorage.removeItem('pdf_user_email');
    localStorage.removeItem('pdf_session_expiry');
    setToken('');
    setUserEmail('');
    setSessionExpiryTime(0);
    setOtpSent(false);
    setOtpDigits(['', '', '', '', '', '']);
    setAuthError(msg || '');
    setAuthSuccess('');
  };

  // ── Handlers: Document & Questions ───────────────────────────────────
  const handleAsk = async (e) => {
    e?.preventDefault();
    if (!documentId.trim()) {
      setQaError('Please enter a Document ID.');
      return;
    }
    if (!question.trim()) {
      setQaError('Please enter a Question.');
      return;
    }

    setQaError('');
    setAnswerData(null);
    setStreamedText('');
    setIsAsking(true);
    localStorage.setItem('last_doc_id', documentId.trim());

    if (useStreaming) {
      // ── Streaming mode (/ask/stream) ──
      try {
        const response = await fetch(`${API_BASE}/ask/stream`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({
            document_id: documentId.trim(),
            question: question.trim(),
          }),
        });

        if (!response.ok) {
          const errData = await response.json();
          throw new Error(errData.detail || 'Streaming query failed.');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let accumulated = '';

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const dataContent = line.replace('data: ', '');
              if (dataContent === '[DONE]') {
                break;
              } else if (dataContent.startsWith('[ERROR]')) {
                throw new Error(dataContent.replace('[ERROR] ', ''));
              } else {
                accumulated += dataContent;
                setStreamedText(accumulated);
              }
            }
          }
        }

        setAnswerData({
          answer: accumulated,
          source: 'streamed_rag',
          cached: false,
          sources: [],
        });
      } catch (err) {
        setQaError(err.message || 'Streaming failed.');
      } finally {
        setIsAsking(false);
      }
    } else {
      // ── Standard sync mode (/ask) ──
      try {
        const res = await fetch(`${API_BASE}/ask`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({
            document_id: documentId.trim(),
            question: question.trim(),
          }),
        });

        const data = await res.json();
        if (!res.ok) {
          if (res.status === 401) {
            handleLogout('Session expired. Please log in again.');
            return;
          }
          throw new Error(data.detail || 'Failed to answer question.');
        }

        setAnswerData(data);
      } catch (err) {
        setQaError(err.message || 'Failed to get answer.');
      } finally {
        setIsAsking(false);
      }
    }
  };

  const handlePdfUpload = async (e) => {
    e.preventDefault();
    if (!uploadFile) return;

    setIsUploading(true);
    setUploadMessage('');

    const formData = new FormData();
    formData.append('file', uploadFile);

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Upload failed.');
      }

      const newId = data.document_id || data.id;
      setDocumentId(newId);
      localStorage.setItem('last_doc_id', newId);
      setUploadMessage(` Uploaded! Document ID: ${newId}`);
      setShowUploader(false);
    } catch (err) {
      setUploadMessage(`❌ ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const formatTimer = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
  };

  // ═════════════════════════════════════════════════════════════════════════
  // RENDER
  // ═════════════════════════════════════════════════════════════════════════

  return (
    <div className="app-container">
      {/* ── Top Header ──────────────────────────────────────────────── */}
      <header className="top-header">
        <div className="brand-section">
          <div className="brand-icon">
            <FileText size={22} />
          </div>
          <div>
            <h1 className="brand-title">PDF Chatbot AI</h1>
            <p className="brand-subtitle">MongoDB Atlas • Semantic Cache • Gemini RAG</p>
          </div>
        </div>

        {token && (
          <div className="header-status">
            <div
              className={`session-badge ${
                sessionRemaining < 300 ? 'warning' : ''
              }`}
              title="Session duration is 1 hour"
            >
              <Clock size={14} />
              <span>Session: {formatTimer(sessionRemaining)}</span>
            </div>
            <button
              id="logout-btn"
              onClick={() => handleLogout('Logged out successfully.')}
              className="btn-secondary"
              title="Logout"
            >
              <LogOut size={14} />
              <span>Logout</span>
            </button>
          </div>
        )}
      </header>

      {/* ── Stepper indicator ────────────────────────────────────────── */}
      <div className="steps-stepper">
        <div className={`step-item ${!token ? 'active' : 'completed'}`}>
          <span className="step-num">{token ? '✓' : '1'}</span>
          <span>Step 1: Email OTP Auth</span>
        </div>
        <div className="step-line" />
        <div className={`step-item ${token ? 'active' : ''}`}>
          <span className="step-num">2</span>
          <span>Step 2: Ask Document</span>
        </div>
      </div>

      {/* ── STEP 1: AUTHENTICATION SECTION ───────────────────────────── */}
      {!token ? (
        <div className="glass-card" id="step-1-auth-card">
          <div className="card-header">
            <h2 className="card-title">
              <KeyRound className="text-indigo-400" size={24} />
              Step 1: Email OTP Verification
            </h2>
            <p className="card-desc">
              Enter your email to receive a 6-digit verification code. OTP is valid for{' '}
              <strong style={{ color: 'var(--text-primary)' }}>10 minutes</strong>. Once verified, your session will remain active for{' '}
              <strong style={{ color: 'var(--text-primary)' }}>1 hour</strong> with direct access.
            </p>
          </div>

          {authError && (
            <div className="alert-box error" id="auth-error-alert">
              <AlertCircle size={18} />
              <span>{authError}</span>
            </div>
          )}

          {authSuccess && (
            <div className="alert-box success" id="auth-success-alert">
              <CheckCircle2 size={18} />
              <span>{authSuccess}</span>
            </div>
          )}

          {!otpSent ? (
            /* Email Input Form */
            <form onSubmit={handleSendOtp}>
              <div className="form-group">
                <label className="form-label" htmlFor="email-input">
                  <span>Work or Personal Email</span>
                  <span className="form-hint">OTP will be sent here</span>
                </label>
                <div className="input-wrapper">
                  <Mail className="input-icon" size={18} />
                  <input
                    id="email-input"
                    type="email"
                    required
                    placeholder="you@example.com"
                    value={emailInput}
                    onChange={(e) => setEmailInput(e.target.value)}
                    className="form-input"
                    disabled={isSendingOtp}
                  />
                </div>
              </div>

              <button
                id="send-otp-btn"
                type="submit"
                className="btn-primary"
                disabled={isSendingOtp || !emailInput}
              >
                {isSendingOtp ? (
                  <>
                    <RefreshCw className="spinner" size={18} />
                    <span>Sending Code...</span>
                  </>
                ) : (
                  <>
                    <span>Send Verification Code</span>
                    <ArrowRight size={18} />
                  </>
                )}
              </button>
            </form>
          ) : (
            /* 6-Digit OTP Verification Form */
            <div>
              <div className="form-group">
                <div className="form-label">
                  <span>Enter 6-Digit Code sent to {emailInput}</span>
                  <span
                    style={{
                      color: otpRemaining < 60 ? 'var(--accent-rose)' : 'var(--accent-amber)',
                      fontFamily: 'var(--font-mono)',
                    }}
                  >
                    Expires in: {formatTimer(otpRemaining)} (10 min total)
                  </span>
                </div>

                <div className="otp-input-row" onPaste={handleOtpPaste}>
                  {otpDigits.map((digit, idx) => (
                    <input
                      key={idx}
                      id={`otp-box-${idx}`}
                      ref={(el) => (otpInputRefs.current[idx] = el)}
                      type="text"
                      inputMode="numeric"
                      maxLength={1}
                      value={digit}
                      onChange={(e) => handleOtpChange(idx, e.target.value)}
                      onKeyDown={(e) => handleOtpKeyDown(idx, e)}
                      className="otp-box"
                      disabled={isVerifyingOtp}
                    />
                  ))}
                </div>
              </div>

              <button
                id="verify-otp-btn"
                onClick={() => handleVerifyOtp()}
                className="btn-primary"
                disabled={isVerifyingOtp || otpDigits.some((d) => !d)}
              >
                {isVerifyingOtp ? (
                  <>
                    <RefreshCw className="spinner" size={18} />
                    <span>Verifying Code & Generating Session...</span>
                  </>
                ) : (
                  <>
                    <CheckCircle2 size={18} />
                    <span>Verify & Access Application (1 Hour Session)</span>
                  </>
                )}
              </button>

              <div style={{ marginTop: '16px', display: 'flex', justifyContent: 'space-between' }}>
                <button
                  type="button"
                  onClick={() => setOtpSent(false)}
                  className="btn-secondary"
                  style={{ fontSize: '0.8rem' }}
                >
                  Change Email
                </button>
                <button
                  type="button"
                  onClick={handleSendOtp}
                  disabled={otpRemaining > 540} // 60s cooldown
                  className="btn-secondary"
                  style={{ fontSize: '0.8rem' }}
                >
                  Resend OTP
                </button>
              </div>
            </div>
          )}

          {/* Quick manual token override */}
          <div style={{ marginTop: '28px', borderTop: '1px solid var(--border-subtle)', paddingTop: '16px' }}>
            <button
              type="button"
              onClick={() => setShowManualToken(!showManualToken)}
              className="btn-secondary"
              style={{ fontSize: '0.78rem' }}
            >
              {showManualToken ? 'Hide Token Input' : 'Already have a Bearer Token? Enter directly'}
            </button>

            {showManualToken && (
              <form onSubmit={handleManualTokenSubmit} style={{ marginTop: '12px' }}>
                <div className="input-wrapper" style={{ marginBottom: '10px' }}>
                  <input
                    type="text"
                    placeholder="Paste UUID Bearer token..."
                    value={manualTokenInput}
                    onChange={(e) => setManualTokenInput(e.target.value)}
                    className="form-input no-icon"
                  />
                </div>
                <button type="submit" className="btn-secondary" style={{ width: '100%' }}>
                  Unlock Directly with Token
                </button>
              </form>
            )}
          </div>
        </div>
      ) : (
        /* ── STEP 2: DOCUMENT Q&A SECTION ───────────────────────────── */
        <div className="glass-card" id="step-2-qa-card">
          <div className="card-header">
            <h2 className="card-title">
              <Sparkles className="text-cyan-400" size={24} />
              Step 2: Ask Question on Document
            </h2>
            <p className="card-desc">
              Enter your <strong style={{ color: 'var(--text-primary)' }}>Document ID</strong> from MongoDB Atlas (e.g.{' '}
              <code style={{ background: 'rgba(255,255,255,0.08)', padding: '2px 6px', borderRadius: '4px' }}>
                6aa13ddca2e858b6612f9b9d
              </code>
              ) and your question to get instant RAG-powered answers.
            </p>
          </div>

          {qaError && (
            <div className="alert-box error" id="qa-error-alert">
              <AlertCircle size={18} />
              <span>{qaError}</span>
            </div>
          )}

          {/* Upload Drawer Toggle */}
          <div style={{ marginBottom: '20px', display: 'flex', justifyContent: 'flex-end' }}>
            <button
              id="toggle-upload-drawer-btn"
              type="button"
              onClick={() => setShowUploader(!showUploader)}
              className="btn-secondary"
            >
              <UploadCloud size={16} />
              <span>{showUploader ? 'Close PDF Uploader' : 'Need to Upload a New PDF?'}</span>
            </button>
          </div>

          {showUploader && (
            <form
              onSubmit={handlePdfUpload}
              style={{
                background: 'rgba(15, 23, 42, 0.8)',
                padding: '20px',
                borderRadius: 'var(--radius-md)',
                marginBottom: '20px',
                border: '1px dashed var(--primary)',
              }}
            >
              <h3 style={{ fontSize: '0.95rem', marginBottom: '8px' }}>Upload PDF to MongoDB Atlas</h3>
              <input
                id="pdf-file-input"
                type="file"
                accept=".pdf"
                required
                onChange={(e) => setUploadFile(e.target.files[0])}
                style={{ marginBottom: '12px', display: 'block', color: 'var(--text-secondary)' }}
              />
              <button
                type="submit"
                className="btn-primary"
                disabled={isUploading || !uploadFile}
                style={{ width: 'auto', padding: '8px 16px' }}
              >
                {isUploading ? 'Uploading & Indexing with FAISS...' : 'Upload & Get ID'}
              </button>
              {uploadMessage && (
                <p style={{ marginTop: '8px', fontSize: '0.85rem' }}>{uploadMessage}</p>
              )}
            </form>
          )}

          <form onSubmit={handleAsk}>
            {/* Document ID Input */}
            <div className="form-group">
              <label className="form-label" htmlFor="document-id-input">
                <span>Document ID</span>
                <span className="form-hint">MongoDB Atlas _id or integer</span>
              </label>
              <div className="input-wrapper">
                <input
                  id="document-id-input"
                  type="text"
                  required
                  placeholder="e.g. 6aa13ddca2e858b6612f9b9d"
                  value={documentId}
                  onChange={(e) => setDocumentId(e.target.value)}
                  className="form-input no-icon"
                />
              </div>
            </div>

            {/* Question Textarea */}
            <div className="form-group">
              <label className="form-label" htmlFor="question-input">
                <span>Your Question</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <label style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={useStreaming}
                      onChange={(e) => setUseStreaming(e.target.checked)}
                      style={{ marginRight: '4px' }}
                    />
                    Live SSE Stream
                  </label>
                </div>
              </label>
              <textarea
                id="question-input"
                required
                placeholder="e.g. explain the types of functions mentioned in this document..."
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                className="form-textarea"
                rows={3}
              />
            </div>

            <button
              id="ask-btn"
              type="submit"
              className="btn-primary"
              disabled={isAsking || !documentId || !question}
            >
              {isAsking ? (
                <>
                  <RefreshCw className="spinner" size={18} />
                  <span>Generating RAG Answer...</span>
                </>
              ) : (
                <>
                  <Send size={18} />
                  <span>Ask Question</span>
                </>
              )}
            </button>
          </form>

          {/* ── Answer / Response Section ── */}
          {(answerData || streamedText) && (
            <div className="response-container" id="answer-result-card">
              <div className="response-header">
                <div className="badges-group">
                  <span className="badge rag">
                    <Sparkles size={12} />
                    {answerData?.source || 'streaming'}
                  </span>

                  {answerData?.cached && (
                    <span className="badge cache">
                      <CheckCircle2 size={12} />
                      Cached
                    </span>
                  )}

                  {answerData?.similarity_score && (
                    <span className="badge semantic">
                      Match: {(answerData.similarity_score * 100).toFixed(1)}%
                    </span>
                  )}

                  {answerData?.sources && answerData.sources.length > 0 && (
                    <span className="badge sources">
                      Pages: {answerData.sources.join(', ')}
                    </span>
                  )}
                </div>

                <button
                  onClick={() => copyToClipboard(answerData?.answer || streamedText)}
                  className="btn-secondary"
                  style={{ padding: '4px 10px', fontSize: '0.8rem' }}
                >
                  {copied ? <Check size={14} /> : <Copy size={14} />}
                  <span>{copied ? 'Copied' : 'Copy'}</span>
                </button>
              </div>

              <div className="response-body">
                {answerData?.answer || streamedText}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
