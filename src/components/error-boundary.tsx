'use client';

import React from 'react';

export class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error?: Error }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('ErrorBoundary caught:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="page" style={{ padding: '2rem', textAlign: 'center' }}>
          <h1 className="page-title">Something went wrong.</h1>
          <p className="muted" style={{ marginBottom: '1rem' }}>
            {this.state.error?.message}
          </p>
          <button
            className="button"
            onClick={() => {
              this.setState({ hasError: false });
              window.location.reload();
            }}
            type="button"
          >
            Reload page
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
