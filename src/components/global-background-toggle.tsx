'use client';

import { useVisualBackground } from '@/lib/visual-background-provider';

export function GlobalBackgroundToggle() {
  const { backgroundEnabled, toggleBackgroundEnabled } = useVisualBackground();

  return (
    <button
      aria-label={backgroundEnabled ? 'Turn off animated background' : 'Turn on animated background'}
      className={`background-toggle-btn${backgroundEnabled ? '' : ' is-off'}`}
      onClick={toggleBackgroundEnabled}
      title={backgroundEnabled ? 'Turn off animated background' : 'Turn on animated background'}
      type="button"
    >
      <svg aria-hidden="true" viewBox="0 0 24 24">
        <path
          d="M4.5 12c1.9-3.4 4.8-5.1 7.5-5.1s5.6 1.7 7.5 5.1c-1.9 3.4-4.8 5.1-7.5 5.1S6.4 15.4 4.5 12Z"
          fill="none"
          stroke="currentColor"
          strokeLinejoin="round"
          strokeWidth="1.7"
        />
        <circle cx="12" cy="12" fill="none" r="2.9" stroke="currentColor" strokeWidth="1.7" />
        {!backgroundEnabled ? (
          <path d="M5 5l14 14" fill="none" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
        ) : null}
      </svg>
    </button>
  );
}
